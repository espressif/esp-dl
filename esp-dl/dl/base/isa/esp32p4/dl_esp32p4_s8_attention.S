############################################################################################################################################################
####
#### dl_esp32p4_s8_attention_exp_lut_shift
####
############################################################################################################################################################
# int32_t dl_esp32p4_s8_attention_exp_lut_shift(int8_t *e, const int32_t *dots,
#     int n, int32_t m_new, int32_t sat, int32_t shift, const int32_t *table,
#     int32_t *scratch);
# idx = min(m_new - dots[t], sat) >> shift; e[t] = table[idx]. Returns
# sum(e[0:n]). The caller absorbs the scale into a 512-entry table where
# entry i stands for diff == i << shift.
# Single fused pass (idx + vector gather + store). ESP.LDXQ.32 scales its
# index by 4, so each [0,127] exp value occupies one int32 slot and idx gathers
# it directly. An int8 table would instead need idx>>2 to address the word,
# then (idx & 3) shifted in to select the byte -- 4 extra instructions per 4
# tokens. The older 3-pass path wrote int32 indices to scratch and
# scalar-gathered them; that scratch R/W + dependent lbu chain was the
# softmax hot spot.
# e must be 16-byte-aligned (64-byte row). dots is TILE_K=64 wide.
# scratch holds 16 bytes of broadcast constants only.


    .text
    .global dl_esp32p4_s8_attention_exp_lut_shift
    .type   dl_esp32p4_s8_attention_exp_lut_shift, @function
    .align 4

# One quad: 4 int32 dots at a1 -> 4 gathered table entries, one per 32-bit
# lane of \Rj. Entries are 0..127 so only the low byte is ever set; the two
# VUNZIPs in the caller read even bytes and discard the rest for free.
# LDXQ takes its index from bits [15:0] of the selected 16-bit slot, i.e. the
# low half of each 32-bit lane, and scales it by 4 -- which is exactly the
# int32 table stride, so idx needs no adjustment. idx <= 511 always fits that
# 16-bit field.
# q2 carries diff and then idx; \Rj holds the clamped value and is then
# overwritten by the gather, so a quad needs just one scratch register.
# The four LDXQ share a destination QR and an index register, which looks
# like a latency chain worth software-pipelining. It is not: interleaving two
# independent quads (second index register, results in separate QRs) moved
# 5.37 -> 5.13 cycles/token, so the cost is LDXQ gather throughput -- one
# 4-byte indexed load per token -- not latency. Reverted; do not retry.
# Reads q0=m_new, t4=table, a4=sat, s0=0, SAR=shift.


.macro exp_quad Rj
    esp.vld.128.ip   q2, a1, 16
    esp.vsub.s32     q2, q0, q2        # diff = m_new - dot
    esp.vsat.s32     \Rj, q2, s0, a4
    # clamp to [0, sat]; s0 holds 0. VSAT
    # wants distinct qz/qx and its AR
    # fields only reach x8-x15, so neither
    # x0 nor a6 is usable here.


    esp.vsr.s32      q2, \Rj           # idx = diff >> shift (SAR)
    esp.ldxq.32      \Rj, q2, t4, 0, 0
    esp.ldxq.32      \Rj, q2, t4, 1, 2
    esp.ldxq.32      \Rj, q2, t4, 2, 4
    esp.ldxq.32      \Rj, q2, t4, 3, 6
.endm

dl_esp32p4_s8_attention_exp_lut_shift:
    # a0 e, a1 dots, a2 n, a3 m_new, a4 sat, a5 shift, a6 table, a7 scratch
    beqz a2, .Lexpv_ret0

    addi             sp, sp, -16
    sw               s0, 0(sp)
    sw               a0, 4(sp)   # e; a0 stops at e+n now, see .Lexpv_sum
    li               s0, 0       # VSAT.S32 low bound


    # Truncate shifts in the byte-select path (match dl_esp32p4_s8_lut). The
    # entry value is parked in scratch+16 rather than a register: every temp
    # gets reused before the restore at .Lexpv_sum, so keeping it in t6 wrote
    # an address into CFG and left vxrm/mis_ld set at random for whichever
    # kernel ran next.


    esp.movx.r.cfg   t6
    sw               t6, 16(a7)
    li               t5, 0xffffff0f
    and              t4, t6, t5
    esp.movx.w.cfg   t4

    esp.movx.w.sar   a5
    sw               a3, 0(a7)
    mv               t5, a7
    esp.vldbc.32.ip  q0, t5, 4
    # m_new. The byte-extract constants the int8
    # table needed (3 and 2 per 16-bit lane) are
    # gone with the int32 table.



    mv               t4, a6      # table base for ldxq (needs low reg)


    # Clear e[n & ~15 .. 64) up front, in whole 16-byte chunks. The block loop
    # only writes below n & ~15, and the scalar remainder writes into the first
    # of these chunks after this point, so both still land correctly. e is
    # 16-byte aligned and TILE_K is 64, so every store here is aligned.
    # This used to be a byte loop at the end of the function, 4 instructions
    # per byte: a 36-token tile spent 112 instructions clearing 28 bytes, more
    # than the 108 the two vector blocks of that same tile cost. It cannot just
    # be dropped, because .Lexpv_sum reduces the full 64-byte row to get e_sum
    # and therefore needs e[n..63] reading as zero before it runs.


    li               t5, 64
    beq              a2, t5, .Lexpv_pz_skip   # full tile: nothing to clear
    andi             t1, a2, -16
    add              a0, a0, t1
    esp.zero.q       q4
    bge              t1, t5, .Lexpv_pz_done
    esp.vst.128.ip   q4, a0, 16
    addi             t1, t1, 16
    bge              t1, t5, .Lexpv_pz_done
    esp.vst.128.ip   q4, a0, 16
    addi             t1, t1, 16
    bge              t1, t5, .Lexpv_pz_done
    esp.vst.128.ip   q4, a0, 16
    addi             t1, t1, 16
    bge              t1, t5, .Lexpv_pz_done
    esp.vst.128.ip   q4, a0, 16
.Lexpv_pz_done:
    lw               a0, 4(sp)   # rewind to e for the block loop
.Lexpv_pz_skip:

    andi             t3, a2, 15   # n % 16 remainder (do not clobber t4)
    srli             t0, a2, 4
    beqz             t0, .Lexpv_fused_rem

# 16 tokens per pass. Four quads land in q1/q3/q4, then VUNZIP.16 pairs them
# into 8 x 16-bit and VUNZIP.8 pairs those into 16 bytes for a single 128-bit
# store. The old code instead spilled every quad to scratch and pulled the
# four bytes back out with lbu/sb, i.e. 64 dependent scalar round-trips per
# 64-token row; this keeps the whole tail in the vector pipe. Output is
# bit-identical: same table, same index, same byte select.


.Lexpv_fused_blk:
    exp_quad         q1
    exp_quad         q3
    esp.vunzip.16    q1, q3      # q1 = tokens 0-7, one per 16-bit lane
    exp_quad         q3
    exp_quad         q4
    esp.vunzip.16    q3, q4      # q3 = tokens 8-15
    esp.vunzip.8     q1, q3      # q1 = all 16 bytes
    esp.vst.128.ip   q1, a0, 16
    addi             t0, t0, -1
    bnez             t0, .Lexpv_fused_blk

.Lexpv_fused_rem:
    beqz             t3, .Lexpv_sum
.Lexpv_fused_one:
    lw               t6, 0(a1)
    sub              t6, a3, t6      # diff = m_new - dot
    blez             t6, .Lexpv_f1_cl0
    bgt              t6, a4, .Lexpv_f1_csat
    j                .Lexpv_f1_sh
.Lexpv_f1_cl0:
    li               t6, 0
    j                .Lexpv_f1_sh
.Lexpv_f1_csat:
    mv               t6, a4          # clamp diff to sat (match vmax/vmin)
.Lexpv_f1_sh:
    sra              t6, t6, a5
    slli             t6, t6, 2       # int32 table stride
    add              t6, t4, t6
    lw               t6, 0(t6)
.Lexpv_f1_store:
    sb               t6, 0(a0)
    addi             a0, a0, 1
    addi             a1, a1, 4
    addi             t3, t3, -1
    bnez             t3, .Lexpv_fused_one

.Lexpv_sum:
    li               t2, 0x01010101
    sw               t2, 0(a7)
    sw               t2, 4(a7)
    sw               t2, 8(a7)
    sw               t2, 12(a7)
    mv               t6, a7
    esp.vld.128.ip   q5, t6, 16
    esp.zero.xacc
    lw               t6, 4(sp)           # e row base; a0 stops at e+n
    esp.vld.128.ip   q4, t6, 16
    esp.vmulas.s8.xacc q4, q5
    esp.vld.128.ip   q4, t6, 16
    esp.vmulas.s8.xacc q4, q5
    esp.vld.128.ip   q4, t6, 16
    esp.vmulas.s8.xacc q4, q5
    esp.vld.128.ip   q4, t6, 16
    esp.vmulas.s8.xacc q4, q5
    li               a3, 0
    esp.srs.s.xacc   t4, a3
    lw               t6, 16(a7)
    esp.movx.w.cfg   t6
    mv               a0, t4
    lw               s0, 0(sp)
    addi             sp, sp, 16
    ret

.Lexpv_ret0:
    li               a0, 0
    ret


############################################################################################################################################################
####
#### dl_esp32p4_s8_attention_pv_qacc
####
############################################################################################################################################################
# void dl_esp32p4_s8_attention_pv_qacc(float *acc, const int8_t *e,
#     const int8_t *v_blk, const float *deq, int nq, int D, int acc_stride,
#     int32_t *tmp16);
# Outer-product P·V. Unlike the n64 kernel (which reduces along t with a
# single XACC and therefore pays zero.xacc + srs + store for EVERY acc[d]),
# this walks t as the reduction and keeps 16 d-lanes live in QACC. One
# esp.vsmulas.s8.qacc broadcasts e[t] over V[t][d0:d0+16], so the
# accumulator setup/readout is amortized over 16 outputs instead of 1.
# V must be in d-block-major layout (see load_v_tile_dblk_s8):
#     v_blk[dg*TILE_K*16 + t*16 + dl],  d = dg*16 + dl
# so the 16 lanes of one QACC pass are contiguous and t advances by 16 B.
# That is the natural V[t][d] order re-chunked, so no bit-level transpose
# is needed.
# TILE_K is fixed at 64 (e_row = 64 int8 = four q registers). D must be a
# multiple of 16. tmp16 is 16 int32 of 16-byte-aligned scratch.


    .text
    .global dl_esp32p4_s8_attention_pv_qacc
    .type   dl_esp32p4_s8_attention_pv_qacc, @function
    .align 4

# Shared by the QACC P·V and Q·K kernels: 16 broadcast-scalar MACs against a
# 16-lane QACC, streaming the vector operand from a2 (post-inc 16).
#   P·V : scalar = e[t],    vector = V[t][d0:d0+16], lanes = 16 d
#   Q·K : scalar = Q[iq][d], vector = K^T[d][t0:t0+16], lanes = 16 t
# Four groups of these cover 64 reduction steps and consume exactly the
# 1024-byte block that one QACC pass needs, leaving a2 at the next block.
# q4/q5 alternate so each load lands one instruction before its use.


.macro qacc_mac16 e_reg
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 0
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 1
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 2
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 3
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 4
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 5
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 6
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 7
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 8
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 9
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 10
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 11
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 12
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 13
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 14
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 15
.endm

# Last group: the final two MACs must not load past the 1024 B tile.
.macro qacc_mac16_tail e_reg
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 0
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 1
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 2
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 3
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 4
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 5
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 6
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 7
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 8
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 9
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 10
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 11
    esp.vsmulas.s8.qacc.ld.incp q4, a2, q4, \e_reg, 12
    esp.vsmulas.s8.qacc.ld.incp q5, a2, q5, \e_reg, 13
    esp.vsmulas.s8.qacc              q4, \e_reg, 14
    esp.vsmulas.s8.qacc              q5, \e_reg, 15
.endm

dl_esp32p4_s8_attention_pv_qacc:
    # a0 acc, a1 e, a2 v_blk, a3 deq, a4 nq, a5 D, a6 acc_stride, a7 tmp16
    beqz a4, .Lpvq_end
    beqz a5, .Lpvq_end

    addi sp, sp, -16
    sw   s0, 0(sp)

    mv   t0, a0                 # current acc row
    mv   s0, a2                 # v_blk base; a2 streams through each row
    slli a6, a6, 2              # acc_stride in bytes
    srli a5, a5, 4              # n_dgroup = D / 16

.Lpvq_iq:

    # a1 is the current e row and the four post-increment loads advance it
    # exactly 64 bytes to the next row.
    # PIE load/store addresses must live in an "a" or "s" register (a "t"
    # register is rejected by the assembler), so a1 is reused as the walk
    # pointer here.


    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a1, 16
    esp.vld.128.ip q2, a1, 16
    esp.vld.128.ip q3, a1, 16

    # ft0 = *deq; a3 advances with the query row.
    flw  ft0, 0(a3)

    mv   t6, t0                 # acc walk pointer

    mv   a2, s0                 # v walk pointer
    mv   t5, a5                 # d-group counter

.Lpvq_dg:
    esp.zero.qacc
    # prime the two alternating V registers (bytes 0..31 of this tile)
    esp.vld.128.ip q4, a2, 16
    esp.vld.128.ip q5, a2, 16

    qacc_mac16      q0
    qacc_mac16      q1
    qacc_mac16      q2
    qacc_mac16_tail q3


    # 16 int32 lanes -> tmp16, then acc[d] += lane * deq.
    # Inlined esp32p4_save_qacc_as_32bit (this file does not include
    # dl_esp32p4_common.S). Lane order is 0..15, matching the byte order of
    # the vector operand — same contract avgpool2d relies on.


    mv   a0, a7
    esp.st.qacc.l.l.128.ip  a0, 16
    esp.st.qacc.l.h.128.ip  a0, 16
    esp.st.qacc.h.l.128.ip  a0, 16
    esp.st.qacc.h.h.128.ip  a0, 16

    lw   t1, 0(a7)
    lw   t2, 4(a7)
    lw   t3, 8(a7)
    lw   t4, 12(a7)
    fcvt.s.w ft1, t1
    fcvt.s.w ft2, t2
    fcvt.s.w ft3, t3
    fcvt.s.w ft4, t4
    flw  ft5, 0(t6)
    flw  ft6, 4(t6)
    flw  ft7, 8(t6)
    flw  fa0, 12(t6)
    fmadd.s ft1, ft1, ft0, ft5
    fmadd.s ft2, ft2, ft0, ft6
    fmadd.s ft3, ft3, ft0, ft7
    fmadd.s ft4, ft4, ft0, fa0
    fsw  ft1, 0(t6)
    fsw  ft2, 4(t6)
    fsw  ft3, 8(t6)
    fsw  ft4, 12(t6)

    lw   t1, 16(a7)
    lw   t2, 20(a7)
    lw   t3, 24(a7)
    lw   t4, 28(a7)
    fcvt.s.w ft1, t1
    fcvt.s.w ft2, t2
    fcvt.s.w ft3, t3
    fcvt.s.w ft4, t4
    flw  ft5, 16(t6)
    flw  ft6, 20(t6)
    flw  ft7, 24(t6)
    flw  fa0, 28(t6)
    fmadd.s ft1, ft1, ft0, ft5
    fmadd.s ft2, ft2, ft0, ft6
    fmadd.s ft3, ft3, ft0, ft7
    fmadd.s ft4, ft4, ft0, fa0
    fsw  ft1, 16(t6)
    fsw  ft2, 20(t6)
    fsw  ft3, 24(t6)
    fsw  ft4, 28(t6)

    lw   t1, 32(a7)
    lw   t2, 36(a7)
    lw   t3, 40(a7)
    lw   t4, 44(a7)
    fcvt.s.w ft1, t1
    fcvt.s.w ft2, t2
    fcvt.s.w ft3, t3
    fcvt.s.w ft4, t4
    flw  ft5, 32(t6)
    flw  ft6, 36(t6)
    flw  ft7, 40(t6)
    flw  fa0, 44(t6)
    fmadd.s ft1, ft1, ft0, ft5
    fmadd.s ft2, ft2, ft0, ft6
    fmadd.s ft3, ft3, ft0, ft7
    fmadd.s ft4, ft4, ft0, fa0
    fsw  ft1, 32(t6)
    fsw  ft2, 36(t6)
    fsw  ft3, 40(t6)
    fsw  ft4, 44(t6)

    lw   t1, 48(a7)
    lw   t2, 52(a7)
    lw   t3, 56(a7)
    lw   t4, 60(a7)
    fcvt.s.w ft1, t1
    fcvt.s.w ft2, t2
    fcvt.s.w ft3, t3
    fcvt.s.w ft4, t4
    flw  ft5, 48(t6)
    flw  ft6, 52(t6)
    flw  ft7, 56(t6)
    flw  fa0, 60(t6)
    fmadd.s ft1, ft1, ft0, ft5
    fmadd.s ft2, ft2, ft0, ft6
    fmadd.s ft3, ft3, ft0, ft7
    fmadd.s ft4, ft4, ft0, fa0
    fsw  ft1, 48(t6)
    fsw  ft2, 52(t6)
    fsw  ft3, 56(t6)
    fsw  ft4, 60(t6)

    addi t6, t6, 64             # next 16 d lanes of acc
    addi t5, t5, -1
    bnez t5, .Lpvq_dg

    add  t0, t0, a6             # next acc row
    addi a3, a3, 4              # next dequant scale
    addi a4, a4, -1
    bnez a4, .Lpvq_iq

    lw   s0, 0(sp)
    addi sp, sp, 16

.Lpvq_end:
    ret


############################################################################################################################################################
####
#### dl_esp32p4_s8_attention_pv_qacc_i32
####
############################################################################################################################################################
# void dl_esp32p4_s8_attention_pv_qacc_i32(int32_t *acc, const int8_t *e,
#     const int8_t *v_blk, int nq, int D, int acc_stride, int32_t *tmp16);
# Same outer-product P·V as the kernel above, but accumulates in int32 and
# leaves the dequant scale to the caller. The fp32 variant spends 5
# instructions per lane on lw/fcvt/flw/fmadd/fsw — 80 of its 151 instructions
# per d-group — purely to fold each tile into a float accumulator. On the
# batched path exp_scale is the constant v_scale/127 for every tile, so the
# multiply can be hoisted out of the tile loop entirely and the per-tile fold
# becomes four vld/vadd.s32/vst quads: 16 instructions instead of 80.
# Range: one d-lane of one tile is at most 64*127*127 = 1.03e6, so even a
# 2048-token K (32 tiles) tops out near 3.3e7 — two orders of magnitude below
# int32 overflow.


    .text
    .global dl_esp32p4_s8_attention_pv_qacc_i32
    .type   dl_esp32p4_s8_attention_pv_qacc_i32, @function
    .align 4

# acc[0:16] += the 16 int32 lanes staged at a0; a3 walks the acc row.
.macro pv_i32_fold
    esp.vld.128.ip q6, a0, 16
    esp.vld.128.ip q7, a3, 0
    esp.vadd.s32   q7, q7, q6
    esp.vst.128.ip q7, a3, 16
.endm

dl_esp32p4_s8_attention_pv_qacc_i32:
    # a0 acc, a1 e, a2 v_blk, a3 nq, a4 D, a5 acc_stride, a6 tmp16
    beqz a3, .Lpvi_end
    beqz a4, .Lpvi_end

    mv   t0, a0                 # current acc row
    mv   t1, a2                 # v_blk base
    mv   t2, a3                 # remaining query rows
    slli a5, a5, 2              # acc_stride in bytes
    srli a4, a4, 4              # n_dgroup = D / 16

.Lpvi_iq:
    # The four loads advance a1 by the fixed 64-byte e-row stride.
    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a1, 16
    esp.vld.128.ip q2, a1, 16
    esp.vld.128.ip q3, a1, 16


    # a3 must be an "a"/"s" register because
    # the fold below PIE-loads and stores through it.


    mv   a3, t0

    mv   a2, t1                 # v walk pointer
    mv   t5, a4                 # d-group counter

.Lpvi_dg:
    esp.zero.qacc
    # prime the two alternating V registers (bytes 0..31 of this tile)
    esp.vld.128.ip q4, a2, 16
    esp.vld.128.ip q5, a2, 16

    qacc_mac16      q0
    qacc_mac16      q1
    qacc_mac16      q2
    qacc_mac16_tail q3

    mv   a0, a6
    esp.st.qacc.l.l.128.ip  a0, 16
    esp.st.qacc.l.h.128.ip  a0, 16
    esp.st.qacc.h.l.128.ip  a0, 16
    esp.st.qacc.h.h.128.ip  a0, 16

    mv   a0, a6
    pv_i32_fold
    pv_i32_fold
    pv_i32_fold
    pv_i32_fold

    addi t5, t5, -1
    bnez t5, .Lpvi_dg

    add  t0, t0, a5             # next acc row
    addi t2, t2, -1
    bnez t2, .Lpvi_iq

.Lpvi_end:
    ret


############################################################################################################################################################
####
#### dl_esp32p4_s8_attention_qk_qacc_d64
####
############################################################################################################################################################
# void dl_esp32p4_s8_attention_qk_qacc_d64(int32_t *dots, const int8_t *q,
#     const int8_t *k_blk, int nq, int n_tgroup, int score_stride,
#     int32_t *row_max, int tile_n);
# Outer-product Q·K, the mirror of the QACC P·V kernel. A per-row dot
# reduces along d with one XACC, so every single dots[iq][t] costs a
# zero.xacc + srs + sw on top of its 4 MACs. Here t is the 16 QACC lanes and
# d is the reduction, so that setup/readout is amortized over 16 outputs:
# 64 MACs + 1 zero.qacc + 4 st.qacc per 16 scores instead of 64 MACs + 16
# accumulator round-trips. The MAC count is identical; the overhead is not.
# K must be in t-block-major layout (see build_k_head_tblk_s8):
#     k_blk[tg*D_pad*16 + d*16 + tl],  t = tg*16 + tl
# so one QACC pass sees 16 consecutive tokens per lane while d advances by
# 16 B, exactly the stride qacc_mac16 walks.
# head_dim is fixed at 64 (q0..q3). score_stride must keep each dots row
# 16-byte aligned so st.qacc can write straight into dots — no tmp buffer.
# Token groups past tile_n are zero-padded in k_blk and land in dots as 0;
# softmax only reads the first tile_n, but row_max must ignore them, so the
# max scan is clamped to tile_n.


    .text
    .global dl_esp32p4_s8_attention_qk_qacc_d64
    .type   dl_esp32p4_s8_attention_qk_qacc_d64, @function
    .align 4

# Vector running max: q7 holds four int32 lanes of partial max, folded from
# the 16 scores st.qacc just wrote. 8 instructions per group instead of the
# 32 a scalar lane-by-lane scan needs. Only the final group of the final
# tile can be short, and its zero-padded lanes would poison the max, so that
# one takes the scalar path below.


.macro qk_q_vmax
    esp.vld.128.ip q6, a1, 16
    esp.vmax.s32   q7, q7, q6
    esp.vld.128.ip q6, a1, 16
    esp.vmax.s32   q7, q7, q6
    esp.vld.128.ip q6, a1, 16
    esp.vmax.s32   q7, q7, q6
    esp.vld.128.ip q6, a1, 16
    esp.vmax.s32   q7, q7, q6
.endm

# One QACC pass: 16 scores for tokens [tg*16, tg*16+16) of this tile,
# stored straight into the dots row through a0.


.macro qk_q_group
    esp.zero.qacc
    # prime the two alternating K registers (bytes 0..31 of this group)
    esp.vld.128.ip q4, a2, 16
    esp.vld.128.ip q5, a2, 16

    qacc_mac16      q0
    qacc_mac16      q1
    qacc_mac16      q2
    qacc_mac16_tail q3

    esp.st.qacc.l.l.128.ip  a0, 16
    esp.st.qacc.l.h.128.ip  a0, 16
    esp.st.qacc.h.l.128.ip  a0, 16
    esp.st.qacc.h.h.128.ip  a0, 16
.endm

dl_esp32p4_s8_attention_qk_qacc_d64:

    # a0 dots, a1 q, a2 k_blk, a3 nq, a4 tile_n, a5 score_stride,
    # a6 row_max, a7 tmp16 (>= 8 int32, 16 B aligned)


    beqz a3, .Lqkq_end
    beqz a4, .Lqkq_end

    mv   t3, a1                 # current Q row
    mv   t4, a2                 # k_blk base; a2 streams per query
    slli a5, a5, 2              # score_stride in bytes
    addi t0, a4, 15
    srli t0, t0, 4              # groups written per score row
    slli t0, t0, 6              # each QACC group stores 64 bytes
    sub  a5, a5, t0             # adjustment from row end to next row


    # Stage INT32_MIN x4 at tmp16[4..7] so each query row can reload the
    # vector-max identity with a single vld.


    lui  t0, 0x80000
    sw   t0, 16(a7)
    sw   t0, 20(a7)
    sw   t0, 24(a7)
    sw   t0, 28(a7)

.Lqkq_iq:

    # Q row = q + iq*64 -> q0..q3. PIE addresses must be an "a"/"s"
    # register, so a1 is the walk pointer.


    mv   a1, t3
    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a1, 16
    esp.vld.128.ip q2, a1, 16
    esp.vld.128.ip q3, a1, 16

    # a0 already points at this dots row; st.qacc advances it by 64 B/group.
    mv   a2, t4                 # k walk pointer
    srli t5, a4, 4              # full-group counter

    addi a1, a7, 16             # q7 = INT32_MIN x4
    esp.vld.128.ip q7, a1, 16

    beqz t5, .Lqkq_reduce

.Lqkq_tg:
    qk_q_group
    addi a1, a0, -64            # fold the 16 scores just stored
    qk_q_vmax
    addi t5, t5, -1
    bnez t5, .Lqkq_tg

.Lqkq_reduce:
    # q7's four lanes -> scalar rmax in t6
    mv   a1, a7
    esp.vst.128.ip q7, a1, 16
    lw   t6, 0(a7)
    lw   t1, 4(a7)
    lw   t2, 8(a7)
    lw   t0, 12(a7)
    blt  t1, t6, 93f
    mv   t6, t1
93:
    blt  t2, t6, 94f
    mv   t6, t2
94:
    blt  t0, t6, 95f
    mv   t6, t0
95:


    # Short tail group: its padded lanes are zero, which would beat a
    # genuinely all-negative row, so scan only the real lanes.


    andi t2, a4, 15
    beqz t2, .Lqkq_iq_done
    qk_q_group
    addi t0, a0, -64
.Lqkq_max_one:
    lw   t1, 0(t0)
    blt  t1, t6, 92f
    mv   t6, t1
92:
    addi t0, t0, 4
    addi t2, t2, -1
    bgtz t2, .Lqkq_max_one

.Lqkq_iq_done:
    sw   t6, 0(a6)
    addi a6, a6, 4
    addi t3, t3, 64             # next Q row
    add  a0, a0, a5             # next dots row
    addi a3, a3, -1
    bnez a3, .Lqkq_iq

.Lqkq_end:
    ret


############################################################################################################################################################
####
#### dl_esp32p4_s8_attention_qk_qacc_d32
####
############################################################################################################################################################
# void dl_esp32p4_s8_attention_qk_qacc_d32(int32_t *dots, const int8_t *q,
#     const int8_t *k_blk, int nq, int tile_n, int score_stride,
#     int32_t *row_max, int32_t *tmp16);
# D=32 mirror of dl_esp32p4_s8_attention_qk_qacc_d64: q0..q1 only, two qacc_mac16
# groups (512 B of t-block K per 16-token pass). Same K layout and dots
# contract as the d64 kernel.


    .text
    .global dl_esp32p4_s8_attention_qk_qacc_d32
    .type   dl_esp32p4_s8_attention_qk_qacc_d32, @function
    .align 4

.macro qk_q_group_d32
    esp.zero.qacc
    esp.vld.128.ip q4, a2, 16
    esp.vld.128.ip q5, a2, 16

    qacc_mac16      q0
    qacc_mac16_tail q1

    esp.st.qacc.l.l.128.ip  a0, 16
    esp.st.qacc.l.h.128.ip  a0, 16
    esp.st.qacc.h.l.128.ip  a0, 16
    esp.st.qacc.h.h.128.ip  a0, 16
.endm

dl_esp32p4_s8_attention_qk_qacc_d32:

    # a0 dots, a1 q, a2 k_blk, a3 nq, a4 tile_n, a5 score_stride,
    # a6 row_max, a7 tmp16 (>= 8 int32, 16 B aligned)


    beqz a3, .Lqkq32_end
    beqz a4, .Lqkq32_end

    mv   t3, a1                 # current Q row
    mv   t4, a2                 # k_blk base
    slli a5, a5, 2              # score_stride in bytes
    addi t0, a4, 15
    srli t0, t0, 4
    slli t0, t0, 6              # bytes stored per row
    sub  a5, a5, t0             # adjustment from row end to next row

    lui  t0, 0x80000
    sw   t0, 16(a7)
    sw   t0, 20(a7)
    sw   t0, 24(a7)
    sw   t0, 28(a7)

.Lqkq32_iq:
    mv   a1, t3
    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a1, 16

    mv   a2, t4
    srli t5, a4, 4

    addi a1, a7, 16
    esp.vld.128.ip q7, a1, 16

    beqz t5, .Lqkq32_reduce

.Lqkq32_tg:
    qk_q_group_d32
    addi a1, a0, -64
    qk_q_vmax
    addi t5, t5, -1
    bnez t5, .Lqkq32_tg

.Lqkq32_reduce:
    mv   a1, a7
    esp.vst.128.ip q7, a1, 16
    lw   t6, 0(a7)
    lw   t1, 4(a7)
    lw   t2, 8(a7)
    lw   t0, 12(a7)
    blt  t1, t6, 103f
    mv   t6, t1
103:
    blt  t2, t6, 104f
    mv   t6, t2
104:
    blt  t0, t6, 105f
    mv   t6, t0
105:

    andi t2, a4, 15
    beqz t2, .Lqkq32_iq_done
    qk_q_group_d32
    addi t0, a0, -64
.Lqkq32_max_one:
    lw   t1, 0(t0)
    blt  t1, t6, 102f
    mv   t6, t1
102:
    addi t0, t0, 4
    addi t2, t2, -1
    bgtz t2, .Lqkq32_max_one

.Lqkq32_iq_done:
    sw   t6, 0(a6)
    addi a6, a6, 4
    addi t3, t3, 32             # next Q row
    add  a0, a0, a5             # next dots row
    addi a3, a3, -1
    bnez a3, .Lqkq32_iq

.Lqkq32_end:
    ret


############################################################################################################################################################
####
#### dl_esp32p4_s8_attention_qk_qacc_generic
####
############################################################################################################################################################
# void dl_esp32p4_s8_attention_qk_qacc_generic(int32_t *dots, const int8_t *q,
#     const int8_t *k_blk, int nq, int tile_n, int score_stride,
#     int32_t *row_max, int32_t *tmp16, int D_pad);
# Generic-D_pad variant of dl_esp32p4_s8_attention_qk_qacc_d64 (D_pad % 16 == 0,
# 16 <= D_pad <= 128). Same K layout, dots and row_max contract. Q rows are
# D_pad bytes apart; Q does not fit the register file next to K and the max
# scratch, so each 16-d Q chunk is reloaded per 16-token group (one vld per
# 16 MACs). D_pad is the 9th argument and arrives on the stack.


    .text
    .global dl_esp32p4_s8_attention_qk_qacc_generic
    .type   dl_esp32p4_s8_attention_qk_qacc_generic, @function
    .align 4

# s10 = Q row, s9 = D_pad/16 - 1
.macro qk_q_group_any
    esp.zero.qacc
    esp.vld.128.ip q4, a2, 16
    esp.vld.128.ip q5, a2, 16
    mv   a1, s10
    mv   t6, s9
    beqz t6, 2f
1:
    esp.vld.128.ip q0, a1, 16
    qacc_mac16 q0
    addi t6, t6, -1
    bnez t6, 1b
2:
    esp.vld.128.ip q0, a1, 16
    qacc_mac16_tail q0

    esp.st.qacc.l.l.128.ip  a0, 16
    esp.st.qacc.l.h.128.ip  a0, 16
    esp.st.qacc.h.l.128.ip  a0, 16
    esp.st.qacc.h.h.128.ip  a0, 16
.endm

dl_esp32p4_s8_attention_qk_qacc_generic:

    # a0 dots, a1 q, a2 k_blk, a3 nq, a4 tile_n, a5 score_stride,
    # a6 row_max, a7 tmp16 (>= 8 int32, 16 B aligned), 0(sp) D_pad


    beqz a3, .Lqkqa_end
    beqz a4, .Lqkqa_end
    lw   t4, 0(sp)

    addi sp, sp, -16
    sw   s8, 0(sp)
    sw   s9, 4(sp)
    sw   s10, 8(sp)

    mv   s8, t4                 # Q row bytes
    srli s9, t4, 4
    addi s9, s9, -1             # Q chunks before the tail chunk
    mv   s10, a1                # current Q row
    mv   t3, a2                 # k_blk base
    slli a5, a5, 2              # score_stride in bytes
    addi t0, a4, 15
    srli t0, t0, 4
    slli t0, t0, 6              # bytes stored per row
    sub  a5, a5, t0             # adjustment from row end to next row

    lui  t0, 0x80000
    sw   t0, 16(a7)
    sw   t0, 20(a7)
    sw   t0, 24(a7)
    sw   t0, 28(a7)

.Lqkqa_iq:
    mv   a2, t3
    srli t5, a4, 4

    addi a1, a7, 16
    esp.vld.128.ip q7, a1, 16

    beqz t5, .Lqkqa_reduce

.Lqkqa_tg:
    qk_q_group_any
    addi a1, a0, -64
    qk_q_vmax
    addi t5, t5, -1
    bnez t5, .Lqkqa_tg

.Lqkqa_reduce:
    mv   a1, a7
    esp.vst.128.ip q7, a1, 16
    lw   t4, 0(a7)
    lw   t1, 4(a7)
    lw   t2, 8(a7)
    lw   t0, 12(a7)
    blt  t1, t4, 113f
    mv   t4, t1
113:
    blt  t2, t4, 114f
    mv   t4, t2
114:
    blt  t0, t4, 115f
    mv   t4, t0
115:

    andi t2, a4, 15
    beqz t2, .Lqkqa_iq_done
    qk_q_group_any
    addi t0, a0, -64
.Lqkqa_max_one:
    lw   t1, 0(t0)
    blt  t1, t4, 112f
    mv   t4, t1
112:
    addi t0, t0, 4
    addi t2, t2, -1
    bgtz t2, .Lqkqa_max_one

.Lqkqa_iq_done:
    sw   t4, 0(a6)
    addi a6, a6, 4
    add  s10, s10, s8           # next Q row
    add  a0, a0, a5             # next dots row
    addi a3, a3, -1
    bnez a3, .Lqkqa_iq

    lw   s8, 0(sp)
    lw   s9, 4(sp)
    lw   s10, 8(sp)
    addi sp, sp, 16

.Lqkqa_end:
    ret
