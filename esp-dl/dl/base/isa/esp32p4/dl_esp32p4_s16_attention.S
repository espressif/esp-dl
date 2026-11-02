############################################################################################################################################################
####
#### dl_esp32p4_s16_attention_qk_qacc_d32
####
############################################################################################################################################################
# void dl_esp32p4_s16_attention_qk_qacc_d32(float *scores, const int16_t *q,
#     const int16_t *k_blk, int nq, int tile_n, int score_stride,
#     float dot_scale, int32_t *tmp16);
# scores[iq*score_stride + t] =
#     float(round_half_even((Q[iq].K[t]) / 2^QK16_D32_SHIFT)) * dot_scale
# The caller passes dot_scale = logit_scale * 2^QK16_D32_SHIFT. The last
# group always writes all 8 lanes, so each scores row must hold tile_n
# rounded up to 8; lanes past tile_n are 0.
# Key hardware fact this kernel encodes: st.qacc of the s16 QACC yields 8
# sign-extended 64-bit results, low word at even index, high word at odd
# index (probed 2026-09-30 with |dot| up to 2^35). A D=32 int16 dot needs up
# to 36 bits, so each result is shifted right by QK16_D32_SHIFT with
# round-half-even before the int32 -> float conversion.
# K must be in the int16 t-block layout produced by build_k_head_tblk_s16:
#     k_blk[tg*D*8 + d*8 + tl],  t = tg*8 + tl  (short tail zero-padded).
# Q row is 32 int16 = 64 B, held in q0..q3 (8 s16 lanes each).
# Each 8-token QACC pass:
#   - 32 vsmulas.s16.qacc MACs (one per d), reading 512 B of t-block K.
#   - 30 use ld.incp; the last 2 MACs re-use already-primed q4/q5 so a2
#     never advances past the group.
#   - Four st.qacc halves land 8 s64 in tmp16 (needs 64 B); each is rounded,
#     converted and scaled into scores.


    .set QK16_D32_SHIFT, 5      # 32 * 2^30 >> 5 = 2^30: never exceeds s32
    .text
    .global dl_esp32p4_s16_attention_qk_qacc_d32
    .type   dl_esp32p4_s16_attention_qk_qacc_d32, @function
    .align 4

.macro qacc_mac8_i16 e_reg
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, \e_reg, 0
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, \e_reg, 1
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, \e_reg, 2
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, \e_reg, 3
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, \e_reg, 4
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, \e_reg, 5
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, \e_reg, 6
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, \e_reg, 7
.endm

.macro qacc_mac8_i16_tail e_reg
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, \e_reg, 0
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, \e_reg, 1
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, \e_reg, 2
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, \e_reg, 3
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, \e_reg, 4
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, \e_reg, 5
    esp.vsmulas.s16.qacc           q4, \e_reg, 6
    esp.vsmulas.s16.qacc           q5, \e_reg, 7
.endm

# scores[i] = float(round_half_even((s64 at tmp16[2i..2i+1]) / 2^QK16_D32_SHIFT))
#             * fa0.
# fcvt/fmul use the dynamic FP rounding mode, as the compiler does for
# float(int) * float.
# Clobbers t0, t1, t2, ft0.


.macro qk16_pack_s64 i
    lw       t0, (8 * \i)(a7)
    lw       t1, (8 * \i + 4)(a7)
    andi     t2, t0, ((1 << QK16_D32_SHIFT) - 1)      # remainder
    srli     t0, t0, QK16_D32_SHIFT
    slli     t1, t1, (32 - QK16_D32_SHIFT)
    or       t0, t0, t1                               # floor
    andi     t1, t0, 1
    add      t2, t2, t1                               # rem + odd > half <=> round up
    sltiu    t2, t2, ((1 << (QK16_D32_SHIFT - 1)) + 1)
    xori     t2, t2, 1
    add      t0, t0, t2
    fcvt.s.w ft0, t0
    fmul.s   ft0, ft0, fa0
    fsw      ft0, (4 * \i)(a0)
.endm

# One QACC pass: 8 scores for tokens [tg*8, tg*8+8) into scores[a0..a0+32).
.macro qk_q_group_i16_d32
    esp.zero.qacc
    # prime the two alternating K registers (bytes 0..31 of this group)
    esp.vld.128.ip q4, a2, 16
    esp.vld.128.ip q5, a2, 16

    qacc_mac8_i16      q0
    qacc_mac8_i16      q1
    qacc_mac8_i16      q2
    qacc_mac8_i16_tail q3


    # Hardware-probed s16 QACC layout (ESP32-P4 rev v3.1): result i of
    # `esp.vsmulas.s16.qacc ..., N` is a sign-extended s64 whose low word
    # lands at tmp16[2*i] and high word at tmp16[2*i+1].
    # a1 is free here: the Q row already sits in q0..q3. (The PIE 128.ip
    # forms only accept the a0..a7 compressed encoding, so t-registers cannot
    # be used as the base pointer.)


    mv   a1, a7
    esp.st.qacc.l.l.128.ip  a1, 16
    esp.st.qacc.l.h.128.ip  a1, 16
    esp.st.qacc.h.l.128.ip  a1, 16
    esp.st.qacc.h.h.128.ip  a1, 16
    qk16_pack_s64 0
    qk16_pack_s64 1
    qk16_pack_s64 2
    qk16_pack_s64 3
    qk16_pack_s64 4
    qk16_pack_s64 5
    qk16_pack_s64 6
    qk16_pack_s64 7
    addi a0, a0, 32
.endm

dl_esp32p4_s16_attention_qk_qacc_d32:

    # a0 scores, a1 q, a2 k_blk, a3 nq, a4 tile_n, a5 score_stride (floats),
    # a6 tmp16 (>= 16 int32, 16 B aligned), fa0 dot_scale


    beqz a3, .Lqk16q32_end
    beqz a4, .Lqk16q32_end

    mv   a7, a6                 # tmp16, addressed by qk16_pack_s64
    mv   a6, a2                 # k_blk base; a2 streams per query
    mv   t3, a1                 # current Q row
    slli a5, a5, 2              # score_stride in bytes
    addi t0, a4, 7
    srli t0, t0, 3              # groups written per score row
    slli t0, t0, 5              # each QACC group stores 32 bytes
    sub  a5, a5, t0             # adjustment from row end to next row

.Lqk16q32_iq:
    # Current 32-element Q row -> q0..q3.
    mv   a1, t3
    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a1, 16
    esp.vld.128.ip q2, a1, 16
    esp.vld.128.ip q3, a1, 16

    # a0 already points at this scores row.
    mv   a2, a6                 # k walk pointer
    srli t5, a4, 3              # full-group counter
    beqz t5, .Lqk16q32_tail

.Lqk16q32_tg:
    qk_q_group_i16_d32
    addi t5, t5, -1
    bnez t5, .Lqk16q32_tg

.Lqk16q32_tail:
    andi t5, a4, 7
    beqz t5, .Lqk16q32_iq_done
    qk_q_group_i16_d32

.Lqk16q32_iq_done:
    addi t3, t3, 64             # next Q row
    add  a0, a0, a5             # next scores row
    addi a3, a3, -1
    bnez a3, .Lqk16q32_iq

.Lqk16q32_end:
    ret


############################################################################################################################################################
####
#### dl_esp32p4_s16_attention_qk_qacc_generic
####
############################################################################################################################################################
# void dl_esp32p4_s16_attention_qk_qacc_generic(float *scores, const int16_t *q,
#     const int16_t *k_blk, int nq, int tile_n, int score_stride,
#     int32_t *tmp16, int D_real, float dot_scale);
# Generic-D variant of dl_esp32p4_s16_attention_qk_qacc_d32
# (1 <= D_real <= 128). Q and K are physically padded to
# D_pad=round_up(D_real, 8), while the dot shift is s=ceil(log2(D_real)),
# so
#     scores[iq*score_stride + t] =
#         float(round_half_even((Q[iq].K[t]) / 2^s)) * dot_scale
# and the caller passes dot_scale = logit_scale * 2^s. Q rows are D_pad int16
# apart; K uses the same 8-token t-block layout as the D=32 kernel. Q does
# not fit the register file beyond D=32, so every 8-lane Q chunk is reloaded
# per 8-token group (one vld per 8 MACs).


    .text
    .global dl_esp32p4_s16_attention_qk_qacc_generic
    .type   dl_esp32p4_s16_attention_qk_qacc_generic, @function
    .align 4

# Like qk16_pack_s64 with the shift in registers: s7 = s, s8 = 32 - s,
# s9 = 2^s - 1, s10 = 2^(s-1) + 1; tmp16 at a6.


.macro qk16_pack_s64_r i
    lw       t0, (8 * \i)(a6)
    lw       t1, (8 * \i + 4)(a6)
    beqz     s7, .Lqk16_pack_done\@
    and      t2, t0, s9
    srl      t0, t0, s7
    sll      t1, t1, s8
    or       t0, t0, t1
    andi     t1, t0, 1
    add      t2, t2, t1
    sltu     t2, t2, s10
    xori     t2, t2, 1
    add      t0, t0, t2
.Lqk16_pack_done\@:
    fcvt.s.w ft0, t0
    fmul.s   ft0, ft0, fa0
    fsw      ft0, (4 * \i)(a0)
.endm

# 8 scores for one t-block group; t4 = D/8 - 1, s11 = Q row.
.macro qk_q_group_i16_any
    esp.zero.qacc
    esp.vld.128.ip q4, a2, 16
    esp.vld.128.ip q5, a2, 16
    mv   a1, s11
    mv   t6, t4
    beqz t6, 2f
1:
    esp.vld.128.ip q0, a1, 16
    qacc_mac8_i16 q0
    addi t6, t6, -1
    bnez t6, 1b
2:
    esp.vld.128.ip q0, a1, 16
    qacc_mac8_i16_tail q0

    mv   a1, a6
    esp.st.qacc.l.l.128.ip  a1, 16
    esp.st.qacc.l.h.128.ip  a1, 16
    esp.st.qacc.h.l.128.ip  a1, 16
    esp.st.qacc.h.h.128.ip  a1, 16
    qk16_pack_s64_r 0
    qk16_pack_s64_r 1
    qk16_pack_s64_r 2
    qk16_pack_s64_r 3
    qk16_pack_s64_r 4
    qk16_pack_s64_r 5
    qk16_pack_s64_r 6
    qk16_pack_s64_r 7
    addi a0, a0, 32
.endm

dl_esp32p4_s16_attention_qk_qacc_generic:

    # a0 scores, a1 q, a2 k_blk, a3 nq, a4 tile_n, a5 score_stride (floats),
    # a6 tmp16 (>= 16 int32, 16 B aligned), a7 D_real, fa0 dot_scale


    beqz a3, .Lqk16any_end
    beqz a4, .Lqk16any_end

    addi sp, sp, -32
    sw   s7, 0(sp)
    sw   s8, 4(sp)
    sw   s9, 8(sp)
    sw   s10, 12(sp)
    sw   s11, 16(sp)

    # s7 = ceil(log2(D))
    li   s7, 0
    li   t0, 1
.Lqk16any_log2:
    bge  t0, a7, .Lqk16any_log2_done
    slli t0, t0, 1
    addi s7, s7, 1
    j    .Lqk16any_log2
.Lqk16any_log2_done:
    li   t0, 32
    sub  s8, t0, s7
    li   t0, 1
    sll  s9, t0, s7
    addi s9, s9, -1
    addi t1, s7, -1
    sll  s10, t0, t1
    addi s10, s10, 1

    mv   s11, a1                # current Q row
    mv   t3, a2                 # k_blk base
    slli a5, a5, 2              # score_stride in bytes
    addi t0, a4, 7
    srli t0, t0, 3
    slli t0, t0, 5              # bytes stored per score row
    sub  a5, a5, t0             # adjustment from row end to next row
    addi t0, a7, 7
    andi t0, t0, -8             # D_pad = round_up(D_real, 8)
    srli t4, t0, 3
    addi t4, t4, -1             # Q chunks before the tail chunk
    slli a7, t0, 1              # padded Q row bytes

.Lqk16any_iq:
    mv   a2, t3
    srli t5, a4, 3
    beqz t5, .Lqk16any_tail

.Lqk16any_tg:
    qk_q_group_i16_any
    addi t5, t5, -1
    bnez t5, .Lqk16any_tg

.Lqk16any_tail:
    andi t5, a4, 7
    beqz t5, .Lqk16any_iq_done
    qk_q_group_i16_any

.Lqk16any_iq_done:
    add  s11, s11, a7           # next Q row
    add  a0, a0, a5             # next scores row
    addi a3, a3, -1
    bnez a3, .Lqk16any_iq

    lw   s7, 0(sp)
    lw   s8, 4(sp)
    lw   s9, 8(sp)
    lw   s10, 12(sp)
    lw   s11, 16(sp)
    addi sp, sp, 32

.Lqk16any_end:
    ret


############################################################################################################################################################
####
#### dl_esp32p4_s16_attention_pv_qacc
####
############################################################################################################################################################
# void dl_esp32p4_s16_attention_pv_qacc(float *acc, const int16_t *e,
#     const int16_t *v_dblk, int nq, int n_tok, int D, int acc_stride,
#     int32_t *tmp32, float scale);
# int16 P·V with each softmax weight split into two int16 halves,
#     e[t] ~= (e_hi[t] * 2^15 + e_lo[t]) * 2^-29,
# so both halves are exact integer dots on the s16 QACC and
#     acc[iq*acc_stride + d] += scale * (2^15 * sum_t e_hi[t] * V[t][d]
#                                        + sum_t e_lo[t] * V[t][d])
# with the caller passing scale = v_scale * 2^-29. The two s64 lane sums are
# combined in integers (|total| < 2^52) and converted to fp32 once, with a
# relative error within 2^-24.
# n_tok % 8 == 0, D % 8 == 0. Per query row the e buffer holds
# e_hi[n_tok] followed by e_lo[n_tok] (row stride 2*n_tok int16).
# v_dblk[dc][t][8] = V[t][dc*8 .. dc*8+8): one 16 B vector per token per
# 8-lane d-block, so the MAC walks v_dblk with ld.incp.
# tmp32: 16 B aligned, >= 32 int32 (both passes' st.qacc dumps).


    .text
    .global dl_esp32p4_s16_attention_pv_qacc
    .type   dl_esp32p4_s16_attention_pv_qacc, @function
    .align 4

# One pass of sum_t e[t] * V[t][chunk] into QACC, dumped at a6 + off;
# a1 = e half, a2 = chunk start, t4 = n_tok/8 - 1.


.macro pv16_pass off
    esp.zero.qacc
    esp.vld.128.ip q4, a2, 16
    esp.vld.128.ip q5, a2, 16
    mv   t6, t4
    beqz t6, 2f

    # Keep only the hardware-loop region uncompressed: lp.setup encodes the
    # address of its last 4-byte instruction.


    .option push
    .option norvc
    esp.lp.setup 0, t6, .Lpv16_pass_end\@
    esp.vld.128.ip q0, a1, 16
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, q0, 0
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, q0, 1
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, q0, 2
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, q0, 3
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, q0, 4
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, q0, 5
    esp.vsmulas.s16.qacc.ld.incp q4, a2, q4, q0, 6
.Lpv16_pass_end\@:
    esp.vsmulas.s16.qacc.ld.incp q5, a2, q5, q0, 7
    .option pop
2:
    esp.vld.128.ip q0, a1, 16
    qacc_mac8_i16_tail q0
    addi a1, a6, \off
    esp.st.qacc.l.l.128.ip  a1, 16
    esp.st.qacc.l.h.128.ip  a1, 16
    esp.st.qacc.h.l.128.ip  a1, 16
    esp.st.qacc.h.h.128.ip  a1, 16
.endm

# Lane i: total = (S_hi << 15) + S_lo as (rh:rl), then rh += rl[31] so the
# value reads rh * 2^32 + signed rl (an int32-range total has rh = 0 and
# converts exactly; an unsigned rl would cancel against rh = -1).
# S_hi at a6 + 8i, S_lo at a6 + 64 + 8i. Clobbers t0, t1, t2.


.macro pv16_comb i, rh, rl
    lw   t0, (8 * \i)(a6)
    lw   t1, (8 * \i + 4)(a6)
    lw   \rl, (64 + 8 * \i)(a6)
    lw   \rh, (64 + 8 * \i + 4)(a6)
    slli t1, t1, 15
    srli t2, t0, 17
    or   t1, t1, t2
    slli t0, t0, 15
    add  \rl, \rl, t0
    sltu t2, \rl, t0
    add  \rh, \rh, t1
    add  \rh, \rh, t2
    srli t2, \rl, 31
    add  \rh, \rh, t2
.endm

# acc[b..b+3] += float(total) * fa0, four lanes interleaved to hide the FPU
# latency; s11 acc chunk, fa2 = 2^32.


.macro pv16_acc4 b0, b1, b2, b3
    pv16_comb \b0, a0, a1
    pv16_comb \b1, a2, a7
    pv16_comb \b2, t3, t5
    pv16_comb \b3, t6, s10
    fcvt.s.w ft0, a1
    fcvt.s.w ft1, a0
    fcvt.s.w ft2, a7
    fcvt.s.w ft3, a2
    fcvt.s.w ft4, t5
    fcvt.s.w ft5, t3
    fcvt.s.w ft6, s10
    fcvt.s.w ft7, t6
    flw      ft8, (4 * \b0)(s11)
    flw      ft9, (4 * \b1)(s11)
    flw      ft10, (4 * \b2)(s11)
    flw      ft11, (4 * \b3)(s11)
    fmadd.s  ft0, ft1, fa2, ft0
    fmadd.s  ft2, ft3, fa2, ft2
    fmadd.s  ft4, ft5, fa2, ft4
    fmadd.s  ft6, ft7, fa2, ft6
    fmadd.s  ft8, ft0, fa0, ft8
    fmadd.s  ft9, ft2, fa0, ft9
    fmadd.s  ft10, ft4, fa0, ft10
    fmadd.s  ft11, ft6, fa0, ft11
    fsw      ft8, (4 * \b0)(s11)
    fsw      ft9, (4 * \b1)(s11)
    fsw      ft10, (4 * \b2)(s11)
    fsw      ft11, (4 * \b3)(s11)
.endm

dl_esp32p4_s16_attention_pv_qacc:

    # a0 acc, a1 e, a2 v_dblk, a3 nq, a4 n_tok, a5 D, a6 acc_stride (floats),
    # a7 tmp32, fa0 scale


    beqz a3, .Lpv16_end
    beqz a4, .Lpv16_end
    beqz a5, .Lpv16_end

    addi sp, sp, -48
    sw   s0, 0(sp)
    sw   s1, 4(sp)
    sw   s2, 8(sp)
    sw   s3, 12(sp)
    sw   s4, 16(sp)
    sw   s5, 20(sp)
    sw   s6, 24(sp)
    sw   s10, 28(sp)
    sw   s11, 32(sp)

    li   t0, 0x4f800000         # 2^32
    fmv.w.x fa2, t0

    mv   s0, a0                 # current acc row
    mv   s1, a1                 # current e_hi row
    mv   s2, a2                 # v_dblk base
    slli s3, a6, 2              # acc row bytes
    slli s4, a4, 1              # e half bytes
    slli s5, a4, 4              # v_dblk chunk bytes
    srli s6, a5, 3              # d chunks
    srli t4, a4, 3
    addi t4, t4, -1             # e chunks before the tail chunk
    mv   a6, a7                 # tmp32

.Lpv16_iq:
    mv   s11, s0                # current acc chunk
    mv   a4, s2                 # v chunk
    mv   a5, s6

.Lpv16_dc:
    mv   a1, s1
    mv   a2, a4
    pv16_pass 0
    add  a1, s1, s4
    mv   a2, a4
    pv16_pass 64
    pv16_acc4 0, 1, 2, 3
    pv16_acc4 4, 5, 6, 7

    addi s11, s11, 32
    add  a4, a4, s5
    addi a5, a5, -1
    bnez a5, .Lpv16_dc

    add  s0, s0, s3             # next acc row
    add  s1, s1, s4
    add  s1, s1, s4             # next e_hi row
    addi a3, a3, -1
    bnez a3, .Lpv16_iq

    lw   s0, 0(sp)
    lw   s1, 4(sp)
    lw   s2, 8(sp)
    lw   s3, 12(sp)
    lw   s4, 16(sp)
    lw   s5, 20(sp)
    lw   s6, 24(sp)
    lw   s10, 28(sp)
    lw   s11, 32(sp)
    addi sp, sp, 48

.Lpv16_end:
    ret
