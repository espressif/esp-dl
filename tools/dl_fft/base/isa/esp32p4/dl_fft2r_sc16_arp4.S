// Copyright 2026 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



    .text
    .option norvc

/*
 * Macro: reduce_abs_max_arp4_macro
 *
 * SIMD reduce-abs-max over a 16-byte aligned int16_t array.
 * Computes max(|data[i]|) for i in [0, count). abs(INT16_MIN) saturates
 * to INT16_MAX (handled by the saturated subtract). count must be >= 8
 * and a multiple of 8.
 *
 * Parameters (must all be PIE-valid GPRs: a0-a5/s0-s1/s8-s11/t3-t6):
 *   data_ptr - pointer to int16_t array (TRASHED: advanced past the data)
 *   count    - number of int16_t elements (TRASHED: used as loop counter)
 *   result   - receives the scalar max-abs value (int16_t, sign-extended)
 *   tmp      - scratch register (TRASHED)
 *
 * Clobbers: q0, q1, q2, q3, SAR_BYTE
 */
.macro reduce_abs_max_arp4_macro data_ptr count result tmp
    srli            \count, \count, 3
    esp.zero.q      q2
    esp.vld.128.ip  q0, \data_ptr, 16
    esp.vsub.s16    q1, q2, q0
    esp.vmax.s16    q0, q0, q1
    addi            \count, \count, -1
    beqz            \count, .Lram_done\@
.Lram_loop\@:
    esp.vld.128.ip  q1, \data_ptr, 16
    esp.vsub.s16    q3, q2, q1
    esp.vmax.s16    q1, q1, q3
    esp.vmax.s16    q0, q0, q1
    addi            \count, \count, -1
    bnez            \count, .Lram_loop\@
.Lram_done\@:
    li              \tmp, 8
    esp.movx.w.sar.bytes \tmp
    esp.src.q       q1, q0, q0
    esp.vmax.s16    q0, q0, q1
    li              \tmp, 4
    esp.movx.w.sar.bytes \tmp
    esp.src.q       q1, q0, q0
    esp.vmax.s16    q0, q0, q1
    li              \tmp, 2
    esp.movx.w.sar.bytes \tmp
    esp.src.q       q1, q0, q0
    esp.vmax.s16    q0, q0, q1
    esp.movi.32.a   q0, \result, 0
    slli            \result, \result, 16
    srai            \result, \result, 16
.endm



.macro dif_hp_butterfly_inner_arp4 sar4_val
    addi            t0, s9, 0
    beqz            t0, .Ldif_hp_in_end\@
.Ldif_hp_in_loop\@:
    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, \sar4_val
    esp.fft.cmul.s16.ld.xp   q4, s1, t6, q5, q3, q2, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 2
    esp.fft.cmul.s16.ld.xp   q1, a5, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t3, t6, 6, 0, \sar4_val

    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, \sar4_val
    esp.fft.cmul.s16.ld.xp   q2, s1, t6, q5, q3, q4, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q4, 2
    esp.fft.cmul.s16.ld.xp   q1, a5, t6, q5, q3, q4, 4
    esp.fft.cmul.s16.st.xp   q3, q4, q5, t3, t6, 6, 0, \sar4_val

    addi            t0, t0, -1
    bnez            t0, .Ldif_hp_in_loop\@
.Ldif_hp_in_end\@:
.endm



.macro ifft_hp_butterfly_inner_arp4 sar4_val
    addi            t0, s9, 0
    beqz            t0, .Lifft_hp_in_end\@
.Lifft_hp_in_loop\@:
    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, \sar4_val
    esp.fft.cmul.s16.ld.xp   q4, s1, t6, q5, q3, q2, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 3
    esp.fft.cmul.s16.ld.xp   q1, a5, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t3, t6, 7, 0, \sar4_val

    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, \sar4_val
    esp.fft.cmul.s16.ld.xp   q2, s1, t6, q5, q3, q4, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q4, 3
    esp.fft.cmul.s16.ld.xp   q1, a5, t6, q5, q3, q4, 5
    esp.fft.cmul.s16.st.xp   q3, q4, q5, t3, t6, 7, 0, \sar4_val

    addi            t0, t0, -1
    bnez            t0, .Lifft_hp_in_loop\@
.Lifft_hp_in_end\@:
.endm


/* void dl_bitrev2r_sc16_arp4_(int16_t *data, int cpx_points, int log2N);
 *
 * In-place bit-reversal permutation for 16-bit complex samples
 * (interleaved Re/Im, 32 bits per cpx point). cpx_points must be a power
 * of 2 and >= 8. log2N selects the FFT_BIT_WIDTH operating mode of
 * ESP.BITREV (3..10).
 *
 * Each outer iteration consumes 8 cpx points: ESP.BITREV produces 8
 * bit-reversed indices in q2 (lanes 0..7) and auto-increments the rs1
 * counter by 8. sel8=0..3 are used by the first half (4 ldxq/stxq) and
 * sel8=4..7 by the second half. Therefore ESP.BITREV must be called
 * exactly ONCE per outer iteration.
 *
 * Args (RV32 ABI):
 *   a0 = data, a1 = cpx_points, a2 = log2N
 *
 * GPR usage (all PIE-valid):
 *   a0  : data base for ldxq.32/stxq.32 (must NOT be advanced)
 *   a3  : vld pointer (post-incremented by esp.vld.128.ip)
 *   a4  : vst pointer (post-incremented by esp.vst.128.ip)
 *   t3  : ESP.BITREV index counter (auto += 8 per call)
 *   a1  : outer-loop counter (only used by base-RV insns)
 */
    .align 4
    .global dl_bitrev2r_sc16_arp4_
    .type   dl_bitrev2r_sc16_arp4_, @function
dl_bitrev2r_sc16_arp4_:
    srli            a1, a1, 3
    esp.movx.w.fft.bit.width a2
    li              t3, 0
    mv              a3, a0
    mv              a4, a0

    beqz            a1, .Ldl_bitrev_loop_end
.Ldl_bitrev_loop:
    esp.vld.128.ip  q0, a3, 16
    esp.fft.bitrev  q2, t3
    esp.ldxq.32     q1, q2, a0, 0, 0
    esp.ldxq.32     q1, q2, a0, 1, 1
    esp.ldxq.32     q1, q2, a0, 2, 2
    esp.ldxq.32     q1, q2, a0, 3, 3
    esp.stxq.32     q0, q2, a0, 0, 0
    esp.stxq.32     q0, q2, a0, 1, 1
    esp.stxq.32     q0, q2, a0, 2, 2
    esp.stxq.32     q0, q2, a0, 3, 3
    esp.vst.128.ip  q1, a4, 16

    esp.vld.128.ip  q0, a3, 16
    esp.ldxq.32     q1, q2, a0, 0, 4
    esp.ldxq.32     q1, q2, a0, 1, 5
    esp.ldxq.32     q1, q2, a0, 2, 6
    esp.ldxq.32     q1, q2, a0, 3, 7
    esp.stxq.32     q0, q2, a0, 0, 4
    esp.stxq.32     q0, q2, a0, 1, 5
    esp.stxq.32     q0, q2, a0, 2, 6
    esp.stxq.32     q0, q2, a0, 3, 7
    esp.vst.128.ip  q1, a4, 16

    addi            a1, a1, -1
    bnez            a1, .Ldl_bitrev_loop
.Ldl_bitrev_loop_end:
    ret


    .align 4
    .global dl_fft_reduce_max_arp4_
    .type   dl_fft_reduce_max_arp4_, @function
dl_fft_reduce_max_arp4_:
    esp.vld.128.ip  q0, a0, 16
    srli            a1, a1, 3
    addi            a1, a1, -1
    beqz            a1, .Lreduce_max_hreduce
.Lreduce_max_loop:
    esp.vld.128.ip  q1, a0, 16
    esp.vmax.s16    q0, q0, q1
    addi            a1, a1, -1
    bnez            a1, .Lreduce_max_loop
.Lreduce_max_hreduce:
    /* Horizontal reduction: 8 lanes -> 1 */
    li              t3, 8
    esp.movx.w.sar.bytes t3
    esp.src.q       q1, q0, q0
    esp.vmax.s16    q0, q0, q1
    li              t3, 4
    esp.movx.w.sar.bytes t3
    esp.src.q       q1, q0, q0
    esp.vmax.s16    q0, q0, q1
    li              t3, 2
    esp.movx.w.sar.bytes t3
    esp.src.q       q1, q0, q0
    esp.vmax.s16    q0, q0, q1
    esp.movi.32.a   q0, a0, 0
    slli            a0, a0, 16
    srai            a0, a0, 16
    ret


/* int16_t dl_reduce_abs_max_arp4_(const int16_t *inputs, int len);
 *
 * Maximum absolute value over a 16-byte aligned int16_t array.
 * abs(INT16_MIN) saturates to INT16_MAX. len must be >= 8 and a
 * multiple of 8. Thin wrapper around reduce_abs_max_arp4_macro.
 */
    .align 4
    .global dl_reduce_abs_max_arp4_
    .type   dl_reduce_abs_max_arp4_, @function
dl_reduce_abs_max_arp4_:
    reduce_abs_max_arp4_macro a0, a1, a0, t3
    ret


/* void dl_fft2r_sc16_dif_arp4_(int16_t *data, int16_t *table,
 *                              int shift, int num_stages, int cpx_points);
 *
 * DIF radix-2 FFT butterfly using ESP32-P4 PIE FFT instructions.
 * `table` must be in concatenated DIF layout (not bit-reversed). `shift`
 * is the per-multiply right-shift programmed into SAR (typical: 14).
 * cpx_points must be a power of 2 and >= 16.
 *
 * Args (RV32 ABI):
 *   a0 = data, a1 = table, a2 = shift, a3 = num_stages, a4 = cpx_points
 *
 * GPR usage (all PIE-valid):
 *   a0  : data base (preserved)
 *   a1  : working twiddle ptr (per-stage), broadcast base for last1
 *   a2  : working data ptr (per-box)
 *   a3  : outer-stage counter (= num_stages - 3, decremented to 0)
 *   a4  : working cpx_points (halved each stage)
 *   a5  : upper-half read ptr           (Xtensa a7)
 *   t3  : upper-half write ptr          (Xtensa a8)
 *   t4  : lower-half read ptr           (Xtensa a9)
 *   t5  : lower-half write ptr          (Xtensa a10)
 *   t6  : constant 16                   (Xtensa a11)
 *   s0  : box count per stage           (Xtensa a12, callee-saved)
 *   s1  : current twiddle ptr           (Xtensa a13, callee-saved)
 *   s8  : box counter                   (Xtensa a14, callee-saved)
 *   s9  : inner butterfly count         (Xtensa a15, callee-saved)
 *   t0  : scratch (loop counter / addx2/addx4 multiplier)
 */
    .text
    .align 4
    .global dl_fft2r_sc16_dif_arp4_
    .type   dl_fft2r_sc16_dif_arp4_, @function
dl_fft2r_sc16_dif_arp4_:
    addi            sp, sp, -16
    sw              s0, 12(sp)
    sw              s1, 8(sp)
    sw              s8, 4(sp)
    sw              s9, 0(sp)

    esp.movx.w.sar  a2
    mv              s1, a1                /* s1 = a1 (table)         */
    addi            a3, a3, -3            /* a3 = num_stages - 3     */
    li              s0, 1                 /* s0 = box count          */
    mv              s8, s0                /* s8 = box counter        */
    srli            s9, a4, 4             /* s9 = cpx_points / 16    */
    li              t6, 16

.Ldl_dif_stage:
    mv              a2, a0                /* per-stage data working ptr  */
    mv              a1, s1                /* per-stage twiddle backup    */

.Ldl_dif_box:
    mv              t4, a2                /* t4 = lower read ptr     */
    mv              t5, t4                /* t5 = lower write ptr    */
    slli            a5, a4, 1             /* a5 = a4*2 (= addx2 a7)  */
    add             a5, a5, a2            /* a5 = a2 + 2*cpx_points  */
    mv              t3, a5                /* t3 = upper write ptr    */
    mv              s1, a1                /* reset s1 from backup    */

    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, a5, 16
    esp.vld.128.ip  q2, s1, 16

    addi            t0, s9, 0
    beqz            t0, .Ldl_dif_bf_end
.Ldl_dif_bf_loop:
    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
    esp.fft.cmul.s16.ld.xp   q4, s1, t6, q5, q3, q2, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 2
    esp.fft.cmul.s16.ld.xp   q1, a5, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t3, t6, 6, 0, 1

    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
    esp.fft.cmul.s16.ld.xp   q2, s1, t6, q5, q3, q4, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q4, 2
    esp.fft.cmul.s16.ld.xp   q1, a5, t6, q5, q3, q4, 4
    esp.fft.cmul.s16.st.xp   q3, q4, q5, t3, t6, 6, 0, 1

    addi            t0, t0, -1
    bnez            t0, .Ldl_dif_bf_loop
.Ldl_dif_bf_end:

    slli            t0, a4, 2             /* a2 += 4*cpx_points (addx4)*/
    add             a2, t0, a2
    addi            s8, s8, -1
    sub             s1, s1, t6
    bnez            s8, .Ldl_dif_box

    srli            s9, s9, 1
    srli            a4, a4, 1
    slli            s0, s0, 1
    mv              s8, s0
    addi            a3, a3, -1
    bnez            a3, .Ldl_dif_stage

    /* ===== Last 3rd stage ===== */
    mv              t4, a0
    mv              t5, t4
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vld.128.ip  q2, s1, 16
    li              s8, 0                 /* zero offset             */

    addi            t0, s0, 0
    beqz            t0, .Ldl_dif_last3_end
.Ldl_dif_last3_loop:
    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
    esp.fft.cmul.s16.ld.xp   q4, s1, s8, q5, q3, q2, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 2
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 6, 0, 1
    addi            t0, t0, -1
    bnez            t0, .Ldl_dif_last3_loop
.Ldl_dif_last3_end:

    /* ===== Last 2nd stage ===== */
    mv              t4, a0
    mv              t5, t4
    mv              a1, s1                /* save twiddle ptr for last1 */

    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vld.h.64.ip q2, s1, 0

    addi            t0, s0, 0
    beqz            t0, .Ldl_dif_last2_end
.Ldl_dif_last2_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 6, 1, 1
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 4
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 6, 1, 1
    addi            t0, t0, -1
    bnez            t0, .Ldl_dif_last2_loop
.Ldl_dif_last2_end:

    /* ===== Last 1st stage ===== */
    mv              t4, a0
    mv              t5, t4
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vldbc.32.ip q2, a1, 0

    addi            t0, s0, 0
    beqz            t0, .Ldl_dif_last1_end
.Ldl_dif_last1_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 6, 2, 1
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 4
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 6, 2, 1
    addi            t0, t0, -1
    bnez            t0, .Ldl_dif_last1_loop
.Ldl_dif_last1_end:

    lw              s9, 0(sp)
    lw              s8, 4(sp)
    lw              s1, 8(sp)
    lw              s0, 12(sp)
    addi            sp, sp, 16
    ret


/* void dl_fft2r_sc16_dif_hp_arp4_(int16_t *data, int16_t *table,
 *                                 int num_stages, int cpx_points,
 *                                 int *out_shift);
 *
 * High-precision DIF radix-2 FFT with dynamic per-stage SAR (sar4) and
 * per-stage SAR (post-multiply shift) selection driven by an SIMD
 * absolute-max scan of the live data range.
 *
 * For each stage we pick:
 *   sar=15, sar4=1   if max_abs(data) >= 16384 (data uses full range)
 *   sar=14, sar4=0   otherwise (extra precision)
 *
 *
 * Args (RV32 ABI):
 *   a0 = data, a1 = table, a2 = num_stages, a3 = cpx_points, a4 = out_shift
 *
 * GPR usage (PIE-valid):
 *   a0  : data base (preserved)
 *   a1  : working twiddle ptr (per-stage, broadcast base for last1)
 *   a2  : outer-stage counter (= num_stages - 3)
 *   a3  : working cpx_points (halved each stage)
 *   a4  : *out_shift pointer (preserved)
 *   a5  : upper read ptr (Xtensa a7) / abs-max temp / shift temp
 *   t3  : SAR shift / upper write ptr (Xtensa a8)
 *   t4  : lower read ptr (Xtensa a9)
 *   t5  : lower write ptr (Xtensa a10)
 *   t6  : constant 16 (Xtensa a11)
 *   s0  : box count
 *   s1  : current twiddle ptr (also used by butterfly macro)
 *   s8  : box counter / zero offset for last3
 *   s9  : inner butterfly count
 *   s10 : per-stage data working ptr (corresponds to Xtensa "mov a4, a2")
 *   s11 : per-stage twiddle backup (Xtensa "mov a3, a13")
 */
    .text
    .align 4
    .global dl_fft2r_sc16_dif_hp_arp4_
    .type   dl_fft2r_sc16_dif_hp_arp4_, @function
dl_fft2r_sc16_dif_hp_arp4_:
    addi            sp, sp, -32
    sw              s0, 28(sp)
    sw              s1, 24(sp)
    sw              s8, 20(sp)
    sw              s9, 16(sp)
    sw              s10, 12(sp)
    sw              s11, 8(sp)

    /* *out_shift = 0 */
    sw              x0, 0(a4)

    mv              s1, a1                /* s1 = table             */
    addi            a2, a2, -3            /* a2 = num_stages - 3    */
    li              s0, 1
    mv              s8, s0
    srli            s9, a3, 4
    li              t6, 16

    /* if (num_stages-3) < 1 -> jump to last3 */
    li              t0, 1
    blt             a2, t0, .Ldl_dif_hp_last3

.Ldl_dif_hp_stage:
    /* --- 1. SIMD abs-max scan over data[0 .. cpx_points * box_count) --- */
    mul             a5, a3, s0            /* a5 = cpx_points * box_cnt */
    slli            t3, a5, 1             /* t3 = 2*a5 (int16 count)   */
    mv              a5, a0                /* a5 = data ptr             */
    reduce_abs_max_arp4_macro a5, t3, a5, t4

    /* --- 2. decide loop_shift: max_abs >= 16384 => 15, else 14 --- */
    li              t3, 15
    srli            a5, a5, 13
    bnez            a5, .Ldl_hp_shift_done
    li              t3, 14
.Ldl_hp_shift_done:
    esp.movx.w.sar  t3                    /* SAR = loop_shift          */

    /* --- 3. *out_shift += (loop_shift - 14) --- */
    lw              t4, 0(a4)
    addi            t5, t3, -14
    add             t4, t4, t5
    sw              t4, 0(a4)

    li              t6, 16

    /* --- 4. branch to sar4=0 (loop_shift=14) or sar4=1 (loop_shift=15) --- */
    andi            t0, t3, 1
    beqz            t0, .Ldl_hp_bf_sar0

    /* ===== sar4 = 1 ===== */
    mv              s10, a0
    mv              s11, s1
.Ldl_dif_hp_box_s1:
    mv              t4, s10
    mv              t5, t4
    slli            a5, a3, 1
    add             a5, a5, s10
    mv              t3, a5
    mv              s1, s11

    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, a5, 16
    esp.vld.128.ip  q2, s1, 16

    dif_hp_butterfly_inner_arp4 1

    slli            t0, a3, 2
    add             s10, t0, s10
    addi            s8, s8, -1
    sub             s1, s1, t6              /* roll back working twiddle ptr by 16 (mirrors S3 sub a13,a11) */
    bnez            s8, .Ldl_dif_hp_box_s1
    mv              s11, s1                 /* propagate final twiddle ptr to next stage */
    j               .Ldl_hp_stage_done

.Ldl_hp_bf_sar0:
    /* ===== sar4 = 0 ===== */
    mv              s10, a0
    mv              s11, s1
.Ldl_dif_hp_box_s0:
    mv              t4, s10
    mv              t5, t4
    slli            a5, a3, 1
    add             a5, a5, s10
    mv              t3, a5
    mv              s1, s11

    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, a5, 16
    esp.vld.128.ip  q2, s1, 16

    dif_hp_butterfly_inner_arp4 0

    slli            t0, a3, 2
    add             s10, t0, s10
    addi            s8, s8, -1
    sub             s1, s1, t6              /* roll back working twiddle ptr by 16 */
    bnez            s8, .Ldl_dif_hp_box_s0
    mv              s11, s1                 /* propagate final twiddle ptr to next stage */

.Ldl_hp_stage_done:
    srli            s9, s9, 1
    srli            a3, a3, 1
    slli            s0, s0, 1
    mv              s8, s0
    addi            a2, a2, -1
    bnez            a2, .Ldl_dif_hp_stage

.Ldl_dif_hp_last3:
    /* ----- Last 3rd stage ----- */
    mul             a5, a3, s0
    slli            t3, a5, 1
    mv              a5, a0
    reduce_abs_max_arp4_macro a5, t3, a5, t4

    li              t3, 15
    srli            a5, a5, 13
    bnez            a5, .Ldl_hp_l3_shift_done
    li              t3, 14
.Ldl_hp_l3_shift_done:
    esp.movx.w.sar  t3

    lw              t4, 0(a4)
    addi            t5, t3, -14
    add             t4, t4, t5
    sw              t4, 0(a4)

    li              t6, 16
    li              s8, 0                 /* zero offset for cmul   */
    mv              t4, a0
    mv              t5, t4
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vld.128.ip  q2, s1, 16

    andi            t0, t3, 1
    beqz            t0, .Ldl_hp_l3_sar0

    /* sar4=1 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_dif_hp_l3_done
.Ldl_dif_hp_l3s1_loop:
    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
    esp.fft.cmul.s16.ld.xp   q4, s1, s8, q5, q3, q2, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 2
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 6, 0, 0
    addi            t0, t0, -1
    bnez            t0, .Ldl_dif_hp_l3s1_loop
    j               .Ldl_dif_hp_l3_done

.Ldl_hp_l3_sar0:
    /* sar4=0 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_dif_hp_l3_done
.Ldl_dif_hp_l3s0_loop:
    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 0
    esp.fft.cmul.s16.ld.xp   q4, s1, s8, q5, q3, q2, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 2
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 6, 0, 0
    addi            t0, t0, -1
    bnez            t0, .Ldl_dif_hp_l3s0_loop
.Ldl_dif_hp_l3_done:

    /* ----- Last 2nd stage ----- */
    mul             a5, a3, s0
    slli            t3, a5, 1
    mv              a5, a0
    reduce_abs_max_arp4_macro a5, t3, a5, t4

    li              t3, 15
    srli            a5, a5, 13
    bnez            a5, .Ldl_hp_l2_shift_done
    li              t3, 14
.Ldl_hp_l2_shift_done:
    esp.movx.w.sar  t3

    lw              t4, 0(a4)
    addi            t5, t3, -14
    add             t4, t4, t5
    sw              t4, 0(a4)

    li              t6, 16
    mv              t4, a0
    mv              t5, t4
    mv              a1, s1                /* save twiddle for last1 */
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vld.h.64.ip q2, s1, 0

    andi            t0, t3, 1
    beqz            t0, .Ldl_hp_l2_sar0

    /* sar4=1 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_dif_hp_l2_done
.Ldl_dif_hp_l2s1_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 6, 1, 1
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 4
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 6, 1, 1
    addi            t0, t0, -1
    bnez            t0, .Ldl_dif_hp_l2s1_loop
    j               .Ldl_dif_hp_l2_done

.Ldl_hp_l2_sar0:
    /* sar4=0 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_dif_hp_l2_done
.Ldl_dif_hp_l2s0_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 6, 1, 0
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 4
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 6, 1, 0
    addi            t0, t0, -1
    bnez            t0, .Ldl_dif_hp_l2s0_loop
.Ldl_dif_hp_l2_done:

    /* ----- Last 1st stage ----- */
    mul             a5, a3, s0
    slli            t3, a5, 1
    mv              a5, a0
    reduce_abs_max_arp4_macro a5, t3, a5, t4

    li              t3, 15
    srli            a5, a5, 13
    bnez            a5, .Ldl_hp_l1_shift_done
    li              t3, 14
.Ldl_hp_l1_shift_done:
    esp.movx.w.sar  t3

    lw              t4, 0(a4)
    addi            t5, t3, -14
    add             t4, t4, t5
    sw              t4, 0(a4)

    li              t6, 16
    mv              t4, a0
    mv              t5, t4
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vldbc.32.ip q2, a1, 0

    andi            t0, t3, 1
    beqz            t0, .Ldl_hp_l1_sar0

    /* sar4=1 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_dif_hp_l1_done
.Ldl_dif_hp_l1s1_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 6, 2, 1
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 4
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 6, 2, 1
    addi            t0, t0, -1
    bnez            t0, .Ldl_dif_hp_l1s1_loop
    j               .Ldl_dif_hp_l1_done

.Ldl_hp_l1_sar0:
    /* sar4=0 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_dif_hp_l1_done
.Ldl_dif_hp_l1s0_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 4
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 6, 2, 0
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 4
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 6, 2, 0
    addi            t0, t0, -1
    bnez            t0, .Ldl_dif_hp_l1s0_loop
.Ldl_dif_hp_l1_done:

    lw              s11, 8(sp)
    lw              s10, 12(sp)
    lw              s9, 16(sp)
    lw              s8, 20(sp)
    lw              s1, 24(sp)
    lw              s0, 28(sp)
    addi            sp, sp, 32
    ret


/* void dl_ifft2r_sc16_dif_arp4_(int16_t *data, int16_t *table,
 *                               int shift, int num_stages, int cpx_points);
 *
 * DIF radix-2 IFFT butterfly using ESP32-P4 PIE FFT instructions.
 * Mirrors dl_fft2r_sc16_dif_arp4_; the only difference is the cmul
 * sel8 immediate (incremented by 1) which switches the hardware
 * complex multiplier to its conjugate path (e^{+jθ}). Same calling
 * convention and register usage as the forward variant.
 *
 * Args (RV32 ABI):
 *   a0 = data, a1 = table, a2 = shift, a3 = num_stages, a4 = cpx_points
 */
    .text
    .align 4
    .global dl_ifft2r_sc16_dif_arp4_
    .type   dl_ifft2r_sc16_dif_arp4_, @function
dl_ifft2r_sc16_dif_arp4_:
    addi            sp, sp, -16
    sw              s0, 12(sp)
    sw              s1, 8(sp)
    sw              s8, 4(sp)
    sw              s9, 0(sp)

    esp.movx.w.sar  a2
    mv              s1, a1
    addi            a3, a3, -3
    li              s0, 1
    mv              s8, s0
    srli            s9, a4, 4
    li              t6, 16

.Ldl_idif_stage:
    mv              a2, a0
    mv              a1, s1

.Ldl_idif_box:
    mv              t4, a2
    mv              t5, t4
    slli            a5, a4, 1
    add             a5, a5, a2
    mv              t3, a5
    mv              s1, a1

    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, a5, 16
    esp.vld.128.ip  q2, s1, 16

    addi            t0, s9, 0
    beqz            t0, .Ldl_idif_bf_end
.Ldl_idif_bf_loop:
    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
    esp.fft.cmul.s16.ld.xp   q4, s1, t6, q5, q3, q2, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 3
    esp.fft.cmul.s16.ld.xp   q1, a5, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t3, t6, 7, 0, 1

    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
    esp.fft.cmul.s16.ld.xp   q2, s1, t6, q5, q3, q4, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q4, 3
    esp.fft.cmul.s16.ld.xp   q1, a5, t6, q5, q3, q4, 5
    esp.fft.cmul.s16.st.xp   q3, q4, q5, t3, t6, 7, 0, 1

    addi            t0, t0, -1
    bnez            t0, .Ldl_idif_bf_loop
.Ldl_idif_bf_end:

    slli            t0, a4, 2
    add             a2, t0, a2
    addi            s8, s8, -1
    sub             s1, s1, t6
    bnez            s8, .Ldl_idif_box

    srli            s9, s9, 1
    srli            a4, a4, 1
    slli            s0, s0, 1
    mv              s8, s0
    addi            a3, a3, -1
    bnez            a3, .Ldl_idif_stage

    /* ===== Last 3rd stage ===== */
    mv              t4, a0
    mv              t5, t4
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vld.128.ip  q2, s1, 16
    li              s8, 0

    addi            t0, s0, 0
    beqz            t0, .Ldl_idif_last3_end
.Ldl_idif_last3_loop:
    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
    esp.fft.cmul.s16.ld.xp   q4, s1, s8, q5, q3, q2, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 3
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 7, 0, 1
    addi            t0, t0, -1
    bnez            t0, .Ldl_idif_last3_loop
.Ldl_idif_last3_end:

    /* ===== Last 2nd stage ===== */
    mv              t4, a0
    mv              t5, t4
    mv              a1, s1

    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vld.h.64.ip q2, s1, 0

    addi            t0, s0, 0
    beqz            t0, .Ldl_idif_last2_end
.Ldl_idif_last2_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 7, 1, 1
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 5
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 7, 1, 1
    addi            t0, t0, -1
    bnez            t0, .Ldl_idif_last2_loop
.Ldl_idif_last2_end:

    /* ===== Last 1st stage ===== */
    mv              t4, a0
    mv              t5, t4
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vldbc.32.ip q2, a1, 0

    addi            t0, s0, 0
    beqz            t0, .Ldl_idif_last1_end
.Ldl_idif_last1_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 7, 2, 1
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 5
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 7, 2, 1
    addi            t0, t0, -1
    bnez            t0, .Ldl_idif_last1_loop
.Ldl_idif_last1_end:

    lw              s9, 0(sp)
    lw              s8, 4(sp)
    lw              s1, 8(sp)
    lw              s0, 12(sp)
    addi            sp, sp, 16
    ret


/* void dl_ifft2r_sc16_dif_hp_arp4_(int16_t *data, int16_t *table,
 *                                  int num_stages, int cpx_points,
 *                                  int *out_shift);
 *
 * High-precision DIF radix-2 IFFT with dynamic per-stage SAR (sar4)
 * driven by an SIMD absolute-max scan, mirroring
 * dl_fft2r_sc16_dif_hp_arp4_. The forward butterfly is replaced with
 * the IFFT macro `ifft_hp_butterfly_inner_arp4`; per-stage last1/last2/
 * last3 sections also use sel8 ∈ {1,3,5,7} (conjugate cmul path).
 *
 * Args (RV32 ABI):
 *   a0 = data, a1 = table, a2 = num_stages, a3 = cpx_points, a4 = out_shift
 */
    .text
    .align 4
    .global dl_ifft2r_sc16_dif_hp_arp4_
    .type   dl_ifft2r_sc16_dif_hp_arp4_, @function
dl_ifft2r_sc16_dif_hp_arp4_:
    addi            sp, sp, -32
    sw              s0, 28(sp)
    sw              s1, 24(sp)
    sw              s8, 20(sp)
    sw              s9, 16(sp)
    sw              s10, 12(sp)
    sw              s11, 8(sp)

    sw              x0, 0(a4)

    mv              s1, a1
    addi            a2, a2, -3
    li              s0, 1
    mv              s8, s0
    srli            s9, a3, 4
    li              t6, 16

    li              t0, 1
    blt             a2, t0, .Ldl_idif_hp_last3

.Ldl_idif_hp_stage:
    /* --- 1. SIMD abs-max scan --- */
    mul             a5, a3, s0
    slli            t3, a5, 1
    mv              a5, a0
    reduce_abs_max_arp4_macro a5, t3, a5, t4

    /* --- 2. decide loop_shift --- */
    li              t3, 15
    srli            a5, a5, 13
    bnez            a5, .Ldl_ihp_shift_done
    li              t3, 14
.Ldl_ihp_shift_done:
    esp.movx.w.sar  t3

    /* --- 3. *out_shift += (loop_shift - 14) --- */
    lw              t4, 0(a4)
    addi            t5, t3, -14
    add             t4, t4, t5
    sw              t4, 0(a4)

    li              t6, 16

    /* --- 4. branch on sar4 --- */
    andi            t0, t3, 1
    beqz            t0, .Ldl_ihp_bf_sar0

    /* ===== sar4 = 1 ===== */
    mv              s10, a0
    mv              s11, s1
.Ldl_idif_hp_box_s1:
    mv              t4, s10
    mv              t5, t4
    slli            a5, a3, 1
    add             a5, a5, s10
    mv              t3, a5
    mv              s1, s11

    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, a5, 16
    esp.vld.128.ip  q2, s1, 16

    ifft_hp_butterfly_inner_arp4 1

    slli            t0, a3, 2
    add             s10, t0, s10
    addi            s8, s8, -1
    sub             s1, s1, t6
    bnez            s8, .Ldl_idif_hp_box_s1
    mv              s11, s1
    j               .Ldl_ihp_stage_done

.Ldl_ihp_bf_sar0:
    /* ===== sar4 = 0 ===== */
    mv              s10, a0
    mv              s11, s1
.Ldl_idif_hp_box_s0:
    mv              t4, s10
    mv              t5, t4
    slli            a5, a3, 1
    add             a5, a5, s10
    mv              t3, a5
    mv              s1, s11

    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, a5, 16
    esp.vld.128.ip  q2, s1, 16

    ifft_hp_butterfly_inner_arp4 0

    slli            t0, a3, 2
    add             s10, t0, s10
    addi            s8, s8, -1
    sub             s1, s1, t6
    bnez            s8, .Ldl_idif_hp_box_s0
    mv              s11, s1

.Ldl_ihp_stage_done:
    srli            s9, s9, 1
    srli            a3, a3, 1
    slli            s0, s0, 1
    mv              s8, s0
    addi            a2, a2, -1
    bnez            a2, .Ldl_idif_hp_stage

.Ldl_idif_hp_last3:
    /* ----- Last 3rd stage ----- */
    mul             a5, a3, s0
    slli            t3, a5, 1
    mv              a5, a0
    reduce_abs_max_arp4_macro a5, t3, a5, t4

    li              t3, 15
    srli            a5, a5, 13
    bnez            a5, .Ldl_ihp_l3_shift_done
    li              t3, 14
.Ldl_ihp_l3_shift_done:
    esp.movx.w.sar  t3

    lw              t4, 0(a4)
    addi            t5, t3, -14
    add             t4, t4, t5
    sw              t4, 0(a4)

    li              t6, 16
    li              s8, 0
    mv              t4, a0
    mv              t5, t4
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vld.128.ip  q2, s1, 16

    andi            t0, t3, 1
    beqz            t0, .Ldl_ihp_l3_sar0

    /* sar4=1 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_idif_hp_l3_done
.Ldl_idif_hp_l3s1_loop:
    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
    esp.fft.cmul.s16.ld.xp   q4, s1, s8, q5, q3, q2, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 3
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 7, 0, 0
    addi            t0, t0, -1
    bnez            t0, .Ldl_idif_hp_l3s1_loop
    j               .Ldl_idif_hp_l3_done

.Ldl_ihp_l3_sar0:
    /* sar4=0 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_idif_hp_l3_done
.Ldl_idif_hp_l3s0_loop:
    esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 0
    esp.fft.cmul.s16.ld.xp   q4, s1, s8, q5, q3, q2, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 3
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 7, 0, 0
    addi            t0, t0, -1
    bnez            t0, .Ldl_idif_hp_l3s0_loop
.Ldl_idif_hp_l3_done:

    /* ----- Last 2nd stage ----- */
    mul             a5, a3, s0
    slli            t3, a5, 1
    mv              a5, a0
    reduce_abs_max_arp4_macro a5, t3, a5, t4

    li              t3, 15
    srli            a5, a5, 13
    bnez            a5, .Ldl_ihp_l2_shift_done
    li              t3, 14
.Ldl_ihp_l2_shift_done:
    esp.movx.w.sar  t3

    lw              t4, 0(a4)
    addi            t5, t3, -14
    add             t4, t4, t5
    sw              t4, 0(a4)

    li              t6, 16
    mv              t4, a0
    mv              t5, t4
    mv              a1, s1
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vld.h.64.ip q2, s1, 0

    andi            t0, t3, 1
    beqz            t0, .Ldl_ihp_l2_sar0

    /* sar4=1 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_idif_hp_l2_done
.Ldl_idif_hp_l2s1_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 7, 1, 1
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 5
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 7, 1, 1
    addi            t0, t0, -1
    bnez            t0, .Ldl_idif_hp_l2s1_loop
    j               .Ldl_idif_hp_l2_done

.Ldl_ihp_l2_sar0:
    /* sar4=0 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_idif_hp_l2_done
.Ldl_idif_hp_l2s0_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 0
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 7, 1, 0
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 5
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 7, 1, 0
    addi            t0, t0, -1
    bnez            t0, .Ldl_idif_hp_l2s0_loop
.Ldl_idif_hp_l2_done:

    /* ----- Last 1st stage ----- */
    mul             a5, a3, s0
    slli            t3, a5, 1
    mv              a5, a0
    reduce_abs_max_arp4_macro a5, t3, a5, t4

    li              t3, 15
    srli            a5, a5, 13
    bnez            a5, .Ldl_ihp_l1_shift_done
    li              t3, 14
.Ldl_ihp_l1_shift_done:
    esp.movx.w.sar  t3

    lw              t4, 0(a4)
    addi            t5, t3, -14
    add             t4, t4, t5
    sw              t4, 0(a4)

    li              t6, 16
    mv              t4, a0
    mv              t5, t4
    esp.vld.128.ip  q0, t4, 16
    esp.vld.128.ip  q1, t4, 16
    esp.vldbc.32.ip q2, a1, 0

    andi            t0, t3, 1
    beqz            t0, .Ldl_ihp_l1_sar0

    /* sar4=1 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_idif_hp_l1_done
.Ldl_idif_hp_l1s1_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 7, 2, 1
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 5
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 7, 2, 1
    addi            t0, t0, -1
    bnez            t0, .Ldl_idif_hp_l1s1_loop
    j               .Ldl_idif_hp_l1_done

.Ldl_ihp_l1_sar0:
    /* sar4=0 */
    addi            t0, s0, 0
    beqz            t0, .Ldl_idif_hp_l1_done
.Ldl_idif_hp_l1s0_loop:
    esp.fft.r2bf.s16         q3, q4, q0, q1, 1
    esp.fft.cmul.s16.ld.xp   q0, t4, t6, q5, q3, q2, 5
    esp.fft.cmul.s16.st.xp   q3, q2, q5, t5, t6, 7, 2, 0
    esp.fft.cmul.s16.ld.xp   q1, t4, t6, q5, q4, q2, 5
    esp.fft.cmul.s16.st.xp   q4, q2, q5, t5, t6, 7, 2, 0
    addi            t0, t0, -1
    bnez            t0, .Ldl_idif_hp_l1s0_loop
.Ldl_idif_hp_l1_done:

    lw              s11, 8(sp)
    lw              s10, 12(sp)
    lw              s9, 16(sp)
    lw              s8, 20(sp)
    lw              s1, 24(sp)
    lw              s0, 28(sp)
    addi            sp, sp, 32
    ret
