/*
 * FPU-accelerated RMSNormalization requantization kernels for ESP32-P4 (PIE V2).
 *
 * These kernels fuse the per-element weight multiply, the per-slice scalar (rms) multiply,
 * the rounding and the saturation of the inner loop:
 *
 *     for (int j = 0; j < n; j++)
 *         output[j] = clip( round_half_even(input[j] * (*rms) * scale[j]), TYPE_MIN, TYPE_MAX );
 *
 * ESP32-P4 uses "round half to even", which is exactly what the RISC-V `fcvt.w.s rd, fs, rne`
 * instruction does, so this scalar-FPU path stays bit-exact with the C reference
 * (dl::tool::round -> round_half_even -> fcvt.w.s rne, plus dl::tool::truncate). The work is
 * 4-way unrolled to keep the FP pipeline busy and to avoid the per-element non-inlined
 * dl::tool::round() function call paid by the C loop.
 *
 * The PIE SIMD engine on the P4 is integer-only and cannot reproduce the per-slice float
 * scalar multiply with bit-exact round-half-even, so the scalar FPU is used here (no PIE
 * vector registers are touched, hence no 16-byte alignment requirement). The remainder
 * (n % 4) is handled by an in-kernel tail loop.
 *
 * Saturation is performed in the float domain with fmin.s/fmax.s before the conversion. This
 * is equivalent to "round then integer-clip": any value that rounding would push outside the
 * type range is already pinned to the boundary, and boundary values are exactly representable
 * as float, so the rounded result is identical.
 */

/* void dl_esp32p4_rmsnorm_s16(int16_t *output, int16_t *input, float *scale, float *rms, int n); */
    .align 2
    .text
    .global dl_esp32p4_rmsnorm_s16
    .type   dl_esp32p4_rmsnorm_s16, @function
    .balign 4
    .option norvc
dl_esp32p4_rmsnorm_s16:
    # a0 = output, a1 = input, a2 = scale, a3 = rms (float*), a4 = n
    flw      ft0, 0(a3)              # ft0 = *rms
    li       t0, 32767
    fcvt.s.w ft2, t0                 # ft2 = 32767.0f  (INT16_MAX)
    li       t1, -32768
    fcvt.s.w ft3, t1                 # ft3 = -32768.0f (INT16_MIN)

    srai     t2, a4, 2               # t2 = n / 4
    beqz     t2, .Lrmsnorm_s16_tail
.Lrmsnorm_s16_main:
    lh       t3, 0(a1)
    lh       t4, 2(a1)
    lh       t5, 4(a1)
    lh       t6, 6(a1)
    fcvt.s.w ft4, t3
    fcvt.s.w ft5, t4
    fcvt.s.w ft6, t5
    fcvt.s.w ft7, t6
    flw      ft8,  0(a2)             # scale is float
    flw      ft9,  4(a2)
    flw      ft10, 8(a2)
    flw      ft11, 12(a2)
    fmul.s   ft4, ft4, ft0           # input * rms
    fmul.s   ft5, ft5, ft0
    fmul.s   ft6, ft6, ft0
    fmul.s   ft7, ft7, ft0
    fmul.s   ft4, ft4, ft8           # (input * rms) * scale
    fmul.s   ft5, ft5, ft9
    fmul.s   ft6, ft6, ft10
    fmul.s   ft7, ft7, ft11
    fmin.s   ft4, ft4, ft2           # clip to int16 range
    fmax.s   ft4, ft4, ft3
    fmin.s   ft5, ft5, ft2
    fmax.s   ft5, ft5, ft3
    fmin.s   ft6, ft6, ft2
    fmax.s   ft6, ft6, ft3
    fmin.s   ft7, ft7, ft2
    fmax.s   ft7, ft7, ft3
    fcvt.w.s t3, ft4, rne            # round half to even
    fcvt.w.s t4, ft5, rne
    fcvt.w.s t5, ft6, rne
    fcvt.w.s t6, ft7, rne
    sh       t3, 0(a0)
    sh       t4, 2(a0)
    sh       t5, 4(a0)
    sh       t6, 6(a0)
    addi     a1, a1, 8
    addi     a2, a2, 16              # scale is float (4 bytes each)
    addi     a0, a0, 8
    addi     t2, t2, -1
    bnez     t2, .Lrmsnorm_s16_main

.Lrmsnorm_s16_tail:
    andi     t2, a4, 3               # t2 = n % 4
    beqz     t2, .Lrmsnorm_s16_done
.Lrmsnorm_s16_tail_loop:
    lh       t3, 0(a1)
    flw      ft8, 0(a2)              # scale is float
    fcvt.s.w ft4, t3
    fmul.s   ft4, ft4, ft0
    fmul.s   ft4, ft4, ft8
    fmin.s   ft4, ft4, ft2
    fmax.s   ft4, ft4, ft3
    fcvt.w.s t3, ft4, rne
    sh       t3, 0(a0)
    addi     a1, a1, 2
    addi     a2, a2, 4               # scale is float (4 bytes each)
    addi     a0, a0, 2
    addi     t2, t2, -1
    bnez     t2, .Lrmsnorm_s16_tail_loop
.Lrmsnorm_s16_done:
    ret


/* void dl_esp32p4_rmsnorm_s8(int8_t *output, int8_t *input, float *scale, float *rms, int n); */
    .align 2
    .text
    .global dl_esp32p4_rmsnorm_s8
    .type   dl_esp32p4_rmsnorm_s8, @function
    .balign 4
    .option norvc
dl_esp32p4_rmsnorm_s8:
    # a0 = output, a1 = input, a2 = scale, a3 = rms (float*), a4 = n
    flw      ft0, 0(a3)              # ft0 = *rms
    li       t0, 127
    fcvt.s.w ft2, t0                 # ft2 = 127.0f  (INT8_MAX)
    li       t1, -128
    fcvt.s.w ft3, t1                 # ft3 = -128.0f (INT8_MIN)

    srai     t2, a4, 2               # t2 = n / 4
    beqz     t2, .Lrmsnorm_s8_tail
.Lrmsnorm_s8_main:
    lb       t3, 0(a1)               # lb sign-extends int8
    lb       t4, 1(a1)
    lb       t5, 2(a1)
    lb       t6, 3(a1)
    fcvt.s.w ft4, t3
    fcvt.s.w ft5, t4
    fcvt.s.w ft6, t5
    fcvt.s.w ft7, t6
    flw      ft8,  0(a2)             # scale is float
    flw      ft9,  4(a2)
    flw      ft10, 8(a2)
    flw      ft11, 12(a2)
    fmul.s   ft4, ft4, ft0           # input * rms
    fmul.s   ft5, ft5, ft0
    fmul.s   ft6, ft6, ft0
    fmul.s   ft7, ft7, ft0
    fmul.s   ft4, ft4, ft8           # (input * rms) * scale
    fmul.s   ft5, ft5, ft9
    fmul.s   ft6, ft6, ft10
    fmul.s   ft7, ft7, ft11
    fmin.s   ft4, ft4, ft2           # clip to int8 range
    fmax.s   ft4, ft4, ft3
    fmin.s   ft5, ft5, ft2
    fmax.s   ft5, ft5, ft3
    fmin.s   ft6, ft6, ft2
    fmax.s   ft6, ft6, ft3
    fmin.s   ft7, ft7, ft2
    fmax.s   ft7, ft7, ft3
    fcvt.w.s t3, ft4, rne            # round half to even
    fcvt.w.s t4, ft5, rne
    fcvt.w.s t5, ft6, rne
    fcvt.w.s t6, ft7, rne
    sb       t3, 0(a0)
    sb       t4, 1(a0)
    sb       t5, 2(a0)
    sb       t6, 3(a0)
    addi     a1, a1, 4
    addi     a2, a2, 16              # scale is float (4 bytes each)
    addi     a0, a0, 4
    addi     t2, t2, -1
    bnez     t2, .Lrmsnorm_s8_main

.Lrmsnorm_s8_tail:
    andi     t2, a4, 3               # t2 = n % 4
    beqz     t2, .Lrmsnorm_s8_done
.Lrmsnorm_s8_tail_loop:
    lb       t3, 0(a1)
    flw      ft8, 0(a2)              # scale is float
    fcvt.s.w ft4, t3
    fmul.s   ft4, ft4, ft0
    fmul.s   ft4, ft4, ft8
    fmin.s   ft4, ft4, ft2
    fmax.s   ft4, ft4, ft3
    fcvt.w.s t3, ft4, rne
    sb       t3, 0(a0)
    addi     a1, a1, 1
    addi     a2, a2, 4               # scale is float (4 bytes each)
    addi     a0, a0, 1
    addi     t2, t2, -1
    bnez     t2, .Lrmsnorm_s8_tail_loop
.Lrmsnorm_s8_done:
    ret
