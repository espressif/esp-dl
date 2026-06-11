/*
 * FPU-accelerated RMSNormalization requantization kernels for ESP32-S3 (PIE / tie728).
 *
 * These kernels fuse the per-element weight multiply, the per-slice scalar (rms) multiply,
 * the round-half-up and the saturation of the inner loop:
 *
 *     for (int j = 0; j < n; j++)
 *         output[j] = clip( floor(input[j] * (*rms) * scale[j] + 0.5f), TYPE_MIN, TYPE_MAX );
 *
 * ESP32-S3 uses "round half up", which is exactly floor(x + 0.5f), so the scalar FPU path
 * stays bit-exact with the C reference (dl::tool::round + dl::tool::truncate). The work is
 * 4-way unrolled to keep the FP pipeline busy and to avoid the per-element non-inlined
 * dl::tool::round() function call paid by the C loop.
 *
 * Only scalar loads/stores are used, so there is NO 16-byte alignment requirement and the
 * remainder (n % 4) is handled by an in-kernel tail loop.
 */

/* void dl_tie728_rmsnorm_s16(int16_t *output, int16_t *input, float *scale, float *rms, int n); */
    .text
    .align 4
    .global dl_tie728_rmsnorm_s16
    .type   dl_tie728_rmsnorm_s16, @function
dl_tie728_rmsnorm_s16:
    /* a2 = output
     * a3 = input
     * a4 = scale
     * a5 = rms (float*)
     * a6 = n
     */
    entry sp, 16

    lsi f0, a5, 0                  // f0 = *rms
    movi a7, 0x3f000000            // 0.5f bit pattern
    wfr f1, a7                     // f1 = 0.5f
    movi a12, 32767                // INT16_MAX (kept live across both loops)
    movi a13, -32768               // INT16_MIN

    srai a7, a6, 2                 // a7 = n / 4
    loopnez a7, dl_tie728_rmsnorm_s16_main_end
        l16si a8,  a3, 0
        l16si a9,  a3, 2
        l16si a10, a3, 4
        l16si a11, a3, 6
        float.s f2, a8, 0
        float.s f3, a9, 0
        float.s f4, a10, 0
        float.s f5, a11, 0
        lsi f6, a4, 0              // scale is float
        lsi f7, a4, 4
        lsi f8, a4, 8
        lsi f9, a4, 12
        mul.s f2, f2, f0           // input * rms
        mul.s f3, f3, f0
        mul.s f4, f4, f0
        mul.s f5, f5, f0
        mul.s f2, f2, f6           // (input * rms) * scale
        mul.s f3, f3, f7
        mul.s f4, f4, f8
        mul.s f5, f5, f9
        add.s f2, f2, f1           // + 0.5f
        add.s f3, f3, f1
        add.s f4, f4, f1
        add.s f5, f5, f1
        floor.s a8,  f2, 0         // round half up
        floor.s a9,  f3, 0
        floor.s a10, f4, 0
        floor.s a11, f5, 0
        min a8,  a8,  a12          // clip to int16 range
        max a8,  a8,  a13
        min a9,  a9,  a12
        max a9,  a9,  a13
        min a10, a10, a12
        max a10, a10, a13
        min a11, a11, a12
        max a11, a11, a13
        s16i a8,  a2, 0
        s16i a9,  a2, 2
        s16i a10, a2, 4
        s16i a11, a2, 6
        addi a3, a3, 8
        addi a4, a4, 16            // scale is float (4 bytes each)
        addi a2, a2, 8
    dl_tie728_rmsnorm_s16_main_end:

    extui a7, a6, 0, 2             // a7 = n % 4
    loopnez a7, dl_tie728_rmsnorm_s16_tail_end
        l16si a8, a3, 0
        float.s f2, a8, 0
        lsi f6, a4, 0              // scale is float
        mul.s f2, f2, f0
        mul.s f2, f2, f6
        add.s f2, f2, f1
        floor.s a8, f2, 0
        min a8, a8, a12
        max a8, a8, a13
        s16i a8, a2, 0
        addi a3, a3, 2
        addi a4, a4, 4             // scale is float (4 bytes each)
        addi a2, a2, 2
    dl_tie728_rmsnorm_s16_tail_end:

    retw


/* void dl_tie728_rmsnorm_s8(int8_t *output, int8_t *input, float *scale, float *rms, int n); */
    .text
    .align 4
    .global dl_tie728_rmsnorm_s8
    .type   dl_tie728_rmsnorm_s8, @function
dl_tie728_rmsnorm_s8:
    /* a2 = output
     * a3 = input
     * a4 = scale
     * a5 = rms (float*)
     * a6 = n
     */
    entry sp, 16

    lsi f0, a5, 0                  // f0 = *rms
    movi a7, 0x3f000000            // 0.5f bit pattern
    wfr f1, a7                     // f1 = 0.5f
    movi a12, 127                  // INT8_MAX (kept live across both loops)
    movi a13, -128                 // INT8_MIN

    srai a7, a6, 2                 // a7 = n / 4
    loopnez a7, dl_tie728_rmsnorm_s8_main_end
        l8ui a8,  a3, 0            // l8ui is unsigned, sign-extend below
        l8ui a9,  a3, 1
        l8ui a10, a3, 2
        l8ui a11, a3, 3
        sext a8,  a8,  7
        sext a9,  a9,  7
        sext a10, a10, 7
        sext a11, a11, 7
        float.s f2, a8, 0
        float.s f3, a9, 0
        float.s f4, a10, 0
        float.s f5, a11, 0
        lsi f6, a4, 0              // scale is float
        lsi f7, a4, 4
        lsi f8, a4, 8
        lsi f9, a4, 12
        mul.s f2, f2, f0           // input * rms
        mul.s f3, f3, f0
        mul.s f4, f4, f0
        mul.s f5, f5, f0
        mul.s f2, f2, f6           // (input * rms) * scale
        mul.s f3, f3, f7
        mul.s f4, f4, f8
        mul.s f5, f5, f9
        add.s f2, f2, f1           // + 0.5f
        add.s f3, f3, f1
        add.s f4, f4, f1
        add.s f5, f5, f1
        floor.s a8,  f2, 0         // round half up
        floor.s a9,  f3, 0
        floor.s a10, f4, 0
        floor.s a11, f5, 0
        min a8,  a8,  a12          // clip to int8 range
        max a8,  a8,  a13
        min a9,  a9,  a12
        max a9,  a9,  a13
        min a10, a10, a12
        max a10, a10, a13
        min a11, a11, a12
        max a11, a11, a13
        s8i a8,  a2, 0
        s8i a9,  a2, 1
        s8i a10, a2, 2
        s8i a11, a2, 3
        addi a3, a3, 4
        addi a4, a4, 16            // scale is float (4 bytes each)
        addi a2, a2, 4
    dl_tie728_rmsnorm_s8_main_end:

    extui a7, a6, 0, 2             // a7 = n % 4
    loopnez a7, dl_tie728_rmsnorm_s8_tail_end
        l8ui a8, a3, 0
        sext a8, a8, 7
        float.s f2, a8, 0
        lsi f6, a4, 0              // scale is float
        mul.s f2, f2, f0
        mul.s f2, f2, f6
        add.s f2, f2, f1
        floor.s a8, f2, 0
        min a8, a8, a12
        max a8, a8, a13
        s8i a8, a2, 0
        addi a3, a3, 1
        addi a4, a4, 4             // scale is float (4 bytes each)
        addi a2, a2, 1
    dl_tie728_rmsnorm_s8_tail_end:

    retw
