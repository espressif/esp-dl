/*
 * ESP32-S3 SIMD optimized FFT operations for 16-bit fixed-point data.
 * DIF FFT butterfly, bit-reversal, and RFFT post-processing.
 */


/*
 * Macro: dif_hp_butterfly_inner
 *
 * DIF radix-2 butterfly inner loop, parameterized by sar4.
 * sar4 controls the right-shift of (A+B) in EE.FFT.R2BF.S16.ST.INCP.
 * For upd4=0 in CMUL.ST.XP, sar4 is not used by hardware but is passed
 * for source consistency.
 *
 * Expects registers set up by the group-loop caller:
 *   a7=lower_rd, a8=lower_wr, a9=upper_rd, a10=upper_wr,
 *   a11=16, a13=twiddle_ptr, a15=iteration_count, q0/q1/q2 preloaded
 */
.macro dif_hp_butterfly_inner sar4_val
    loopnez a15, 0f
        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, \sar4_val
        ee.fft.cmul.s16.ld.xp q4, a13, a11, q5, q3, q2, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 2
        ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q2, 4
        ee.fft.cmul.s16.st.xp q3, q2, q5, a8, a11, 6, 0, \sar4_val

        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, \sar4_val
        ee.fft.cmul.s16.ld.xp q2, a13, a11, q5, q3, q4, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q4, 2
        ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q4, 4
        ee.fft.cmul.s16.st.xp q3, q4, q5, a8, a11, 6, 0, \sar4_val
0:
.endm


/*
 * Macro: ifft_hp_butterfly_inner
 *
 * DIF radix-2 IFFT butterfly inner loop.
 * Uses sel8=1,3,5 for LD and 7 for ST to perform conjugate complex multiply.
 */
.macro ifft_hp_butterfly_inner sar4_val
    loopnez a15, 0f
        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, \sar4_val
        ee.fft.cmul.s16.ld.xp q4, a13, a11, q5, q3, q2, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 3
        ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q2, 5
        ee.fft.cmul.s16.st.xp q3, q2, q5, a8, a11, 7, 0, \sar4_val

        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, \sar4_val
        ee.fft.cmul.s16.ld.xp q2, a13, a11, q5, q3, q4, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q4, 3
        ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q4, 5
        ee.fft.cmul.s16.st.xp q3, q4, q5, a8, a11, 7, 0, \sar4_val
0:
.endm


/*
 * Macro: reduce_abs_max_aes3_macro
 *
 * SIMD reduce-abs-max over a 16-byte aligned int16_t array.
 * Computes max(|data[i]|) for i in [0, count).  abs(INT16_MIN) saturates to
 * INT16_MAX.  count must be >= 8 and a multiple of 8.
 *
 * Parameters (all AR registers):
 *   data_ptr – pointer to int16_t array (TRASHED: advanced past the data)
 *   count    – number of int16_t elements (TRASHED)
 *   result   – receives the scalar max-abs value (int16_t)
 *   tmp      – scratch register (TRASHED)
 *
 * Clobbers: q0, q1, q2, q3, SAR_BYTE
 */
.macro reduce_abs_max_aes3_macro data_ptr count result tmp
    srai \count, \count, 3
    ee.zero.q q2
    ee.vld.128.ip q0, \data_ptr, 16
    ee.vsubs.s16 q1, q2, q0
    ee.vmax.s16 q0, q0, q1
    addi \count, \count, -1
    loopgtz \count, 0f
        ee.vld.128.ip q1, \data_ptr, 16
        ee.vsubs.s16 q3, q2, q1
        ee.vmax.s16 q1, q1, q3
        ee.vmax.s16 q0, q0, q1
0:
    movi \tmp, 8
    wur.sar_byte \tmp
    ee.src.q q1, q0, q0
    ee.vmax.s16 q0, q0, q1
    movi \tmp, 4
    wur.sar_byte \tmp
    ee.src.q q1, q0, q0
    ee.vmax.s16 q0, q0, q1
    movi \tmp, 2
    wur.sar_byte \tmp
    ee.src.q q1, q0, q0
    ee.vmax.s16 q0, q0, q1
    ee.movi.32.a q0, \result, 0
    sext \result, \result, 15
.endm


/* void dl_bitrev2r_sc16_aes3_(int16_t *data, int cpx_points, int log2N); */
    .text
    .align 4
    .global dl_bitrev2r_sc16_aes3_
    .type   dl_bitrev2r_sc16_aes3_, @function
dl_bitrev2r_sc16_aes3_:
    /* a2 = data, a3 = cpx_points, a4 = log2N */
    .align 4
    entry sp, 16

    srli a3, a3, 3
    wur.fft_bit_width a4
    movi a14, 0
    mov a7, a2
    mov a6, a2

    loopnez a3, .Ldl_bitrev_loop_end
        ee.vld.128.ip q0, a7, 16
        ee.bitrev q2, a14
        ee.ldxq.32 q1, q2, a2, 0, 0
        ee.ldxq.32 q1, q2, a2, 1, 1
        ee.ldxq.32 q1, q2, a2, 2, 2
        ee.ldxq.32 q1, q2, a2, 3, 3
        ee.stxq.32 q0, q2, a2, 0, 0
        ee.stxq.32 q0, q2, a2, 1, 1
        ee.stxq.32 q0, q2, a2, 2, 2
        ee.stxq.32 q0, q2, a2, 3, 3
        ee.vst.128.ip q1, a6, 16

        ee.vld.128.ip q0, a7, 16
        ee.ldxq.32 q1, q2, a2, 0, 4
        ee.ldxq.32 q1, q2, a2, 1, 5
        ee.ldxq.32 q1, q2, a2, 2, 6
        ee.ldxq.32 q1, q2, a2, 3, 7
        ee.stxq.32 q0, q2, a2, 0, 4
        ee.stxq.32 q0, q2, a2, 1, 5
        ee.stxq.32 q0, q2, a2, 2, 6
        ee.stxq.32 q0, q2, a2, 3, 7
        ee.vst.128.ip q1, a6, 16
    .Ldl_bitrev_loop_end:

    retw


/* void dl_fft2r_sc16_dif_aes3_(int16_t *data, int16_t *table, int shift, int num_stages, int cpx_points);
 * DIF radix-2 FFT butterfly using TIE728 SIMD.
 * Table must be in concatenated DIF layout (not bit-reversed). */
    .text
    .align 4
    .global dl_fft2r_sc16_dif_aes3_
    .type   dl_fft2r_sc16_dif_aes3_, @function
dl_fft2r_sc16_dif_aes3_:
    /* a2 = data, a3 = win, a4 = shift, a5 = num_stages, a6 = cpx_points */
    .align 4
    entry sp, 16

    ssr a4

    mov a13, a3
    addi a5, a5, -3
    movi a12, 1
    mov a14, a12
    srli a15, a6, 4
    movi a11, 16

    .Ldl_dif_stage:
        mov a4, a2
        mov a3, a13

        .Ldl_dif_box:
            mov a9, a4
            mov a10, a9
            addx2 a7, a6, a4
            mov a8, a7
            mov a13, a3

            ee.vld.128.ip q0, a9, 16
            ee.vld.128.ip q1, a7, 16
            ee.vld.128.ip q2, a13, 16

            loopnez a15, .Ldl_dif_bf_end
                ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
                ee.fft.cmul.s16.ld.xp q4, a13, a11, q5, q3, q2, 0
                ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 2
                ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q2, 4
                ee.fft.cmul.s16.st.xp q3, q2, q5, a8, a11, 6, 0, 1

                ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
                ee.fft.cmul.s16.ld.xp q2, a13, a11, q5, q3, q4, 0
                ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q4, 2
                ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q4, 4
                ee.fft.cmul.s16.st.xp q3, q4, q5, a8, a11, 6, 0, 1
            .Ldl_dif_bf_end:

            addx4 a4, a6, a4
            addi a14, a14, -1
            sub a13, a13, a11
        bnez a14, .Ldl_dif_box

        srli a15, a15, 1
        srli a6, a6, 1
        slli a12, a12, 1
        mov a14, a12
        addi a5, a5, -1
    bnez a5, .Ldl_dif_stage

    /* Last 3 stages */
    mov a9, a2
    mov a10, a9
    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.128.ip q2, a13, 16
    movi a14, 0

    loopnez a12, .Ldl_dif_last3_end
        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
        ee.fft.cmul.s16.ld.xp q4, a13, a14, q5, q3, q2, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 2
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q3, q2, 4
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 0, 1
    .Ldl_dif_last3_end:

    mov a9, a2
    mov a10, a9
    mov a14, a2
    mov a3, a13

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.h.64.ip q2, a13, 0

    loopnez a12, .Ldl_dif_last2_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 4
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 1, 1
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 4
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 6, 1, 1
    .Ldl_dif_last2_end:

    mov a9, a2
    mov a10, a9
    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vldbc.32 q2, a3

    loopnez a12, .Ldl_dif_last1_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 4
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 2, 1
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 4
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 6, 2, 1
    .Ldl_dif_last1_end:

    retw

/* int16_t dl_fft_reduce_max_aes3_(int16_t *inputs, int len);
 * Find the maximum value in a 16-byte aligned int16_t array.
 * len must be >= 8 and a multiple of 8. */
    .text
    .align 4
    .global dl_fft_reduce_max_aes3_
    .type   dl_fft_reduce_max_aes3_, @function
dl_fft_reduce_max_aes3_:
    /* a2 = inputs (16-byte aligned), a3 = len (element count) */
    entry sp, 16

    ee.vld.128.ip q0, a2, 16
    srai a3, a3, 3
    addi a3, a3, -1

    loopgtz a3, .Lreduce_max_loop_end
        ee.vld.128.ip q1, a2, 16
        ee.vmax.s16 q0, q0, q1
    .Lreduce_max_loop_end:

    /* Horizontal reduction: 8 lanes -> 1 */
    movi a4, 8
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s16 q0, q0, q1

    movi a4, 4
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s16 q0, q0, q1

    movi a4, 2
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s16 q0, q0, q1

    ee.movi.32.a q0, a2, 0
    sext a2, a2, 15

    retw


/* int16_t dl_reduce_abs_max_aes3_(int16_t *inputs, int len);
 * Find the maximum absolute value in a 16-byte aligned int16_t array.
 * abs(INT16_MIN) saturates to INT16_MAX.
 * len must be >= 8 and a multiple of 8. */
    .text
    .align 4
    .global dl_reduce_abs_max_aes3_
    .type   dl_reduce_abs_max_aes3_, @function
dl_reduce_abs_max_aes3_:
    /* a2 = inputs (16-byte aligned), a3 = len (element count) */
    entry sp, 16
    reduce_abs_max_aes3_macro a2, a3, a2, a4
    retw


/* void dl_fft2r_sc16_dif_hp_aes3_(int16_t *data, int16_t *table,
 *                                 int num_stages, int cpx_points, int *out_shift);
 * High-precision DIF radix-2 FFT with dynamic per-stage sar4 + SAR switching.
 */
    .text
    .align 4
    .global dl_fft2r_sc16_dif_hp_aes3_
    .type   dl_fft2r_sc16_dif_hp_aes3_, @function
dl_fft2r_sc16_dif_hp_aes3_:
    /* a2 = data, a3 = table, a4 = num_stages, a5 = cpx_points, a6 = out_shift */
    entry sp, 16

    movi a7, 0
    s32i a7, a6, 0
    wur.accx_0 a6

    mov a6, a5
    mov a13, a3
    addi a5, a4, -3
    movi a12, 1
    mov a14, a12
    srli a15, a6, 4
    movi a11, 16

    blti a5, 1, .Ldl_dif_hp_last3

    .Ldl_dif_hp_stage:
        /* --- 1. SIMD abs-max scan --- */
        mull a7, a6, a12
        slli a8, a7, 1
        mov a7, a2
        reduce_abs_max_aes3_macro a7, a8, a7, a9

        /* --- 2. decide shift: max_abs < 16384 → shift=15, else shift=16 --- */
        movi a8, 15
        srli a7, a7, 13
        bnez a7, .Ldl_hp_shift_done
        movi a8, 14
    .Ldl_hp_shift_done:
        ssr a8

        /* --- 3. update *out_shift += (loop_shift - 15) --- */
        rur.accx_0 a7
        l32i a9, a7, 0
        addi a10, a8, -14
        add a9, a9, a10
        s32i a9, a7, 0

        movi a11, 16

        /* --- 4. branch to sar4=0 or sar4=1 butterfly --- */
        bbsi a8, 0, .Ldl_hp_bf_sar1

        /* ===== sar4=0 butterfly (loop_shift=15) ===== */
        mov a4, a2
        mov a3, a13
        .Ldl_dif_hp_box_s0:
            mov a9, a4
            mov a10, a9
            addx2 a7, a6, a4
            mov a8, a7
            mov a13, a3
            ee.vld.128.ip q0, a9, 16
            ee.vld.128.ip q1, a7, 16
            ee.vld.128.ip q2, a13, 16
            dif_hp_butterfly_inner 0
            addx4 a4, a6, a4
            addi a14, a14, -1
            sub a13, a13, a11
        bnez a14, .Ldl_dif_hp_box_s0
        j .Ldl_hp_stage_done

    .Ldl_hp_bf_sar1:
        /* ===== sar4=1 butterfly (loop_shift=16, standard) ===== */
        mov a4, a2
        mov a3, a13
        .Ldl_dif_hp_box_s1:
            mov a9, a4
            mov a10, a9
            addx2 a7, a6, a4
            mov a8, a7
            mov a13, a3
            ee.vld.128.ip q0, a9, 16
            ee.vld.128.ip q1, a7, 16
            ee.vld.128.ip q2, a13, 16
            dif_hp_butterfly_inner 1
            addx4 a4, a6, a4
            addi a14, a14, -1
            sub a13, a13, a11
        bnez a14, .Ldl_dif_hp_box_s1

    .Ldl_hp_stage_done:
        srli a15, a15, 1
        srli a6, a6, 1
        slli a12, a12, 1
        mov a14, a12
        addi a5, a5, -1
    bnez a5, .Ldl_dif_hp_stage

    .Ldl_dif_hp_last3:
    /* --- Stage num_stages-2 (Last 3rd) --- */
    mull a7, a6, a12
    slli a8, a7, 1
    mov a7, a2
    reduce_abs_max_aes3_macro a7, a8, a7, a9

    movi a8, 15
    srli a7, a7, 13
    bnez a7, .Ldl_hp_l3_shift_done
    movi a8, 14
    .Ldl_hp_l3_shift_done:
    ssr a8

    /* update *out_shift: +=(loop_shift-15) */
    rur.accx_0 a7
    l32i a9, a7, 0
    addi a10, a8, -14
    add a9, a9, a10
    s32i a9, a7, 0

    movi a11, 16
    movi a14, 0
    mov a9, a2
    mov a10, a9
    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.128.ip q2, a13, 16

    bbsi a8, 0, .Ldl_hp_l3_sar1

    /* sar4=0 branch */
    loopnez a12, .Ldl_dif_hp_l3s0_end
        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 0
        ee.fft.cmul.s16.ld.xp q4, a13, a14, q5, q3, q2, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 2
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q3, q2, 4
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 0, 0
    .Ldl_dif_hp_l3s0_end:
    j .Ldl_dif_hp_l3_done

    .Ldl_hp_l3_sar1:
    /* sar4=1 branch */
    loopnez a12, .Ldl_dif_hp_l3s1_end
        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
        ee.fft.cmul.s16.ld.xp q4, a13, a14, q5, q3, q2, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 2
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q3, q2, 4
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 0, 0
    .Ldl_dif_hp_l3s1_end:

    .Ldl_dif_hp_l3_done:

    /* --- Stage num_stages-1 (Last 2nd) --- */
    mull a7, a6, a12
    slli a8, a7, 1
    mov a7, a2
    reduce_abs_max_aes3_macro a7, a8, a7, a9

    movi a8, 15
    srli a7, a7, 13
    bnez a7, .Ldl_hp_l2_shift_done
    movi a8, 14
    .Ldl_hp_l2_shift_done:
    ssr a8

    /* update *out_shift: +=(loop_shift-15) */
    rur.accx_0 a7
    l32i a9, a7, 0
    addi a10, a8, -14
    add a9, a9, a10
    s32i a9, a7, 0

    movi a11, 16
    mov a9, a2
    mov a10, a9
    mov a3, a13
    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.h.64.ip q2, a13, 0

    bbsi a8, 0, .Ldl_hp_l2_sar1

    /* sar4=0 branch */
    loopnez a12, .Ldl_dif_hp_l2s0_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 4
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 1, 0
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 4
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 6, 1, 0
    .Ldl_dif_hp_l2s0_end:
    j .Ldl_dif_hp_l2_done

    .Ldl_hp_l2_sar1:
    /* sar4=1 branch */
    loopnez a12, .Ldl_dif_hp_l2s1_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 4
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 1, 1
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 4
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 6, 1, 1
    .Ldl_dif_hp_l2s1_end:

    .Ldl_dif_hp_l2_done:

    /* --- Stage num_stages (Last 1st) --- */
    mull a7, a6, a12
    slli a8, a7, 1
    mov a7, a2
    reduce_abs_max_aes3_macro a7, a8, a7, a9

    movi a8, 15
    srli a7, a7, 13
    bnez a7, .Ldl_hp_l1_shift_done
    movi a8, 14
    .Ldl_hp_l1_shift_done:
    ssr a8

    /* update *out_shift: +=(loop_shift-15) */
    rur.accx_0 a7
    l32i a9, a7, 0
    addi a10, a8, -14
    add a9, a9, a10
    s32i a9, a7, 0

    movi a11, 16
    mov a9, a2
    mov a10, a9
    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vldbc.32 q2, a3

    bbsi a8, 0, .Ldl_hp_l1_sar1

    /* sar4=0 branch */
    loopnez a12, .Ldl_dif_hp_l1s0_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 4
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 2, 0
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 4
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 6, 2, 0
    .Ldl_dif_hp_l1s0_end:
    retw

    .Ldl_hp_l1_sar1:
    /* sar4=1 branch */
    loopnez a12, .Ldl_dif_hp_l1s1_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 4
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 2, 1
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 4
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 6, 2, 1
    .Ldl_dif_hp_l1s1_end:

    retw


/* void dl_ifft2r_sc16_dif_aes3_(int16_t *data, int16_t *table, int shift, int num_stages, int cpx_points); */
    .text
    .align 4
    .global dl_ifft2r_sc16_dif_aes3_
    .type   dl_ifft2r_sc16_dif_aes3_, @function
dl_ifft2r_sc16_dif_aes3_:
    /* a2 = data, a3 = win, a4 = shift, a5 = num_stages, a6 = cpx_points */
    .align 4
    entry sp, 16

    ssr a4

    mov a13, a3
    addi a5, a5, -3
    movi a12, 1
    mov a14, a12
    srli a15, a6, 4
    movi a11, 16

    .Ldl_idif_stage:
        mov a4, a2
        mov a3, a13

        .Ldl_idif_box:
            mov a9, a4
            mov a10, a9
            addx2 a7, a6, a4
            mov a8, a7
            mov a13, a3

            ee.vld.128.ip q0, a9, 16
            ee.vld.128.ip q1, a7, 16
            ee.vld.128.ip q2, a13, 16

            loopnez a15, .Ldl_idif_bf_end
                ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
                ee.fft.cmul.s16.ld.xp q4, a13, a11, q5, q3, q2, 1
                ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 3
                ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q2, 5
                ee.fft.cmul.s16.st.xp q3, q2, q5, a8, a11, 7, 0, 1

                ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
                ee.fft.cmul.s16.ld.xp q2, a13, a11, q5, q3, q4, 1
                ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q4, 3
                ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q4, 5
                ee.fft.cmul.s16.st.xp q3, q4, q5, a8, a11, 7, 0, 1
            .Ldl_idif_bf_end:

            addx4 a4, a6, a4
            addi a14, a14, -1
            sub a13, a13, a11
        bnez a14, .Ldl_idif_box

        srli a15, a15, 1
        srli a6, a6, 1
        slli a12, a12, 1
        mov a14, a12
        addi a5, a5, -1
    bnez a5, .Ldl_idif_stage

    /* Last 3 stages */
    mov a9, a2
    mov a10, a9
    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.128.ip q2, a13, 16
    movi a14, 0

    loopnez a12, .Ldl_idif_last3_end
        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
        ee.fft.cmul.s16.ld.xp q4, a13, a14, q5, q3, q2, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 3
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q3, q2, 5
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 0, 1
    .Ldl_idif_last3_end:

    mov a9, a2
    mov a10, a9
    mov a14, a2
    mov a3, a13

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.h.64.ip q2, a13, 0

    loopnez a12, .Ldl_idif_last2_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 5
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 1, 1
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 5
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 7, 1, 1
    .Ldl_idif_last2_end:

    mov a9, a2
    mov a10, a9
    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vldbc.32 q2, a3

    loopnez a12, .Ldl_idif_last1_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 5
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 2, 1
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 5
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 7, 2, 1
    .Ldl_idif_last1_end:

    retw

/* void dl_ifft2r_sc16_dif_hp_aes3_(int16_t *data, int16_t *table,
 *                                  int num_stages, int cpx_points, int *out_shift);
 * High-precision DIF radix-2 IFFT with dynamic per-stage sar4 + SAR switching.
 */
    .text
    .align 4
    .global dl_ifft2r_sc16_dif_hp_aes3_
    .type   dl_ifft2r_sc16_dif_hp_aes3_, @function
dl_ifft2r_sc16_dif_hp_aes3_:
    /* a2 = data, a3 = table, a4 = num_stages, a5 = cpx_points, a6 = out_shift */
    entry sp, 16

    movi a7, 0
    s32i a7, a6, 0
    wur.accx_0 a6

    mov a6, a5
    mov a13, a3
    addi a5, a4, -3
    movi a12, 1
    mov a14, a12
    srli a15, a6, 4
    movi a11, 16

    blti a5, 1, .Ldl_idif_hp_last3

    .Ldl_idif_hp_stage:
        /* --- 1. SIMD abs-max scan --- */
        mull a7, a6, a12
        slli a8, a7, 1
        mov a7, a2
        reduce_abs_max_aes3_macro a7, a8, a7, a9

        /* --- 2. decide shift: max_abs < 16384 → shift=15, else shift=16 --- */
        movi a8, 15
        srli a7, a7, 13
        bnez a7, .Ldl_hp_ishift_done
        movi a8, 14
    .Ldl_hp_ishift_done:
        ssr a8

        /* --- 3. update *out_shift += (loop_shift - 15) --- */
        rur.accx_0 a7
        l32i a9, a7, 0
        addi a10, a8, -14
        add a9, a9, a10
        s32i a9, a7, 0

        movi a11, 16

        /* --- 4. branch to sar4=0 or sar4=1 butterfly --- */
        bbsi a8, 0, .Ldl_hp_ibf_sar1

        /* ===== sar4=0 butterfly (loop_shift=15) ===== */
        mov a4, a2
        mov a3, a13
        .Ldl_idif_hp_box_s0:
            mov a9, a4
            mov a10, a9
            addx2 a7, a6, a4
            mov a8, a7
            mov a13, a3
            ee.vld.128.ip q0, a9, 16
            ee.vld.128.ip q1, a7, 16
            ee.vld.128.ip q2, a13, 16
            ifft_hp_butterfly_inner 0
            addx4 a4, a6, a4
            addi a14, a14, -1
            sub a13, a13, a11
        bnez a14, .Ldl_idif_hp_box_s0
        j .Ldl_hp_istage_done

    .Ldl_hp_ibf_sar1:
        /* ===== sar4=1 butterfly (loop_shift=16, standard) ===== */
        mov a4, a2
        mov a3, a13
        .Ldl_idif_hp_box_s1:
            mov a9, a4
            mov a10, a9
            addx2 a7, a6, a4
            mov a8, a7
            mov a13, a3
            ee.vld.128.ip q0, a9, 16
            ee.vld.128.ip q1, a7, 16
            ee.vld.128.ip q2, a13, 16
            ifft_hp_butterfly_inner 1
            addx4 a4, a6, a4
            addi a14, a14, -1
            sub a13, a13, a11
        bnez a14, .Ldl_idif_hp_box_s1

    .Ldl_hp_istage_done:
        srli a15, a15, 1
        srli a6, a6, 1
        slli a12, a12, 1
        mov a14, a12
        addi a5, a5, -1
    bnez a5, .Ldl_idif_hp_stage

    .Ldl_idif_hp_last3:
    /* --- Stage num_stages-2 (Last 3rd) --- */
    mull a7, a6, a12
    slli a8, a7, 1
    mov a7, a2
    reduce_abs_max_aes3_macro a7, a8, a7, a9

    movi a8, 15
    srli a7, a7, 13
    bnez a7, .Ldl_hp_il3_shift_done
    movi a8, 14
    .Ldl_hp_il3_shift_done:
    ssr a8

    /* update *out_shift: +=(loop_shift-15) */
    rur.accx_0 a7
    l32i a9, a7, 0
    addi a10, a8, -14
    add a9, a9, a10
    s32i a9, a7, 0

    movi a11, 16
    movi a14, 0
    mov a9, a2
    mov a10, a9
    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.128.ip q2, a13, 16

    bbsi a8, 0, .Ldl_hp_il3_sar1

    /* sar4=0 branch */
    loopnez a12, .Ldl_idif_hp_l3s0_end
        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 0
        ee.fft.cmul.s16.ld.xp q4, a13, a14, q5, q3, q2, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 3
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q3, q2, 5
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 0, 0
    .Ldl_idif_hp_l3s0_end:
    j .Ldl_idif_hp_l3_done

    .Ldl_hp_il3_sar1:
    /* sar4=1 branch */
    loopnez a12, .Ldl_idif_hp_l3s1_end
        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
        ee.fft.cmul.s16.ld.xp q4, a13, a14, q5, q3, q2, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 3
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q3, q2, 5
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 0, 0
    .Ldl_idif_hp_l3s1_end:

    .Ldl_idif_hp_l3_done:

    /* --- Stage num_stages-1 (Last 2nd) --- */
    mull a7, a6, a12
    slli a8, a7, 1
    mov a7, a2
    reduce_abs_max_aes3_macro a7, a8, a7, a9

    movi a8, 15
    srli a7, a7, 13
    bnez a7, .Ldl_hp_il2_shift_done
    movi a8, 14
    .Ldl_hp_il2_shift_done:
    ssr a8

    /* update *out_shift: +=(loop_shift-15) */
    rur.accx_0 a7
    l32i a9, a7, 0
    addi a10, a8, -14
    add a9, a9, a10
    s32i a9, a7, 0

    movi a11, 16
    mov a9, a2
    mov a10, a9
    mov a3, a13
    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.h.64.ip q2, a13, 0

    bbsi a8, 0, .Ldl_hp_il2_sar1

    /* sar4=0 branch */
    loopnez a12, .Ldl_idif_hp_l2s0_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 5
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 1, 0
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 5
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 7, 1, 0
    .Ldl_idif_hp_l2s0_end:
    j .Ldl_idif_hp_l2_done

    .Ldl_hp_il2_sar1:
    /* sar4=1 branch */
    loopnez a12, .Ldl_idif_hp_l2s1_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 0
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 5
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 1, 1
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 5
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 7, 1, 1
    .Ldl_idif_hp_l2s1_end:

    .Ldl_idif_hp_l2_done:

    /* --- Stage num_stages (Last 1st) --- */
    mull a7, a6, a12
    slli a8, a7, 1
    mov a7, a2
    reduce_abs_max_aes3_macro a7, a8, a7, a9

    movi a8, 15
    srli a7, a7, 13
    bnez a7, .Ldl_hp_il1_shift_done
    movi a8, 14
    .Ldl_hp_il1_shift_done:
    ssr a8

    /* update *out_shift: +=(loop_shift-15) */
    rur.accx_0 a7
    l32i a9, a7, 0
    addi a10, a8, -14
    add a9, a9, a10
    s32i a9, a7, 0

    movi a11, 16
    mov a9, a2
    mov a10, a9
    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vldbc.32 q2, a3

    bbsi a8, 0, .Ldl_hp_il1_sar1

    /* sar4=0 branch */
    loopnez a12, .Ldl_idif_hp_l1s0_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 5
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 2, 0
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 5
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 7, 2, 0
    .Ldl_idif_hp_l1s0_end:
    retw

    .Ldl_hp_il1_sar1:
    /* sar4=1 branch */
    loopnez a12, .Ldl_idif_hp_l1s1_end
        ee.fft.r2bf.s16 q3, q4, q0, q1, 1
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 5
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 2, 1
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 5
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 7, 2, 1
    .Ldl_idif_hp_l1s1_end:

    retw


/* esp_err_t dl_rfft_post_proc_sc16_aes3_(int16_t *data, int N, int16_t *table);
 *
 * NOTE: The AMS hardware truncates the intermediate (sum_r + tw_r) addition to
 * 16 bits BEFORE the built-in `>> 1` shift in the store, so when
 * |sum_r + tw_r| > INT16_MAX bit 16 is dropped and the result differs from
 * the int32 C reference by 32768. To prevent this, callers must pre-shift the
 * input data by 1 right shift whenever the data uses (close to) the full int16
 * range (this is handled by the `dl_rfft_post_proc_sc16` wrapper and the
 * caller compensates the output exponent by +1).
 */
    .text
    .align 4
    .global dl_rfft_post_proc_sc16_aes3_
    .type   dl_rfft_post_proc_sc16_aes3_, @function
dl_rfft_post_proc_sc16_aes3_:
    /* a2 = data, a3 = N (cpx_point), a4 = table */
.align 4
    entry sp, 16

	movi a5, 14
    ssr a5
    mov a6, a2
    addi a7, a2, 16  //ptr1 = data + 16;
    addx4 a9, a3, a2  //ptr_inv1 = data + cpx_points*2
    addi a8, a9, -16  //ptr_inv = data + cpx_points*2 - 8


	l16si a10, a2, 0
	l16si a11, a2, 2

	add a13, a10, a11
	sub a12, a10, a11
	movi a11, 0

	addi a9, a9, -16

	ee.vld.128.ip q2, a4, 16
	ee.fft.ams.s16.ld.r32.decp q1, a8, q6, q7, q3, q4, q5, 0 // only use the load

    ee.vld.128.ip q3, a2, 0
    movi a10, 4
    wur.sar_byte a10
    ee.vld.128.ip q4, a7, 16
    ee.src.q q0, q3, q4

    addi a10, a2, 16

    ee.fft.ams.s16.ld.incp.uaup q3, a10, q6, q7, q0, q1, q2, 0 //only use the load

    srli a3, a3, 4
    loopnez a3, .Lpost_loop_end
        ee.fft.ams.s16.ld.incp.uaup q3, a7, q6, q7, q0, q1, q2, 0
        ee.fft.ams.s16.ld.incp q5, a4, q6, q7, q0, q1, q2, 0
        ee.fft.ams.s16.ld.r32.decp q4, a8, q6, q7, q0, q1, q2, 0
        ee.fft.ams.s16.st.incp q6, q7, a11, a6, q0, q1, q2, 0
        ee.fft.vst.r32.decp q7, a9, 1

        ee.fft.ams.s16.ld.incp.uaup q0, a7, q6, q7, q3, q4, q5, 0
        ee.fft.ams.s16.ld.incp q2, a4, q6, q7, q3, q4, q5, 0
        ee.fft.ams.s16.ld.r32.decp q1, a8, q6, q7, q3, q4, q5, 0
        ee.fft.ams.s16.st.incp q6, q7, a11, a6, q3, q4, q5, 0
        ee.fft.vst.r32.decp q7, a9, 1
    .Lpost_loop_end:

	s16i a13, a2, 0
	s16i a12, a2, 2

    retw

/* esp_err_t dl_rfft_pre_proc_sc16_aes3_(int16_t *data, int N, int16_t *table); */
    .text
    .align 4
    .global dl_rfft_pre_proc_sc16_aes3_
    .type   dl_rfft_pre_proc_sc16_aes3_, @function
dl_rfft_pre_proc_sc16_aes3_:
    /* a2 = data, a3 = N (cpx_point), a4 = table */
    entry sp, 16

    movi a5, 14
    ssr a5

    mov a6, a2
    addi a7, a2, 16  //ptr1 = data + 16;
    addx4 a9, a3, a2  //ptr_inv1 = data + cpx_points*2
    addi a8, a9, -16  //ptr_inv = data + cpx_points*2 - 8


    l16si a10, a2, 0
    l16si a11, a2, 2

    add a13, a10, a11
    sub a12, a10, a11
    //srai a13, a13, 1
    //srai a12, a12, 1

    addi a9, a9, -16

    ee.vld.128.ip q2, a4, 16
    ee.fft.ams.s16.ld.r32.decp q1, a8, q6, q7, q3, q4, q5, 0 // only use the load


    ee.vld.128.ip q3, a2, 0
    movi a10, 4
    wur.sar_byte a10
    ee.vld.128.ip q4, a7, 16
    ee.src.q q0, q3, q4

    addi a10, a2, 16

    ee.fft.ams.s16.ld.incp.uaup q3, a10, q6, q7, q0, q1, q2, 0 //only use the load

    movi a11, 0

    srli a3, a3, 4
    loopnez a3, .Lpre_loop_end
        ee.fft.ams.s16.ld.incp.uaup q3, a7, q6, q7, q0, q1, q2, 1
        ee.fft.ams.s16.ld.incp q5, a4, q6, q7, q0, q1, q2, 1
        ee.fft.ams.s16.ld.r32.decp q4, a8, q6, q7, q0, q1, q2, 1
        ee.fft.ams.s16.st.incp q6, q7, a11, a6, q0, q1, q2, 1
        ee.fft.vst.r32.decp q7, a9, 0

        ee.fft.ams.s16.ld.incp.uaup q0, a7, q6, q7, q3, q4, q5, 1
        ee.fft.ams.s16.ld.incp q2, a4, q6, q7, q3, q4, q5, 1
        ee.fft.ams.s16.ld.r32.decp q1, a8, q6, q7, q3, q4, q5, 1
        ee.fft.ams.s16.st.incp q6, q7, a11, a6, q3, q4, q5, 1
        ee.fft.vst.r32.decp q7, a9, 0
    .Lpre_loop_end:

    s16i a13, a2, 0
    s16i a12, a2, 2

    movi a2, 0
    retw
