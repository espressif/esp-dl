/*
 * SIMD accelerated reduce_sum (element-wise sum) kernels for ESP32-S3 (PIE / tie728).
 *
 * These kernels only handle the contiguous (stride == 1) case. The input pointer
 * must be 16-byte aligned and `size` is expected to be a multiple of the lane
 * count (16 for int8, 8 for int16). The scalar prologue / tail that takes care of
 * unaligned heads and remainders is handled by the C dispatcher in
 * dl_base_reduce.cpp.
 *
 * Technique: use ACCX accumulation with a multiplier of 1.
 *   ACCX += sum(q0[i] * 1) = sum(q0[i])
 * The all-ones vector is created by broadcast-loading a stack constant.
 *
 * int8 kernel:  the result is extracted with EE.SRS.ACCX (arithmetic right shift
 *               by 0), which saturates ACCX to a 32-bit signed value directly
 *               into the return register a2.
 *
 * int16 kernel: the accumulated value is read back by reading ACCX_0 and ACCX_1
 *               special registers (40-bit split across two 32-bit halves).
 *               The C dispatcher accumulates chunk results into a 64-bit total.
 */

/* int32_t dl_tie728_reduce_sum_s8_aligned(int8_t *input, int32_t size);
 *
 *     int32_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += input[i];
 *     return sum;
 */
    .text
    .align 4
    .global dl_tie728_reduce_sum_s8_aligned
    .type   dl_tie728_reduce_sum_s8_aligned, @function
dl_tie728_reduce_sum_s8_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 16)
     */
    entry sp, 16

    movi a5, 1
    s8i  a5, sp, 0                   // store byte 1 to stack
    ee.vldbc.8.ip q1, sp, 0          // q1 = broadcast 1 to all 16 bytes

    ee.zero.accx
    srai a4, a3, 4                   // a4 = size / 16 (16 int8 per 128-bit vector)
    beqz a4, dl_tie728_reduce_sum_s8_aligned_end

    dl_tie728_reduce_sum_s8_aligned_loop:
        ee.vld.128.ip   q0, a2, 16   // load 16 int8, a2 += 16
        ee.vmulas.s8.accx q0, q1     // ACCX += sum(q0[i] * 1)
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_sum_s8_aligned_loop
    dl_tie728_reduce_sum_s8_aligned_end:

    ee.srs.accx a2, a4, 0            // a4 == 0 here: arithmetic right shift ACCX by 0,
                                     // saturate to int32, write result into a2 (return value)
    retw



/* int64_t dl_tie728_reduce_sum_s16_aligned(int16_t *input, int32_t size);
 *
 *     int64_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += input[i];
 *     return sum;
 */
    .text
    .align 4
    .global dl_tie728_reduce_sum_s16_aligned
    .type   dl_tie728_reduce_sum_s16_aligned, @function
dl_tie728_reduce_sum_s16_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 8)
     */
    entry sp, 16

    movi a5, 1
    s16i a5, sp, 0                   // store halfword 1 to stack
    ee.vldbc.16.ip q1, sp, 0         // q1 = broadcast 1 to all 8 lanes

    ee.zero.accx
    srai a4, a3, 3                   // a4 = size / 8 (8 int16 per 128-bit vector)
    beqz a4, dl_tie728_reduce_sum_s16_aligned_end

    dl_tie728_reduce_sum_s16_aligned_loop:
        ee.vld.128.ip    q0, a2, 16  // load 8 int16, a2 += 16
        ee.vmulas.s16.accx q0, q1    // ACCX += sum(q0[i] * 1)
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_sum_s16_aligned_loop
    dl_tie728_reduce_sum_s16_aligned_end:

    // Read 40-bit ACCX as two 32-bit halves: ACCX_0 = low 32, ACCX_1 = high 8.
    rur.accx_0 a2                     // low 32 bits
    rur.accx_1 a3                     // high 8 bits (sign-extended by hardware)
    retw
