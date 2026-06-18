/*
 * SIMD accelerated L1 reduce (sum of absolute values) kernels for ESP32-S3 (PIE / tie728).
 *
 * These kernels only handle the contiguous (stride == 1) case. The input pointer
 * must be 16-byte aligned and `size` is expected to be a multiple of the lane
 * count (16 for int8, 8 for int16). The scalar prologue / tail that takes care of
 * unaligned heads and remainders is handled by the C dispatcher in
 * dl_base_reduce.cpp.
 *
 * Technique: compute absolute value per element (negate + max), then accumulate
 * into ACCX using the multiply-by-1 trick (ACCX += sum(abs_val[i] * 1)).
 *
 * int8 kernel:  the result is extracted with EE.SRS.ACCX (arithmetic right shift
 *               by 0), which saturates ACCX to a 32-bit signed value directly
 *               into the return register a2.
 *
 * int16 kernel: the accumulated value is read back by reading ACCX_0 and ACCX_1
 *               special registers (40-bit split across two 32-bit halves).
 *               The C dispatcher accumulates chunk results into a 64-bit total.
 */

/* int32_t dl_tie728_reduce_l1_s8_aligned(int8_t *input, int32_t size);
 *
 *     int32_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += abs(input[i]);
 *     return sum;
 */
    .text
    .align 4
    .global dl_tie728_reduce_l1_s8_aligned
    .type   dl_tie728_reduce_l1_s8_aligned, @function
dl_tie728_reduce_l1_s8_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 16)
     */
    entry sp, 16

    movi a5, 1
    s8i  a5, sp, 0                   // store byte 1 to stack
    ee.vldbc.8.ip q3, sp, 0          // q3 = broadcast 1 to all 16 bytes
    ee.zero.q q2                      // q2 = 0 (for negate)

    ee.zero.accx
    srai a4, a3, 4                   // a4 = size / 16 (16 int8 per 128-bit vector)
    beqz a4, dl_tie728_reduce_l1_s8_aligned_end

    dl_tie728_reduce_l1_s8_aligned_loop:
        ee.vld.128.ip   q0, a2, 16   // load 16 int8, a2 += 16
        ee.vsubs.s8      q1, q2, q0   // q1 = 0 - q0 = -q0
        ee.vmax.s8      q1, q0, q1   // q1 = max(q0, -q0) = abs(q0)
        ee.vmulas.s8.accx q1, q3     // ACCX += sum(abs(q0[i]) * 1)
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_l1_s8_aligned_loop
    dl_tie728_reduce_l1_s8_aligned_end:

    ee.srs.accx a2, a4, 0            // a4 == 0 here: arithmetic right shift ACCX by 0,
                                     // saturate to int32, write result into a2 (return value)
    retw



/* int64_t dl_tie728_reduce_l1_s16_aligned(int16_t *input, int32_t size);
 *
 *     int64_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += abs(input[i]);
 *     return sum;
 */
    .text
    .align 4
    .global dl_tie728_reduce_l1_s16_aligned
    .type   dl_tie728_reduce_l1_s16_aligned, @function
dl_tie728_reduce_l1_s16_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 8)
     */
    entry sp, 16

    movi a5, 1
    s16i a5, sp, 0                   // store halfword 1 to stack
    ee.vldbc.16.ip q3, sp, 0         // q3 = broadcast 1 to all 8 lanes
    ee.zero.q q2                      // q2 = 0 (for negate)

    ee.zero.accx
    srai a4, a3, 3                   // a4 = size / 8 (8 int16 per 128-bit vector)
    beqz a4, dl_tie728_reduce_l1_s16_aligned_end

    dl_tie728_reduce_l1_s16_aligned_loop:
        ee.vld.128.ip    q0, a2, 16  // load 8 int16, a2 += 16
        ee.vsubs.s16      q1, q2, q0  // q1 = 0 - q0 = -q0
        ee.vmax.s16      q1, q0, q1  // q1 = max(q0, -q0) = abs(q0)
        ee.vmulas.s16.accx q1, q3    // ACCX += sum(abs(q0[i]) * 1)
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_l1_s16_aligned_loop
    dl_tie728_reduce_l1_s16_aligned_end:

    // Read 40-bit ACCX as two 32-bit halves: ACCX_0 = low 32, ACCX_1 = high 8.
    rur.accx_0 a2                     // low 32 bits
    rur.accx_1 a3                     // high 8 bits (sign-extended by hardware)
    retw
