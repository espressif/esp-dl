/*
 * SIMD accelerated L2 reduce (sum of squares) kernels for ESP32-S3 (PIE / tie728).
 *
 * These kernels only handle the contiguous (stride == 1) case. The input pointer
 * must be 16-byte aligned and `size` is expected to be a multiple of the lane
 * count (16 for int8, 8 for int16). The scalar prologue / tail that takes care of
 * unaligned heads and remainders is handled by the C dispatcher in
 * dl_base_reduce.cpp.
 */

/* int32_t dl_tie728_reduce_l2_s8_aligned(int8_t *input, int32_t size);
 *
 *     int32_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += input[i] * input[i];
 *     return sum;
 */
    .text
    .align 4
    .global dl_tie728_reduce_l2_s8_aligned
    .type   dl_tie728_reduce_l2_s8_aligned, @function
dl_tie728_reduce_l2_s8_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 16)
     */
    entry sp, 16

    ee.zero.accx
    srai a4, a3, 4                 // a4 = size / 16 (16 int8 per 128-bit vector)
    beqz a4, dl_tie728_reduce_l2_s8_aligned_end

    dl_tie728_reduce_l2_s8_aligned_loop:
        ee.vld.128.ip   q0, a2, 16 // load 16 int8, a2 += 16
        ee.vmulas.s8.accx q0, q0   // ACCX += sum(q0[i] * q0[i])
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_l2_s8_aligned_loop
    dl_tie728_reduce_l2_s8_aligned_end:

    ee.srs.accx a2, a4, 0          // a4 == 0 here: arithmetic right shift ACCX by 0,
                                   // saturate to int32, write result into a2 (return value)
    retw



/* int64_t dl_tie728_reduce_l2_s16_aligned(int16_t *input, int32_t size);
 *
 *     int64_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += (int64_t)input[i] * input[i];
 *     return sum;
 */
    .text
    .align 4
    .global dl_tie728_reduce_l2_s16_aligned
    .type   dl_tie728_reduce_l2_s16_aligned, @function
dl_tie728_reduce_l2_s16_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 8)
     */
    entry sp, 16

    ee.zero.accx
    srai a4, a3, 3                 // a4 = size / 8 (8 int16 per 128-bit vector)
    beqz a4, dl_tie728_reduce_l2_s16_aligned_end

    dl_tie728_reduce_l2_s16_aligned_loop:
        ee.vld.128.ip    q0, a2, 16 // load 8 int16, a2 += 16
        ee.vmulas.s16.accx q0, q0   // ACCX += sum(q0[i] * q0[i])
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_l2_s16_aligned_loop
    dl_tie728_reduce_l2_s16_aligned_end:

    rur.accx_0 a2                  // low 32 bits  -> int64 low word
    rur.accx_1 a3                  // bits 39:32   -> int64 high word
    extui a3, a3, 0, 8             // keep only the valid 8 bits (sum of squares >= 0)
    retw
