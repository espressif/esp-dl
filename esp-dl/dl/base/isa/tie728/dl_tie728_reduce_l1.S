/*
 * SIMD accelerated L1 reduce (sum of absolute values) kernels for ESP32-S3 (PIE / tie728).
 *
 * These kernels only handle the contiguous (stride == 1) case. The input pointer
 * must be 16-byte aligned and `size` is expected to be a multiple of the lane
 * count (16 for int8, 8 for int16). The scalar prologue / tail that takes care of
 * unaligned heads and remainders is handled by the C dispatcher in
 * dl_base_reduce.cpp.
 *
 * Technique: accumulate sum(input[i] * sign(input[i])) into ACCX, where the sign
 * vector holds +1 for non-negative lanes and -1 for negative lanes. Because the
 * MAC computes each product in a wider intermediate before accumulating, this
 * yields the exact absolute value even for INT8_MIN/INT16_MIN (e.g. -128 * -1 =
 * +128) instead of the saturated result a per-lane negate would produce.
 *
 * The sign vector is built per iteration with a signed compare-less-than against
 * zero (0xFF for negative lanes, 0x00 otherwise) OR'd with a broadcast 1, giving
 * 0xFF == -1 for negatives and 0x01 == +1 for non-negatives.
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
    ee.zero.q q2                      // q2 = 0 (compare reference)

    ee.zero.accx
    srai a4, a3, 4                   // a4 = size / 16 (16 int8 per 128-bit vector)
    beqz a4, dl_tie728_reduce_l1_s8_aligned_end

    dl_tie728_reduce_l1_s8_aligned_loop:
        ee.vld.128.ip   q0, a2, 16   // load 16 int8, a2 += 16
        ee.vcmp.lt.s8    q1, q0, q2   // q1 = (q0 < 0) ? 0xFF : 0x00
        ee.orq           q1, q1, q3   // q1 = sign(q0): 0xFF (-1) if neg, 0x01 (+1) if non-neg
        ee.vmulas.s8.accx q0, q1     // ACCX += sum(q0[i] * sign[i]) = sum(|q0[i]|), exact for -128
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
    ee.zero.q q2                      // q2 = 0 (compare reference)

    ee.zero.accx
    srai a4, a3, 3                   // a4 = size / 8 (8 int16 per 128-bit vector)
    beqz a4, dl_tie728_reduce_l1_s16_aligned_end

    dl_tie728_reduce_l1_s16_aligned_loop:
        ee.vld.128.ip    q0, a2, 16  // load 8 int16, a2 += 16
        ee.vcmp.lt.s16    q1, q0, q2  // q1 = (q0 < 0) ? 0xFFFF : 0x0000
        ee.orq            q1, q1, q3  // q1 = sign(q0): 0xFFFF (-1) if neg, 0x0001 (+1) if non-neg
        ee.vmulas.s16.accx q0, q1    // ACCX += sum(q0[i] * sign[i]) = sum(|q0[i]|), exact for -32768
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_l1_s16_aligned_loop
    dl_tie728_reduce_l1_s16_aligned_end:

    rur.accx_0 a2                     // low 32 bits
    rur.accx_1 a3                     // high 8 bits (zero-extended from bit 39:32)
    slli a3, a3, 24                  // sign-extend: bit 7 → bit 31
    srai a3, a3, 24                  // fill upper 24 bits with sign
    retw                              // return int64_t as a3:a2
