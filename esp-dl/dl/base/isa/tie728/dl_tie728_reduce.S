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
    .balign 4
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
    .balign 4
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



/* int8_t dl_tie728_reduce_max_s8_aligned(int8_t *input, int32_t size);
 *
 *     int8_t result = INT8_MIN;
 *     for (int i = 0; i < size; i++) {
 *         if (input[i] > result) result = input[i];
 *     }
 *     return result;
 */
    .text
    .balign 4
    .global dl_tie728_reduce_max_s8_aligned
    .type   dl_tie728_reduce_max_s8_aligned, @function
dl_tie728_reduce_max_s8_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 16)
     */
    entry sp, 16

    srai a4, a3, 4                 // a4 = size / 16 (16 int8 per 128-bit vector)
    ee.vld.128.ip q0, a2, 16       // load first vector
    addi a4, a4, -1
    beqz a4, dl_tie728_reduce_max_s8_aligned_end

    dl_tie728_reduce_max_s8_aligned_loop:
        ee.vld.128.ip q1, a2, 16   // load next vector
        ee.vmax.s8 q0, q0, q1      // element-wise running max
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_max_s8_aligned_loop

    dl_tie728_reduce_max_s8_aligned_end:
    // Horizontal tree reduction on q0 (16 int8 -> 1).
    movi a4, 8
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s8 q0, q0, q1
    movi a4, 4
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s8 q0, q0, q1
    movi a4, 2
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s8 q0, q0, q1
    movi a4, 1
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s8 q0, q0, q1
    // All 16 bytes now hold the global max; extract and sign-extend.
    ee.movi.32.a q0, a2, 0
    slli a2, a2, 24
    srai a2, a2, 24
    retw



/* int16_t dl_tie728_reduce_max_s16_aligned(int16_t *input, int32_t size);
 *
 *     int16_t result = INT16_MIN;
 *     for (int i = 0; i < size; i++) {
 *         if (input[i] > result) result = input[i];
 *     }
 *     return result;
 */
    .text
    .balign 4
    .global dl_tie728_reduce_max_s16_aligned
    .type   dl_tie728_reduce_max_s16_aligned, @function
dl_tie728_reduce_max_s16_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 8)
     */
    entry sp, 16

    srai a4, a3, 3                 // a4 = size / 8 (8 int16 per 128-bit vector)
    ee.vld.128.ip q0, a2, 16       // load first vector
    addi a4, a4, -1
    beqz a4, dl_tie728_reduce_max_s16_aligned_end

    dl_tie728_reduce_max_s16_aligned_loop:
        ee.vld.128.ip q1, a2, 16   // load next vector
        ee.vmax.s16 q0, q0, q1     // element-wise running max
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_max_s16_aligned_loop

    dl_tie728_reduce_max_s16_aligned_end:
    // Horizontal tree reduction on q0 (8 int16 -> 1).
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
    // All 8 lanes now hold the global max; extract and sign-extend.
    ee.movi.32.a q0, a2, 0
    slli a2, a2, 16
    srai a2, a2, 16
    retw



/* int8_t dl_tie728_reduce_min_s8_aligned(int8_t *input, int32_t size);
 *
 *     int8_t result = INT8_MAX;
 *     for (int i = 0; i < size; i++) {
 *         if (input[i] < result) result = input[i];
 *     }
 *     return result;
 */
    .text
    .balign 4
    .global dl_tie728_reduce_min_s8_aligned
    .type   dl_tie728_reduce_min_s8_aligned, @function
dl_tie728_reduce_min_s8_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 16)
     */
    entry sp, 16

    srai a4, a3, 4                 // a4 = size / 16 (16 int8 per 128-bit vector)
    ee.vld.128.ip q0, a2, 16       // load first vector
    addi a4, a4, -1
    beqz a4, dl_tie728_reduce_min_s8_aligned_end

    dl_tie728_reduce_min_s8_aligned_loop:
        ee.vld.128.ip q1, a2, 16
        ee.vmin.s8 q0, q0, q1
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_min_s8_aligned_loop

    dl_tie728_reduce_min_s8_aligned_end:
    movi a4, 8
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmin.s8 q0, q0, q1
    movi a4, 4
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmin.s8 q0, q0, q1
    movi a4, 2
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmin.s8 q0, q0, q1
    movi a4, 1
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmin.s8 q0, q0, q1
    ee.movi.32.a q0, a2, 0
    slli a2, a2, 24
    srai a2, a2, 24
    retw



/* int16_t dl_tie728_reduce_min_s16_aligned(int16_t *input, int32_t size);
 *
 *     int16_t result = INT16_MAX;
 *     for (int i = 0; i < size; i++) {
 *         if (input[i] < result) result = input[i];
 *     }
 *     return result;
 */
    .text
    .balign 4
    .global dl_tie728_reduce_min_s16_aligned
    .type   dl_tie728_reduce_min_s16_aligned, @function
dl_tie728_reduce_min_s16_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 8)
     */
    entry sp, 16

    srai a4, a3, 3                 // a4 = size / 8 (8 int16 per 128-bit vector)
    ee.vld.128.ip q0, a2, 16       // load first vector
    addi a4, a4, -1
    beqz a4, dl_tie728_reduce_min_s16_aligned_end

    dl_tie728_reduce_min_s16_aligned_loop:
        ee.vld.128.ip q1, a2, 16
        ee.vmin.s16 q0, q0, q1
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_min_s16_aligned_loop

    dl_tie728_reduce_min_s16_aligned_end:
    movi a4, 8
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmin.s16 q0, q0, q1
    movi a4, 4
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmin.s16 q0, q0, q1
    movi a4, 2
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmin.s16 q0, q0, q1
    ee.movi.32.a q0, a2, 0
    slli a2, a2, 16
    srai a2, a2, 16
    retw



/* int8_t dl_tie728_reduce_abs_max_s8_aligned(int8_t *input, int32_t size);
 *
 *     int8_t result = 0;
 *     for (int i = 0; i < size; i++) {
 *         int8_t abs_val = abs(input[i]);  // saturates at INT8_MAX
 *         if (abs_val > result) result = abs_val;
 *     }
 *     return result;
 */
    .text
    .balign 4
    .global dl_tie728_reduce_abs_max_s8_aligned
    .type   dl_tie728_reduce_abs_max_s8_aligned, @function
dl_tie728_reduce_abs_max_s8_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 16)
     */
    entry sp, 16

    srai a4, a3, 4                 // a4 = size / 16
    ee.vld.128.ip q0, a2, 16       // load first vector
    ee.zero.q q2                   // q2 = 0
    ee.vsubs.s8 q1, q2, q0         // q1 = -q0  (saturating negation)
    ee.vmax.s8 q0, q0, q1          // q0 = max(q0, -q0) = abs(q0)
    addi a4, a4, -1
    beqz a4, dl_tie728_reduce_abs_max_s8_aligned_end

    dl_tie728_reduce_abs_max_s8_aligned_loop:
        ee.vld.128.ip q1, a2, 16   // load next vector
        ee.vsubs.s8 q3, q2, q1     // q3 = -q1
        ee.vmax.s8 q1, q1, q3      // q1 = abs(q1)
        ee.vmax.s8 q0, q0, q1      // element-wise running max of abs values
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_abs_max_s8_aligned_loop

    dl_tie728_reduce_abs_max_s8_aligned_end:
    // Horizontal tree reduction (16 int8 -> 1).
    movi a4, 8
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s8 q0, q0, q1
    movi a4, 4
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s8 q0, q0, q1
    movi a4, 2
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s8 q0, q0, q1
    movi a4, 1
    wur.sar_byte a4
    ee.src.q q1, q0, q0
    ee.vmax.s8 q0, q0, q1
    ee.movi.32.a q0, a2, 0
    slli a2, a2, 24
    srai a2, a2, 24
    retw



/* int16_t dl_tie728_reduce_abs_max_s16_aligned(int16_t *input, int32_t size);
 *
 *     int16_t result = 0;
 *     for (int i = 0; i < size; i++) {
 *         int16_t abs_val = abs(input[i]);  // saturates at INT16_MAX
 *         if (abs_val > result) result = abs_val;
 *     }
 *     return result;
 */
    .text
    .balign 4
    .global dl_tie728_reduce_abs_max_s16_aligned
    .type   dl_tie728_reduce_abs_max_s16_aligned, @function
dl_tie728_reduce_abs_max_s16_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 8)
     */
    entry sp, 16

    srai a4, a3, 3                 // a4 = size / 8
    ee.vld.128.ip q0, a2, 16       // load first vector
    ee.zero.q q2                   // q2 = 0
    ee.vsubs.s16 q1, q2, q0        // q1 = -q0  (saturating negation)
    ee.vmax.s16 q0, q0, q1         // q0 = max(q0, -q0) = abs(q0)
    addi a4, a4, -1
    beqz a4, dl_tie728_reduce_abs_max_s16_aligned_end

    dl_tie728_reduce_abs_max_s16_aligned_loop:
        ee.vld.128.ip q1, a2, 16   // load next vector
        ee.vsubs.s16 q3, q2, q1    // q3 = -q1
        ee.vmax.s16 q1, q1, q3     // q1 = abs(q1)
        ee.vmax.s16 q0, q0, q1     // element-wise running max of abs values
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_abs_max_s16_aligned_loop

    dl_tie728_reduce_abs_max_s16_aligned_end:
    // Horizontal tree reduction (8 int16 -> 1).
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
    slli a2, a2, 16
    srai a2, a2, 16
    retw


/* int32_t dl_tie728_reduce_l1_s8_aligned(int8_t *input, int32_t size);
 *
 *     int32_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += abs(input[i]);
 *     return sum;
 */
    .text
    .balign 4
    .global dl_tie728_reduce_l1_s8_aligned
    .type   dl_tie728_reduce_l1_s8_aligned, @function
dl_tie728_reduce_l1_s8_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 16)
     */
    entry sp, 16

    ee.zero.q q2                      // q2 = 0 (compare reference)
    ee.vcmp.eq.s8 q3, q2, q2          // q3 = -1 in all 16 lanes
    ee.vsubs.s8 q3, q2, q3            // q3 = broadcast 1 to all 16 lanes

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
    .balign 4
    .global dl_tie728_reduce_l1_s16_aligned
    .type   dl_tie728_reduce_l1_s16_aligned, @function
dl_tie728_reduce_l1_s16_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 8)
     */
    entry sp, 16

    ee.zero.q q2                      // q2 = 0 (compare reference)
    ee.vcmp.eq.s16 q3, q2, q2         // q3 = -1 in all 8 lanes
    ee.vsubs.s16 q3, q2, q3           // q3 = broadcast 1 to all 8 lanes

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


/* int32_t dl_tie728_reduce_sum_s8_aligned(int8_t *input, int32_t size);
 *
 *     int32_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += input[i];
 *     return sum;
 */
    .text
    .balign 4
    .global dl_tie728_reduce_sum_s8_aligned
    .type   dl_tie728_reduce_sum_s8_aligned, @function
dl_tie728_reduce_sum_s8_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 16)
     */
    entry sp, 16

    ee.zero.q q2
    ee.vcmp.eq.s8 q1, q2, q2          // q1 = -1 in all 16 lanes
    ee.vsubs.s8 q1, q2, q1            // q1 = broadcast 1 to all 16 lanes

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
    .balign 4
    .global dl_tie728_reduce_sum_s16_aligned
    .type   dl_tie728_reduce_sum_s16_aligned, @function
dl_tie728_reduce_sum_s16_aligned:
    /* a2 = input (16-byte aligned)
     * a3 = size  (multiple of 8)
     */
    entry sp, 16

    ee.zero.q q2
    ee.vcmp.eq.s16 q1, q2, q2         // q1 = -1 in all 8 lanes
    ee.vsubs.s16 q1, q2, q1           // q1 = broadcast 1 to all 8 lanes

    ee.zero.accx
    srai a4, a3, 3                   // a4 = size / 8 (8 int16 per 128-bit vector)
    beqz a4, dl_tie728_reduce_sum_s16_aligned_end

    dl_tie728_reduce_sum_s16_aligned_loop:
        ee.vld.128.ip    q0, a2, 16  // load 8 int16, a2 += 16
        ee.vmulas.s16.accx q0, q1    // ACCX += sum(q0[i] * 1)
        addi a4, a4, -1
        bnez a4, dl_tie728_reduce_sum_s16_aligned_loop
    dl_tie728_reduce_sum_s16_aligned_end:

    rur.accx_0 a2                     // low 32 bits
    rur.accx_1 a3                     // high 8 bits (zero-extended from bit 39:32)
    slli a3, a3, 24                  // sign-extend: bit 7 → bit 31
    srai a3, a3, 24                  // fill upper 24 bits with sign
    retw                              // return int64_t as a3:a2
