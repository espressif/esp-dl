/*
 * SIMD accelerated L2 reduce (sum of squares) kernels for ESP32-P4 (PIE V2).
 *
 * These kernels only handle the contiguous (stride == 1) case. The input pointer
 * must be 16-byte aligned and `size` is expected to be a multiple of the lane
 * count (16 for int8, 8 for int16). The scalar prologue / tail that takes care of
 * unaligned heads and remainders is handled by the C dispatcher in
 * dl_base_reduce.cpp.
 *
 * The XACC accumulator is 40-bit saturating (same as PIE V1 / tie728), so the int16
 * caller must bound the per-call element count (the C dispatcher chunks by 256) to
 * avoid saturation.
 *
 * int8 kernel: the result is extracted with ESP.SRS.S.XACC (arithmetic right shift
 * by 0), which saturates XACC to a 32-bit signed value directly into the return
 * register a0.
 *
 * int16 kernel: the accumulated value is read back with ESP.ST.S.XACC.IP, which
 * sign-extends XACC to 8 bytes, so the 40-bit result is returned as a full int64.
 * Note: the PIE engine cannot address sp (x2) directly, so the stack slot pointer
 * is first copied into a PIE-addressable AR (a3) before the XACC store.
 */

/* int32_t dl_esp32p4_reduce_l2_s8_aligned(int8_t *input, int32_t size);
 *
 *     int32_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += input[i] * input[i];
 *     return sum;
 */
    .text
    .align 4
    .global dl_esp32p4_reduce_l2_s8_aligned
    .type   dl_esp32p4_reduce_l2_s8_aligned, @function
dl_esp32p4_reduce_l2_s8_aligned:
    /* a0 = input (16-byte aligned)
     * a1 = size  (multiple of 16)
     */
    esp.zero.xacc
    srli a2, a1, 4                 // a2 = size / 16 (16 int8 per 128-bit vector)
    beqz a2, 1f
0:
    esp.vld.128.ip     q0, a0, 16  // load 16 int8, a0 += 16
    esp.vmulas.s8.xacc q0, q0      // XACC += sum(q0[i] * q0[i])
    addi a2, a2, -1
    bnez a2, 0b
1:
    esp.srs.s.xacc a0, a2          // a2 == 0 here: arithmetic right shift XACC by 0,
                                   // saturate to int32, write result into a0 (return value)
    ret



/* int64_t dl_esp32p4_reduce_l2_s16_aligned(int16_t *input, int32_t size);
 *
 *     int64_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += (int64_t)input[i] * input[i];
 *     return sum;
 */
    .text
    .align 4
    .global dl_esp32p4_reduce_l2_s16_aligned
    .type   dl_esp32p4_reduce_l2_s16_aligned, @function
dl_esp32p4_reduce_l2_s16_aligned:
    /* a0 = input (16-byte aligned)
     * a1 = size  (multiple of 8)
     */
    esp.zero.xacc
    srli a2, a1, 3                 // a2 = size / 8 (8 int16 per 128-bit vector)
    beqz a2, 1f
0:
    esp.vld.128.ip      q0, a0, 16 // load 8 int16, a0 += 16
    esp.vmulas.s16.xacc q0, q0     // XACC += sum(q0[i] * q0[i])
    addi a2, a2, -1
    bnez a2, 0b
1:
    addi sp, sp, -16
    mv   a3, sp                    // sp (x2) is not PIE-addressable
    esp.st.s.xacc.ip a3, 8         // store sign-extended 64-bit XACC to stack
    lw   a0, 0(sp)                 // low 32 bits  -> int64 low word
    lw   a1, 4(sp)                 // high 32 bits -> int64 high word
    addi sp, sp, 16
    ret



/* int8_t dl_esp32p4_reduce_max_s8_aligned(int8_t *input, int32_t size);
 *
 *     int8_t result = INT8_MIN;
 *     for (int i = 0; i < size; i++) {
 *         if (input[i] > result) result = input[i];
 *     }
 *     return result;
 */
    .text
    .align 4
    .global dl_esp32p4_reduce_max_s8_aligned
    .type   dl_esp32p4_reduce_max_s8_aligned, @function
dl_esp32p4_reduce_max_s8_aligned:
    /* a0 = input (16-byte aligned)
     * a1 = size  (multiple of 16)
     */
    srli a2, a1, 4                 // a2 = size / 16 (16 int8 per 128-bit vector)
    esp.vld.128.ip q0, a0, 16      // load first vector
    addi a2, a2, -1
    beqz a2, 1f                    // only one vector
0:
    esp.vld.128.ip q1, a0, 16      // load next vector
    esp.vmax.s8 q0, q0, q1         // element-wise running max
    addi a2, a2, -1
    bnez a2, 0b
1:
    // Horizontal tree reduction on q0 (16 int8 -> 1).
    // Step 1: max with self shifted by 8 bytes
    li   a3, 8
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s8 q0, q0, q1
    // Step 2: max with self shifted by 4 bytes
    li   a3, 4
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s8 q0, q0, q1
    // Step 3: max with self shifted by 2 bytes
    li   a3, 2
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s8 q0, q0, q1
    // Step 4: max with self shifted by 1 byte
    li   a3, 1
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s8 q0, q0, q1
    // All 16 bytes now hold the global max; extract and sign-extend.
    esp.movi.32.a q0, a0, 0
    slli a0, a0, 24
    srai a0, a0, 24
    ret



/* int16_t dl_esp32p4_reduce_max_s16_aligned(int16_t *input, int32_t size);
 *
 *     int16_t result = INT16_MIN;
 *     for (int i = 0; i < size; i++) {
 *         if (input[i] > result) result = input[i];
 *     }
 *     return result;
 */
    .text
    .align 4
    .global dl_esp32p4_reduce_max_s16_aligned
    .type   dl_esp32p4_reduce_max_s16_aligned, @function
dl_esp32p4_reduce_max_s16_aligned:
    /* a0 = input (16-byte aligned)
     * a1 = size  (multiple of 8)
     */
    srli a2, a1, 3                 // a2 = size / 8 (8 int16 per 128-bit vector)
    esp.vld.128.ip q0, a0, 16      // load first vector
    addi a2, a2, -1
    beqz a2, 1f
0:
    esp.vld.128.ip q1, a0, 16      // load next vector
    esp.vmax.s16 q0, q0, q1        // element-wise running max
    addi a2, a2, -1
    bnez a2, 0b
1:
    // Horizontal tree reduction on q0 (8 int16 -> 1).
    // Step 1: max with self shifted by 8 bytes (= 4 elements)
    li   a3, 8
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s16 q0, q0, q1
    // Step 2: max with self shifted by 4 bytes (= 2 elements)
    li   a3, 4
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s16 q0, q0, q1
    // Step 3: max with self shifted by 2 bytes (= 1 element)
    li   a3, 2
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s16 q0, q0, q1
    // All 8 lanes now hold the global max; extract and sign-extend.
    esp.movi.32.a q0, a0, 0
    slli a0, a0, 16
    srai a0, a0, 16
    ret



/* int8_t dl_esp32p4_reduce_min_s8_aligned(int8_t *input, int32_t size);
 *
 *     int8_t result = INT8_MAX;
 *     for (int i = 0; i < size; i++) {
 *         if (input[i] < result) result = input[i];
 *     }
 *     return result;
 */
    .text
    .align 4
    .global dl_esp32p4_reduce_min_s8_aligned
    .type   dl_esp32p4_reduce_min_s8_aligned, @function
dl_esp32p4_reduce_min_s8_aligned:
    /* a0 = input (16-byte aligned)
     * a1 = size  (multiple of 16)
     */
    srli a2, a1, 4                 // a2 = size / 16 (16 int8 per 128-bit vector)
    esp.vld.128.ip q0, a0, 16      // load first vector
    addi a2, a2, -1
    beqz a2, 1f
0:
    esp.vld.128.ip q1, a0, 16      // load next vector
    esp.vmin.s8 q0, q0, q1         // element-wise running min
    addi a2, a2, -1
    bnez a2, 0b
1:
    // Horizontal tree reduction on q0 (16 int8 -> 1).
    li   a3, 8
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmin.s8 q0, q0, q1
    li   a3, 4
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmin.s8 q0, q0, q1
    li   a3, 2
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmin.s8 q0, q0, q1
    li   a3, 1
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmin.s8 q0, q0, q1
    esp.movi.32.a q0, a0, 0
    slli a0, a0, 24
    srai a0, a0, 24
    ret



/* int16_t dl_esp32p4_reduce_min_s16_aligned(int16_t *input, int32_t size);
 *
 *     int16_t result = INT16_MAX;
 *     for (int i = 0; i < size; i++) {
 *         if (input[i] < result) result = input[i];
 *     }
 *     return result;
 */
    .text
    .align 4
    .global dl_esp32p4_reduce_min_s16_aligned
    .type   dl_esp32p4_reduce_min_s16_aligned, @function
dl_esp32p4_reduce_min_s16_aligned:
    /* a0 = input (16-byte aligned)
     * a1 = size  (multiple of 8)
     */
    srli a2, a1, 3                 // a2 = size / 8 (8 int16 per 128-bit vector)
    esp.vld.128.ip q0, a0, 16      // load first vector
    addi a2, a2, -1
    beqz a2, 1f
0:
    esp.vld.128.ip q1, a0, 16      // load next vector
    esp.vmin.s16 q0, q0, q1        // element-wise running min
    addi a2, a2, -1
    bnez a2, 0b
1:
    // Horizontal tree reduction on q0 (8 int16 -> 1).
    li   a3, 8
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmin.s16 q0, q0, q1
    li   a3, 4
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmin.s16 q0, q0, q1
    li   a3, 2
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmin.s16 q0, q0, q1
    esp.movi.32.a q0, a0, 0
    slli a0, a0, 16
    srai a0, a0, 16
    ret



/* int8_t dl_esp32p4_reduce_abs_max_s8_aligned(int8_t *input, int32_t size);
 *
 *     int8_t result = 0;
 *     for (int i = 0; i < size; i++) {
 *         int8_t abs_val = abs(input[i]);  // saturates at INT8_MAX
 *         if (abs_val > result) result = abs_val;
 *     }
 *     return result;
 */
    .text
    .align 4
    .global dl_esp32p4_reduce_abs_max_s8_aligned
    .type   dl_esp32p4_reduce_abs_max_s8_aligned, @function
dl_esp32p4_reduce_abs_max_s8_aligned:
    /* a0 = input (16-byte aligned)
     * a1 = size  (multiple of 16)
     */
    srli a2, a1, 4                 // a2 = size / 16
    esp.zero.q q2                  // q2 = 0
    esp.vld.128.ip q0, a0, 16      // load first vector
    esp.vsub.s8 q1, q2, q0         // q1 = 0 - q0 = -q0 (hardware saturating sub)
    esp.vmax.s8 q0, q0, q1         // q0 = max(q0, -q0) = abs(q0)
    addi a2, a2, -1
    beqz a2, 1f
0:
    esp.vld.128.ip q1, a0, 16      // load next vector
    esp.vsub.s8 q3, q2, q1         // q3 = -q1
    esp.vmax.s8 q1, q1, q3         // q1 = abs(q1)
    esp.vmax.s8 q0, q0, q1         // element-wise running max of abs
    addi a2, a2, -1
    bnez a2, 0b
1:
    // Horizontal tree reduction (16 int8 -> 1).
    li   a3, 8
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s8 q0, q0, q1
    li   a3, 4
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s8 q0, q0, q1
    li   a3, 2
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s8 q0, q0, q1
    li   a3, 1
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s8 q0, q0, q1
    esp.movi.32.a q0, a0, 0
    slli a0, a0, 24
    srai a0, a0, 24
    ret



/* int16_t dl_esp32p4_reduce_abs_max_s16_aligned(int16_t *input, int32_t size);
 *
 *     int16_t result = 0;
 *     for (int i = 0; i < size; i++) {
 *         int16_t abs_val = abs(input[i]);  // saturates at INT16_MAX
 *         if (abs_val > result) result = abs_val;
 *     }
 *     return result;
 */
    .text
    .align 4
    .global dl_esp32p4_reduce_abs_max_s16_aligned
    .type   dl_esp32p4_reduce_abs_max_s16_aligned, @function
dl_esp32p4_reduce_abs_max_s16_aligned:
    /* a0 = input (16-byte aligned)
     * a1 = size  (multiple of 8)
     */
    srli a2, a1, 3                 // a2 = size / 8
    esp.zero.q q2                  // q2 = 0
    esp.vld.128.ip q0, a0, 16      // load first vector
    esp.vsub.s16 q1, q2, q0        // q1 = 0 - q0 = -q0 (hardware saturating sub)
    esp.vmax.s16 q0, q0, q1        // q0 = max(q0, -q0) = abs(q0)
    addi a2, a2, -1
    beqz a2, 1f
0:
    esp.vld.128.ip q1, a0, 16      // load next vector
    esp.vsub.s16 q3, q2, q1        // q3 = -q1
    esp.vmax.s16 q1, q1, q3        // q1 = abs(q1)
    esp.vmax.s16 q0, q0, q1        // element-wise running max of abs
    addi a2, a2, -1
    bnez a2, 0b
1:
    // Horizontal tree reduction (8 int16 -> 1).
    li   a3, 8
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s16 q0, q0, q1
    li   a3, 4
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s16 q0, q0, q1
    li   a3, 2
    esp.movx.w.sar.bytes a3
    esp.src.q q1, q0, q0
    esp.vmax.s16 q0, q0, q1
    esp.movi.32.a q0, a0, 0
    slli a0, a0, 16
    srai a0, a0, 16
    ret
