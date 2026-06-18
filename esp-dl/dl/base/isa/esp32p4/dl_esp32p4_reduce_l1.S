/*
 * SIMD accelerated L1 reduce (sum of absolute values) kernels for ESP32-P4 (PIE V2).
 *
 * These kernels only handle the contiguous (stride == 1) case. The input pointer
 * must be 16-byte aligned and `size` is expected to be a multiple of the lane
 * count (16 for int8, 8 for int16). The scalar prologue / tail that takes care of
 * unaligned heads and remainders is handled by the C dispatcher in
 * dl_base_reduce.cpp.
 *
 * Technique: accumulate sum(input[i] * sign(input[i])) into XACC, where the sign
 * vector holds +1 for non-negative lanes and -1 for negative lanes. Because the
 * MAC computes each product in a wider intermediate before accumulating, this
 * yields the exact absolute value even for INT8_MIN/INT16_MIN (e.g. -128 * -1 =
 * +128) instead of the saturated result a per-lane negate would produce.
 *
 * The sign vector is built per iteration with a signed compare-less-than against
 * zero (0xFF for negative lanes, 0x00 otherwise) OR'd with a broadcast 1, giving
 * 0xFF == -1 for negatives and 0x01 == +1 for non-negatives.
 *
 * The XACC accumulator is 40-bit saturating, so the int16 caller must bound the
 * per-call element count (the C dispatcher chunks by 256) to avoid saturation.
 *
 * int8 kernel:  the result is extracted with ESP.SRS.S.XACC (arithmetic right
 *               shift by 0), which saturates XACC to a 32-bit signed value
 *               directly into the return register a0.
 *
 * int16 kernel: the accumulated value is read back with ESP.ST.S.XACC.IP, which
 *               sign-extends XACC to 8 bytes, so the 40-bit result is returned
 *               as a full int64. The C dispatcher accumulates chunk results into
 *               a 64-bit total.
 */

/* int32_t dl_esp32p4_reduce_l1_s8_aligned(int8_t *input, int32_t size);
 *
 *     int32_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += abs(input[i]);
 *     return sum;
 */
    .text
    .align 4
    .global dl_esp32p4_reduce_l1_s8_aligned
    .type   dl_esp32p4_reduce_l1_s8_aligned, @function
dl_esp32p4_reduce_l1_s8_aligned:
    /* a0 = input (16-byte aligned)
     * a1 = size  (multiple of 16)
     */
    addi sp, sp, -16
    li   a4, 1
    sb   a4, 0(sp)                   // store byte 1 to stack
    mv   a4, sp                      // sp (x2) is not PIE-addressable
    esp.vldbc.8.ip q2, a4, 0         // q2 = broadcast 1 to all 16 bytes
    esp.zero.q q3                     // q3 = 0 (compare reference)

    esp.zero.xacc
    srli a2, a1, 4                   // a2 = size / 16 (16 int8 per 128-bit vector)
    beqz a2, 1f
0:
    esp.vld.128.ip     q0, a0, 16    // load 16 int8, a0 += 16
    esp.vcmp.lt.s8     q1, q0, q3    // q1 = (q0 < 0) ? 0xFF : 0x00
    esp.orq            q1, q1, q2    // q1 = sign(q0): 0xFF (-1) if neg, 0x01 (+1) if non-neg
    esp.vmulas.s8.xacc q0, q1        // XACC += sum(q0[i] * sign[i]) = sum(|q0[i]|), exact for -128
    addi a2, a2, -1
    bnez a2, 0b
1:
    esp.srs.s.xacc a0, a2            // a2 == 0 here: arithmetic right shift XACC by 0,
                                     // saturate to int32, write result into a0 (return value)
    addi sp, sp, 16
    ret



/* int64_t dl_esp32p4_reduce_l1_s16_aligned(int16_t *input, int32_t size);
 *
 *     int64_t sum = 0;
 *     for (int i = 0; i < size; i++) sum += abs(input[i]);
 *     return sum;
 */
    .text
    .align 4
    .global dl_esp32p4_reduce_l1_s16_aligned
    .type   dl_esp32p4_reduce_l1_s16_aligned, @function
dl_esp32p4_reduce_l1_s16_aligned:
    /* a0 = input (16-byte aligned)
     * a1 = size  (multiple of 8)
     */
    addi sp, sp, -16
    li   a4, 1
    sh   a4, 0(sp)                   // store halfword 1 to stack
    mv   a4, sp                      // sp (x2) is not PIE-addressable
    esp.vldbc.16.ip q2, a4, 0        // q2 = broadcast 1 to all 8 lanes
    esp.zero.q q3                     // q3 = 0 (compare reference)

    esp.zero.xacc
    srli a2, a1, 3                   // a2 = size / 8 (8 int16 per 128-bit vector)
    beqz a2, 1f
0:
    esp.vld.128.ip      q0, a0, 16   // load 8 int16, a0 += 16
    esp.vcmp.lt.s16     q1, q0, q3   // q1 = (q0 < 0) ? 0xFFFF : 0x0000
    esp.orq             q1, q1, q2   // q1 = sign(q0): 0xFFFF (-1) if neg, 0x0001 (+1) if non-neg
    esp.vmulas.s16.xacc q0, q1       // XACC += sum(q0[i] * sign[i]) = sum(|q0[i]|), exact for -32768
    addi a2, a2, -1
    bnez a2, 0b
1:
    addi sp, sp, -16
    mv   a3, sp                      // sp (x2) is not PIE-addressable
    esp.st.s.xacc.ip a3, 8           // store sign-extended 64-bit XACC to stack
    lw   a0, 0(sp)                   // low 32 bits  -> int64 low word
    lw   a1, 4(sp)                   // high 32 bits -> int64 high word
    addi sp, sp, 16
    addi sp, sp, 16
    ret
