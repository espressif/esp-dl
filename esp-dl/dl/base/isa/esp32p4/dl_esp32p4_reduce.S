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
