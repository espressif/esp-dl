// ============================================================================
// ESP32-P4 SIMD kernel for non-interleaved RoPE, int8 X + int16 caches + int8 Y.
//
// Mixed-precision variant: int8 activations (memory-cheap) with int16 cos/sin
// tables (high-frequency precision). No native mixed-precision MAC exists on
// P4, so both x_first and x_second are widened to int16 via VLDEXT.S8 and the
// full computation runs on the S16 MAC path. The narrowing back to int8 uses
// the same "SRCMB.S8.QACC after S16 MAC" + "VUNZIP.8 pack" trick as
// dl_esp32p4_s8_s16_requantize_linear.
//
// Preconditions (caller MUST verify — no runtime checks here):
//   * non-interleaved layout (pair = (k, k+half))
//   * same-k: cos / sin share exponent
//   * ``half`` is a multiple of 16 (call with ``iters = half / 16``)
//   * all 6 pointers are 16-byte aligned
//   * CFG.vxrm = HALF_EVEN so srcmb.s8.qacc matches tool::round()
//
// Per 16 pairs the kernel issues:
//   2 vldext.s8 (widening loads for x_first, x_second)
//   4 vld.128    (cos_lo, sin_lo, cos_hi, sin_hi)
//   2 vsub.s16   (build -x2_lo, -x2_hi)
//   4 zero.qacc + 8 vmulas.s16 + 4 srcmb.s8  (four y outputs)
//   2 vunzip.8   (pack lo+hi halves)
//   2 vst.128    (store 16 int8 per side)
// That is 26 SIMD ops for 32 output elements. The int8×3 kernel does the same
// work in 15 ops but with an int8 MAC that cannot accept int16 caches, so
// this is the minimum required to combine the two widths.
//
// Range analysis (per accumulator segment before shift):
//   Each 64-bit QACC segment ends holding at most
//     |127 * 32767| + |127 * 32767| = 8,322,818 < 2^24.
//   (The INT8_MIN case widens cleanly to int16, so ``-x2`` fits without
//   saturation — this variant does not carry the off-by-1 edge that the pure
//   int8 kernel has at x2 == INT8_MIN.)
//   Values fit entirely in the low 32 bits of the segment, so SRCMB.S8.QACC
//   correctly extracts them from even 32-bit slots; odd slots hold pure sign
//   extension (0x00…0 or 0xFF…F) whose low byte is 0 or 0xFF — discarded by
//   the subsequent VUNZIP.8.
//
// Signature (RISC-V calling convention):
//   void dl_esp32p4_s8_s16c_rope_neox_simd(
//       int8_t        *y_first,      // a0 — writes yr + i,        i in [0, half)
//       int8_t        *y_second,     // a1 — writes yr + i + half
//       const int8_t  *x_first,      // a2 — reads xr + i
//       const int8_t  *x_second,     // a3 — reads xr + i + half
//       const int16_t *cos_row,      // a4 — reads cr + k,         k = i
//       const int16_t *sin_row,      // a5 — reads sr + k
//       int            iters,        // a6 — half / 16
//       int            mac_shift);   // a7 — QACC shift for both y1 and y2
// ============================================================================

    .align 2
    .text
    .global dl_esp32p4_s8_s16c_rope_neox_simd
    .type   dl_esp32p4_s8_s16c_rope_neox_simd, @function
dl_esp32p4_s8_s16c_rope_neox_simd:
    // Shuttle iters / mac_shift to PIE-accessible registers (t3, t4).
    mv   t3, a6                     // t3 = iters
    mv   t4, a7                     // t4 = mac_shift
    beqz t3, .Ldl_s8_s16c_rope_done

    // q7 = 0.
    esp.zero.q     q7               // q7 = 0

.Ldl_s8_s16c_rope_loop:
    // Widening loads: 16 int8 → 16 int16 across two QRs.
    // vldext.s8.ip q_lo, q_hi, addr, imm — advances addr by imm bytes.
    esp.vldext.s8.ip q0, q1, a2, 16   // q0 = x1[0..7]s16, q1 = x1[8..15]s16
    esp.vldext.s8.ip q2, q3, a3, 16   // q2 = x2[0..7]s16, q3 = x2[8..15]s16

    // --- Sub-block A: pairs 0..7 (produces y1_lo in q6, y2_lo in q0) ---

    esp.vld.128.ip q4, a4, 16       // q4 = cos[0..7] int16
    esp.vld.128.ip q5, a5, 16       // q5 = sin[0..7] int16

    // y1_lo = round((x1_lo*c_lo - x2_lo*s_lo) >> mac_shift), saturated int8.
    esp.vsub.s16 q6, q7, q2         // q6 = -x2_lo (no saturation: int8 fits)
    esp.zero.qacc
    esp.vmulas.s16.qacc q0, q4      // QACC  = x1_lo * c_lo
    esp.vmulas.s16.qacc q6, q5      // QACC += (-x2_lo) * s_lo
    esp.srcmb.s8.qacc   q6, t4, 1   // q6 has y1_lo bytes at even positions

    // y2_lo = round((x1_lo*s_lo + x2_lo*c_lo) >> mac_shift), saturated int8.
    // OVERWRITE q0 (x1_lo) since it is not needed after this MAC pair.
    esp.zero.qacc
    esp.vmulas.s16.qacc q0, q5      // QACC  = x1_lo * s_lo
    esp.vmulas.s16.qacc q2, q4      // QACC += x2_lo * c_lo
    esp.srcmb.s8.qacc   q0, t4, 1   // q0 has y2_lo bytes at even positions

    // --- Sub-block B: pairs 8..15 (produces y1_hi in q2, y2_hi in q1) ---

    esp.vld.128.ip q4, a4, 16       // q4 = cos[8..15] int16
    esp.vld.128.ip q5, a5, 16       // q5 = sin[8..15] int16

    // y1_hi: overwrite q2 (x2_lo, no longer needed) to hold -x2_hi, then
    // the SRCMB result. This is safe because x2_lo's last use was the y2_lo
    // MAC above.
    esp.vsub.s16 q2, q7, q3         // q2 = -x2_hi
    esp.zero.qacc
    esp.vmulas.s16.qacc q1, q4      // QACC  = x1_hi * c_hi
    esp.vmulas.s16.qacc q2, q5      // QACC += (-x2_hi) * s_hi
    esp.srcmb.s8.qacc   q2, t4, 1   // q2 has y1_hi bytes at even positions

    // Pack y1_lo (q6) and y1_hi (q2) into 16 packed int8 and store to y_first.
    // vunzip.8 gathers the even bytes of (q6 || q2) into q6 and the odd bytes
    // (sign-extension junk) into q2, giving us 8 real bytes from each source
    // packed contiguously.
    esp.vunzip.8       q6, q2
    esp.vst.128.ip     q6, a0, 16   // store y_first[i..i+15]

    // y2_hi: overwrite q1 (x1_hi, no longer needed).
    esp.zero.qacc
    esp.vmulas.s16.qacc q1, q5      // QACC  = x1_hi * s_hi
    esp.vmulas.s16.qacc q3, q4      // QACC += x2_hi * c_hi
    esp.srcmb.s8.qacc   q1, t4, 1   // q1 has y2_hi bytes at even positions

    // Pack y2_lo (q0) and y2_hi (q1) and store to y_second.
    esp.vunzip.8       q0, q1
    esp.vst.128.ip     q0, a1, 16   // store y_second[i..i+15]

    addi t3, t3, -1
    bnez t3, .Ldl_s8_s16c_rope_loop

.Ldl_s8_s16c_rope_done:
    ret
    .size dl_esp32p4_s8_s16c_rope_neox_simd, .-dl_esp32p4_s8_s16c_rope_neox_simd

// ============================================================================
// Eight-pair mixed-precision variant.
//
// The 16-pair kernel above requires both half rows to be 16-byte aligned. That
// excludes common head sizes such as D=48/80: their half rows are only
// 8-byte aligned, even though the cache rows remain 16-byte aligned. Loading
// one 64-bit half into each side of a QR and widening them together removes
// that restriction without introducing an unaligned 128-bit access.
//
// Preconditions:
//   * ``half`` is a multiple of 8 (call with ``iters = half / 8``)
//   * x/y pointers are 8-byte aligned
//   * cos/sin pointers are 16-byte aligned
//   * same-k, ``0 < mac_shift <= 31``, CFG.vxrm = HALF_EVEN
//
// The signature is identical to the 16-pair kernel.
// ============================================================================

    .align 2
    .text
    .global dl_esp32p4_s8_s16c8_rope_neox_simd
    .type   dl_esp32p4_s8_s16c8_rope_neox_simd, @function
dl_esp32p4_s8_s16c8_rope_neox_simd:
    mv   t3, a6                     // t3 = iters
    mv   t4, a7                     // t4 = mac_shift
    beqz t3, .Ldl_s8_s16c8_rope_done

    esp.zero.q q7                   // q7 = 0

.Ldl_s8_s16c8_rope_loop:
    // Assemble x1[0..7] and x2[0..7] in one QR, then sign-extend each
    // 64-bit half into a full eight-lane int16 vector.
    esp.vld.l.64.ip q0, a2, 8
    esp.vld.h.64.ip q0, a3, 8
    esp.vext.s8 q1, q2, q0         // q1 = x1 s16, q2 = x2 s16

    esp.vld.128.ip q3, a4, 16      // q3 = cos[0..7] int16
    esp.vld.128.ip q4, a5, 16      // q4 = sin[0..7] int16

    // y1 = round((x1*c - x2*s) >> mac_shift), saturated int8.
    esp.vsub.s16 q5, q7, q2
    esp.zero.qacc
    esp.vmulas.s16.qacc q1, q3
    esp.vmulas.s16.qacc q5, q4
    esp.srcmb.s8.qacc q5, t4, 1
    // SRCMB leaves the eight result bytes in even byte positions. Compact
    // them into the low 64 bits before the 8-byte store.
    esp.zero.q q6
    esp.vunzip.8 q5, q6
    esp.vst.l.64.ip q5, a0, 8

    // y2 = round((x1*s + x2*c) >> mac_shift), saturated int8.
    esp.zero.qacc
    esp.vmulas.s16.qacc q1, q4
    esp.vmulas.s16.qacc q2, q3
    esp.srcmb.s8.qacc q5, t4, 1
    esp.zero.q q6
    esp.vunzip.8 q5, q6
    esp.vst.l.64.ip q5, a1, 8

    addi t3, t3, -1
    bnez t3, .Ldl_s8_s16c8_rope_loop

.Ldl_s8_s16c8_rope_done:
    ret
    .size dl_esp32p4_s8_s16c8_rope_neox_simd, .-dl_esp32p4_s8_s16c8_rope_neox_simd
