// ============================================================================
// ESP32-P4 SIMD kernel for non-interleaved RoPE, int8 X + int8 caches + int8 Y.
//
// Preconditions (caller MUST verify — no runtime checks here):
//   * non-interleaved layout (pair = (k, k+half))
//   * same-k: cos / sin share exponent, so a single mac_shift covers both
//     y1 = round((x1*c - x2*s) * kc) and y2 = round((x1*s + x2*c) * kc)
//     where ``kc = 2^-mac_shift`` (esp-dl int8 tensors use pow-of-2 scales).
//   * ``half`` is a multiple of 16 (call with ``iters = half / 16``).
//   * all 6 pointers are 16-byte aligned.
//
// Per 16 pairs the kernel issues:
//   4 vld.128.ip, 1 notq (build ~x2), 2 zero.qacc, 5 vmulas, 2 srcmb, 2 vst.
// That is 16 SIMD ops for 32 output elements, versus ~20 scalar ops per pair
// (fmul / fmsub / fcvt chain) in the fallback — 15-20x fewer ops in flight.
//
// Edge case INT8_MIN (-128):
//   Two's complement gives -x2 == ~x2 + 1 for every int8 value, including
//   INT8_MIN (~(-128) == +127). Computing (~x2)*s + 1*s in the 32-bit QACC
//   accumulator is therefore exact across the entire [-128, 127] range without
//   saturation or scalar fallback.
//
// Signature (RISC-V calling convention):
//   void dl_esp32p4_s8_rope_neox_simd(
//       int8_t       *y_first,       // a0 — writes to yr + i,        i in [0, half)
//       int8_t       *y_second,      // a1 — writes to yr + i + half
//       const int8_t *x_first,       // a2 — reads xr + i
//       const int8_t *x_second,      // a3 — reads xr + i + half
//       const int8_t *cos_row,       // a4 — reads cr + k,            k = i
//       const int8_t *sin_row,       // a5 — reads sr + k
//       int           iters,         // a6 — half / 16
//       int           mac_shift);    // a7 — QACC shift for both y1 and y2
// ============================================================================

    .align 2
    .text
    .global dl_esp32p4_s8_rope_neox_simd
    .type   dl_esp32p4_s8_rope_neox_simd, @function
dl_esp32p4_s8_rope_neox_simd:
    // PIE instructions may only reference x8-x15 (s0/s1, a0-a5) and x24-x31
    // (s8-s11, t3-t6). ``iters`` (a6) and ``mac_shift`` (a7) live on the wrong
    // side of that fence so we shuttle them into t3 / t4 up front. This is the
    // one-time cost of following the standard RISC-V ABI at the call boundary.
    mv   t3, a6                     // t3 = iters
    mv   t4, a7                     // t4 = mac_shift
    beqz t3, .Ldl_s8_rope_done

    // q7 = +1 in all 16 int8 lanes (q6 = 0, q7 = ~0 = -1, q7 = 0 - (-1) = +1).
    esp.zero.q     q6               // q6 = 0
    esp.notq       q7, q6           // q7 = -1 (0xFF in every byte)
    esp.vsub.s8    q7, q6, q7       // q7 = +1 (0x01 in every byte)

.Ldl_s8_rope_loop:
    esp.vld.128.ip q0, a2, 16       // q0 = x[i..i+15]
    esp.vld.128.ip q1, a3, 16       // q1 = x[i+half..i+half+15]
    esp.vld.128.ip q2, a4, 16       // q2 = cos[k..k+15]
    esp.vld.128.ip q3, a5, 16       // q3 = sin[k..k+15]

    // y1 = round((x1*c - x2*s) >> mac_shift), saturated to int8.
    esp.notq           q4, q1       // q4 = ~x2 = -x2 - 1 (exact on [-128, 127])
    esp.zero.qacc
    esp.vmulas.s8.qacc q0, q2       // QACC  = x1 * c
    esp.vmulas.s8.qacc q4, q3       // QACC += (-x2 - 1) * s
    esp.vmulas.s8.qacc q7, q3       // QACC += 1 * s = x1 * c - x2 * s
    esp.srcmb.s8.qacc  q5, t4, 1    // q5 = QACC >> t4, saturating to int8
    esp.vst.128.ip     q5, a0, 16   // store 16 lanes of y[i..i+15]

    // y2 = round((x1*s + x2*c) >> mac_shift), saturated to int8.
    esp.zero.qacc
    esp.vmulas.s8.qacc q0, q3       // QACC  = x1 * s
    esp.vmulas.s8.qacc q1, q2       // QACC += x2 * c
    esp.srcmb.s8.qacc  q5, t4, 1    // q5 = QACC >> t4, saturating to int8
    esp.vst.128.ip     q5, a1, 16   // store 16 lanes of y[i+half..i+half+15]

    addi t3, t3, -1
    bnez t3, .Ldl_s8_rope_loop

.Ldl_s8_rope_done:
    ret
    .size dl_esp32p4_s8_rope_neox_simd, .-dl_esp32p4_s8_rope_neox_simd
