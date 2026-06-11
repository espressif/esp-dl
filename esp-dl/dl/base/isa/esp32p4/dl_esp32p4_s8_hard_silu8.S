// ═══════════════════════════════════════════════════════════════════
// Author: Boumedine Billal (https://github.com/BoumedineBillal)
//
// Contribution:
//   - Designed the HardSiluPie8 activation: a custom HardSiLU variant
//     using /8 (2^3) instead of /6, enabling pure shift-based quantization
//     with zero multiply overhead for the divisor. Same accuracy as the
//     /6 variant.
//   - Derived the integer-domain formula from the float definition,
//     mapping all constants to powers of 2 for PIE SIMD compatibility.
//   - Wrote the SIMD kernel using 6 vector registers (q0-q5).
//   - Validated bit-exact on ESP32-P4 silicon (256/256 INT8 values correct)
//     using the P4-JIT hardware debugging framework.
//     (https://github.com/BoumedineBillal/esp32-p4-jit)
// ═══════════════════════════════════════════════════════════════════
//
// ═══════════════════════════════════════════════════════════════════
// dl_esp32p4_s8_hard_silu8
//
// Float:   y = x × clamp(x/8 + 0.5, 0, 1) × (scale_int / 256)
//
// Integer: xs_q   = x_q × s_int                                  (exact, 16-bit)
//          gate_q = clamp(x_q + 2^(-e1+2), 0, 2^(-e1+3))
//          y_q    = xs_q × gate_q × 2^(2*e1 - 3 - e2 - 8)       (single rounding)
//          e1 = input exponent, e2 = output exponent
//          SAR_total = -2*e1 + 3 + e2 + 8  (absorbs both hardsilu shift and scale /256)
//
// Pre-multiply approach (single rounding):
//   xs   = x × scale_int >> 0          (exact, no rounding)
//   gate = clamp(x + half, 0, max)
//   y    = xs × gate >> SAR_total      (single rounding)
//
// SAR_total = (-2*e1 + 3 + e2) + 8     (caller must pass this!)
//
// 12 PIE instructions per 16 INT8 elements. 6 vector registers (q0-q5).
// Operates in INT16 domain: INT8 input is sign-extended to INT16 for
// arithmetic (vadd, vsat, vmul), then packed back to INT8 via vunzip.8.
// No output clamping needed: gate ∈ [0,1] and scale < 1 guarantees |y| ≤ |x| ≤ 127.
// esp.vmul.s16 uses banker's rounding (HALF_EVEN) via PIE CFG register.
//
// Overflow handling:
//   xs = x × s_int >> 0:  max = 127 × 255 = 32,385 → fits INT16 (< 32,767) ✓
//   xs × gate:            max = 32,385 × 1024 = 33M → fits INT32 (< 2^31)   ✓
//   >> SAR_total:         result may exceed INT8 when e2 < e1 (finer output scale)
//   → explicit esp.vsat.s16 [-128, 127] before vunzip.8 pack
//
// Arguments (RISC-V calling convention — all PIE-compatible):
//   a0: int8_t  *output_ptr
//   a1: int8_t  *input_ptr
//   a2: int      n_16          (number of 16-element chunks)
//   a3: int16_t *half_buf      (16-byte aligned, 8× INT16 broadcast of 2^(-e1+2))
//   a4: int      clamp_hi      (upper clamp bound = 2^(-e1+3))
//   a5: int      sar_total     (SAR shift = -2*e1 + 3 + e2 + 8)
//   a6: int16_t *scale_buf     (16-byte aligned, 8× INT16 broadcast of scale_int)
//
// PIE-compatible GP registers used:
//   a0 (x10) — output pointer, auto-incremented
//   a1 (x11) — input pointer, auto-incremented
//   a2 (x12) — clamp_lo = 0 (gate clamping)
//   a3 (x13) — half_buf pointer (setup only)
//   a4 (x14) — clamp_hi (gate clamping)
//   a5 (x15) — sar_total
//   a6 (x16) — scale_buf pointer (setup only)
//   t3 (x28) — loop counter
//   t4 (x29) — SAR=0 for scale pre-multiply
//   t5 (x30) — INT8_MIN = -128 (output saturation)
//   t6 (x31) — INT8_MAX = 127  (output saturation)
//
// Vector registers:
//   q0 — input / gate_lo / y_lo / output
//   q1 — x_lo / xs_lo
//   q2 — x_hi / xs_hi
//   q3 — const half_offset (set once)
//   q4 — const scale_int broadcast (set once)
//   q5 — gate_hi
// ═══════════════════════════════════════════════════════════════════

    .text
    .global dl_esp32p4_s8_hard_silu8
    .type   dl_esp32p4_s8_hard_silu8, @function
    .balign 4
    .option norvc

dl_esp32p4_s8_hard_silu8:

    // ── Setup (once) ──────────────────────────────
    esp.vld.128.ip  q3, a3, 0               // q3 = broadcast(half_offset)
    mv              t4, a6                   // t4 = scale_buf ptr (PIE-compatible)
    esp.vld.128.ip  q4, t4, 0               // q4 = broadcast(scale_int)
    li              t4, 0                    // t4 = 0 (SAR for scale pre-multiply, exact)
    mv              t3, a2                   // t3 = n_16 (save loop count)
    li              a2, 0                    // a2 = clamp_lo = 0
    li              t5, -128                 // t5 = INT8_MIN (for output saturation)
    li              t6, 127                  // t6 = INT8_MAX (for output saturation)
    // a4 already holds clamp_hi
    // a5 already holds sar_total

    // ── Software loop (branch-based) ──────────────
    beqz            t3, .L_hard_silu_done   // skip if n_16 == 0

.L_hard_silu_loop:
        esp.vld.128.ip  q0, a1, 16          // load 16 × INT8, a1 += 16
        esp.vext.s8     q1, q2, q0          // sign-extend INT8→INT16: x_lo=q1, x_hi=q2
        // Compute BOTH gates from original x FIRST (before x is overwritten)
        esp.vadd.s16    q0, q1, q3          // gate_lo = x_lo + half → q0
        esp.vsat.s16    q0, q0, a2, a4      // clamp gate_lo [0, clamp_hi]
        esp.vadd.s16    q5, q2, q3          // gate_hi = x_hi + half → q5
        esp.vsat.s16    q5, q5, a2, a4      // clamp gate_hi [0, clamp_hi]
        // Scale pre-multiply: xs = x × s_int (SAR=0, exact, no rounding)
        esp.movx.w.sar  t4                  // SAR = 0
        esp.vmul.s16    q1, q1, q4          // xs_lo = x_lo × s_int >> 0 (exact)
        esp.vmul.s16    q2, q2, q4          // xs_hi = x_hi × s_int >> 0 (exact)
        // Final multiply: y = xs × gate (SAR_total, single rounding)
        esp.movx.w.sar  a5                  // SAR = sar_total
        esp.vmul.s16    q0, q1, q0          // y_lo = xs_lo × gate_lo >> SAR_total
        esp.vmul.s16    q2, q2, q5          // y_hi = xs_hi × gate_hi >> SAR_total
        // Saturate to INT8 range before pack (prevents overflow when e2 < e1)
        esp.vsat.s16    q0, q0, t5, t6      // clamp y_lo to [-128, 127]
        esp.vsat.s16    q2, q2, t5, t6      // clamp y_hi to [-128, 127]
        // Pack + store
        esp.vunzip.8    q0, q2              // pack INT16→INT8 (now safe)
        esp.vst.128.ip  q0, a0, 16          // store 16 × INT8, a0 += 16
        addi            t3, t3, -1          // decrement counter
        bnez            t3, .L_hard_silu_loop

.L_hard_silu_done:
    ret

    .size dl_esp32p4_s8_hard_silu8, .-dl_esp32p4_s8_hard_silu8

