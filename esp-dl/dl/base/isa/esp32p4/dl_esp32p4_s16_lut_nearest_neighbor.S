    .text
    .global dl_esp32p4_s16_lut_nearest_neighbor
    .type   dl_esp32p4_s16_lut_nearest_neighbor, @function
    .balign 4
    .option norvc
// ===================================================================
// Author: Boumedine Billal (https://github.com/BoumedineBillal)
//
// dl_esp32p4_s16_lut_nearest_neighbor  (v1 - interleaved pipeline)
//
// INT16 LUT activation with SIMD load/store, no interpolation.
// Nearest-neighbor lookup with HALF_EVEN rounding.
//
// v1 optimization: 4-element interleaving for ILP.
//   - Extract 4 indices in parallel (no dependencies)
//   - Compute 4 addresses in parallel (no dependencies)
//   - Issue 4 loads back-to-back (memory pipelining)
//   - Insert 4 values in parallel (no dependencies)
//
// Arguments:
//   a0: int16_t  *output_ptr
//   a1: int16_t  *input_ptr
//   a2: int       n_8         (number of 8-element chunks)
//   a3: int16_t  *table_ptr
//   a4: int       shift       (SAR = log2(step))
//
// Register allocation (after setup):
//   a0 -- output ptr (auto-inc)
//   a1 -- input ptr (auto-inc)
//   a2, a3, a5, t5, t6 -- temps (for 4-way interleave)
//   a4 -- table_base
//   t3 -- loop counter
//   t4 -- reused for address computation
// ===================================================================
dl_esp32p4_s16_lut_nearest_neighbor:
    // -- Setup --
    esp.zero.q      q0                  // q0 = 0
    esp.vcmp.eq.s16 q5, q0, q0         // q5 = -1 in all 8 lanes
    esp.vsub.s16    q5, q0, q5         // q5 = broadcast(1)
    li              t4, 15
    esp.movx.w.sar  t4
    esp.vsl.32      q7, q5              // q7 = broadcast(0x8000)
    mv              t4, a4             // t4 = shift (a4 not PIE-addressable)
    esp.movx.w.sar  t4                 // SAR = shift
    mv              t3, a2             // t3 = n_8
    mv              a4, a3             // a4 = table_base

    // -- Software loop --
.L_lut_loop:
        // SIMD: load 8 int16 + compute rounded indices
        esp.vld.128.ip  q0, a1, 16     // q0 = input[0..7]
        esp.xorq        q0, q0, q7    // signed → unsigned: +32768
        esp.vmul.u16    q0, q0, q5    // q0 = (q0 * 1) >> SAR = rounded index

        // ---- Wave 1: elements 0-3 (4-way interleave) ----
        // Extract 4 indices (all independent)
        esp.movi.16.a   q0, a2, 0     // idx[0] → a2
        esp.movi.16.a   q0, a5, 1     // idx[1] → a5
        esp.movi.16.a   q0, t5, 2     // idx[2] → t5
        esp.movi.16.a   q0, t6, 3     // idx[3] → t6
        // Compute 4 addresses (all independent)
        esp.addx2       a3, a4, a2    // addr[0] = table + idx[0]*2
        esp.addx2       a2, a4, a5    // addr[1] = table + idx[1]*2
        esp.addx2       a5, a4, t5    // addr[2] = table + idx[2]*2
        esp.addx2       t4, a4, t6    // addr[3] = table + idx[3]*2
        // Load 4 table values (back-to-back for memory pipelining)
        lh              t5, 0(a3)     // val[0]
        lh              t6, 0(a2)     // val[1]
        lh              a3, 0(a5)     // val[2]
        lh              a2, 0(t4)     // val[3]
        // Insert 4 values into output register
        esp.movi.16.q   q1, t5, 0
        esp.movi.16.q   q1, t6, 1
        esp.movi.16.q   q1, a3, 2
        esp.movi.16.q   q1, a2, 3

        // ---- Wave 2: elements 4-7 (4-way interleave) ----
        // Extract 4 indices
        esp.movi.16.a   q0, a2, 4
        esp.movi.16.a   q0, a5, 5
        esp.movi.16.a   q0, t5, 6
        esp.movi.16.a   q0, t6, 7
        // Compute 4 addresses
        esp.addx2       a3, a4, a2
        esp.addx2       a2, a4, a5
        esp.addx2       a5, a4, t5
        esp.addx2       t4, a4, t6
        // Load 4 table values
        lh              t5, 0(a3)
        lh              t6, 0(a2)
        lh              a3, 0(a5)
        lh              a2, 0(t4)
        // Insert 4 values
        esp.movi.16.q   q1, t5, 4
        esp.movi.16.q   q1, t6, 5
        esp.movi.16.q   q1, a3, 6
        esp.movi.16.q   q1, a2, 7

        // SIMD: store 8 results
        esp.vst.128.ip  q1, a0, 16
        addi            t3, t3, -1
        bnez            t3, .L_lut_loop

    ret

    .size dl_esp32p4_s16_lut_nearest_neighbor, .-dl_esp32p4_s16_lut_nearest_neighbor
