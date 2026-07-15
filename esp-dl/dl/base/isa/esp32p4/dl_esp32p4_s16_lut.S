    .text
    .global dl_esp32p4_s16_lut_nearest_neighbor
    .type   dl_esp32p4_s16_lut_nearest_neighbor, @function
    .balign 4
    .option norvc
// ===================================================================
// Author: Boumedine Billal (https://github.com/BoumedineBillal)
//
// dl_esp32p4_s16_lut_nearest_neighbor  (v2 - indexed vector lookup)
//
// INT16 LUT activation with SIMD load/store, no interpolation.
// Nearest-neighbor lookup with HALF_EVEN rounding.
//
// v2 optimization: gather two adjacent int16 LUT entries per ESP.LDXQ.32.
//   - Use floor(index / 2) as the 32-bit gather index.
//   - Select the low/high int16 with a per-lane shift from index & 1.
//   - Unzip the selected low halves into one 8 x int16 output vector.
// table_ptr must be 16-byte aligned and padded for the final 32-bit load.
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
//   t3 -- loop counter
//   a3 -- table base (16-byte aligned)
//   q5 -- broadcast(1)
//   q6 -- broadcast(2)
//   q7 -- broadcast(0x8000)
// ===================================================================
dl_esp32p4_s16_lut_nearest_neighbor:
    // -- Setup --
    esp.zero.q      q0                  // q0 = 0
    esp.vcmp.eq.s16 q5, q0, q0         // q5 = -1 in all 8 lanes
    esp.vsub.s16    q5, q0, q5         // q5 = broadcast(1)
    esp.vadd.u16    q6, q5, q5         // q6 = broadcast(2)
    li              t4, 15
    esp.movx.w.sar  t4
    esp.vsl.32      q7, q5              // q7 = broadcast(0x8000)
    mv              t4, a4             // t4 = shift (a4 not PIE-addressable)
    esp.movx.w.sar  t4                 // SAR = shift
    mv              t3, a2             // t3 = n_8
    beqz            t3, .L_lut_done

    // -- Software loop --
.L_lut_loop:
        // SIMD: load 8 int16 + compute rounded indices
        esp.vld.128.ip  q0, a1, 16     // q0 = input[0..7]
        esp.xorq        q0, q0, q7     // signed -> unsigned: +32768
        esp.vmul.u16    q0, q0, q5     // q0 = rounded LUT index

        // Zero-extend before shifting because ESP.VSRD.16 treats indices >= 0x8000
        // as negative. HALF_EVEN then rounds up only when index % 4 == 3.
        esp.vext.u16    q4, q1, q0
        esp.vsrd.32     q4, q4, q5
        esp.vsrd.32     q1, q1, q5
        esp.vunzip.16   q4, q1         // q4 = round_half_even(index / 2)
        esp.andq        q1, q0, q6     // q1 = index & 2
        esp.vsrd.16     q1, q1, q5     // q1 = (index >> 1) & 1
        esp.andq        q1, q1, q0     // q1 = (index % 4 == 3)
        esp.vsub.u16    q4, q4, q1     // q4 = floor(index / 2)

        // Gather table[floor(index / 2) * 2 .. +1] as 8 x uint32.
        esp.ldxq.32     q2, q4, a3, 0, 0
        esp.ldxq.32     q2, q4, a3, 1, 1
        esp.ldxq.32     q2, q4, a3, 2, 2
        esp.ldxq.32     q2, q4, a3, 3, 3
        esp.ldxq.32     q3, q4, a3, 0, 4
        esp.ldxq.32     q3, q4, a3, 1, 5
        esp.ldxq.32     q3, q4, a3, 2, 6
        esp.ldxq.32     q3, q4, a3, 3, 7

        // Unzip low/high halves, then select high for odd indices. Avoid a
        // right shift by 16 because HALF_EVEN would round from the low half.
        esp.andq        q4, q0, q5     // q4 = index & 1
        esp.vcmp.eq.s16 q4, q4, q5     // q4 = 0xffff for odd indices
        esp.vunzip.16   q2, q3         // q2 = low halves, q3 = high halves
        esp.andq        q3, q3, q4
        esp.notq        q4, q4
        esp.andq        q2, q2, q4
        esp.orq         q2, q2, q3     // q2 = selected 8 x int16
        esp.vst.128.ip  q2, a0, 16

        addi            t3, t3, -1
        bnez            t3, .L_lut_loop

.L_lut_done:
    ret
