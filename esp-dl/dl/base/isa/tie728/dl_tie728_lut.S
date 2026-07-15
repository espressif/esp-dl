    .text
    .global dl_tie728_s16_lut_nearest_neighbor
    .type   dl_tie728_s16_lut_nearest_neighbor, @function
    .balign 4
// ===================================================================
// ESP32-S3 INT16 LUT nearest-neighbor lookup.
//
// Uses 128-bit input/output and two four-way interleaved scalar lookup waves.
// HALF_UP is computed explicitly because EE.VMUL.U16 truncates on ESP32-S3.
//
// table_ptr must be 16-byte aligned and padded for the final 32-bit load.
//
// Arguments:
//   a2: int16_t *output_ptr
//   a3: int16_t *input_ptr
//   a4: int       n_8
//   a5: int16_t *table_ptr
//   a6: int       shift       (log2(step))
//
// Register allocation:
//   a7 -- HALF_UP rounding offset
//   a8-a11 -- indices / table addresses
//   a12-a15 -- table values
//   q7 -- broadcast(0x8000)
// ===================================================================
dl_tie728_s16_lut_nearest_neighbor:
    entry           sp, 16

    // Build vector constants without memory buffers.
    ee.zero.q       q0
    ee.vcmp.eq.s16  q5, q0, q0         // q5 = broadcast(-1)
    ee.vsubs.s16    q5, q0, q5         // q5 = broadcast(1)
    movi            a8, 15
    wsr.sar         a8
    ee.vsl.32       q7, q5              // q7 = broadcast(0x8000)

    // a7 = 1 << (shift - 1), SAR = shift for scalar SRL instructions.
    movi            a7, 1
    addi            a8, a6, -1
    ssl             a8
    sll             a7, a7
    ssr             a6
    beqz            a4, .L_lut_done

.L_lut_loop:
    ee.vld.128.ip   q0, a3, 16
    ee.xorq         q0, q0, q7          // signed -> unsigned: +32768

    // Elements 0-3: extract, HALF_UP round, address, load, and insert.
    ee.movi.32.a    q0, a8, 0
    ee.movi.32.a    q0, a10, 1
    extui           a9, a8, 16, 16
    extui           a8, a8, 0, 16
    extui           a11, a10, 16, 16
    extui           a10, a10, 0, 16
    add             a8, a8, a7
    add             a9, a9, a7
    add             a10, a10, a7
    add             a11, a11, a7
    srl             a8, a8
    srl             a9, a9
    srl             a10, a10
    srl             a11, a11
    addx2           a8, a8, a5
    addx2           a9, a9, a5
    addx2           a10, a10, a5
    addx2           a11, a11, a5
    l16si           a12, a8, 0
    l16si           a13, a9, 0
    l16si           a14, a10, 0
    l16si           a15, a11, 0
    extui           a12, a12, 0, 16
    slli            a13, a13, 16
    or              a12, a12, a13
    extui           a14, a14, 0, 16
    slli            a15, a15, 16
    or              a14, a14, a15
    ee.movi.32.q    q1, a12, 0
    ee.movi.32.q    q1, a14, 1

    // Elements 4-7.
    ee.movi.32.a    q0, a8, 2
    ee.movi.32.a    q0, a10, 3
    extui           a9, a8, 16, 16
    extui           a8, a8, 0, 16
    extui           a11, a10, 16, 16
    extui           a10, a10, 0, 16
    add             a8, a8, a7
    add             a9, a9, a7
    add             a10, a10, a7
    add             a11, a11, a7
    srl             a8, a8
    srl             a9, a9
    srl             a10, a10
    srl             a11, a11
    addx2           a8, a8, a5
    addx2           a9, a9, a5
    addx2           a10, a10, a5
    addx2           a11, a11, a5
    l16si           a12, a8, 0
    l16si           a13, a9, 0
    l16si           a14, a10, 0
    l16si           a15, a11, 0
    extui           a12, a12, 0, 16
    slli            a13, a13, 16
    or              a12, a12, a13
    extui           a14, a14, 0, 16
    slli            a15, a15, 16
    or              a14, a14, a15
    ee.movi.32.q    q1, a12, 2
    ee.movi.32.q    q1, a14, 3

    ee.vst.128.ip   q1, a2, 16

    addi            a4, a4, -1
    bnez            a4, .L_lut_loop

.L_lut_done:
    retw

    .size dl_tie728_s16_lut_nearest_neighbor, .-dl_tie728_s16_lut_nearest_neighbor
