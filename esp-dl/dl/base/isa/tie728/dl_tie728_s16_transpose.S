// dl_tie728_s16_transpose
// Kernel 1 — half-word zip transpose for INT16 (ESP32-S3 TIE728)
// Transposes N×M INT16 matrix to M×N (row-major)
//   a2 = output pointer
//   a3 = input pointer
//   a4 = N (must be multiple of 8)
//   a5 = M (must be multiple of 8)
//
// 8×8 tiles, 2-stage EE.VZIP butterfly. Each output row is two 64-bit halves.

    .align 4
    .text
    .global dl_tie728_s16_transpose
    .type   dl_tie728_s16_transpose, @function
    .section .iram1
dl_tie728_s16_transpose:
    .align 4
    entry   sp, 16

    slli    a8, a5, 1           // a8 = M*2 (input row stride)
    slli    a9, a4, 1           // a9 = N*2 (output row stride)
    addi    a7, a9, -8          // a7 = output stride - 8
    srli    a10, a4, 3          // a10 = N/8
    mov     a11, a3             // a11 = input row-strip base
    mov     a12, a2             // a12 = output col base

.L_s16_row:
    srli    a13, a5, 3          // a13 = M/8
    mov     a3, a11             // a3 = input tile origin
    mov     a2, a12             // a2 = output tile origin

.L_s16_col:
    mov     a14, a3
    EE.VLD.128.XP  q0, a14, a8
    EE.VLD.128.XP  q1, a14, a8
    EE.VLD.128.XP  q2, a14, a8
    EE.VLD.128.XP  q3, a14, a8
    EE.VLD.128.XP  q4, a14, a8
    EE.VLD.128.XP  q5, a14, a8
    EE.VLD.128.XP  q6, a14, a8
    EE.VLD.128.XP  q7, a14, a8

    EE.VZIP.16  q0, q1
    EE.VZIP.16  q2, q3
    EE.VZIP.16  q4, q5
    EE.VZIP.16  q6, q7
    EE.VZIP.32  q0, q2
    EE.VZIP.32  q1, q3
    EE.VZIP.32  q4, q6
    EE.VZIP.32  q5, q7

    // Register order: 0/4, 2/6, 1/5, 3/7
    mov     a15, a2
    EE.VST.L.64.IP  q0, a15, 8
    EE.VST.L.64.IP  q4, a15, 0
    add     a15, a15, a7
    EE.VST.H.64.IP  q0, a15, 8
    EE.VST.H.64.IP  q4, a15, 0
    add     a15, a15, a7
    EE.VST.L.64.IP  q2, a15, 8
    EE.VST.L.64.IP  q6, a15, 0
    add     a15, a15, a7
    EE.VST.H.64.IP  q2, a15, 8
    EE.VST.H.64.IP  q6, a15, 0
    add     a15, a15, a7
    EE.VST.L.64.IP  q1, a15, 8
    EE.VST.L.64.IP  q5, a15, 0
    add     a15, a15, a7
    EE.VST.H.64.IP  q1, a15, 8
    EE.VST.H.64.IP  q5, a15, 0
    add     a15, a15, a7
    EE.VST.L.64.IP  q3, a15, 8
    EE.VST.L.64.IP  q7, a15, 0
    add     a15, a15, a7
    EE.VST.H.64.IP  q3, a15, 8
    EE.VST.H.64.IP  q7, a15, 0

    addi    a3, a3, 16          // next 8 input columns
    slli    a14, a9, 3          // skip 8 output rows × N*2
    add     a2, a2, a14

    addi    a13, a13, -1
    bnez    a13, .L_s16_col

    slli    a14, a8, 3          // skip 8 input rows × M*2
    add     a11, a11, a14
    addi    a12, a12, 16        // next 8 output columns

    addi    a10, a10, -1
    bnez    a10, .L_s16_row

    retw
    .size dl_tie728_s16_transpose, .-dl_tie728_s16_transpose
