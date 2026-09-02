// dl_tie728_s8_transpose
// Kernel 1 — byte zip transpose for INT8 (ESP32-S3 TIE728)
// Transposes N×M INT8 matrix to M×N (row-major)
//   a2 = output pointer
//   a3 = input pointer
//   a4 = N (must be multiple of 8)
//   a5 = M (must be multiple of 16)
//
// 8×16 tiles, 3-stage EE.VZIP butterfly, store order 0,4,2,6,1,5,3,7.

    .align 4
    .text
    .global dl_tie728_s8_transpose
    .type   dl_tie728_s8_transpose, @function
    .section .iram1
dl_tie728_s8_transpose:
    .align 4
    entry   sp, 16

    mov     a8, a5              // a8 = M (input row stride)
    mov     a9, a4              // a9 = N (output row stride)
    srli    a10, a4, 3          // a10 = N/8
    mov     a11, a3             // a11 = input row-strip base
    mov     a12, a2             // a12 = output col base

.L_k1_row:
    srli    a13, a5, 4          // a13 = M/16
    mov     a7, a11             // a7 = input tile origin
    mov     a6, a12             // a6 = output tile origin

.L_k1_col:
    mov     a14, a7
    EE.VLD.128.XP  q0, a14, a8
    EE.VLD.128.XP  q1, a14, a8
    EE.VLD.128.XP  q2, a14, a8
    EE.VLD.128.XP  q3, a14, a8
    EE.VLD.128.XP  q4, a14, a8
    EE.VLD.128.XP  q5, a14, a8
    EE.VLD.128.XP  q6, a14, a8
    EE.VLD.128.XP  q7, a14, a8

    EE.VZIP.8   q0, q1
    EE.VZIP.8   q2, q3
    EE.VZIP.8   q4, q5
    EE.VZIP.8   q6, q7
    EE.VZIP.16  q0, q2
    EE.VZIP.16  q1, q3
    EE.VZIP.16  q4, q6
    EE.VZIP.16  q5, q7
    EE.VZIP.32  q0, q4
    EE.VZIP.32  q1, q5
    EE.VZIP.32  q2, q6
    EE.VZIP.32  q3, q7

    // Store 16 output rows × 8 bytes. 64-bit stores need 8-byte alignment
    // (N%8==0 and 16-byte-aligned output guarantee that).
    mov     a15, a6
    EE.VST.L.64.IP  q0, a15, 0
    add     a15, a15, a9
    EE.VST.H.64.IP  q0, a15, 0
    add     a15, a15, a9
    EE.VST.L.64.IP  q4, a15, 0
    add     a15, a15, a9
    EE.VST.H.64.IP  q4, a15, 0
    add     a15, a15, a9
    EE.VST.L.64.IP  q2, a15, 0
    add     a15, a15, a9
    EE.VST.H.64.IP  q2, a15, 0
    add     a15, a15, a9
    EE.VST.L.64.IP  q6, a15, 0
    add     a15, a15, a9
    EE.VST.H.64.IP  q6, a15, 0
    add     a15, a15, a9
    EE.VST.L.64.IP  q1, a15, 0
    add     a15, a15, a9
    EE.VST.H.64.IP  q1, a15, 0
    add     a15, a15, a9
    EE.VST.L.64.IP  q5, a15, 0
    add     a15, a15, a9
    EE.VST.H.64.IP  q5, a15, 0
    add     a15, a15, a9
    EE.VST.L.64.IP  q3, a15, 0
    add     a15, a15, a9
    EE.VST.H.64.IP  q3, a15, 0
    add     a15, a15, a9
    EE.VST.L.64.IP  q7, a15, 0
    add     a15, a15, a9
    EE.VST.H.64.IP  q7, a15, 0

    addi    a7, a7, 16          // next 16 input columns
    slli    a14, a9, 4          // skip 16 output rows × N
    add     a6, a6, a14

    addi    a13, a13, -1
    bnez    a13, .L_k1_col

    slli    a14, a8, 3          // skip 8 input rows × M
    add     a11, a11, a14
    addi    a12, a12, 8         // next 8 output columns

    addi    a10, a10, -1
    bnez    a10, .L_k1_row

    retw
    .size dl_tie728_s8_transpose, .-dl_tie728_s8_transpose


// dl_tie728_block_transpose
// Kernel 2 — N×M grid of K-byte blocks
//   a2 = output, a3 = input, a4 = N, a5 = M, a6 = K (bytes, any K >= 1)
//
// S3 128-bit load/store force 16-byte alignment, so the SIMD paths require
// K%16==0 (consecutive blocks stay aligned). Other K uses a word/byte copy.

    .align 4
    .text
    .global dl_tie728_block_transpose
    .type   dl_tie728_block_transpose, @function
    .section .iram1
dl_tie728_block_transpose:
    .align 4
    entry   sp, 16

.L_block_body:
    mull    a7, a4, a6          // a7 = N*K (output row stride)
    mull    a8, a5, a6          // a8 = M*K (input row stride)
    mov     a9, a4              // a9 = N
    mov     a10, a3             // a10 = input row base
    mov     a11, a2             // a11 = output col base
    mov     a12, a6             // a12 = K

    movi    a13, 16
    bne     a12, a13, .L_k2_general

.L_k2_k16_row:
    srli    a13, a5, 3          // M/8
    mov     a14, a10
    mov     a15, a11
    beqz    a13, .L_k2_k16_tail

.L_k2_k16_col8:
    EE.VLD.128.IP  q0, a14, 16
    EE.VLD.128.IP  q1, a14, 16
    EE.VLD.128.IP  q2, a14, 16
    EE.VLD.128.IP  q3, a14, 16
    EE.VLD.128.IP  q4, a14, 16
    EE.VLD.128.IP  q5, a14, 16
    EE.VLD.128.IP  q6, a14, 16
    EE.VLD.128.IP  q7, a14, 16
    EE.VST.128.XP  q0, a15, a7
    EE.VST.128.XP  q1, a15, a7
    EE.VST.128.XP  q2, a15, a7
    EE.VST.128.XP  q3, a15, a7
    EE.VST.128.XP  q4, a15, a7
    EE.VST.128.XP  q5, a15, a7
    EE.VST.128.XP  q6, a15, a7
    EE.VST.128.XP  q7, a15, a7
    addi    a13, a13, -1
    bnez    a13, .L_k2_k16_col8

.L_k2_k16_tail:
    movi    a2, 7
    and     a2, a5, a2          // M%8
    beqz    a2, .L_k2_k16_next

.L_k2_k16_tail_loop:
    EE.VLD.128.IP  q0, a14, 16
    EE.VST.128.XP  q0, a15, a7
    addi    a2, a2, -1
    bnez    a2, .L_k2_k16_tail_loop

.L_k2_k16_next:
    add     a10, a10, a8
    addi    a11, a11, 16
    addi    a9, a9, -1
    bnez    a9, .L_k2_k16_row
    j       .L_k2_done

.L_k2_general:
    movi    a2, 15
    and     a2, a12, a2         // K%16
    bnez    a2, .L_k2_scalar

.L_k2_gen_row:
    mov     a3, a5              // M
    mov     a14, a10
    mov     a15, a11

.L_k2_gen_block:
    srli    a2, a12, 4          // K/16
.L_k2_gen_sub:
    EE.VLD.128.IP  q0, a14, 16
    EE.VST.128.IP  q0, a15, 16
    addi    a2, a2, -1
    bnez    a2, .L_k2_gen_sub
    sub     a2, a7, a12         // next output row: +N*K - K
    add     a15, a15, a2
    addi    a3, a3, -1
    bnez    a3, .L_k2_gen_block

    add     a10, a10, a8
    add     a11, a11, a12
    addi    a9, a9, -1
    bnez    a9, .L_k2_gen_row
    j       .L_k2_done

.L_k2_scalar:
.L_k2_sc_row:
    mov     a3, a5
    mov     a14, a10
    mov     a2, a11

.L_k2_sc_block:
    mov     a13, a12
    mov     a15, a2
    or      a4, a14, a15
    movi    a6, 3
    and     a4, a4, a6
    bnez    a4, .L_k2_sc_byte

.L_k2_sc_word:
    movi    a4, 4
    blt     a13, a4, .L_k2_sc_half
    l32i    a6, a14, 0
    s32i    a6, a15, 0
    addi    a14, a14, 4
    addi    a15, a15, 4
    addi    a13, a13, -4
    j       .L_k2_sc_word

.L_k2_sc_half:
    movi    a4, 2
    blt     a13, a4, .L_k2_sc_b1
    l16ui   a6, a14, 0
    s16i    a6, a15, 0
    addi    a14, a14, 2
    addi    a15, a15, 2
    addi    a13, a13, -2
.L_k2_sc_b1:
    beqz    a13, .L_k2_sc_next
    l8ui    a6, a14, 0
    s8i     a6, a15, 0
    addi    a14, a14, 1
    j       .L_k2_sc_next

.L_k2_sc_byte:
    beqz    a13, .L_k2_sc_next
    l8ui    a6, a14, 0
    s8i     a6, a15, 0
    addi    a14, a14, 1
    addi    a15, a15, 1
    addi    a13, a13, -1
    j       .L_k2_sc_byte

.L_k2_sc_next:
    add     a2, a2, a7
    addi    a3, a3, -1
    bnez    a3, .L_k2_sc_block

    add     a10, a10, a8
    add     a11, a11, a12
    addi    a9, a9, -1
    bnez    a9, .L_k2_sc_row

.L_k2_done:
    retw
    .size dl_tie728_block_transpose, .-dl_tie728_block_transpose


    .align 4
    .text
    .global dl_tie728_s16_block_transpose
    .type   dl_tie728_s16_block_transpose, @function
    .section .iram1
dl_tie728_s16_block_transpose:
    .align 4
    entry   sp, 16
    slli    a6, a6, 1           // K_bytes = K_elems * 2
    j       .L_block_body
    .size dl_tie728_s16_block_transpose, .-dl_tie728_s16_block_transpose
