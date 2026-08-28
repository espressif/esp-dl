#include "dl_tie728_s16.S"

    # Load 8 unaligned int16s from ptr into dest (Conv 11cn pattern).
    .macro tie728_s16_loadu128 dest, ptr, front, back
    ee.ld.128.usar.ip \front, \ptr, 16
    ee.ld.128.usar.ip \back, \ptr, 16
    ee.src.q.qup \dest, \front, \back
    .endm

    .text
    .align 4
    .global dl_tie728_s16_unaligned_matmul_kn
    .type   dl_tie728_s16_unaligned_matmul_kn, @function
dl_tie728_s16_unaligned_matmul_kn:
    # a2: const MatMulArgs<int16_t> *
    # Native KN outer-product for B/C that are not 16-byte aligned or
    # whose row stride is not a multiple of 8 elements. Each K-row of B
    # is loaded with USAR+SRC.Q because SAR_BYTE can change every row.
    entry   sp, 48

    l32i    a6, a2, 0           # A
    l32i    a7, a2, 4           # B
    l32i    a8, a2, 8           # C
    l32i    a9, a2, 16          # SIMD-covered N (multiple of 8)
    l32i    a10, a2, 24         # A row stride, elements
    l32i    a11, a2, 28         # B row stride, elements
    l32i    a12, a2, 32         # C row stride, elements
    l32i    a3, a2, 36          # row_start
    l32i    a13, a2, 40         # row_end
    l32i    a15, a2, 44         # mac_shift
    l32i    a4, a2, 20          # K
    s32i    a4, sp, 4
    l32i    a4, a2, 48          # activation
    s32i    a4, sp, 0

    slli    a10, a10, 1
    slli    a11, a11, 1
    slli    a12, a12, 1
    sub     a13, a13, a3
    mull    a4, a3, a10
    add     a6, a6, a4
    mull    a4, a3, a12
    add     a8, a8, a4

.Lu16_row:
    mov     a3, a7              # current B column-block base
    mov     a4, a8              # current output pointer
    srli    a14, a9, 3
    s32i    a14, sp, 8

.Lu16_block:
    s32i    a3, sp, 16
    mov     a5, a6
    l32i    a14, sp, 4
    ee.zero.qacc

    mov     a2, a3
    tie728_s16_loadu128 q0, a2, q2, q3
    add     a3, a3, a11
    ee.vldbc.16.ip q1, a5, 2
    addi    a14, a14, -1
    loopgtz a14, .Lu16_last
    ee.vmulas.s16.qacc q0, q1
    mov     a2, a3
    tie728_s16_loadu128 q0, a2, q2, q3
    add     a3, a3, a11
    ee.vldbc.16.ip q1, a5, 2
.Lu16_last:
    ee.vmulas.s16.qacc q0, q1
    tie728_s16_vector_round_result q0, a15, a2, q7
    l32i    a14, sp, 0
    addi    a2, a14, -1
    bnez    a2, .Lu16_store
    ee.vrelu.s16 q0, a2, a2
.Lu16_store:
    dl_tie728_128b_unaligned_store0 q0, a4, a2
    l32i    a3, sp, 16
    addi    a3, a3, 16
    l32i    a14, sp, 8
    addi    a14, a14, -1
    s32i    a14, sp, 8
    bnez    a14, .Lu16_block

    add     a6, a6, a10
    add     a8, a8, a12
    addi    a13, a13, -1
    bnez    a13, .Lu16_row

    retw
    .size dl_tie728_s16_unaligned_matmul_kn, .-dl_tie728_s16_unaligned_matmul_kn
