#include "dl_tie728_s8.S"

    .text
    .align 4
    .global dl_tie728_s8_matmul_kn
    .type   dl_tie728_s8_matmul_kn, @function
dl_tie728_s8_matmul_kn:
    # a2: const MatMulArgs<int8_t> *
    # Native KN outer-product, same algorithm as the ESP32-P4 kernel.
    # 16 int8 lanes along N. B/C must be 16-byte aligned and the row
    # strides a multiple of 16 so every K-step stays aligned.
    # Inner K loop: software-pipelined fused MAC.
    #   ee.vmulas.s8.qacc.ld.xp  MAC current B*A, prefetch next B row (stride N)
    #   ee.vldbc.8.ip            prefetch next A scalar
    # B ping-pongs in q0/q2 so the next 128-bit load is not used in the same MAC.
    entry   sp, 32

    l32i    a6, a2, 0           # A
    l32i    a7, a2, 4           # B
    l32i    a8, a2, 8           # C
    l32i    a9, a2, 16          # SIMD-covered N (multiple of 16)
    l32i    a10, a2, 24         # A row stride, bytes
    l32i    a11, a2, 28         # B row stride, bytes
    l32i    a12, a2, 32         # C row stride, bytes
    l32i    a3, a2, 36          # row_start
    l32i    a13, a2, 40         # row_end
    l32i    a15, a2, 44         # mac_shift
    l32i    a4, a2, 20          # K
    s32i    a4, sp, 4
    l32i    a4, a2, 48          # activation
    s32i    a4, sp, 0

    sub     a13, a13, a3
    mull    a4, a3, a10
    add     a6, a6, a4
    mull    a4, a3, a12
    add     a8, a8, a4

.Ls8_row:
    mov     a3, a7              # current B column-block base
    mov     a4, a8              # current output pointer
    srli    a14, a9, 4
    s32i    a14, sp, 8

.Ls8_block:
    mov     a2, a3
    mov     a5, a6
    l32i    a14, sp, 4
    ee.zero.qacc

    ee.vld.128.xp q0, a2, a11
    ee.vldbc.8.ip q1, a5, 1
    addi    a14, a14, -1
    s32i    a14, sp, 12         # K - 1
    srli    a14, a14, 2
    loopgtz a14, .Ls8_k4_end
    ee.vmulas.s8.qacc.ld.xp q2, a2, a11, q0, q1
    ee.vldbc.8.ip q1, a5, 1
    ee.vmulas.s8.qacc.ld.xp q0, a2, a11, q2, q1
    ee.vldbc.8.ip q1, a5, 1
    ee.vmulas.s8.qacc.ld.xp q2, a2, a11, q0, q1
    ee.vldbc.8.ip q1, a5, 1
    ee.vmulas.s8.qacc.ld.xp q0, a2, a11, q2, q1
    ee.vldbc.8.ip q1, a5, 1
.Ls8_k4_end:
    l32i    a14, sp, 12
    extui   a14, a14, 0, 2
    loopgtz a14, .Ls8_last
    ee.vmulas.s8.qacc.ld.xp q0, a2, a11, q0, q1
    ee.vldbc.8.ip q1, a5, 1
.Ls8_last:
    ee.vmulas.s8.qacc q0, q1
    tie728_s8_vector_round_result q0, a15, a2, q7
    l32i    a14, sp, 0
    addi    a2, a14, -1
    bnez    a2, .Ls8_store
    ee.vrelu.s8 q0, a2, a2
.Ls8_store:
    ee.vst.128.ip q0, a4, 16
    addi    a3, a3, 16
    l32i    a14, sp, 8
    addi    a14, a14, -1
    s32i    a14, sp, 8
    bnez    a14, .Ls8_block

    add     a6, a6, a10
    add     a8, a8, a12
    addi    a13, a13, -1
    bnez    a13, .Ls8_row

    retw
    .size dl_tie728_s8_matmul_kn, .-dl_tie728_s8_matmul_kn
