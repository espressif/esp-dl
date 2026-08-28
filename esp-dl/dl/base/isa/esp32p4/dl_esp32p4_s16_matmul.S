    .text
    .global dl_esp32p4_s16_matmul_kn
    .type dl_esp32p4_s16_matmul_kn, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_matmul_kn:
    # a0: const MatMulArgs<int16_t> *
    # Inner K loop: fused MAC + esp.lp.setup (zero-overhead, same as 1x1 Conv).
    #   esp.vmulas.s16.qacc.ld.xp  MAC current B*A, prefetch next B row (stride N)
    #   esp.vldbc.16.ip            prefetch next A scalar
    # B ping-pongs in q0/q2 so the next 128-bit load is not used in the same MAC.
    addi sp, sp, -32
    sw s0, 28(sp)
    sw s1, 24(sp)
    sw s8, 20(sp)
    sw s9, 16(sp)
    sw s10, 12(sp)
    sw s11, 8(sp)

    lw s9, 0(a0)       # A base
    lw s8, 4(a0)       # B base
    lw s10, 8(a0)      # C base
    lw s1, 16(a0)      # SIMD-covered N (multiple of 8)
    lw t3, 24(a0)      # A row stride, elements
    lw t4, 28(a0)      # B row stride, elements
    lw t5, 32(a0)      # C row stride, elements
    lw a4, 36(a0)      # row_start
    lw s11, 40(a0)     # row_end
    lw s0, 44(a0)      # mac_shift
    lw a5, 20(a0)      # K
    sw a5, 4(sp)
    lw a5, 48(a0)      # activation (Linear=0, ReLU=1)
    sw a5, 0(sp)

    slli t3, t3, 1
    slli t4, t4, 1
    slli t5, t5, 1
    sub s11, s11, a4
    mul a5, a4, t3
    add s9, s9, a5
    mul a5, a4, t5
    add s10, s10, a5

.Ls16_row:
    mv a4, s8          # current B column-block base
    mv a3, s10         # current output pointer
    srli t6, s1, 3

.Ls16_block:
    mv a0, a4
    mv a1, s9
    lw a2, 4(sp)
    esp.zero.qacc

    # Prologue: B[0] and broadcast A[0]
    # Inner K uses esp.lp.setup (same zero-overhead loop as 1x1 Conv).
    esp.vld.128.xp q0, a0, t4
    esp.vldbc.16.ip q1, a1, 2
    addi a5, a2, -1            # fused iterations = K - 1
    beqz a5, .Ls16_last

    srli a5, a5, 2             # (K - 1) / 4
    beqz a5, .Ls16_k_tail
    esp.lp.setup 0, a5, .Ls16_k4_end
    esp.vmulas.s16.qacc.ld.xp q2, a0, t4, q0, q1
    esp.vldbc.16.ip q1, a1, 2
    esp.vmulas.s16.qacc.ld.xp q0, a0, t4, q2, q1
    esp.vldbc.16.ip q1, a1, 2
    esp.vmulas.s16.qacc.ld.xp q2, a0, t4, q0, q1
    esp.vldbc.16.ip q1, a1, 2
    esp.vmulas.s16.qacc.ld.xp q0, a0, t4, q2, q1
.Ls16_k4_end:
    esp.vldbc.16.ip q1, a1, 2

.Ls16_k_tail:
    addi a5, a2, -1
    andi a5, a5, 3
    beqz a5, .Ls16_last
    esp.lp.setup 0, a5, .Ls16_k1_end
    esp.vmulas.s16.qacc.ld.xp q0, a0, t4, q0, q1
.Ls16_k1_end:
    esp.vldbc.16.ip q1, a1, 2

.Ls16_last:
    esp.vmulas.s16.qacc q0, q1
    esp.srcmb.s16.qacc q0, s0, 1
    lw a5, 0(sp)
    addi a2, a5, -1
    bnez a2, .Ls16_store
    esp.vrelu.s16 q0, a2, a2
.Ls16_store:
    esp.vst.128.ip q0, a3, 16
    addi a4, a4, 16
    addi t6, t6, -1
    bnez t6, .Ls16_block

    add s9, s9, t3
    add s10, s10, t5
    addi s11, s11, -1
    bnez s11, .Ls16_row

    lw s0, 28(sp)
    lw s1, 24(sp)
    lw s8, 20(sp)
    lw s9, 16(sp)
    lw s10, 12(sp)
    lw s11, 8(sp)
    addi sp, sp, 32
    ret
    .size dl_esp32p4_s16_matmul_kn, .-dl_esp32p4_s16_matmul_kn
