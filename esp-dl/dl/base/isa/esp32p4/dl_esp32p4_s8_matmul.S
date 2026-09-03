#include "esp_idf_version.h"

#if (ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 5, 2))
#define VLDBC_8_STEP 4 /* xesppie: write bytes*4 */
#else
#define VLDBC_8_STEP 1
#endif

    .text
    .global dl_esp32p4_s8_matmul_kn
    .type dl_esp32p4_s8_matmul_kn, @function
    .balign 4
    .option norvc
dl_esp32p4_s8_matmul_kn:
    # a0: const MatMulArgs<int8_t> *
    # Inner K loop: fused MAC + esp.lp.setup (zero-overhead, same as 1x1 Conv).
    #   esp.vmulas.s8.qacc.ld.xp  MAC current B*A, prefetch next B row (stride N)
    #   esp.vldbc.8.ip            prefetch next A scalar
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
    lw s1, 16(a0)      # SIMD-covered N (multiple of 16)
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

    sub s11, s11, a4
    mul a5, a4, t3
    add s9, s9, a5
    mul a5, a4, t5
    add s10, s10, a5

.Ls8_row:
    mv a4, s8          # current B column-block base
    mv a3, s10         # current output pointer
    srli t6, s1, 4

.Ls8_block:
    mv a0, a4
    mv a1, s9
    lw a2, 4(sp)
    esp.zero.qacc

    # Prologue: B[0] and broadcast A[0]
    # Inner K uses esp.lp.setup (same zero-overhead loop as 1x1 Conv).
    # Count register must be PIE-capable (not t0-t2). Label sits on the
    # last instruction of the body. Count 0 is skipped with beqz.
    esp.vld.128.xp q0, a0, t4
    esp.vldbc.8.ip q1, a1, VLDBC_8_STEP
    addi a5, a2, -1            # fused iterations = K - 1
    beqz a5, .Ls8_last

    srli a5, a5, 2             # (K - 1) / 4
    beqz a5, .Ls8_k_tail
    esp.lp.setup 0, a5, .Ls8_k4_end
    esp.vmulas.s8.qacc.ld.xp q2, a0, t4, q0, q1
    esp.vldbc.8.ip q1, a1, VLDBC_8_STEP
    esp.vmulas.s8.qacc.ld.xp q0, a0, t4, q2, q1
    esp.vldbc.8.ip q1, a1, VLDBC_8_STEP
    esp.vmulas.s8.qacc.ld.xp q2, a0, t4, q0, q1
    esp.vldbc.8.ip q1, a1, VLDBC_8_STEP
    esp.vmulas.s8.qacc.ld.xp q0, a0, t4, q2, q1
.Ls8_k4_end:
    esp.vldbc.8.ip q1, a1, VLDBC_8_STEP

.Ls8_k_tail:
    addi a5, a2, -1
    andi a5, a5, 3
    beqz a5, .Ls8_last
    esp.lp.setup 0, a5, .Ls8_k1_end
    esp.vmulas.s8.qacc.ld.xp q0, a0, t4, q0, q1
.Ls8_k1_end:
    esp.vldbc.8.ip q1, a1, VLDBC_8_STEP

.Ls8_last:
    esp.vmulas.s8.qacc q0, q1
    esp.srcmb.s8.qacc q0, s0, 1
    lw a5, 0(sp)
    addi a2, a5, -1
    bnez a2, .Ls8_store
    esp.vrelu.s8 q0, a2, a2
.Ls8_store:
    esp.vst.128.ip q0, a3, 16
    addi a4, a4, 16
    addi t6, t6, -1
    bnez t6, .Ls8_block

    add s9, s9, t3
    add s10, s10, t5
    addi s11, s11, -1
    bnez s11, .Ls8_row

    lw s0, 28(sp)
    lw s1, 24(sp)
    lw s8, 20(sp)
    lw s9, 16(sp)
    lw s10, 12(sp)
    lw s11, 8(sp)
    addi sp, sp, 32
    ret
    .size dl_esp32p4_s8_matmul_kn, .-dl_esp32p4_s8_matmul_kn
