#if defined(__riscv)

############################################################################################################################################################
####
#### 1x1 conv micro-kernels for 4 interleaved pixels, xp[i * 4 + p]. Both live in IRAM: where flash and PSRAM share one
#### cache, streaming PSRAM weights would evict them.
####
############################################################################################################################################################

# void dl_riscv_w8a16_mac_3x4(const int8_t *f0, const int8_t *f1, const int8_t *f2, const int16_t *xp, const int16_t *xe,
#                             int32_t *acc)
# acc[j * 4 + p] = sum_i f_j[i] * xp[i * 4 + p] over [xp, xe), 3 output channels x 4 pixels.
# Scheduled for a 2-cycle load and mul latency: every result is consumed at least two instructions after it is produced.
# The 12 accumulators live in s0-s11 and the main loop covers 4 input channels per iteration, with ra holding its end.
.section .iram1.dl_riscv_w8a16_mac_3x4,"ax",@progbits
.global dl_riscv_w8a16_mac_3x4
.type dl_riscv_w8a16_mac_3x4, @function
.p2align 2
dl_riscv_w8a16_mac_3x4:
    addi sp, sp, -64
    sw   s0, 0(sp)
    sw   s1, 4(sp)
    sw   s2, 8(sp)
    sw   s3, 12(sp)
    sw   s4, 16(sp)
    sw   s5, 20(sp)
    sw   s6, 24(sp)
    sw   s7, 28(sp)
    sw   s8, 32(sp)
    sw   s9, 36(sp)
    sw   s10, 40(sp)
    sw   s11, 44(sp)
    sw   ra, 48(sp)
    li   s0, 0
    li   s1, 0
    li   s2, 0
    li   s3, 0
    li   s4, 0
    li   s5, 0
    li   s6, 0
    li   s7, 0
    li   s8, 0
    li   s9, 0
    li   s10, 0
    li   s11, 0
    sub  t0, a4, a3
    andi t0, t0, -32
    add  ra, a3, t0
    bgeu a3, ra, 3f
.p2align 2
1:
    lb   t0, 0(a0)
    lb   t1, 0(a1)
    lh   t3, 0(a3)
    lh   t4, 2(a3)
    lh   t5, 4(a3)
    lh   t6, 6(a3)
    mul  a6, t0, t3
    mul  a7, t0, t4
    add  s0, s0, a6
    mul  a6, t0, t5
    add  s1, s1, a7
    mul  a7, t0, t6
    add  s2, s2, a6
    lb   t2, 0(a2)
    add  s3, s3, a7
    mul  a6, t1, t3
    mul  a7, t1, t4
    add  s4, s4, a6
    mul  a6, t1, t5
    add  s5, s5, a7
    mul  a7, t1, t6
    add  s6, s6, a6
    mul  a6, t2, t3
    add  s7, s7, a7
    mul  a7, t2, t4
    add  s8, s8, a6
    mul  a6, t2, t5
    add  s9, s9, a7
    mul  a7, t2, t6
    add  s10, s10, a6
    lb   t0, 1(a0)
    lb   t1, 1(a1)
    add  s11, s11, a7
    lh   t3, 8(a3)
    lh   t4, 10(a3)
    lh   t5, 12(a3)
    lh   t6, 14(a3)
    mul  a6, t0, t3
    mul  a7, t0, t4
    add  s0, s0, a6
    mul  a6, t0, t5
    add  s1, s1, a7
    mul  a7, t0, t6
    add  s2, s2, a6
    lb   t2, 1(a2)
    add  s3, s3, a7
    mul  a6, t1, t3
    mul  a7, t1, t4
    add  s4, s4, a6
    mul  a6, t1, t5
    add  s5, s5, a7
    mul  a7, t1, t6
    add  s6, s6, a6
    mul  a6, t2, t3
    add  s7, s7, a7
    mul  a7, t2, t4
    add  s8, s8, a6
    mul  a6, t2, t5
    add  s9, s9, a7
    mul  a7, t2, t6
    add  s10, s10, a6
    lb   t0, 2(a0)
    lb   t1, 2(a1)
    add  s11, s11, a7
    lh   t3, 16(a3)
    lh   t4, 18(a3)
    lh   t5, 20(a3)
    lh   t6, 22(a3)
    mul  a6, t0, t3
    mul  a7, t0, t4
    add  s0, s0, a6
    mul  a6, t0, t5
    add  s1, s1, a7
    mul  a7, t0, t6
    add  s2, s2, a6
    lb   t2, 2(a2)
    add  s3, s3, a7
    mul  a6, t1, t3
    mul  a7, t1, t4
    add  s4, s4, a6
    mul  a6, t1, t5
    add  s5, s5, a7
    mul  a7, t1, t6
    add  s6, s6, a6
    mul  a6, t2, t3
    add  s7, s7, a7
    mul  a7, t2, t4
    add  s8, s8, a6
    mul  a6, t2, t5
    add  s9, s9, a7
    mul  a7, t2, t6
    add  s10, s10, a6
    lb   t0, 3(a0)
    lb   t1, 3(a1)
    add  s11, s11, a7
    lh   t3, 24(a3)
    lh   t4, 26(a3)
    lh   t5, 28(a3)
    lh   t6, 30(a3)
    mul  a6, t0, t3
    mul  a7, t0, t4
    add  s0, s0, a6
    mul  a6, t0, t5
    add  s1, s1, a7
    mul  a7, t0, t6
    add  s2, s2, a6
    lb   t2, 3(a2)
    add  s3, s3, a7
    mul  a6, t1, t3
    mul  a7, t1, t4
    add  s4, s4, a6
    mul  a6, t1, t5
    add  s5, s5, a7
    mul  a7, t1, t6
    add  s6, s6, a6
    mul  a6, t2, t3
    add  s7, s7, a7
    mul  a7, t2, t4
    add  s8, s8, a6
    mul  a6, t2, t5
    add  s9, s9, a7
    mul  a7, t2, t6
    add  s10, s10, a6
    addi a0, a0, 4
    addi a1, a1, 4
    addi a2, a2, 4
    addi a3, a3, 32
    add  s11, s11, a7
    bltu a3, ra, 1b
3:
    bgeu a3, a4, 2f
4:
    lb   t0, 0(a0)
    lb   t1, 0(a1)
    lh   t3, 0(a3)
    lh   t4, 2(a3)
    lh   t5, 4(a3)
    lh   t6, 6(a3)
    mul  a6, t0, t3
    mul  a7, t0, t4
    add  s0, s0, a6
    mul  a6, t0, t5
    add  s1, s1, a7
    mul  a7, t0, t6
    add  s2, s2, a6
    lb   t2, 0(a2)
    add  s3, s3, a7
    mul  a6, t1, t3
    mul  a7, t1, t4
    add  s4, s4, a6
    mul  a6, t1, t5
    add  s5, s5, a7
    mul  a7, t1, t6
    add  s6, s6, a6
    mul  a6, t2, t3
    add  s7, s7, a7
    mul  a7, t2, t4
    add  s8, s8, a6
    mul  a6, t2, t5
    add  s9, s9, a7
    mul  a7, t2, t6
    add  s10, s10, a6
    addi a0, a0, 1
    addi a1, a1, 1
    addi a2, a2, 1
    addi a3, a3, 8
    add  s11, s11, a7
    bltu a3, a4, 4b
2:
    sw   s0, 0(a5)
    sw   s1, 4(a5)
    sw   s2, 8(a5)
    sw   s3, 12(a5)
    sw   s4, 16(a5)
    sw   s5, 20(a5)
    sw   s6, 24(a5)
    sw   s7, 28(a5)
    sw   s8, 32(a5)
    sw   s9, 36(a5)
    sw   s10, 40(a5)
    sw   s11, 44(a5)
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    lw   s2, 8(sp)
    lw   s3, 12(sp)
    lw   s4, 16(sp)
    lw   s5, 20(sp)
    lw   s6, 24(sp)
    lw   s7, 28(sp)
    lw   s8, 32(sp)
    lw   s9, 36(sp)
    lw   s10, 40(sp)
    lw   s11, 44(sp)
    lw   ra, 48(sp)
    addi sp, sp, 64
    ret
.size dl_riscv_w8a16_mac_3x4, .-dl_riscv_w8a16_mac_3x4

# void dl_riscv_s16_mac_2x4(const int16_t *f0, const int16_t *f1, const int16_t *xp, const int16_t *xe, int32_t *sh)
# sh[j * 4 + p] = wrapping sum S, sh[8 + j * 4 + p] = sum of p >> 16 H, for output channel j of 2 and pixel p of 4 over
# [xp, xe). All 28 allocatable registers are used: the 16 accumulators take s0-s11 and t3-t6, and the output pointer
# and ra are parked on the stack for the duration of the loop.
.section .iram1.dl_riscv_s16_mac_2x4,"ax",@progbits
.global dl_riscv_s16_mac_2x4
.type dl_riscv_s16_mac_2x4, @function
.p2align 2
dl_riscv_s16_mac_2x4:
    addi sp, sp, -64
    sw   s0, 0(sp)
    sw   s1, 4(sp)
    sw   s2, 8(sp)
    sw   s3, 12(sp)
    sw   s4, 16(sp)
    sw   s5, 20(sp)
    sw   s6, 24(sp)
    sw   s7, 28(sp)
    sw   s8, 32(sp)
    sw   s9, 36(sp)
    sw   s10, 40(sp)
    sw   s11, 44(sp)
    sw   ra, 48(sp)
    sw   a4, 52(sp)
    li   s0, 0
    li   s1, 0
    li   s2, 0
    li   s3, 0
    li   s4, 0
    li   s5, 0
    li   s6, 0
    li   s7, 0
    li   s8, 0
    li   s9, 0
    li   s10, 0
    li   s11, 0
    li   t3, 0
    li   t4, 0
    li   t5, 0
    li   t6, 0
    addi a3, a3, -24
    bgeu a2, a3, 3f
.p2align 2
1:
    lh   a4, 0(a0)
    lh   a5, 0(a1)
    lh   a6, 0(a2)
    lh   a7, 2(a2)
    mul  t0, a4, a6
    mul  t1, a5, a6
    lh   a6, 4(a2)
    add  s0, s0, t0
    srai t2, t0, 16
    add  s4, s4, t1
    srai ra, t1, 16
    add  s8, s8, t2
    add  t3, t3, ra
    mul  t0, a4, a7
    mul  t1, a5, a7
    lh   a7, 6(a2)
    add  s1, s1, t0
    srai t2, t0, 16
    add  s5, s5, t1
    srai ra, t1, 16
    add  s9, s9, t2
    add  t4, t4, ra
    mul  t0, a4, a6
    mul  t1, a5, a6
    add  s2, s2, t0
    srai t2, t0, 16
    add  s6, s6, t1
    srai ra, t1, 16
    add  s10, s10, t2
    add  t5, t5, ra
    mul  t0, a4, a7
    mul  t1, a5, a7
    add  s3, s3, t0
    srai t2, t0, 16
    add  s7, s7, t1
    srai ra, t1, 16
    add  s11, s11, t2
    add  t6, t6, ra
    lh   a4, 2(a0)
    lh   a5, 2(a1)
    lh   a6, 8(a2)
    lh   a7, 10(a2)
    mul  t0, a4, a6
    mul  t1, a5, a6
    lh   a6, 12(a2)
    add  s0, s0, t0
    srai t2, t0, 16
    add  s4, s4, t1
    srai ra, t1, 16
    add  s8, s8, t2
    add  t3, t3, ra
    mul  t0, a4, a7
    mul  t1, a5, a7
    lh   a7, 14(a2)
    add  s1, s1, t0
    srai t2, t0, 16
    add  s5, s5, t1
    srai ra, t1, 16
    add  s9, s9, t2
    add  t4, t4, ra
    mul  t0, a4, a6
    mul  t1, a5, a6
    add  s2, s2, t0
    srai t2, t0, 16
    add  s6, s6, t1
    srai ra, t1, 16
    add  s10, s10, t2
    add  t5, t5, ra
    mul  t0, a4, a7
    mul  t1, a5, a7
    add  s3, s3, t0
    srai t2, t0, 16
    add  s7, s7, t1
    srai ra, t1, 16
    add  s11, s11, t2
    add  t6, t6, ra
    lh   a4, 4(a0)
    lh   a5, 4(a1)
    lh   a6, 16(a2)
    lh   a7, 18(a2)
    mul  t0, a4, a6
    mul  t1, a5, a6
    lh   a6, 20(a2)
    add  s0, s0, t0
    srai t2, t0, 16
    add  s4, s4, t1
    srai ra, t1, 16
    add  s8, s8, t2
    add  t3, t3, ra
    mul  t0, a4, a7
    mul  t1, a5, a7
    lh   a7, 22(a2)
    add  s1, s1, t0
    srai t2, t0, 16
    add  s5, s5, t1
    srai ra, t1, 16
    add  s9, s9, t2
    add  t4, t4, ra
    mul  t0, a4, a6
    mul  t1, a5, a6
    add  s2, s2, t0
    srai t2, t0, 16
    add  s6, s6, t1
    srai ra, t1, 16
    add  s10, s10, t2
    add  t5, t5, ra
    mul  t0, a4, a7
    mul  t1, a5, a7
    add  s3, s3, t0
    srai t2, t0, 16
    add  s7, s7, t1
    srai ra, t1, 16
    add  s11, s11, t2
    add  t6, t6, ra
    lh   a4, 6(a0)
    lh   a5, 6(a1)
    lh   a6, 24(a2)
    lh   a7, 26(a2)
    mul  t0, a4, a6
    mul  t1, a5, a6
    lh   a6, 28(a2)
    add  s0, s0, t0
    srai t2, t0, 16
    add  s4, s4, t1
    srai ra, t1, 16
    add  s8, s8, t2
    add  t3, t3, ra
    mul  t0, a4, a7
    mul  t1, a5, a7
    lh   a7, 30(a2)
    add  s1, s1, t0
    srai t2, t0, 16
    add  s5, s5, t1
    srai ra, t1, 16
    add  s9, s9, t2
    add  t4, t4, ra
    mul  t0, a4, a6
    mul  t1, a5, a6
    add  s2, s2, t0
    srai t2, t0, 16
    add  s6, s6, t1
    srai ra, t1, 16
    add  s10, s10, t2
    add  t5, t5, ra
    mul  t0, a4, a7
    mul  t1, a5, a7
    add  s3, s3, t0
    srai t2, t0, 16
    add  s7, s7, t1
    srai ra, t1, 16
    add  s11, s11, t2
    add  t6, t6, ra
    addi a0, a0, 8
    addi a1, a1, 8
    addi a2, a2, 32
    bltu a2, a3, 1b
3:
    addi a3, a3, 24
    bgeu a2, a3, 2f
4:
    lh   a4, 0(a0)
    lh   a5, 0(a1)
    lh   a6, 0(a2)
    lh   a7, 2(a2)
    addi a0, a0, 2
    mul  t0, a4, a6
    mul  t1, a5, a6
    lh   a6, 4(a2)
    add  s0, s0, t0
    srai t2, t0, 16
    add  s4, s4, t1
    srai ra, t1, 16
    add  s8, s8, t2
    add  t3, t3, ra
    mul  t0, a4, a7
    mul  t1, a5, a7
    lh   a7, 6(a2)
    add  s1, s1, t0
    srai t2, t0, 16
    add  s5, s5, t1
    srai ra, t1, 16
    add  s9, s9, t2
    add  t4, t4, ra
    mul  t0, a4, a6
    mul  t1, a5, a6
    addi a1, a1, 2
    add  s2, s2, t0
    srai t2, t0, 16
    add  s6, s6, t1
    srai ra, t1, 16
    add  s10, s10, t2
    add  t5, t5, ra
    mul  t0, a4, a7
    mul  t1, a5, a7
    addi a2, a2, 8
    add  s3, s3, t0
    srai t2, t0, 16
    add  s7, s7, t1
    srai ra, t1, 16
    add  s11, s11, t2
    add  t6, t6, ra
    bltu a2, a3, 4b
2:
    lw   a4, 52(sp)
    sw   s0, 0(a4)
    sw   s1, 4(a4)
    sw   s2, 8(a4)
    sw   s3, 12(a4)
    sw   s4, 16(a4)
    sw   s5, 20(a4)
    sw   s6, 24(a4)
    sw   s7, 28(a4)
    sw   s8, 32(a4)
    sw   s9, 36(a4)
    sw   s10, 40(a4)
    sw   s11, 44(a4)
    sw   t3, 48(a4)
    sw   t4, 52(a4)
    sw   t5, 56(a4)
    sw   t6, 60(a4)
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    lw   s2, 8(sp)
    lw   s3, 12(sp)
    lw   s4, 16(sp)
    lw   s5, 20(sp)
    lw   s6, 24(sp)
    lw   s7, 28(sp)
    lw   s8, 32(sp)
    lw   s9, 36(sp)
    lw   s10, 40(sp)
    lw   s11, 44(sp)
    lw   ra, 48(sp)
    addi sp, sp, 64
    ret
.size dl_riscv_s16_mac_2x4, .-dl_riscv_s16_mac_2x4

#endif
