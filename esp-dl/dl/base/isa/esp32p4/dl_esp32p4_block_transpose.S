// dl_esp32p4_block_transpose
// Kernel 2 — generalized block copy transpose for INT8
// Transposes N×M grid of K-byte blocks (scatter-write)
//   a0 = output pointer
//   a1 = input pointer
//   a2 = N (any value >= 1)
//   a3 = M (any value >= 1)
//   a4 = K (block size in bytes, must be multiple of 16)
//
// Each "element" is a K-byte block (K=16,32,48,...).
// Input:  N rows × M blocks × K bytes (contiguous), row stride = M*K
// Output: M rows × N blocks × K bytes (contiguous), row stride = N*K
//
// Internal fast paths:
//   K==16, M==4:  hard-unrolled 4 load/store (zero loop overhead)
//   K==16, M>=8:  8-block unrolled main + 1-block tail
//   K==16, M<8:   1-block tail loop
//   K>16:         general nested loop (K/16 sub-blocks per block)
//
// Throughput: ~2,153 MB/s on ESP32-P4 (30× scalar) for K==16 path
// Validated on ESP32-P4 silicon.

    .text
    .global dl_esp32p4_block_transpose
    .type   dl_esp32p4_block_transpose, @function
    .balign 4
    .option norvc

dl_esp32p4_block_transpose:
    addi    sp, sp, -16
    sw      s0, 0(sp)
    sw      s1, 4(sp)
    sw      s2, 8(sp)
    sw      s3, 12(sp)

    // Compute strides using K (a4)
    mul     t3, a2, a4              // t3 = N*K (output row stride in bytes)
    mul     t5, a3, a4              // t5 = M*K (input row stride in bytes)
    mv      t0, a2                  // t0 = N (outer loop counter)
    mv      s0, a1                  // s0 = input row base
    mv      s1, a0                  // s1 = output col base

    // Branch on K for fast path
    li      t6, 16
    bne     a4, t6, .L_k2_general   // K != 16 → general path

    // ===== K==16 fast path =====
    // Branch on M for best unrolling
    li      t6, 4
    beq     a3, t6, .L_k2_m4_outer  // M==4 → hard-unrolled

    // Compute main (M/8) and tail (M%8) counts
    srli    s2, a3, 3               // s2 = M/8 (main loop iterations)
    andi    s3, a3, 7               // s3 = M%8 (tail count)

    // Fall through to K16 general M path

// ----- K==16, general M: outer loop over N rows -----
.L_k2_k16_row:
    mv      a4, s0                  // a4 = load cursor (contiguous input)
    mv      a5, s1                  // a5 = store cursor (strided output)
    mv      t1, s2                  // t1 = main loop count (M/8)
    beqz    t1, .L_k2_k16_tail_setup

// ----- Main loop: 8 blocks per iteration -----
.L_k2_k16_col8:
    esp.vld.128.ip  q0, a4, 16
    esp.vld.128.ip  q1, a4, 16
    esp.vld.128.ip  q2, a4, 16
    esp.vld.128.ip  q3, a4, 16
    esp.vld.128.ip  q4, a4, 16
    esp.vld.128.ip  q5, a4, 16
    esp.vld.128.ip  q6, a4, 16
    esp.vld.128.ip  q7, a4, 16

    esp.vst.128.xp  q0, a5, t3
    esp.vst.128.xp  q1, a5, t3
    esp.vst.128.xp  q2, a5, t3
    esp.vst.128.xp  q3, a5, t3
    esp.vst.128.xp  q4, a5, t3
    esp.vst.128.xp  q5, a5, t3
    esp.vst.128.xp  q6, a5, t3
    esp.vst.128.xp  q7, a5, t3

    addi    t1, t1, -1
    bnez    t1, .L_k2_k16_col8

// ----- Tail: 1 block at a time for remaining M%8 -----
.L_k2_k16_tail_setup:
    mv      t2, s3                  // t2 = tail count
    beqz    t2, .L_k2_k16_next_row

.L_k2_k16_tail:
    esp.vld.128.ip  q0, a4, 16
    esp.vst.128.xp  q0, a5, t3

    addi    t2, t2, -1
    bnez    t2, .L_k2_k16_tail

.L_k2_k16_next_row:
    add     s0, s0, t5              // s0 += M*K (next input row)
    addi    s1, s1, 16              // s1 += 16  (next output column, K==16)
    addi    t0, t0, -1
    bnez    t0, .L_k2_k16_row
    j       .L_k2_done

// ===== K==16, M==4: hard-unrolled (zero inner loop overhead) =====
.L_k2_m4_outer:
    mv      a4, s0                  // a4 = load cursor
    mv      a5, s1                  // a5 = store cursor

    // Load 4 contiguous 16-byte blocks (64 bytes)
    esp.vld.128.ip  q0, a4, 16
    esp.vld.128.ip  q1, a4, 16
    esp.vld.128.ip  q2, a4, 16
    esp.vld.128.ip  q3, a4, 16

    // Store to 4 different output rows (stride = N*16)
    esp.vst.128.xp  q0, a5, t3
    esp.vst.128.xp  q1, a5, t3
    esp.vst.128.xp  q2, a5, t3
    esp.vst.128.xp  q3, a5, t3

    add     s0, s0, t5              // s0 += M*K = 4*16 = 64 (next input row)
    addi    s1, s1, 16              // s1 += 16 (next output column)
    addi    t0, t0, -1
    bnez    t0, .L_k2_m4_outer
    j       .L_k2_done

// ===== General K path (K=32, 48, 64, ...): nested loop =====
// For each of N rows, copy M blocks of K bytes each.
// Each block is K/16 contiguous SIMD registers in input,
// scattered to output rows spaced N*K bytes apart.
.L_k2_general:
    srli    s2, a4, 4               // s2 = K/16 (sub-blocks per logical block)

.L_k2_gen_row:
    mv      a5, s1                  // a5 = output store base for this input row
    mv      t4, s0                  // t4 = input load cursor (x29, PIE-valid)
    mv      t1, a3                  // t1 = M (block counter)

.L_k2_gen_block:
    // Copy one K-byte block: K/16 SIMD loads + stores
    mv      t2, s2                  // t2 = sub-block counter (K/16)
    mv      t6, a5                  // t6 = output cursor within this block (x31, PIE-valid)

.L_k2_gen_sub:
    esp.vld.128.ip  q0, t4, 16     // load 16 bytes from input (contiguous)
    esp.vst.128.ip  q0, t6, 16     // store 16 bytes to output (contiguous within block)

    addi    t2, t2, -1
    bnez    t2, .L_k2_gen_sub

    // Advance output to next row (stride = N*K)
    add     a5, a5, t3

    addi    t1, t1, -1
    bnez    t1, .L_k2_gen_block

    // Next input row, next output column
    add     s0, s0, t5              // s0 += M*K (next input row)
    add     s1, s1, a4              // s1 += K   (next output column)
    addi    t0, t0, -1
    bnez    t0, .L_k2_gen_row

.L_k2_done:
    lw      s0, 0(sp)
    lw      s1, 4(sp)
    lw      s2, 8(sp)
    lw      s3, 12(sp)
    addi    sp, sp, 16
    ret

    .size dl_esp32p4_block_transpose, .-dl_esp32p4_block_transpose
