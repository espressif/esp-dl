// dl_esp32p4_s16_transpose
// Kernel 1 — half-word zip transpose for INT16
// Transposes N×M INT16 matrix to M×N (row-major)
//   a0 = output pointer
//   a1 = input pointer
//   a2 = N (must be multiple of 8)
//   a3 = M (must be multiple of 8)
//
// Processes 8×8 tiles using a 2-stage vzip butterfly, then stores each
// output row as two 64-bit halves (no vzip.64). Mirrors the s8 8×16
// kernel, one element-width step coarser.

    .text
    .global dl_esp32p4_s16_transpose
    .type   dl_esp32p4_s16_transpose, @function
    .balign 4
    .option norvc

dl_esp32p4_s16_transpose:
    addi    sp, sp, -16
    sw      s0, 0(sp)
    sw      s1, 4(sp)
    sw      s2, 8(sp)
    sw      s3, 12(sp)

    slli    t3, a3, 1           // t3 = M*2 (input row stride in bytes)
    slli    t4, a2, 1           // t4 = N*2 (output row stride in bytes)
    addi    t5, t4, -8          // t5 = output stride - 8 (pair-store bump)
    srli    t0, a2, 3           // t0 = N/8 (number of row strips)
    mv      s0, a1              // s0 = input row-strip base
    mv      s1, a0              // s1 = output col base

.L_s16_row:
    srli    t1, a3, 3           // t1 = M/8 (number of col strips)
    mv      s2, s0              // s2 = current input tile origin
    mv      s3, s1              // s3 = current output tile origin

.L_s16_col:
    // ---- Load 8 rows × 8 int16 (16 bytes) ----
    mv      a4, s2
    esp.vld.128.xp  q0, a4, t3
    esp.vld.128.xp  q1, a4, t3
    esp.vld.128.xp  q2, a4, t3
    esp.vld.128.xp  q3, a4, t3
    esp.vld.128.xp  q4, a4, t3
    esp.vld.128.xp  q5, a4, t3
    esp.vld.128.xp  q6, a4, t3
    esp.vld.128.xp  q7, a4, t3

    // ---- 2-stage in-register transpose (butterfly) ----
    // Stage 1: interleave 16-bit elements
    esp.vzip.16 q0, q1
    esp.vzip.16 q2, q3
    esp.vzip.16 q4, q5
    esp.vzip.16 q6, q7
    // Stage 2: interleave 32-bit groups
    esp.vzip.32 q0, q2
    esp.vzip.32 q1, q3
    esp.vzip.32 q4, q6
    esp.vzip.32 q5, q7

    // ---- Store 8 output rows × 8 int16 ----
    // Each output row is qX.low || qY.low (or .high), 8+8 bytes.
    // After vst.*.64.ip (+8) + vst.*.64.xp (+t5), cursor += t4.
    // Register order: 0/4, 2/6, 1/5, 3/7
    mv      a5, s3
    esp.vst.l.64.ip q0, a5, 8
    esp.vst.l.64.xp q4, a5, t5
    esp.vst.h.64.ip q0, a5, 8
    esp.vst.h.64.xp q4, a5, t5
    esp.vst.l.64.ip q2, a5, 8
    esp.vst.l.64.xp q6, a5, t5
    esp.vst.h.64.ip q2, a5, 8
    esp.vst.h.64.xp q6, a5, t5
    esp.vst.l.64.ip q1, a5, 8
    esp.vst.l.64.xp q5, a5, t5
    esp.vst.h.64.ip q1, a5, 8
    esp.vst.h.64.xp q5, a5, t5
    esp.vst.l.64.ip q3, a5, 8
    esp.vst.l.64.xp q7, a5, t5
    esp.vst.h.64.ip q3, a5, 8
    esp.vst.h.64.xp q7, a5, t5

    // ---- Advance to next col strip ----
    addi    s2, s2, 16          // input: next 8 columns (16 bytes)
    slli    t2, t4, 3           // output: skip 8 rows × N*2 bytes
    add     s3, s3, t2

    addi    t1, t1, -1
    bnez    t1, .L_s16_col

    // ---- Advance to next row strip ----
    slli    t2, t3, 3           // input: skip 8 rows × M*2 bytes
    add     s0, s0, t2
    addi    s1, s1, 16          // output: next 8 columns (16 bytes)

    addi    t0, t0, -1
    bnez    t0, .L_s16_row

    lw      s0, 0(sp)
    lw      s1, 4(sp)
    lw      s2, 8(sp)
    lw      s3, 12(sp)
    addi    sp, sp, 16
    ret

    .size dl_esp32p4_s16_transpose, .-dl_esp32p4_s16_transpose


// ===================================================================
// dl_esp32p4_s16_block_transpose
// Thin wrapper around the dtype-agnostic byte block kernel.
//   a0 = output pointer
//   a1 = input pointer
//   a2 = N (any value >= 1)
//   a3 = M (any value >= 1)
//   a4 = K (block size in INT16 elements, any K >= 1)
// ===================================================================

    .global dl_esp32p4_s16_block_transpose
    .type   dl_esp32p4_s16_block_transpose, @function
    .balign 4

dl_esp32p4_s16_block_transpose:
    slli    a4, a4, 1           // K_bytes = K_elems * 2
    j       dl_esp32p4_block_transpose

    .size dl_esp32p4_s16_block_transpose, .-dl_esp32p4_s16_block_transpose
