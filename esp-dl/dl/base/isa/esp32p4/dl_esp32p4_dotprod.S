/* void dl_esp32p4_dotprod_i16k16o32(int32_t *out2_int16, int16_t *in1_int16, int16_t *in2_int16, int shift, int n); */
    .text
    .global dl_esp32p4_dotprod_i16k16o32
    .type   dl_esp32p4_dotprod_i16k16o32, @function
    #.section .iram1
    .align 4
dl_esp32p4_dotprod_i16k16o32:
    /* a0 = int32_t *z
     * a1 = int16_t *x
     * a2 = int16_t *y
     * a3 = int shift
     * a4 = int n
     */

    esp.zero.xacc
    beqz a4, end_dl_esp32p4_dotprod_i16k16o32
    srli a5, a4, 4 // a5 = n // 16
    slli t3, a5, 4 // t3 = (a4 // 16) * 16

    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a2, 16

    beqz a5, loop_end_dl_esp32p4_dotprod_i16k16o32_small
    andi t0, a4, 8
    bnez t0, dl_esp32p4_dotprod_i16k16o32_remainder

    // Exact multiple of 16: pipeline all but the final group. The final
    // multiply must not preload 16 bytes beyond either input.
    addi t0, a5, -1
    beqz t0, dl_esp32p4_dotprod_i16k16o32_final
    dl_esp32p4_dotprod_i16k16o32_loop:
        esp.vld.128.ip q2, a1, 16
        esp.vmulas.s16.xacc.ld.ip q3, a2, 16, q0, q1
        esp.vld.128.ip q0, a1, 16
        esp.vmulas.s16.xacc.ld.ip q1, a2, 16, q2, q3
        addi t0, t0, -1
        bnez t0, dl_esp32p4_dotprod_i16k16o32_loop

    dl_esp32p4_dotprod_i16k16o32_final:
        esp.vld.128.ip q2, a1, 16
        esp.vmulas.s16.xacc.ld.ip q3, a2, 16, q0, q1
        esp.vmulas.s16.xacc q2, q3
        j end_dl_esp32p4_dotprod_i16k16o32

    dl_esp32p4_dotprod_i16k16o32_remainder:
        mv t0, a5
    dl_esp32p4_dotprod_i16k16o32_remainder_loop:
        esp.vld.128.ip q2, a1, 16
        esp.vmulas.s16.xacc.ld.ip q3, a2, 16, q0, q1
        esp.vld.128.ip q0, a1, 16
        esp.vmulas.s16.xacc.ld.ip q1, a2, 16, q2, q3
        addi t0, t0, -1
        bnez t0, dl_esp32p4_dotprod_i16k16o32_remainder_loop

    loop_end_dl_esp32p4_dotprod_i16k16o32_small:
        beq a4, t3, end_dl_esp32p4_dotprod_i16k16o32
        esp.vmulas.s16.xacc q0, q1

    end_dl_esp32p4_dotprod_i16k16o32:
    esp.srs.s.xacc t3, a3
    sw t3, 0(a0)
    ret



/* void dl_esp32p4_dotprod_i16k16o16(int16_t *out2_int16, int16_t *in1_int16, int16_t *in2_int16, int shift, int n); */
    .text
    .global dl_esp32p4_dotprod_i16k16o16
    .type   dl_esp32p4_dotprod_i16k16o16, @function
    #.section .iram1
    .align 4
dl_esp32p4_dotprod_i16k16o16:
    /* a0 = int32_t *z
     * a1 = int16_t *x
     * a2 = int16_t *y
     * a3 = int shift
     * a4 = int n
     */

    esp.zero.xacc
    beqz a4, end_dl_esp32p4_dotprod_i16k16o16
    srli a5, a4, 4 // a5 = n // 16
    slli t3, a5, 4 // t3 = (a4 // 16) * 16

    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a2, 16

    beqz a5, loop_end_dl_esp32p4_dotprod_i16k16o16_small
    andi t0, a4, 8
    bnez t0, dl_esp32p4_dotprod_i16k16o16_remainder

    addi t0, a5, -1
    beqz t0, dl_esp32p4_dotprod_i16k16o16_final
    dl_esp32p4_dotprod_i16k16o16_loop:
        esp.vld.128.ip q2, a1, 16
        esp.vmulas.s16.xacc.ld.ip q3, a2, 16, q0, q1
        esp.vld.128.ip q0, a1, 16
        esp.vmulas.s16.xacc.ld.ip q1, a2, 16, q2, q3
        addi t0, t0, -1
        bnez t0, dl_esp32p4_dotprod_i16k16o16_loop

    dl_esp32p4_dotprod_i16k16o16_final:
        esp.vld.128.ip q2, a1, 16
        esp.vmulas.s16.xacc.ld.ip q3, a2, 16, q0, q1
        esp.vmulas.s16.xacc q2, q3
        j end_dl_esp32p4_dotprod_i16k16o16

    dl_esp32p4_dotprod_i16k16o16_remainder:
        mv t0, a5
    dl_esp32p4_dotprod_i16k16o16_remainder_loop:
        esp.vld.128.ip q2, a1, 16
        esp.vmulas.s16.xacc.ld.ip q3, a2, 16, q0, q1
        esp.vld.128.ip q0, a1, 16
        esp.vmulas.s16.xacc.ld.ip q1, a2, 16, q2, q3
        addi t0, t0, -1
        bnez t0, dl_esp32p4_dotprod_i16k16o16_remainder_loop

    loop_end_dl_esp32p4_dotprod_i16k16o16_small:

        beq a4, t3, end_dl_esp32p4_dotprod_i16k16o16
        esp.vmulas.s16.xacc q0, q1

    end_dl_esp32p4_dotprod_i16k16o16:
    esp.srs.s.xacc t3, a3

    // max(min(t3, Q16_MAX), Q16_MIN)
    li t4, 32767
    li t5, -32768
    blt t3, t4, 1f
    mv t3, t4
1:
    bgt t3, t5, 2f
    mv t3, t5
2:

    sh t3, 0(a0)
    ret

/* void dl_esp32p4_dotprod_i16k8o16(int16_t *out2_int16, int16_t *in1_int8, int16_t *in2_int16, int shift, int n); */
    .text
    .global dl_esp32p4_dotprod_i16k8o16
    .type   dl_esp32p4_dotprod_i16k8o16, @function
    #.section .iram1
    .align 4
dl_esp32p4_dotprod_i16k8o16:
    /* a0 = int16_t *z
     * a1 = int8_t *x
     * a2 = int16_t *y
     * a3 = int shift
     * a4 = int n
     */

    esp.zero.xacc
    beqz a4, end_dl_esp32p4_dotprod_i16k8o16
    // The C++ entry only sends multiples of 16 here. vldext.s8 has a
    // 16-byte granularity, so an 8-element tail is handled by dotprod_c.
    srli a5, a4, 4 // a5 = n // 16

    esp.vldext.s8.ip q0, q1, a1, 16 // load 8bit kernel and sign-extend
    esp.vld.128.ip q2, a2, 16

    addi t0, a5, -1
    beqz t0, dl_esp32p4_dotprod_i16k8o16_final
    dl_esp32p4_dotprod_i16k8o16_loop:
        esp.vld.128.ip q3, a2, 16
        esp.vmulas.s16.xacc.ld.ip q4, a1, 16, q0, q2
        esp.vmulas.s16.xacc.ld.ip q2, a2, 16, q1, q3
        esp.vext.s8 q0, q1, q4
        addi t0, t0, -1
        bnez t0, dl_esp32p4_dotprod_i16k8o16_loop

    dl_esp32p4_dotprod_i16k8o16_final:
        esp.vld.128.ip q3, a2, 16
        esp.vmulas.s16.xacc q0, q2
        esp.vmulas.s16.xacc q1, q3

    end_dl_esp32p4_dotprod_i16k8o16:
    esp.srs.s.xacc t3, a3

    // max(min(t3, Q16_MAX), Q16_MIN)
    li t4, 32767
    li t5, -32768
    blt t3, t4, 1f
    mv t3, t4
1:
    bgt t3, t5, 2f
    mv t3, t5
2:

    sh t3, 0(a0)
    ret



/* void dl_esp32p4_dotprod_i8k8o16(int16_t *out2_int16, int8_t *in1_int8, int8_t *in2_int8, int shift, int n); */
    .text
    .global dl_esp32p4_dotprod_i8k8o16
    .type   dl_esp32p4_dotprod_i8k8o16, @function
    #.section .iram1
    .align 4
dl_esp32p4_dotprod_i8k8o16:
    /* a0 = int32_t *z
     * a1 = int16_t *x
     * a2 = int16_t *y
     * a3 = int shift
     * a4 = int n
     */

    esp.zero.xacc
    beqz a4, end_dl_esp32p4_dotprod_i8k8o16
    srli a5, a4, 5 // a5 = n // 32
    slli t3, a5, 5 // t3 = (a4 // 32) * 32

    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a2, 16

    beqz a5, loop_end_dl_esp32p4_dotprod_i8k8o16_small
    andi t0, a4, 16
    bnez t0, dl_esp32p4_dotprod_i8k8o16_remainder

    // For an exact multiple of 32, do not issue the usual preload after the
    // final 16-byte block.
    addi t0, a5, -1
    beqz t0, dl_esp32p4_dotprod_i8k8o16_final
    dl_esp32p4_dotprod_i8k8o16_loop:
        esp.vld.128.ip q2, a1, 16
        esp.vmulas.s8.xacc.ld.ip q3, a2, 16, q0, q1
        esp.vld.128.ip q0, a1, 16
        esp.vmulas.s8.xacc.ld.ip q1, a2, 16, q2, q3
        addi t0, t0, -1
        bnez t0, dl_esp32p4_dotprod_i8k8o16_loop

    dl_esp32p4_dotprod_i8k8o16_final:
        esp.vld.128.ip q2, a1, 16
        esp.vmulas.s8.xacc.ld.ip q3, a2, 16, q0, q1
        esp.vmulas.s8.xacc q2, q3
        j end_dl_esp32p4_dotprod_i8k8o16

    dl_esp32p4_dotprod_i8k8o16_remainder:
        mv t0, a5
    dl_esp32p4_dotprod_i8k8o16_remainder_loop:
        esp.vld.128.ip q2, a1, 16
        esp.vmulas.s8.xacc.ld.ip q3, a2, 16, q0, q1
        esp.vld.128.ip q0, a1, 16
        esp.vmulas.s8.xacc.ld.ip q1, a2, 16, q2, q3
        addi t0, t0, -1
        bnez t0, dl_esp32p4_dotprod_i8k8o16_remainder_loop

    loop_end_dl_esp32p4_dotprod_i8k8o16_small:

        beq a4, t3, end_dl_esp32p4_dotprod_i8k8o16
        esp.vmulas.s8.xacc q0, q1

    end_dl_esp32p4_dotprod_i8k8o16:
    esp.srs.s.xacc t3, a3

    // max(min(t3, Q16_MAX), Q16_MIN)
    li t4, 32767
    li t5, -32768
    blt t3, t4, 1f
    mv t3, t4
1:
    bgt t3, t5, 2f
    mv t3, t5
2:

    sh t3, 0(a0)
    ret



/* void dl_esp32p4_dotprod_f32(float *output_ptr, float *input0_ptr, float *input1_ptr, int length); */
/* The PIE SIMD extension only supports integer datatypes, so the float dot   */
/* product is implemented with the RISC-V single-precision FPU. The main loop  */
/* is unrolled by 4 with four independent accumulators to hide fmadd latency.  */
    .text
    .global dl_esp32p4_dotprod_f32
    .type   dl_esp32p4_dotprod_f32, @function
    #.section .iram1
    .align 4
dl_esp32p4_dotprod_f32:
    /* a0 = float *z (output)
     * a1 = float *x (input0)
     * a2 = float *y (input1)
     * a3 = int length
     */

    fmv.w.x ft0, zero       // acc0 = 0.0f
    fmv.w.x ft1, zero       // acc1 = 0.0f
    fmv.w.x ft2, zero       // acc2 = 0.0f
    fmv.w.x ft3, zero       // acc3 = 0.0f

    srli t0, a3, 2          // t0 = length / 4 (unrolled iterations)
    andi t1, a3, 3          // t1 = length % 4 (remainder count)

    beqz t0, loop_tail_dl_esp32p4_dotprod_f32

    loop_main_dl_esp32p4_dotprod_f32:
        flw ft4, 0(a1)
        flw ft5, 0(a2)
        flw ft6, 4(a1)
        flw ft7, 4(a2)
        flw ft8, 8(a1)
        flw ft9, 8(a2)
        flw ft10, 12(a1)
        flw ft11, 12(a2)

        fmadd.s ft0, ft4, ft5, ft0
        fmadd.s ft1, ft6, ft7, ft1
        fmadd.s ft2, ft8, ft9, ft2
        fmadd.s ft3, ft10, ft11, ft3

        addi a1, a1, 16
        addi a2, a2, 16
        addi t0, t0, -1
        bnez t0, loop_main_dl_esp32p4_dotprod_f32

    loop_tail_dl_esp32p4_dotprod_f32:
        beqz t1, reduce_dl_esp32p4_dotprod_f32
        flw ft4, 0(a1)
        flw ft5, 0(a2)
        fmadd.s ft0, ft4, ft5, ft0
        addi a1, a1, 4
        addi a2, a2, 4
        addi t1, t1, -1
        bnez t1, loop_tail_dl_esp32p4_dotprod_f32

    reduce_dl_esp32p4_dotprod_f32:
    fadd.s ft0, ft0, ft1
    fadd.s ft2, ft2, ft3
    fadd.s ft0, ft0, ft2
    fsw ft0, 0(a0)
    ret



/* out[v * rows + i] = sat16(round_shift(dot(matrix[i], vecs[v]), shift)).
 * Preconditions: cols > 0, cols % 16 == 0, shift >= 0.
 * Each vector uses one accumulator; matrix rows are reloaded for each vector.
 *
 * vmulas.*.ld.ip accumulates the current qA*qB, then loads the next
 * 16-byte block into qD and advances p.
 * Exact-multiple-of-32 and remainder paths avoid reading past the input.
 */
    .text
    .global dl_esp32p4_mat_vecs_i8k8o16
    .type   dl_esp32p4_mat_vecs_i8k8o16, @function
    #.section .iram1
    .align 4
dl_esp32p4_mat_vecs_i8k8o16:
    /* a0 = int16_t *out (row i)
     * a1 = int8_t *matrix (row i)
     * a2 = int8_t *vecs
     * a3 = int shift
     * a4 = int cols
     * a5 = int rows (counts down)
     * a6 = int nvec
     * a7 = cols // 32
     * t6 = rows * 2, stride between the outputs of two vectors
     * PIE operands must be a0-a5 / t3-t6: t3 walks the row, t4 walks the vector.
     */
    slli t6, a5, 1
    srli a7, a4, 5

    dl_esp32p4_mat_vecs_i8k8o16_row:
        mv t1, a2 // vector v
        mv t2, a0 // out[v * rows + i]
        mv t5, a6

        dl_esp32p4_mat_vecs_i8k8o16_vec:
            mv t3, a1
            mv t4, t1
            esp.zero.xacc
            esp.vld.128.ip q0, t3, 16
            esp.vld.128.ip q1, t4, 16

            beqz a7, dl_esp32p4_mat_vecs_i8k8o16_small
            andi t0, a4, 16
            bnez t0, dl_esp32p4_mat_vecs_i8k8o16_remainder

            // Exact multiple of 32: the last group must not preload the
            // first 16 bytes after the row and vector.
            addi t0, a7, -1
            beqz t0, dl_esp32p4_mat_vecs_i8k8o16_final
            dl_esp32p4_mat_vecs_i8k8o16_loop:
                esp.vld.128.ip q2, t3, 16
                esp.vmulas.s8.xacc.ld.ip q3, t4, 16, q0, q1
                esp.vld.128.ip q0, t3, 16
                esp.vmulas.s8.xacc.ld.ip q1, t4, 16, q2, q3
                addi t0, t0, -1
                bnez t0, dl_esp32p4_mat_vecs_i8k8o16_loop

            dl_esp32p4_mat_vecs_i8k8o16_final:
                // q0/q1 = [cols-32, cols-17]. Load and consume [cols-16, cols-1] only.
                esp.vld.128.ip q2, t3, 16
                esp.vmulas.s8.xacc.ld.ip q3, t4, 16, q0, q1
                esp.vmulas.s8.xacc q2, q3
                j dl_esp32p4_mat_vecs_i8k8o16_store

            dl_esp32p4_mat_vecs_i8k8o16_remainder:
                mv t0, a7
            dl_esp32p4_mat_vecs_i8k8o16_remainder_loop:
                esp.vld.128.ip q2, t3, 16
                esp.vmulas.s8.xacc.ld.ip q3, t4, 16, q0, q1
                esp.vld.128.ip q0, t3, 16
                esp.vmulas.s8.xacc.ld.ip q1, t4, 16, q2, q3
                addi t0, t0, -1
                bnez t0, dl_esp32p4_mat_vecs_i8k8o16_remainder_loop

            dl_esp32p4_mat_vecs_i8k8o16_small:
            andi t0, a4, 16
            beqz t0, dl_esp32p4_mat_vecs_i8k8o16_store
            // cols == 16 or cols == 32*k+16: q0/q1 are the final valid block.
            esp.vmulas.s8.xacc q0, q1

            dl_esp32p4_mat_vecs_i8k8o16_store:
            esp.srs.s.xacc t3, a3
            li t4, 32767
            blt t3, t4, 1f
            mv t3, t4
1:
            li t4, -32768
            bgt t3, t4, 2f
            mv t3, t4
2:
            sh t3, 0(t2)
            add t2, t2, t6
            add t1, t1, a4
            addi t5, t5, -1
            bnez t5, dl_esp32p4_mat_vecs_i8k8o16_vec

        add a1, a1, a4
        addi a0, a0, 2
        addi a5, a5, -1
        bnez a5, dl_esp32p4_mat_vecs_i8k8o16_row
    ret
