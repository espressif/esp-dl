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
    srli a5, a4, 4 // a5 = n // 16
    slli t3, a5, 4 // t3 = (a4 // 16) * 16, t3 == a4 if n % 16 == 0, otherwise n % 16 == 8

    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a2, 16

    li t0, 0
    beq a5, t0, loop_end_dl_esp32p4_dotprod_i16k16o32_small
        add t0, a5, 0
        loop_end_dl_esp32p4_dotprod_i16k16o32:

            esp.vld.128.ip q2, a1, 16
            esp.vmulas.s16.xacc.ld.ip q3, a2, 16, q0, q1

            esp.vld.128.ip q0, a1, 16
            esp.vmulas.s16.xacc.ld.ip q1, a2, 16, q2, q3

        addi t0, t0, -1
        bnez t0, loop_end_dl_esp32p4_dotprod_i16k16o32

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
    srli a5, a4, 4 // a5 = n // 16
    slli t3, a5, 4 // t3 = (a4 // 16) * 16, t3 == a4 if n % 16 == 0, otherwise n % 16 == 8

    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a2, 16

    li t0, 0
    beq a5, t0, loop_end_dl_esp32p4_dotprod_i16k16o16_small
        add t0, a5, 0
        loop_end_dl_esp32p4_dotprod_i16k16o16:

            esp.vld.128.ip q2, a1, 16
            esp.vmulas.s16.xacc.ld.ip q3, a2, 16, q0, q1

            esp.vld.128.ip q0, a1, 16
            esp.vmulas.s16.xacc.ld.ip q1, a2, 16, q2, q3

        addi t0, t0, -1
        bnez t0, loop_end_dl_esp32p4_dotprod_i16k16o16
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
    srli a5, a4, 4 // a5 = n // 16
    slli t3, a5, 4 // t3 = (a4 // 16) * 16, t3 == a4 if n % 16 == 0, otherwise n % 16 == 8

    esp.vldext.s8.ip q0, q1, a1, 16 // load 8bit kernel and sign-extend
    esp.vld.128.ip q2, a2, 16

    li t0, 0
    beq a5, t0, loop_end_dl_esp32p4_dotprod_i16k8o16_small
        add t0, a5, 0
        loop_end_dl_esp32p4_dotprod_i16k8o16:

            esp.vld.128.ip q3, a2, 16
            esp.vmulas.s16.xacc.ld.ip q4, a1, 16, q0, q2
            esp.vmulas.s16.xacc.ld.ip q2, a2, 16, q1, q3
            esp.vext.s8 q0, q1, q4

        addi t0, t0, -1
        bnez t0, loop_end_dl_esp32p4_dotprod_i16k8o16

    loop_end_dl_esp32p4_dotprod_i16k8o16_small:
        beq a4, t3, end_dl_esp32p4_dotprod_i16k8o16
        esp.vmulas.s16.xacc q0, q2

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
    srli a5, a4, 5 // a5 = n // 32
    slli t3, a5, 5 // t3 = (a4 // 32) * 32, t3 == a4 if n % 32 == 0, otherwise n % 16 == 8

    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a2, 16

    li t0, 0
    beq a5, t0, loop_end_dl_esp32p4_dotprod_i8k8o16_small
    add t0, a5, 0

    loop_end_dl_esp32p4_dotprod_i8k8o16:

            esp.vld.128.ip q2, a1, 16
            esp.vmulas.s8.xacc.ld.ip q3, a2, 16, q0, q1

            esp.vld.128.ip q0, a1, 16
            esp.vmulas.s8.xacc.ld.ip q1, a2, 16, q2, q3

        addi t0, t0, -1
        bnez t0, loop_end_dl_esp32p4_dotprod_i8k8o16

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
