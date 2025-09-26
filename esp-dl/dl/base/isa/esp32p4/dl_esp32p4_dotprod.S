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
