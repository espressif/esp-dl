/* void dl_tie728_s8_dotprod(int32_t *out2_int8, int8_t *in1_int8, int8_t *in2_int8, int shift, int n); */
    .text
    .align 4
    .global dl_tie728_s8_dotprod
    .type   dl_tie728_s8_dotprod, @function
dl_tie728_s8_dotprod:
    /* a2 = out
     * a3 = vin1
     * a4 = vin2
     * a5 = shift
     * a6 = n
     */
    .align 4
    entry sp, 16
    srli a7, a6, 6
    ee.zero.accx
    ee.vld.128.ip q0, a3, 16
    ee.vld.128.ip q1, a4, 16
    ee.vld.128.ip q2, a3, 16
    ee.vld.128.ip q3, a4, 16
    loopnez a7, loop_end_dl_tie728_s8_dotprod
        ee.vmulas.s8.accx.ld.ip q4, a3, 16, q0, q1
        ee.vld.128.ip q5, a4, 16
        ee.vmulas.s8.accx.ld.ip q6, a3, 16, q2, q3
        ee.vld.128.ip q7, a4, 16
        ee.vmulas.s8.accx.ld.ip q0, a3, 16, q4, q5
        ee.vld.128.ip q1, a4, 16
        ee.vmulas.s8.accx.ld.ip q2, a3, 16, q6, q7
        ee.vld.128.ip q3, a4, 16
    loop_end_dl_tie728_s8_dotprod:
    slli a7, a7, 6
    sub a7, a6, a7 // n % 16 = 0, 16, 32, or 48
    movi a8, 16
    blt a7, a8, dl_tie728_s8_dotprod_modres_end // jump if n % 16 < 16, i.e. n % 16 == 0
        ee.vmulas.s8.accx.ld.ip q4, a3, 16, q0, q1
        ee.vld.128.ip q5, a4, 16
    movi a8, 32
    blt a7, a8, dl_tie728_s8_dotprod_modres_end // jump if n % 16 < 32, i.e. n % 16 == 16
        ee.vmulas.s8.accx.ld.ip q6, a3, 16, q2, q3
        ee.vld.128.ip q7, a4, 16
    movi a8, 48
    blt a7, a8, dl_tie728_s8_dotprod_modres_end // jump if n % 16 < 48, i.e. n % 16 == 32
        ee.vmulas.s8.accx.ld.ip q0, a3, 16, q4, q5
        ee.vld.128.ip q1, a4, 16
    dl_tie728_s8_dotprod_modres_end:
    ee.srs.accx a9, a5, 0
    s32i a9, a2, 0
    retw


/* void dotproduct_int16(int32_t *out2_int16, int16_t *in1_int16, int16_t *in2_int16, int shift, int n); */
    .text
    .global dotproduct_int16
    .type   dotproduct_int16, @function
    #.section .iram1
    .align 4
dotproduct_int16:
    /* a2 = int32_t *z
     * a3 = int16_t *x
     * a4 = int16_t *y
     * a5 = int shift
     * a6 = int n
     */
    prologue

    esp.zero.xacc
    srli a5, a4, 4 // a5 = n // 16
    slli t3, a5, 4 // t3 = (a4 // 16) * 16, t3 == a4 if n % 16 == 0, otherwise n % 16 == 8

    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a2, 16

    add t0, a5, 0
    loop_end_dotproduct_int16:

        esp.vld.128.ip q2, a1, 16
        esp.vmulas.s16.xacc.ld.ip q3, a2, 16, q0, q1

        esp.vld.128.ip q0, a1, 16
        esp.vmulas.s16.xacc.ld.ip q1, a2, 16, q2, q3

    addi t0, t0, -1
    bnez t0, loop_end_dotproduct_int16

    beq a4, t3, end_dotproduct_int16
    esp.vmulas.s16.xacc q0, q1

    end_dotproduct_int16:
    esp.srs.s.xacc t3, a3
    sw t3, 0(a0)
    epilogue
    ret
