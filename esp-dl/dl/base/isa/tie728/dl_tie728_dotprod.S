/* void dl_tie728_dotprod_i8k8o16(int16_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, int shift, int n, int64_t rounding_offset) */
    .text
    .align 4
    .global dl_tie728_dotprod_i8k8o16
    .type   dl_tie728_dotprod_i8k8o16, @function
dl_tie728_dotprod_i8k8o16:
    /* a2 = out
     * a3 = vin1
     * a4 = vin2
     * a5 = shift
     * a6 = n
     * a7 = rounding_offset
     */

    entry sp, 16

    ee.ld.accx.ip a7, 0
    srli a7, a6, 5 // a7 = n // 32
    slli a8, a7, 5 // a8 = (a6 // 32) * 32, a8 == a6 if n % 32 == 0, otherwise n % 32 == 16

    ee.vld.128.ip q0, a3, 16
    ee.vld.128.ip q1, a4, 16

    loopnez a7, loop_end_dl_tie728_dotprod_i8k8o16 // compute 32 items per loop

        ee.vld.128.ip q2, a3, 16
        ee.vmulas.s8.accx.ld.ip q3, a4, 16, q0, q1

        ee.vld.128.ip q0, a3, 16
        ee.vmulas.s8.accx.ld.ip q1, a4, 16, q2, q3

    loop_end_dl_tie728_dotprod_i8k8o16:

    beq a6, a8, end_dl_tie728_dotprod_i8k8o16 // if a6 == a8, then n % 32 == 0
    ee.vmulas.s8.accx q0, q1

    end_dl_tie728_dotprod_i8k8o16:
    ee.srs.accx a9, a5, 0

    // max(min(t3, Q16_MAX), Q16_MIN)
    movi    a7, 32767          // Q16_MAX
    movi    a8, -32768         // Q16_MIN
    sub     a6, a9, a7         // a6 = a9 - 32767
    movgez  a9, a7, a6         // if a6>=0, a9 = 32767
    sub     a6, a9, a8         // a6 = a9 - (-32768)
    movltz  a9, a8, a6         // if a6<0, a9 = -32768

    s16i a9, a2, 0
    retw




/* void dl_tie728_dotprod_i16k16o16(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int shift, int n); */
    .text
    .global dl_tie728_dotprod_i16k16o16
    .type   dl_tie728_dotprod_i16k16o16, @function
    .section .iram1
    .align 4
dl_tie728_dotprod_i16k16o16:
    /* a2 = int32_t *z
     * a3 = int16_t *x
     * a4 = int16_t *y
     * a5 = int shift
     * a6 = int n
     */
    entry sp, 16

    //ee.zero.accx
    ee.ld.accx.ip a7, 0
    srli a7, a6, 4 // a7 = n // 16
    slli a8, a7, 4 // a8 = (a6 // 16) * 16, a8 == a6 if n % 16 == 0, otherwise n % 16 == 8

    ee.vld.128.ip q0, a3, 16
    ee.vld.128.ip q1, a4, 16

    loopnez a7, loop_end_dl_tie728_dotprod_i16k16o16 // compute 16 items per loop

        ee.vld.128.ip q2, a3, 16
        ee.vmulas.s16.accx.ld.ip q3, a4, 16, q0, q1

        ee.vld.128.ip q0, a3, 16
        ee.vmulas.s16.accx.ld.ip q1, a4, 16, q2, q3

    loop_end_dl_tie728_dotprod_i16k16o16:

    beq a6, a8, end_dl_tie728_dotprod_i16k16o16 // if a6 == a8, then n % 32 == 0
    ee.vmulas.s16.accx q0, q1

    end_dl_tie728_dotprod_i16k16o16:
    ee.srs.accx a9, a5, 0

    // max(min(t3, Q16_MAX), Q16_MIN)
    movi    a7, 32767          // Q16_MAX
    movi    a8, -32768         // Q16_MIN
    sub     a6, a9, a7         // a6 = a9 - 32767
    movgez  a9, a7, a6         // if a6>=0, a9 = 32767
    sub     a6, a9, a8         // a6 = a9 - (-32768)
    movltz  a9, a8, a6         // if a6<0, a9 = -32768

    s16i a9, a2, 0
    retw



/** void dl_tie728_vsmul_s16f32(int16_t *in_int16, float* scale, int n);
    for (int i=0; i<n; i++)
    {
        in_int16[i] = round(in_int16[i]*scale[0]);
    }
**/

    .text
    .align 4
    .global dl_tie728_vsmul_s16f32
    .type   dl_tie728_vsmul_s16f32, @function
    // .section .iram1
dl_tie728_vsmul_s16f32:
    /* a2 = in
     * a3 = scale
     * a4 = n
     */
    entry sp, 16

    lsip f2, a3, 0
    srai a5, a4, 2 // in_c >> 2 = in_c / 4
    loopnez a5, dl_tie728_vsmul_s16f32_loop
    // load in
    l16si a12, a2, 0
    l16si a13, a2, 2
    l16si a14, a2, 4
    l16si a15, a2, 6
    // in int16->float
    float.s f3, a12, 0
    float.s f4, a13, 0
    float.s f5, a14, 0
    float.s f6, a15, 0
    // mul by scale
    mul.s f3, f3, f2
    mul.s f4, f4, f2
    mul.s f5, f5, f2
    mul.s f6, f6, f2
    // round output
    round.s a12, f3, 0
    round.s a13, f4, 0
    round.s a14, f5, 0
    round.s a15, f6, 0
    // store output
    s16i a12, a2, 0
    s16i a13, a2, 2
    s16i a14, a2, 4
    s16i a15, a2, 6
    addi a2, a2, 8
    dl_tie728_vsmul_s16f32_loop:
    retw



/** void dl_tie728_vsmul_s8f32(int8_t *in_int8, float* scale, int n);
    for (int i = 0; i < n; i++)
    {
        in_int8[i] = round(in_int8[i] * scale[0]);
    }
**/

    .text
    .align 4
    .global dl_tie728_vsmul_s8f32
    .type   dl_tie728_vsmul_s8f32, @function
dl_tie728_vsmul_s8f32:
    /* a2 = in
     * a3 = scale
     * a4 = n
     */
    entry sp, 16

    lsip f2, a3, 0
    srai a5, a4, 2 // in_c >> 2 = in_c / 4
    loopnez a5, dl_tie728_vsmul_s8f32_loop
        // load in, note l8 is unsigned only
        l8ui a12, a2, 0
        l8ui a13, a2, 1
        l8ui a14, a2, 2
        l8ui a15, a2, 3

        // add signed
        movi.n a10, 128
        movi.n a11, -256
        and a10, a12, a10 // in_int8[0] & 128 == 128 if negative, otherwise 0
        beqz a10, dl_tie728_vsmul_s8f32_check_in_sign_0
            or a12, a12, a11 // supplement 1s for in_int8[0]
        dl_tie728_vsmul_s8f32_check_in_sign_0:

        and a10, a13, a10 // in_int8[1] & 128 == 128 if negative, otherwise 0
        beqz a10, dl_tie728_vsmul_s8f32_check_in_sign_1
            or a13, a13, a11 // supplement 1s for in_int8[1]
        dl_tie728_vsmul_s8f32_check_in_sign_1:

        and a10, a14, a10 // in_int8[2] & 128 == 128 if negative, otherwise 0
        beqz a10, dl_tie728_vsmul_s8f32_check_in_sign_2
            or a14, a14, a11 // supplement 1s for in_int8[2]
        dl_tie728_vsmul_s8f32_check_in_sign_2:

        and a10, a15, a10 // in_int8[3] & 128 == 128 if negative, otherwise 0
        beqz a10, dl_tie728_vsmul_s8f32_check_in_sign_3
            or a15, a15, a11 // supplement 1s for in_int8[3]
        dl_tie728_vsmul_s8f32_check_in_sign_3:

        // in int8->float
        float.s f3, a12, 0
        float.s f4, a13, 0
        float.s f5, a14, 0
        float.s f6, a15, 0

        // mul by scale
        mul.s f3, f3, f2
        mul.s f4, f4, f2
        mul.s f5, f5, f2
        mul.s f6, f6, f2

        // round output
        round.s a12, f3, 0
        round.s a13, f4, 0
        round.s a14, f5, 0
        round.s a15, f6, 0

        // a > 127 ? 127 : (a < -128) ? -128 : a
        movi.n a10, 127
        movi.n a11, -128
        min a12, a10, a12
        max a12, a11, a12
        min a13, a10, a13
        max a13, a11, a13
        min a14, a10, a14
        max a14, a11, a14
        min a15, a10, a15
        max a15, a11, a15

        // store output
        s8i a12, a2, 0
        s8i a13, a2, 1
        s8i a14, a2, 2
        s8i a15, a2, 3

        addi a2, a2, 4
    dl_tie728_vsmul_s8f32_loop:

    retw
