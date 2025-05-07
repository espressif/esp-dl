/*
 * test_fft_ins.S
 *
 *  Created on: 2020��12��29��
 *      Author: AI-Image
 */


/* void test_radix2_bit_reverse(int16_t *, int16_t, int16_t);*/
    .text
    .align 4
    .global test_radix2_bit_reverse
    .type   test_radix2_bit_reverse, @function
test_radix2_bit_reverse:

    /* a2 = data
     * a3 = cpx_points
     * a4 = log2N
     */


     .align 4
    entry sp, 16


    srli a3, a3, 3
    wur.fft_bit_width a4
    movi a14, 0

    mov a7, a2          // index for store
    mov a6, a2          // index for load

    loopnez a3, loop_end_radix2_bit_reverse   // each loop switch 8 times(16 cpx)
        ee.vld.128.ip q0, a7, 16
        ee.bitrev q2, a14
        ee.ldxq.32 q1, q2, a2, 0, 0
        ee.ldxq.32 q1, q2, a2, 1, 1
        ee.ldxq.32 q1, q2, a2, 2, 2
        ee.ldxq.32 q1, q2, a2, 3, 3
        ee.stxq.32 q0, q2, a2, 0, 0
        ee.stxq.32 q0, q2, a2, 1, 1
        ee.stxq.32 q0, q2, a2, 2, 2
        ee.stxq.32 q0, q2, a2, 3, 3
        ee.vst.128.ip q1, a6, 16

        ee.vld.128.ip q0, a7, 16
        ee.ldxq.32 q1, q2, a2, 0, 4
        ee.ldxq.32 q1, q2, a2, 1, 5
        ee.ldxq.32 q1, q2, a2, 2, 6
        ee.ldxq.32 q1, q2, a2, 3, 7
        ee.stxq.32 q0, q2, a2, 0, 4
        ee.stxq.32 q0, q2, a2, 1, 5
        ee.stxq.32 q0, q2, a2, 2, 6
        ee.stxq.32 q0, q2, a2, 3, 7
        ee.vst.128.ip q1, a6, 16

    loop_end_radix2_bit_reverse:

    retw


/* void test_radix2_fft_bf_s16(int16_t *, int16_t *, int16_t, int16_t, int16_t); */
    .text
    .align 4
    .global test_radix2_fft_bf_s16
    .type   test_radix2_fft_bf_s16, @function
test_radix2_fft_bf_s16:

    /* a2 = data
     * a3 = win
     * a4 = shift
     * a5 = number of stage
     * a6 = cpx_points
     */

    /* a2 = ptr = data
     * a7 = ptr1
     * a8 = ptr2
     * a9 = ptr3

     */

    .align 4
    entry sp, 16

	//addi a4, a4, 1
    ssr a4


    //addx2 a7, a6, a2    //*ptr1 = data + n        load
    //mov a8, a2          //*ptr2 = ptr       store
    //mov a9, a7          //*ptr3 = ptr1      store


    mov a13, a3
    addi a5, a5, -3 // number of stage - 3
    movi a12, 1
    mov a14, a12		// number of box in current stage
    srli a15, a6, 4 // iteration times: number of bf in each box/8(each iter caculate 16 cpx_points = 8 bf)

    movi a11, 16     // offset of bf


    start_fft_radix2_stage:  // stage loop
        mov a4, a2
        mov a3, a13         // start of current stage for twiddle factor

        start_fft_radix2_stage_box:

            mov a9, a4      // load
            mov a10, a9		// store

            addx2 a7, a6, a4    //*ptr1 = data + n        load
		    mov a8, a7          //  store

            mov a13, a3     // start for twiddle factor

            ee.vld.128.ip q0, a9, 16
		    ee.vld.128.ip q1, a7, 16
		    ee.vld.128.ip q2, a13, 16

            loopnez a15, loop_end_fft_radix2_butterfly
                ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1                  // [a1 a2 a3 a4]
		        ee.fft.cmul.s16.ld.xp q4, a13, a11, q5, q3, q2, 0            // [b1]
		        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 2            // [b2]
		        ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q2, 4            // [b3]
		        ee.fft.cmul.s16.st.xp q3, q2, q5, a8, a11, 6, 0, 1             // [b1 b2 b3 b4]

		        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
		        ee.fft.cmul.s16.ld.xp q2, a13, a11, q5, q3, q4, 0
		        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q4, 2
		        ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q4, 4
		        ee.fft.cmul.s16.st.xp q3, q4, q5, a8, a11, 6, 0, 1
            loop_end_fft_radix2_butterfly:

			addx4 a4, a6, a4  // start index of box
			addi a14, a14, -1
			sub a13, a13, a11

        bnez a14, start_fft_radix2_stage_box

        srli a15, a15, 1                // iteration for bf /2
        srli a6, a6, 1					// number of cpx_points in box
        slli a12, a12, 1				// number of box
        mov a14, a12
        addi a5, a5, -1

    bnez a5, start_fft_radix2_stage     //loop for stage


    mov a9, a2      // load
    mov a10, a9		// store
    //srli a15, a12, 1

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.128.ip q2, a13, 16 //fixed twiddle factor
    movi a14, 0

	loopnez a12, loop_end_radix2_last_third_stage
		ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1                  	// [a1 a2 a3 a4]
        ee.fft.cmul.s16.ld.xp q4, a13, a14, q5, q3, q2, 0            	// [b1] q4- fake load
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 2            	// [b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q3, q2, 4            	// [b3]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 0, 1             	// [b1 b2 b3 b4]
    loop_end_radix2_last_third_stage:

	mov a9, a2      // load
    mov a10, a9		// store
    //slli a15, a15, 1
    mov a14, a2 //fake load
    mov a3, a13

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    //ee.vld.128.incp q2, a13  //fixed twiddle factor
    ee.vld.h.64.ip q2, a13, 0

	loopnez a12, loop_end_radix2_last_second_stage
        ee.fft.r2bf.s16 q3, q4, q0, q1, 0          // [a1 a2 a3 a4]   q5-fake load
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 4           		// [b1]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 1, 1         		// [a1 a2 b1 b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 4            	// [b3]
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 6, 1, 1             	// [a3 a4 b3 b4]
    loop_end_radix2_last_second_stage:


	mov a9, a2      // load
    mov a10, a9		// store

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    //ee.vld.128.incp q2, a13 //fixed twiddle factor
    ee.vldbc.32 q2, a3

	loopnez a12, loop_end_radix2_last_stage
        ee.fft.r2bf.s16 q3, q4, q0, q1, 1          // [a1 a2 a3 a4]
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 4           		// [b1]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 2, 1         		// [a1 a2 b1 b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 4            	// [b3]
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 6, 2, 1             	// [a3 a4 b3 b4]
    loop_end_radix2_last_stage:

    retw


/* void test_radix2_fft_bf_s16_hp(int16_t *, int16_t *, int16_t, int16_t, int16_t); */
    .text
    .align 4
    .global test_radix2_fft_bf_s16_hp
    .type   test_radix2_fft_bf_s16_hp, @function
test_radix2_fft_bf_s16_hp:

    /* a2 = data
     * a3 = win
     * a4 = shift
     * a5 = number of stage
     * a6 = cpx_points
     */

    /* a2 = ptr = data
     * a7 = ptr1
     * a8 = ptr2
     * a9 = ptr3

     */

    .align 4
    entry sp, 16

    //addi a4, a4, 1
    ssr a4


    //addx2 a7, a6, a2    //*ptr1 = data + n        load
    //mov a8, a2          //*ptr2 = ptr       store
    //mov a9, a7          //*ptr3 = ptr1      store


    mov a13, a3
    addi a5, a5, -3 // number of stage - 3
    movi a12, 1
    mov a14, a12        // number of box in current stage
    srli a15, a6, 4 // iteration times: number of bf in each box/8(each iter caculate 16 cpx_points = 8 bf)

    movi a11, 16     // offset of bf

    movi.n a3, 131074 // {int16 2 | int16 2}
    ee.movi.32.q q6, a3, 0
    ee.movi.32.q q6, a3, 1
    ee.movi.32.q q6, a3, 2
    ee.movi.32.q q6, a3, 3

    ee.zero.q q7

    start_fft_radix2_stage_hp:  // stage loop
        mov a4, a2
        mov a3, a13         // start of current stage for twiddle factor

        start_fft_radix2_stage_box_hp:

            mov a9, a4      // load
            mov a10, a9     // store

            addx2 a7, a6, a4    //*ptr1 = data + n        load
            mov a8, a7          //  store

            mov a13, a3     // start for twiddle factor

            ee.vld.128.ip q0, a9, 16
            ee.vld.128.ip q1, a7, 16
            ee.vld.128.ip q2, a13, 16

            loopnez a15, loop_end_fft_radix2_butterfly_hp
                ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1                  // [a1 a2 a3 a4]
                ee.fft.cmul.s16.ld.xp q4, a13, a11, q5, q3, q2, 0            // [b1]
                ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 2            // [b2]
                ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q2, 4            // [b3]
                ee.fft.cmul.s16.st.xp q3, q2, q5, a8, a11, 6, 0, 1             // [b1 b2 b3 b4]

                ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
                ee.fft.cmul.s16.ld.xp q2, a13, a11, q5, q3, q4, 0
                ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q4, 2
                ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q4, 4
                ee.fft.cmul.s16.st.xp q3, q4, q5, a8, a11, 6, 0, 1
            loop_end_fft_radix2_butterfly_hp:

            addx4 a4, a6, a4  // start index of box
            addi a14, a14, -1
            sub a13, a13, a11

        bnez a14, start_fft_radix2_stage_box_hp

        srli a15, a15, 1                // iteration for bf /2
        srli a6, a6, 1                  // number of cpx_points in box
        slli a12, a12, 1                // number of box
        mov a14, a12
        addi a5, a5, -1

        mul16u a3, a6, a12 // number of complex points
        ee.zero.q q0 // max
        ee.zero.q q1 // min
        mov.n a4, a2 // load pointer
        srai a3, a3, 3 // << 1 >> 4
        ee.vld.128.ip q2, a4, 16
        loopnez a3, test_radix2_fft_bf_s16_hp_loop_end_0
            ee.vmax.s16.ld.incp q3, a4, q0, q0, q2
            ee.vmin.s16 q1, q1, q2
            ee.vmax.s16.ld.incp q2, a4, q0, q0, q3
            ee.vmin.s16 q1, q1, q3
        test_radix2_fft_bf_s16_hp_loop_end_0:

        movi.n a4, 0 // abs(max)
        ee.movi.32.a q0, a3, 0
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q0, a3, 1
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q0, a3, 2
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q0, a3, 3
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q1, a3, 0
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        neg a7, a7
        neg a3, a3
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q1, a3, 1
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        neg a7, a7
        neg a3, a3
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q1, a3, 2
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        neg a7, a7
        neg a3, a3
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q1, a3, 3
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        neg a7, a7
        neg a3, a3
        max a4, a4, a3
        max a4, a4, a7

        movi a3, 4096
        bge a4, a3, test_radix2_fft_bf_s16_hp_skip_left_shift
            mul16u a3, a6, a12
            movi.n a4, 0
            ssr a4
            srai a3, a3, 4
            mov.n a4, a2 // load pointer
            mov.n a7, a4 // store pointer
            ee.vld.128.ip q0, a4, 16
            ee.vld.128.ip q1, a4, 16
            loopnez a3, loop_end
                ee.vmul.s16.ld.incp q2, a4, q0, q0, q6
                ee.vmul.s16.ld.incp q3, a4, q1, q1, q6
                ee.vst.128.ip q0, a7, 16
                ee.vst.128.ip q1, a7, 16
                ee.vmul.s16.ld.incp q0, a4, q2, q2, q6
                ee.vmul.s16.ld.incp q1, a4, q3, q3, q6
                ee.vst.128.ip q2, a7, 16
                ee.vst.128.ip q3, a7, 16
            loop_end:
            movi.n a4, 15
            ssr a4
            j test_radix2_fft_bf_s16_hp_skip_left_shift_end
        test_radix2_fft_bf_s16_hp_skip_left_shift:
            // if no need to left shift, add q7[0] by 1 to count how many times result is shifted
            ee.movi.32.a q7, a3, 0
            addi.n a3, a3, 1
            ee.movi.32.q q7, a3, 0
        test_radix2_fft_bf_s16_hp_skip_left_shift_end:

    bnez a5, start_fft_radix2_stage_hp     //loop for stage


    mov a9, a2      // load
    mov a10, a9     // store
    //srli a15, a12, 1

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.128.ip q2, a13, 16 //fixed twiddle factor
    movi a14, 0

    loopnez a12, loop_end_radix2_last_third_stage_hp
        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1                      // [a1 a2 a3 a4]
        ee.fft.cmul.s16.ld.xp q4, a13, a14, q5, q3, q2, 0               // [b1] q4- fake load
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 2                // [b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q3, q2, 4                // [b3]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 0, 1                 // [b1 b2 b3 b4]
    loop_end_radix2_last_third_stage_hp:

    mul16u a3, a6, a12 // number of complex points
    ee.zero.q q0 // max
    ee.zero.q q1 // min
    mov.n a4, a2 // load pointer
    srai a3, a3, 3 // << 1 >> 4
    ee.vld.128.ip q2, a4, 16
    loopnez a3, test_radix2_fft_bf_s16_hp_loop_end_1
        ee.vmax.s16.ld.incp q3, a4, q0, q0, q2
        ee.vmin.s16 q1, q1, q2
        ee.vmax.s16.ld.incp q2, a4, q0, q0, q3
        ee.vmin.s16 q1, q1, q3
    test_radix2_fft_bf_s16_hp_loop_end_1:

    movi.n a4, 0 // abs(max)
    ee.movi.32.a q0, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7

    movi a3, 4096
    bge a4, a3, test_radix2_fft_bf_s16_hp_skip_left_shift_1
        mul16u a3, a6, a12
        movi.n a4, 0
        ssr a4
        srai a3, a3, 4
        mov.n a4, a2 // load pointer
        mov.n a7, a4 // store pointer
        ee.vld.128.ip q0, a4, 16
        ee.vld.128.ip q1, a4, 16
        loopnez a3, loop_end_1
            ee.vmul.s16.ld.incp q2, a4, q0, q0, q6
            ee.vmul.s16.ld.incp q3, a4, q1, q1, q6
            ee.vst.128.ip q0, a7, 16
            ee.vst.128.ip q1, a7, 16
            ee.vmul.s16.ld.incp q0, a4, q2, q2, q6
            ee.vmul.s16.ld.incp q1, a4, q3, q3, q6
            ee.vst.128.ip q2, a7, 16
            ee.vst.128.ip q3, a7, 16
        loop_end_1:
        movi.n a4, 15
        ssr a4
        j test_radix2_fft_bf_s16_hp_skip_left_shift_1_end
    test_radix2_fft_bf_s16_hp_skip_left_shift_1:
        // if no need to left shift, add q7[0] by 1 to count how many times result is shifted
        ee.movi.32.a q7, a3, 0
        addi.n a3, a3, 1
        ee.movi.32.q q7, a3, 0
    test_radix2_fft_bf_s16_hp_skip_left_shift_1_end:

    mov a9, a2      // load
    mov a10, a9     // store
    //slli a15, a15, 1
    mov a14, a2 //fake load
    mov a3, a13

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    //ee.vld.128.incp q2, a13  //fixed twiddle factor
    ee.vld.h.64.ip q2, a13, 0

    loopnez a12, loop_end_radix2_last_second_stage_hp
        ee.fft.r2bf.s16 q3, q4, q0, q1, 0          // [a1 a2 a3 a4]   q5-fake load
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 4                // [b1]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 1, 1                 // [a1 a2 b1 b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 4                // [b3]
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 6, 1, 1                 // [a3 a4 b3 b4]
    loop_end_radix2_last_second_stage_hp:

    mul16u a3, a6, a12 // number of complex points
    ee.zero.q q0 // max
    ee.zero.q q1 // min
    mov.n a4, a2 // load pointer
    srai a3, a3, 3 // << 1 >> 4
    ee.vld.128.ip q2, a4, 16
    loopnez a3, test_radix2_fft_bf_s16_hp_loop_end_2
        ee.vmax.s16.ld.incp q3, a4, q0, q0, q2
        ee.vmin.s16 q1, q1, q2
        ee.vmax.s16.ld.incp q2, a4, q0, q0, q3
        ee.vmin.s16 q1, q1, q3
    test_radix2_fft_bf_s16_hp_loop_end_2:

    movi.n a4, 0 // abs(max)
    ee.movi.32.a q0, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7

    movi a3, 4096
    bge a4, a3, test_radix2_fft_bf_s16_hp_skip_left_shift_2
        mul16u a3, a6, a12
        movi.n a4, 0
        ssr a4
        srai a3, a3, 4
        mov.n a4, a2 // load pointer
        mov.n a7, a4 // store pointer
        ee.vld.128.ip q0, a4, 16
        ee.vld.128.ip q1, a4, 16
        loopnez a3, loop_end_2
            ee.vmul.s16.ld.incp q2, a4, q0, q0, q6
            ee.vmul.s16.ld.incp q3, a4, q1, q1, q6
            ee.vst.128.ip q0, a7, 16
            ee.vst.128.ip q1, a7, 16
            ee.vmul.s16.ld.incp q0, a4, q2, q2, q6
            ee.vmul.s16.ld.incp q1, a4, q3, q3, q6
            ee.vst.128.ip q2, a7, 16
            ee.vst.128.ip q3, a7, 16
        loop_end_2:
        movi.n a4, 15
        ssr a4
        j test_radix2_fft_bf_s16_hp_skip_left_shift_2_end
    test_radix2_fft_bf_s16_hp_skip_left_shift_2:
        // if no need to left shift, add q7[0] by 1 to count how many times result is shifted
        ee.movi.32.a q7, a3, 0
        addi.n a3, a3, 1
        ee.movi.32.q q7, a3, 0
    test_radix2_fft_bf_s16_hp_skip_left_shift_2_end:

    mov a9, a2      // load
    mov a10, a9     // store

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    //ee.vld.128.incp q2, a13 //fixed twiddle factor
    mov a3, a13
    ee.vldbc.32 q2, a3

    loopnez a12, loop_end_radix2_last_stage_hp
        ee.fft.r2bf.s16 q3, q4, q0, q1, 1          // [a1 a2 a3 a4]
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 4                // [b1]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 6, 2, 1                 // [a1 a2 b1 b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 4                // [b3]
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 6, 2, 1                 // [a3 a4 b3 b4]
    loop_end_radix2_last_stage_hp:

    mul16u a3, a6, a12 // number of complex points
    ee.zero.q q0 // max
    ee.zero.q q1 // min
    mov.n a4, a2 // load pointer
    srai a3, a3, 3 // << 1 >> 4
    ee.vld.128.ip q2, a4, 16
    loopnez a3, test_radix2_fft_bf_s16_hp_loop_end_3
        ee.vmax.s16.ld.incp q3, a4, q0, q0, q2
        ee.vmin.s16 q1, q1, q2
        ee.vmax.s16.ld.incp q2, a4, q0, q0, q3
        ee.vmin.s16 q1, q1, q3
    test_radix2_fft_bf_s16_hp_loop_end_3:

    movi.n a4, 0 // abs(max)
    ee.movi.32.a q0, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7

    movi a3, 4096
    bge a4, a3, test_radix2_fft_bf_s16_hp_skip_left_shift_3
        mul16u a3, a6, a12
        movi.n a4, 0
        ssr a4
        srai a3, a3, 4
        mov.n a4, a2 // load pointer
        mov.n a7, a4 // store pointer
        ee.vld.128.ip q0, a4, 16
        ee.vld.128.ip q1, a4, 16
        loopnez a3, loop_end_3
            ee.vmul.s16.ld.incp q2, a4, q0, q0, q6
            ee.vmul.s16.ld.incp q3, a4, q1, q1, q6
            ee.vst.128.ip q0, a7, 16
            ee.vst.128.ip q1, a7, 16
            ee.vmul.s16.ld.incp q0, a4, q2, q2, q6
            ee.vmul.s16.ld.incp q1, a4, q3, q3, q6
            ee.vst.128.ip q2, a7, 16
            ee.vst.128.ip q3, a7, 16
        loop_end_3:
        movi.n a4, 15
        ssr a4
        j test_radix2_fft_bf_s16_hp_skip_left_shift_3_end
    test_radix2_fft_bf_s16_hp_skip_left_shift_3:
        // if no need to left shift, add q7[0] by 1 to count how many times result is shifted
        ee.movi.32.a q7, a3, 0
        addi.n a3, a3, 1
        ee.movi.32.q q7, a3, 0
    test_radix2_fft_bf_s16_hp_skip_left_shift_3_end:

    ee.movi.32.a q7, a2, 0

    retw


/* void test_radix2_ifft_bf_s16(int16_t *, int16_t *, int16_t, int16_t, int16_t); */
    .text
    .align 4
    .global test_radix2_ifft_bf_s16
    .type   test_radix2_ifft_bf_s16, @function
test_radix2_ifft_bf_s16:

    /* a2 = data
     * a3 = win
     * a4 = shift
     * a5 = number of stage
     * a6 = cpx_points
     */

    /* a2 = ptr = data
     * a7 = ptr1
     * a8 = ptr2
     * a9 = ptr3
     */

    .align 4
    entry sp, 16

	//addi a4, a4, 1
    ssr a4


    //addx2 a7, a6, a2    //*ptr1 = data + n        load
    //mov a8, a2          //*ptr2 = ptr       store
    //mov a9, a7          //*ptr3 = ptr1      store


    mov a13, a3
    addi a5, a5, -3 // number of stage - 3
    movi a12, 1
    mov a14, a12		// number of box in current stage
    srli a15, a6, 4 // iteration times: number of bf in each box/8(each iter caculate 16 cpx_points = 8 bf)

    movi a11, 16     // offset of bf

    start_ifft_radix2_stage:  // stage loop
        mov a4, a2
        mov a3, a13         // start of current stage for twiddle factor

        start_ifft_radix2_stage_box:
            mov a9, a4      // load
            mov a10, a9		// store

            addx2 a7, a6, a4    //*ptr1 = data + n        load
		    mov a8, a7          //  store

            mov a13, a3     // start for twiddle factor

            ee.vld.128.ip q0, a9, 16
		    ee.vld.128.ip q1, a7, 16
		    ee.vld.128.ip q2, a13, 16

            loopnez a15, loop_end_ifft_radix2_butterfly
		        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1                  // [a1 a2 a3 a4]
		        ee.fft.cmul.s16.ld.xp q4, a13, a11, q5, q3, q2, 1            // [b1]
		        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 3            // [b2]
		        ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q2, 5            // [b3]
		        ee.fft.cmul.s16.st.xp q3, q2, q5, a8, a11, 7, 0, 1             // [b1 b2 b3 b4]

		        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
		        ee.fft.cmul.s16.ld.xp q2, a13, a11, q5, q3, q4, 1
		        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q4, 3
		        ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q4, 5
		        ee.fft.cmul.s16.st.xp q3, q4, q5, a8, a11, 7, 0, 1

            loop_end_ifft_radix2_butterfly:

			addx4 a4, a6, a4  // start index of box
			addi a14, a14, -1
			sub a13, a13, a11

        bnez a14, start_ifft_radix2_stage_box

        srli a15, a15, 1                // iteration for bf /2
        srli a6, a6, 1					// number of cpx_points in box
        slli a12, a12, 1				// number of box
        mov a14, a12
        addi a5, a5, -1

    bnez a5, start_ifft_radix2_stage     //loop for stage


    mov a9, a2      // load
    mov a10, a9		// store
    //srli a15, a12, 1

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.128.ip q2, a13, 16 //fixed twiddle factor
    movi a14, 0

	loopnez a12, loop_end_radix2_ifft_last_third_stage
		ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1                  	// [a1 a2 a3 a4]
        ee.fft.cmul.s16.ld.xp q4, a13, a14, q5, q3, q2, 1            	// [b1] q4- fake load
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 3            	// [b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q3, q2, 5            	// [b3]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 0, 1             	// [b1 b2 b3 b4]
    loop_end_radix2_ifft_last_third_stage:

	mov a9, a2      // load
    mov a10, a9		// store
    //slli a15, a15, 1
    mov a14, a2 //fake load
    mov a3, a13

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    //ee.vld.128.incp q2, a13  //fixed twiddle factor
    ee.vld.h.64.ip q2, a13, 0

	loopnez a12, loop_end_radix2_ifft_last_second_stage
        ee.fft.r2bf.s16 q3, q4, q0, q1, 0          // [a1 a2 a3 a4]   q5-fake load
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 5           		// [b1]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 1, 1         		// [a1 a2 b1 b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 5            	// [b3]
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 7, 1, 1             	// [a3 a4 b3 b4]
    loop_end_radix2_ifft_last_second_stage:


	mov a9, a2      // load
    mov a10, a9		// store

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    //ee.vld.128.incp q2, a13 //fixed twiddle factor
    ee.vldbc.32 q2, a3

	loopnez a12, loop_end_radix2_ifft_last_stage
        ee.fft.r2bf.s16 q3, q4, q0, q1, 1          // [a1 a2 a3 a4]   q5-fake load
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 5           		// [b1]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 2, 1         		// [a1 a2 b1 b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 5            	// [b3]
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 7, 2, 1             	// [a3 a4 b3 b4]
    loop_end_radix2_ifft_last_stage:

    retw



/* void test_radix2_ifft_bf_s16_hp(int16_t *, int16_t *, int16_t, int16_t, int16_t); */
    .text
    .align 4
    .global test_radix2_ifft_bf_s16_hp
    .type   test_radix2_ifft_bf_s16_hp, @function
test_radix2_ifft_bf_s16_hp:

    /* a2 = data
     * a3 = win
     * a4 = shift
     * a5 = number of stage
     * a6 = cpx_points
     */

    /* a2 = ptr = data
     * a7 = ptr1
     * a8 = ptr2
     * a9 = ptr3
     */

    .align 4
    entry sp, 16

    //addi a4, a4, 1
    ssr a4


    //addx2 a7, a6, a2    //*ptr1 = data + n        load
    //mov a8, a2          //*ptr2 = ptr       store
    //mov a9, a7          //*ptr3 = ptr1      store


    mov a13, a3
    addi a5, a5, -3 // number of stage - 3
    movi a12, 1
    mov a14, a12        // number of box in current stage
    srli a15, a6, 4 // iteration times: number of bf in each box/8(each iter caculate 16 cpx_points = 8 bf)

    movi a11, 16     // offset of bf

    movi.n a3, 131074 // {int16 2 | int16 2}
    ee.movi.32.q q6, a3, 0
    ee.movi.32.q q6, a3, 1
    ee.movi.32.q q6, a3, 2
    ee.movi.32.q q6, a3, 3

    ee.zero.q q7

    start_ifft_radix2_stage_hp:  // stage loop
        mov a4, a2
        mov a3, a13         // start of current stage for twiddle factor

        start_ifft_radix2_stage_box_hp:
            mov a9, a4      // load
            mov a10, a9     // store

            addx2 a7, a6, a4    //*ptr1 = data + n        load
            mov a8, a7          //  store

            mov a13, a3     // start for twiddle factor

            ee.vld.128.ip q0, a9, 16
            ee.vld.128.ip q1, a7, 16
            ee.vld.128.ip q2, a13, 16

            loopnez a15, loop_end_ifft_radix2_butterfly_hp
                ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1                  // [a1 a2 a3 a4]
                ee.fft.cmul.s16.ld.xp q4, a13, a11, q5, q3, q2, 1            // [b1]
                ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 3            // [b2]
                ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q2, 5            // [b3]
                ee.fft.cmul.s16.st.xp q3, q2, q5, a8, a11, 7, 0, 1             // [b1 b2 b3 b4]

                ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1
                ee.fft.cmul.s16.ld.xp q2, a13, a11, q5, q3, q4, 1
                ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q4, 3
                ee.fft.cmul.s16.ld.xp q1, a7, a11, q5, q3, q4, 5
                ee.fft.cmul.s16.st.xp q3, q4, q5, a8, a11, 7, 0, 1

            loop_end_ifft_radix2_butterfly_hp:

            addx4 a4, a6, a4  // start index of box
            addi a14, a14, -1
            sub a13, a13, a11

        bnez a14, start_ifft_radix2_stage_box_hp

        srli a15, a15, 1                // iteration for bf /2
        srli a6, a6, 1                  // number of cpx_points in box
        slli a12, a12, 1                // number of box
        mov a14, a12
        addi a5, a5, -1

        mul16u a3, a6, a12 // number of complex points
        ee.zero.q q0 // max
        ee.zero.q q1 // min
        mov.n a4, a2 // load pointer
        srai a3, a3, 3 // << 1 >> 4
        ee.vld.128.ip q2, a4, 16
        loopnez a3, test_radix2_ifft_bf_s16_hp_loop_end_0
            ee.vmax.s16.ld.incp q3, a4, q0, q0, q2
            ee.vmin.s16 q1, q1, q2
            ee.vmax.s16.ld.incp q2, a4, q0, q0, q3
            ee.vmin.s16 q1, q1, q3
        test_radix2_ifft_bf_s16_hp_loop_end_0:

        movi.n a4, 0 // abs(max)
        ee.movi.32.a q0, a3, 0
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q0, a3, 1
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q0, a3, 2
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q0, a3, 3
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q1, a3, 0
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        neg a7, a7
        neg a3, a3
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q1, a3, 1
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        neg a7, a7
        neg a3, a3
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q1, a3, 2
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        neg a7, a7
        neg a3, a3
        max a4, a4, a3
        max a4, a4, a7
        ee.movi.32.a q1, a3, 3
        srai a7, a3, 16
        slli a3, a3, 16
        srai a3, a3, 16
        neg a7, a7
        neg a3, a3
        max a4, a4, a3
        max a4, a4, a7

        movi a3, 4096
        bge a4, a3, test_radix2_ifft_bf_s16_hp_skip_left_shift
            mul16u a3, a6, a12
            movi.n a4, 0
            ssr a4
            srai a3, a3, 4
            mov.n a4, a2 // load pointer
            mov.n a7, a4 // store pointer
            ee.vld.128.ip q0, a4, 16
            ee.vld.128.ip q1, a4, 16
            loopnez a3, ifft_loop_end
                ee.vmul.s16.ld.incp q2, a4, q0, q0, q6
                ee.vmul.s16.ld.incp q3, a4, q1, q1, q6
                ee.vst.128.ip q0, a7, 16
                ee.vst.128.ip q1, a7, 16
                ee.vmul.s16.ld.incp q0, a4, q2, q2, q6
                ee.vmul.s16.ld.incp q1, a4, q3, q3, q6
                ee.vst.128.ip q2, a7, 16
                ee.vst.128.ip q3, a7, 16
            ifft_loop_end:
            movi.n a4, 15
            ssr a4
            j test_radix2_ifft_bf_s16_hp_skip_left_shift_end
        test_radix2_ifft_bf_s16_hp_skip_left_shift:
            // if no need to left shift, add q7[0] by 1 to count how many times result is shifted
            ee.movi.32.a q7, a3, 0
            addi.n a3, a3, 1
            ee.movi.32.q q7, a3, 0
        test_radix2_ifft_bf_s16_hp_skip_left_shift_end:

    bnez a5, start_ifft_radix2_stage_hp     //loop for stage


    mov a9, a2      // load
    mov a10, a9     // store
    //srli a15, a12, 1

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    ee.vld.128.ip q2, a13, 16 //fixed twiddle factor
    movi a14, 0

    loopnez a12, loop_end_radix2_ifft_last_third_stage_hp
        ee.fft.r2bf.s16.st.incp q3, q0, q1, a10, 1                      // [a1 a2 a3 a4]
        ee.fft.cmul.s16.ld.xp q4, a13, a14, q5, q3, q2, 1               // [b1] q4- fake load
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 3                // [b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q3, q2, 5                // [b3]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 0, 1                 // [b1 b2 b3 b4]
    loop_end_radix2_ifft_last_third_stage_hp:

    mul16u a3, a6, a12 // number of complex points
    ee.zero.q q0 // max
    ee.zero.q q1 // min
    mov.n a4, a2 // load pointer
    srai a3, a3, 3 // << 1 >> 4
    ee.vld.128.ip q2, a4, 16
    loopnez a3, test_radix2_ifft_bf_s16_hp_loop_end_1
        ee.vmax.s16.ld.incp q3, a4, q0, q0, q2
        ee.vmin.s16 q1, q1, q2
        ee.vmax.s16.ld.incp q2, a4, q0, q0, q3
        ee.vmin.s16 q1, q1, q3
    test_radix2_ifft_bf_s16_hp_loop_end_1:

    movi.n a4, 0 // abs(max)
    ee.movi.32.a q0, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7

    movi a3, 4096
    bge a4, a3, test_radix2_ifft_bf_s16_hp_skip_left_shift_1
        mul16u a3, a6, a12
        movi.n a4, 0
        ssr a4
        srai a3, a3, 4
        mov.n a4, a2 // load pointer
        mov.n a7, a4 // store pointer
        ee.vld.128.ip q0, a4, 16
        ee.vld.128.ip q1, a4, 16
        loopnez a3, ifft_loop_end_1
            ee.vmul.s16.ld.incp q2, a4, q0, q0, q6
            ee.vmul.s16.ld.incp q3, a4, q1, q1, q6
            ee.vst.128.ip q0, a7, 16
            ee.vst.128.ip q1, a7, 16
            ee.vmul.s16.ld.incp q0, a4, q2, q2, q6
            ee.vmul.s16.ld.incp q1, a4, q3, q3, q6
            ee.vst.128.ip q2, a7, 16
            ee.vst.128.ip q3, a7, 16
        ifft_loop_end_1:
        movi.n a4, 15
        ssr a4
        j test_radix2_ifft_bf_s16_hp_skip_left_shift_1_end
    test_radix2_ifft_bf_s16_hp_skip_left_shift_1:
        // if no need to left shift, add q7[0] by 1 to count how many times result is shifted
        ee.movi.32.a q7, a3, 0
        addi.n a3, a3, 1
        ee.movi.32.q q7, a3, 0
    test_radix2_ifft_bf_s16_hp_skip_left_shift_1_end:

    mov a9, a2      // load
    mov a10, a9     // store
    //slli a15, a15, 1
    mov a14, a2 //fake load
    mov a3, a13

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    //ee.vld.128.incp q2, a13  //fixed twiddle factor
    ee.vld.h.64.ip q2, a13, 0

    loopnez a12, loop_end_radix2_ifft_last_second_stage_hp
        ee.fft.r2bf.s16 q3, q4, q0, q1, 0          // [a1 a2 a3 a4]   q5-fake load
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 5                // [b1]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 1, 1                 // [a1 a2 b1 b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 5                // [b3]
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 7, 1, 1                 // [a3 a4 b3 b4]
    loop_end_radix2_ifft_last_second_stage_hp:

    mul16u a3, a6, a12 // number of complex points
    ee.zero.q q0 // max
    ee.zero.q q1 // min
    mov.n a4, a2 // load pointer
    srai a3, a3, 3 // << 1 >> 4
    ee.vld.128.ip q2, a4, 16
    loopnez a3, test_radix2_ifft_bf_s16_hp_loop_end_2
        ee.vmax.s16.ld.incp q3, a4, q0, q0, q2
        ee.vmin.s16 q1, q1, q2
        ee.vmax.s16.ld.incp q2, a4, q0, q0, q3
        ee.vmin.s16 q1, q1, q3
    test_radix2_ifft_bf_s16_hp_loop_end_2:

    movi.n a4, 0 // abs(max)
    ee.movi.32.a q0, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7

    movi a3, 4096
    bge a4, a3, test_radix2_ifft_bf_s16_hp_skip_left_shift_2
        mul16u a3, a6, a12
        movi.n a4, 0
        ssr a4
        srai a3, a3, 4
        mov.n a4, a2 // load pointer
        mov.n a7, a4 // store pointer
        ee.vld.128.ip q0, a4, 16
        ee.vld.128.ip q1, a4, 16
        loopnez a3, ifft_loop_end_2
            ee.vmul.s16.ld.incp q2, a4, q0, q0, q6
            ee.vmul.s16.ld.incp q3, a4, q1, q1, q6
            ee.vst.128.ip q0, a7, 16
            ee.vst.128.ip q1, a7, 16
            ee.vmul.s16.ld.incp q0, a4, q2, q2, q6
            ee.vmul.s16.ld.incp q1, a4, q3, q3, q6
            ee.vst.128.ip q2, a7, 16
            ee.vst.128.ip q3, a7, 16
        ifft_loop_end_2:
        movi.n a4, 15
        ssr a4
        j test_radix2_ifft_bf_s16_hp_skip_left_shift_2_end
    test_radix2_ifft_bf_s16_hp_skip_left_shift_2:
        // if no need to left shift, add q7[0] by 1 to count how many times result is shifted
        ee.movi.32.a q7, a3, 0
        addi.n a3, a3, 1
        ee.movi.32.q q7, a3, 0
    test_radix2_ifft_bf_s16_hp_skip_left_shift_2_end:

    mov a9, a2      // load
    mov a10, a9     // store

    ee.vld.128.ip q0, a9, 16
    ee.vld.128.ip q1, a9, 16
    //ee.vld.128.incp q2, a13 //fixed twiddle factor
    mov.n a3, a13
    ee.vldbc.32 q2, a3

    loopnez a12, loop_end_radix2_ifft_last_stage_hp
        ee.fft.r2bf.s16 q3, q4, q0, q1, 1          // [a1 a2 a3 a4]   q5-fake load
        ee.fft.cmul.s16.ld.xp q0, a9, a11, q5, q3, q2, 5                // [b1]
        ee.fft.cmul.s16.st.xp q3, q2, q5, a10, a11, 7, 2, 1                 // [a1 a2 b1 b2]
        ee.fft.cmul.s16.ld.xp q1, a9, a11, q5, q4, q2, 5                // [b3]
        ee.fft.cmul.s16.st.xp q4, q2, q5, a10, a11, 7, 2, 1                 // [a3 a4 b3 b4]
    loop_end_radix2_ifft_last_stage_hp:

    mul16u a3, a6, a12 // number of complex points
    ee.zero.q q0 // max
    ee.zero.q q1 // min
    mov.n a4, a2 // load pointer
    srai a3, a3, 3 // << 1 >> 4
    ee.vld.128.ip q2, a4, 16
    loopnez a3, test_radix2_ifft_bf_s16_hp_loop_end_3
        ee.vmax.s16.ld.incp q3, a4, q0, q0, q2
        ee.vmin.s16 q1, q1, q2
        ee.vmax.s16.ld.incp q2, a4, q0, q0, q3
        ee.vmin.s16 q1, q1, q3
    test_radix2_ifft_bf_s16_hp_loop_end_3:

    movi.n a4, 0 // abs(max)
    ee.movi.32.a q0, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q0, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 0
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 1
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 2
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7
    ee.movi.32.a q1, a3, 3
    srai a7, a3, 16
    slli a3, a3, 16
    srai a3, a3, 16
    neg a7, a7
    neg a3, a3
    max a4, a4, a3
    max a4, a4, a7

    movi a3, 4096
    bge a4, a3, test_radix2_ifft_bf_s16_hp_skip_left_shift_3
        mul16u a3, a6, a12
        movi.n a4, 0
        ssr a4
        srai a3, a3, 4
        mov.n a4, a2 // load pointer
        mov.n a7, a4 // store pointer
        ee.vld.128.ip q0, a4, 16
        ee.vld.128.ip q1, a4, 16
        loopnez a3, ifft_loop_end_3
            ee.vmul.s16.ld.incp q2, a4, q0, q0, q6
            ee.vmul.s16.ld.incp q3, a4, q1, q1, q6
            ee.vst.128.ip q0, a7, 16
            ee.vst.128.ip q1, a7, 16
            ee.vmul.s16.ld.incp q0, a4, q2, q2, q6
            ee.vmul.s16.ld.incp q1, a4, q3, q3, q6
            ee.vst.128.ip q2, a7, 16
            ee.vst.128.ip q3, a7, 16
        ifft_loop_end_3:
        movi.n a4, 15
        ssr a4
        j test_radix2_ifft_bf_s16_hp_skip_left_shift_3_end
    test_radix2_ifft_bf_s16_hp_skip_left_shift_3:
        // if no need to left shift, add q7[0] by 1 to count how many times result is shifted
        ee.movi.32.a q7, a3, 0
        addi.n a3, a3, 1
        ee.movi.32.q q7, a3, 0
    test_radix2_ifft_bf_s16_hp_skip_left_shift_3_end:

    ee.movi.32.a q7, a2, 0

    retw



/* void test_fftr_s16(int16_t *, int16_t *, int16_t); */
    .text
    .align 4
    .global test_fftr_s16
    .type   test_fftr_s16, @function
test_fftr_s16:

    /* a2 = data
     * a3 = win
     * a4 = cpx_points
     * a5 = shift
     */

    /* a2 = ptr = data
     * a7 = ptr1
     * a8 = ptr_inv
     * a9 = ptr_inv1

     * a3 = win
     * a4 = n
     */


    .align 4
    entry sp, 16

	movi a5, 14
    ssr a5

    mov a6, a2
    addi a7, a2, 16  //ptr1 = data + 16;
    addx4 a9, a4, a2  //ptr_inv1 = data + cpx_points*2
    addi a8, a9, -16  //ptr_inv = data + cpx_points*2 - 8


	l16si a10, a2, 0
	l16si a11, a2, 2

	add a13, a10, a11
	sub a11, a10, a11

	s16i a11, a9, 0
	movi a11, 0
	s16i a11, a9, 2

	addi a9, a9, -16

	ee.vld.128.ip q2, a3, 16
	ee.fft.ams.s16.ld.r32.decp q1, a8, q6, q7, q3, q4, q5, 0 // only use the load

    ee.vld.128.ip q3, a2, 0
    movi a10, 4
    wur.sar_byte a10
    ee.vld.128.ip q4, a7, 16
    ee.src.q q0, q3, q4

    addi a10, a2, 16

    ee.fft.ams.s16.ld.incp.uaup q3, a10, q6, q7, q0, q1, q2, 0 //only use the load

    srli a4, a4, 4
    loopnez a4, loop_end_fftr
        ee.fft.ams.s16.ld.incp.uaup q3, a7, q6, q7, q0, q1, q2, 0
        ee.fft.ams.s16.ld.incp q5, a3, q6, q7, q0, q1, q2, 0
        ee.fft.ams.s16.ld.r32.decp q4, a8, q6, q7, q0, q1, q2, 0
        ee.fft.ams.s16.st.incp q6, q7, a11, a6, q0, q1, q2, 0
        ee.fft.vst.r32.decp q7, a9, 1

        ee.fft.ams.s16.ld.incp.uaup q0, a7, q6, q7, q3, q4, q5, 0
        ee.fft.ams.s16.ld.incp q2, a3, q6, q7, q3, q4, q5, 0
        ee.fft.ams.s16.ld.r32.decp q1, a8, q6, q7, q3, q4, q5, 0
        ee.fft.ams.s16.st.incp q6, q7, a11, a6, q3, q4, q5, 0
        ee.fft.vst.r32.decp q7, a9, 1

    loop_end_fftr:

	s32i a13, a2, 0

    retw


/* void test_ffti_s16(int16_t *, int16_t *, int16_t); */
    .text
    .align 4
    .global test_ffti_s16
    .type   test_ffti_s16, @function
test_ffti_s16:

    /* a2 = data
     * a3 = win
     * a4 = cpx_points
     * a5 = shift
     */

    /* a2 = ptr = data
     * a7 = ptr1
     * a8 = ptr_inv
     * a9 = ptr_inv1

     * a3 = win
     * a4 = n
     */


    .align 4
    entry sp, 16

	movi a5, 14
    ssr a5

    mov a6, a2
    addi a7, a2, 16  //ptr1 = data + 16;
    addx4 a9, a4, a2  //ptr_inv1 = data + cpx_points*2
    addi a8, a9, -16  //ptr_inv = data + cpx_points*2 - 8


	l16si a10, a2, 0
	l16si a11, a9, 0

	add a13, a10, a11
	sub a12, a10, a11

	addi a9, a9, -16

	ee.vld.128.ip q2, a3, 16
	ee.fft.ams.s16.ld.r32.decp q1, a8, q6, q7, q3, q4, q5, 0 // only use the load


    ee.vld.128.ip q3, a2, 0
    movi a10, 4
    wur.sar_byte a10
    ee.vld.128.ip q4, a7, 16
    ee.src.q q0, q3, q4

    addi a10, a2, 16

    ee.fft.ams.s16.ld.incp.uaup q3, a10, q6, q7, q0, q1, q2, 0 //only use the load


    srli a4, a4, 4
    loopnez a4, loop_end_ffti
		ee.fft.ams.s16.ld.incp.uaup q3, a7, q6, q7, q0, q1, q2, 1
        ee.fft.ams.s16.ld.incp q5, a3, q6, q7, q0, q1, q2, 1
        ee.fft.ams.s16.ld.r32.decp q4, a8, q6, q7, q0, q1, q2, 1
        ee.fft.ams.s16.st.incp q6, q7, a11, a6, q0, q1, q2, 1
        ee.fft.vst.r32.decp q7, a9, 0

        ee.fft.ams.s16.ld.incp.uaup q0, a7, q6, q7, q3, q4, q5, 1
        ee.fft.ams.s16.ld.incp q2, a3, q6, q7, q3, q4, q5, 1
        ee.fft.ams.s16.ld.r32.decp q1, a8, q6, q7, q3, q4, q5, 1
        ee.fft.ams.s16.st.incp q6, q7, a11, a6, q3, q4, q5, 1
        ee.fft.vst.r32.decp q7, a9, 0

    loop_end_ffti:

	s16i a13, a2, 0
	s16i a12, a2, 2


    retw
