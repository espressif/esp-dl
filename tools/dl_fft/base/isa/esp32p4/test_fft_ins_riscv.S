
.macro prologue
    addi sp, sp, -16
    sw s0, 12(sp)
    sw s1, 8(sp)
    sw s8, 4(sp)
    sw s9, 0(sp)
.endm

.macro epilogue
    lw s9, 0(sp)
    lw s8, 4(sp)
    lw s1, 8(sp)
    lw s0, 12(sp)
    addi sp, sp, 16
.endm





    .text
    .align 4
    .global test_radix2_bit_reverse
    .type   test_radix2_bit_reverse, @function
test_radix2_bit_reverse:
    .option norvc

    prologue


    srli a1, a1, 3
    esp.movx.w.fft.bit.width a2
    li s8, 0

    mv a5, a0
    mv a4, a0

    addi  t0, a1, 0
    loop_end_radix2_bit_reverse:
        esp.vld.128.ip q0, a5, 16
        esp.fft.bitrev q2, s8
        esp.ldxq.32 q1, q2, a0, 0, 0
        esp.ldxq.32 q1, q2, a0, 1, 1
        esp.ldxq.32 q1, q2, a0, 2, 2
        esp.ldxq.32 q1, q2, a0, 3, 3
        esp.stxq.32 q0, q2, a0, 0, 0
        esp.stxq.32 q0, q2, a0, 1, 1
        esp.stxq.32 q0, q2, a0, 2, 2
        esp.stxq.32 q0, q2, a0, 3, 3
        esp.vst.128.ip q1, a4, 16

        esp.vld.128.ip q0, a5, 16
        esp.ldxq.32 q1, q2, a0, 0, 4
        esp.ldxq.32 q1, q2, a0, 1, 5
        esp.ldxq.32 q1, q2, a0, 2, 6
        esp.ldxq.32 q1, q2, a0, 3, 7
        esp.stxq.32 q0, q2, a0, 0, 4
        esp.stxq.32 q0, q2, a0, 1, 5
        esp.stxq.32 q0, q2, a0, 2, 6
        esp.stxq.32 q0, q2, a0, 3, 7
        esp.vst.128.ip q1, a4, 16

    addi t0, t0, -1
    bnez t0, loop_end_radix2_bit_reverse

    epilogue
    .align 4
    ret



    .text
    .align 4
    .global test_radix2_fft_bf_s16
    .type   test_radix2_fft_bf_s16, @function
test_radix2_fft_bf_s16:
    .option norvc
    prologue
    esp.movx.w.sar a2
    mv s1, a1
    addi a3, a3, -3
    li s0, 1
    mv s8, s0
    srli s9, a4, 4

    li t6, 16


    start_fft_stage:
        mv a2, a0
        mv a1, s1

        start_fft_box:

            mv t4, a2
            mv t5, t4
            slli t0, a4, 1
            add a5, t0, a2
		    mv t3, a5
            mv s1, a1
            esp.vld.128.ip q0, t4, 16
		    esp.vld.128.ip q1, a5, 16
		    esp.vld.128.ip q2, s1, 16

            addi t0, s9, 0
            loop_end_fft_butterfly:

                esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
		        esp.fft.cmul.s16.ld.xp q4, s1, t6, q5, q3, q2, 0
		        esp.fft.cmul.s16.ld.xp q0, t4, t6, q5, q3, q2, 2
		        esp.fft.cmul.s16.ld.xp q1, a5, t6, q5, q3, q2, 4

		        esp.fft.cmul.s16.st.xp q3, q2, q5, t3, t6, 6, 0, 1

		        esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
		        esp.fft.cmul.s16.ld.xp q2, s1, t6, q5, q3, q4, 0
		        esp.fft.cmul.s16.ld.xp q0, t4, t6, q5, q3, q4, 2
		        esp.fft.cmul.s16.ld.xp q1, a5, t6, q5, q3, q4, 4

		        esp.fft.cmul.s16.st.xp q3, q4, q5, t3, t6, 6, 0, 1


                addi t0, t0, -1
            bnez t0, loop_end_fft_butterfly

			slli t0, a4, 2
			add a2, t0, a2

			addi s8, s8, -1
			sub s1, s1, t6

        bnez s8, start_fft_box

        srli s9, s9, 1
        srli a4, a4, 1
        slli s0, s0, 1
        mv s8, s0
        addi a3, a3, -1

    bnez a3, start_fft_stage


    mv t4, a0
    mv t5, t4


    esp.vld.128.ip q0, t4, 16
    esp.vld.128.ip q1, t4, 16
    esp.vld.128.ip q2, s1, 16
    li s8, 0

	addi t0, s0, 0
	loop_end_radix2_last_third_stage:
		esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
        esp.fft.cmul.s16.ld.xp q4, s1, s8, q5, q3, q2, 0
        esp.fft.cmul.s16.ld.xp q0, t4, t6, q5, q3, q2, 2
        esp.fft.cmul.s16.ld.xp q1, t4, t6, q5, q3, q2, 4

        esp.fft.cmul.s16.st.xp q3, q2, q5, t5, t6, 6, 0, 1
	addi t0, t0, -1
	bnez t0, loop_end_radix2_last_third_stage




	mv t4, a0
    mv t5, t4

    mv s8, a0
    mv a1, s1



    esp.vld.128.ip q0, t4, 16
    esp.vld.128.ip q1, t4, 16
    esp.vld.h.64.ip q2, s1, 0

	addi t0, s0, 0
	loop_end_radix2_last_second_stage:
        esp.fft.r2bf.s16 q3, q4, q0, q1, 0
        esp.fft.cmul.s16.ld.xp q0, t4, t6, q5, q3, q2, 4
        esp.fft.cmul.s16.st.xp q3, q2, q5, t5, t6, 6, 1, 1
        esp.fft.cmul.s16.ld.xp q1, t4, t6, q5, q4, q2, 4
        esp.fft.cmul.s16.st.xp q4, q2, q5, t5, t6, 6, 1, 1
	addi t0, t0, -1
	bnez t0, loop_end_radix2_last_second_stage




	mv t4, a0
    mv t5, t4

    esp.vld.128.ip q0, t4, 16
    esp.vld.128.ip q1, t4, 16

    esp.vldbc.32.ip q2, a1, 0

	addi t0, s0, 0
	loop_end_radix2_last_stage:
        esp.fft.r2bf.s16 q3, q4, q0, q1, 1
        esp.fft.cmul.s16.ld.xp q0, t4, t6, q5, q3, q2, 4

        esp.fft.cmul.s16.st.xp q3, q2, q5, t5, t6, 6, 2, 1

        esp.fft.cmul.s16.ld.xp q1, t4, t6, q5, q4, q2, 4

        esp.fft.cmul.s16.st.xp q4, q2, q5, t5, t6, 6, 2, 1
	addi t0, t0, -1
	bnez t0, loop_end_radix2_last_stage

    epilogue
    ret



    .text
    .align 4
    .global test_radix2_ifft_bf_s16
    .type   test_radix2_ifft_bf_s16, @function
test_radix2_ifft_bf_s16:
    .option norvc



    .align 4
    prologue
    esp.movx.w.sar a2
    mv s1, a1
    addi a3, a3, -3
    li s0, 1
    mv s8, s0
    srli s9, a4, 4

    li t6, 16

    start_ifft_radix2_stage:
        mv a2, a0
        mv a1, s1
        start_ifft_radix2_stage_box:
            mv t4, a2
            mv t5, t4
            slli t0, a4, 1
            add a5, t0, a2

		    mv t3, a5

            mv s1, a1

            esp.vld.128.ip q0, t4, 16
		    esp.vld.128.ip q1, a5, 16
		    esp.vld.128.ip q2, s1, 16

            add t0, s9, x0
            loop_end_ifft_radix2_butterfly:
		        esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
		        esp.fft.cmul.s16.ld.xp q4, s1, t6, q5, q3, q2, 1
		        esp.fft.cmul.s16.ld.xp q0, t4, t6, q5, q3, q2, 3
		        esp.fft.cmul.s16.ld.xp q1, a5, t6, q5, q3, q2, 5

		        esp.fft.cmul.s16.st.xp q3, q2, q5, t3, t6, 7, 0, 1

		        esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
		        esp.fft.cmul.s16.ld.xp q2, s1, t6, q5, q3, q4, 1
		        esp.fft.cmul.s16.ld.xp q0, t4, t6, q5, q3, q4, 3
		        esp.fft.cmul.s16.ld.xp q1, a5, t6, q5, q3, q4, 5

		        esp.fft.cmul.s16.st.xp q3, q4, q5, t3, t6, 7, 0, 1

            addi t0, t0, -1
            bnez t0, loop_end_ifft_radix2_butterfly

			slli t0, a4, 2
			add a2, t0, a2

			addi s8, s8, -1
			sub s1, s1, t6

        bnez s8, start_ifft_radix2_stage_box

        srli s9, s9, 1
        srli a4, a4, 1
        slli s0, s0, 1
        mv s8, s0
        addi a3, a3, -1

    bnez a3, start_ifft_radix2_stage


    mv t4, a0
    mv t5, t4


    esp.vld.128.ip q0, t4, 16
    esp.vld.128.ip q1, t4, 16
    esp.vld.128.ip q2, s1, 16
    li s8, 0

	add t0, s0, x0
	loop_end_radix2_ifft_last_third_stage:
		esp.fft.r2bf.s16.st.incp q3, q0, q1, t5, 1
        esp.fft.cmul.s16.ld.xp q4, s1, s8, q5, q3, q2, 1
        esp.fft.cmul.s16.ld.xp q0, t4, t6, q5, q3, q2, 3
        esp.fft.cmul.s16.ld.xp q1, t4, t6, q5, q3, q2, 5

        esp.fft.cmul.s16.st.xp q3, q2, q5, t5, t6, 7, 0, 1
	addi t0, t0, -1
	bnez t0, loop_end_radix2_ifft_last_third_stage

	mv t4, a0
    mv t5, t4

    mv s8, a0
    mv a1, s1

    esp.vld.128.ip q0, t4, 16
    esp.vld.128.ip q1, t4, 16

    esp.vld.h.64.ip q2, s1, 0

	add t0, s0, x0
	loop_end_radix2_ifft_last_second_stage:
        esp.fft.r2bf.s16 q3, q4, q0, q1, 0
        esp.fft.cmul.s16.ld.xp q0, t4, t6, q5, q3, q2, 5

        esp.fft.cmul.s16.st.xp q3, q2, q5, t5, t6, 7, 1, 1

        esp.fft.cmul.s16.ld.xp q1, t4, t6, q5, q4, q2, 5

        esp.fft.cmul.s16.st.xp q4, q2, q5, t5, t6, 7, 1, 1
	addi t0, t0, -1
	bnez t0, loop_end_radix2_ifft_last_second_stage


	mv t4, a0
    mv t5, t4

    esp.vld.128.ip q0, t4, 16
    esp.vld.128.ip q1, t4, 16

    esp.vldbc.32.ip q2, a1, 0

	add t0, s0, x0
	loop_end_radix2_ifft_last_stage:
        esp.fft.r2bf.s16 q3, q4, q0, q1, 1
        esp.fft.cmul.s16.ld.xp q0, t4, t6, q5, q3, q2, 5

        esp.fft.cmul.s16.st.xp q3, q2, q5, t5, t6, 7, 2, 1

        esp.fft.cmul.s16.ld.xp q1, t4, t6, q5, q4, q2, 5

        esp.fft.cmul.s16.st.xp q4, q2, q5, t5, t6, 7, 2, 1
	addi t0, t0, -1
	bnez t0, loop_end_radix2_ifft_last_stage

    epilogue
    ret




    .text
    .align 4
    .global test_fftr_s16
    .type   test_fftr_s16, @function
test_fftr_s16:
    .option norvc
    prologue
    li a3, 14
    esp.movx.w.sar a3

    mv a4, a0
    addi a5, a0, 16
    slli t0, a2, 2
    add t4, t0, a0
    addi t3, t4, -16

	lh t5, 0(a0)
	lh t6, 2(a0)

	add s1, t5, t6
	sub t6, t5, t6

	sh t6, 0(t4)
	li t6, 0
	sh t6, 2(t4)

	addi t4, t4, -16

	esp.vld.128.ip q2, a1, 16
	esp.fft.ams.s16.ld.r32.decp q1, t3, q6, q7, q3, q4, q5, 0
    esp.vld.128.ip q3, a0, 0
    li t5, 4
    esp.movx.w.sar.bytes t5
    esp.vld.128.ip q4, a5, 16
    esp.src.q q0, q3, q4

    addi t5, a0, 16
    esp.fft.ams.s16.ld.incp.uaup q3, t5, q6, q7, q0, q1, q2, 0

    srli a2, a2, 4
    addi t0, a2, 0
    loop_end_fftr:
        esp.fft.ams.s16.ld.incp.uaup q3, a5, q6, q7, q0, q1, q2, 0
        esp.fft.ams.s16.ld.incp q5, a1, q6, q7, q0, q1, q2, 0
        esp.fft.ams.s16.ld.r32.decp q4, t3, q6, q7, q0, q1, q2, 0

        esp.fft.ams.s16.st.incp q6, q7, t6, a4, q0, q1, q2, 0
        esp.fft.vst.r32.decp q7, t4, 1

        esp.fft.ams.s16.ld.incp.uaup q0, a5, q6, q7, q3, q4, q5, 0
        esp.fft.ams.s16.ld.incp q2, a1, q6, q7, q3, q4, q5, 0
        esp.fft.ams.s16.ld.r32.decp q1, t3, q6, q7, q3, q4, q5, 0

        esp.fft.ams.s16.st.incp q6, q7, t6, a4, q3, q4, q5, 0
        esp.fft.vst.r32.decp q7, t4, 1

    addi t0, t0, -1
    bnez t0, loop_end_fftr

	sw s1, 0(a0)

    epilogue
    ret



    .text
    .align 4
    .global test_ffti_s16
    .type   test_ffti_s16, @function
test_ffti_s16:
    .option norvc




    .align 4
    prologue
    li a3, 14
    esp.movx.w.sar a3

    mv a4, a0
    addi a5, a0, 16
    slli t0, a2, 2
    add t4, t0, a0

    addi t3, t4, -16


	lh t5, 0(a0)
	lh t6, 0(t4)

	add s1, t5, t6
	sub s0, t5, t6

	addi t4, t4, -16

	esp.vld.128.ip q2, a1, 16
	esp.fft.ams.s16.ld.r32.decp q1, t3, q6, q7, q3, q4, q5, 0


    esp.vld.128.ip q3, a0, 0
    li t5, 4
    esp.movx.w.sar.bytes t5
    esp.vld.128.ip q4, a5, 16
    esp.src.q q0, q3, q4

    addi t5, a0, 16


    esp.fft.ams.s16.ld.incp.uaup q3, t5, q6, q7, q0, q1, q2, 0


    srli a2, a2, 4
    add t0, a2, x0
    loop_end_ffti:
		esp.fft.ams.s16.ld.incp.uaup q3, a5, q6, q7, q0, q1, q2, 1
        esp.fft.ams.s16.ld.incp q5, a1, q6, q7, q0, q1, q2, 1
        esp.fft.ams.s16.ld.r32.decp q4, t3, q6, q7, q0, q1, q2, 1
        esp.fft.ams.s16.st.incp q6, q7, t6, a4, q0, q1, q2, 1
        esp.fft.vst.r32.decp q7, t4, 0

        esp.fft.ams.s16.ld.incp.uaup q0, a5, q6, q7, q3, q4, q5, 1
        esp.fft.ams.s16.ld.incp q2, a1, q6, q7, q3, q4, q5, 1
        esp.fft.ams.s16.ld.r32.decp q1, t3, q6, q7, q3, q4, q5, 1

        esp.fft.ams.s16.st.incp q6, q7, t6, a4, q3, q4, q5, 1
        esp.fft.vst.r32.decp q7, t4, 0

    addi t0, t0, -1
    bnez t0, loop_end_ffti

	sh s1, 0(a0)
	sh s0, 2(a0)


    epilogue
    ret
