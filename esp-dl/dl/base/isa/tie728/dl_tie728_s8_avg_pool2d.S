############################################################################################################################################################
####
#### dl_tie728_s8_avg_pool2d series
####
############################################################################################################################################################

#include "dl_tie728_s8.S"


	.align 4
    .text
    .global dl_tie728_s8_avg_pool2d_22c1
    .type	dl_tie728_s8_avg_pool2d_22c1, @function
    .section .iram1
dl_tie728_s8_avg_pool2d_22c1:
    .align      4
    entry       sp,     16

    # a2: int16_t *output_ptr
    # a3: int16_t *input_ptr
    # a4: void *args

    l32i a5, a4, 16     # input_y_offset
    l32i a6, a4, 20     # input_x_offset
    l32i a10, a4, 4     # input_channel
    l32i a11, a4, 104   # c_div_x_1
    l32i a13, a4, 56    # shift

    addi a14, a4, 64
    EE.VLDBC.8 q0, a14  # avg_pool_area_inv

    add a7, a3, a6
    add a8, a3, a5
    add a9, a8, a6

    EE.VLD.128.IP q1, a3, 16
    EE.VLD.128.IP q2, a7, 16
    loopgtz a11, 0f
    	EE.ZERO.QACC
	    EE.VMULAS.S8.QACC.LD.IP q3, a8, 16, q0, q1
	    EE.VMULAS.S8.QACC.LD.IP q4, a9, 16, q0, q2
	    EE.VMULAS.S8.QACC.LD.IP q1, a3, 16, q0, q3
	    EE.VMULAS.S8.QACC.LD.IP q2, a7, 16, q0, q4
	    # EE.SRCMB.S8.QACC q7, a13, 0
        tie728_s8_vector_round_result q7, a13, a15, q6
	    EE.VST.128.IP q7, a2, 16
	0:

	EE.ZERO.QACC
    EE.VMULAS.S8.QACC.LD.IP q3, a8, 16, q0, q1
    EE.VMULAS.S8.QACC.LD.IP q4, a9, 16, q0, q2
    EE.VMULAS.S8.QACC.LD.IP q1, a3, 16, q0, q3
    EE.VMULAS.S8.QACC.LD.IP q2, a7, 16, q0, q4
    # EE.SRCMB.S8.QACC q7, a13, 0
    tie728_s8_vector_round_result q7, a13, a15, q6

    EE.VST.128.IP q7, a2, 16
    retw






    .align 4
    .text
    .global dl_tie728_s8_unaligned_avg_pool2d_22c1
    .type   dl_tie728_s8_unaligned_avg_pool2d_22c1, @function
    .section .iram1
dl_tie728_s8_unaligned_avg_pool2d_22c1:
    .align      4
    entry       sp,     16
    # a2: int16_t *output_ptr
    # a3: int16_t *input_ptr
    # a4: void *args

    l32i a5, a4, 16     # input_y_offset
    l32i a6, a4, 20     # input_x_offset
    l32i a10, a4, 4     # input_channel
    l32i a11, a4, 104   # c_div_x_1
    l32i a12, a4, 60    # c_remainder
    l32i a13, a4, 56    # shift

    addi a14, a4, 64
    EE.VLDBC.8 q6, a14  # avg_pool_area_inv

    add a7, a3, a6
    add a8, a3, a5
    add a9, a8, a6

    blti a11, 0, dl_tie728_s8_unaligned_avg_pool2d_22c1_remainder #channel < 16

    EE.LD.128.USAR.IP q7, a2, 0 #get output_ptr sar_byte
    rur.sar_byte a15

    EE.LD.128.USAR.IP q0, a3, 16
    EE.LD.128.USAR.IP q1, a3, 0

    beqi a15, 0, 1f
    beqi a15, 8, 2f

    loopgtz a11, 0f
    	EE.ZERO.QACC
        EE.SRC.Q.LD.IP q2, a7, 16, q0, q1

        EE.LD.128.USAR.IP q3, a7, 0
        EE.VMULAS.S8.QACC q6, q0
        EE.SRC.Q.LD.IP q4, a8, 16, q2, q3

        EE.LD.128.USAR.IP q5, a8, 0
        EE.VMULAS.S8.QACC q6, q2
        EE.SRC.Q.LD.IP q2, a9, 16, q4, q5

        EE.LD.128.USAR.IP q3, a9, 0
        EE.VMULAS.S8.QACC q6, q4
        EE.SRC.Q.LD.IP q0, a3, 16, q2, q3

        EE.LD.128.USAR.IP q1, a3, 0
        EE.VMULAS.S8.QACC q6, q2
        # EE.SRCMB.S8.QACC q7, a13, 0
        tie728_s8_vector_round_result q7, a13, a15, q5
        dl_tie728_s8_unaligned_store0 q7, a2, a14
    0:
    j dl_tie728_s8_unaligned_avg_pool2d_22c1_loop_end


1:
    loopgtz a11, 0f
    	EE.ZERO.QACC
        EE.SRC.Q.LD.IP q2, a7, 16, q0, q1

        EE.LD.128.USAR.IP q3, a7, 0
        EE.VMULAS.S8.QACC q6, q0
        EE.SRC.Q.LD.IP q4, a8, 16, q2, q3

        EE.LD.128.USAR.IP q5, a8, 0
        EE.VMULAS.S8.QACC q6, q2
        EE.SRC.Q.LD.IP q2, a9, 16, q4, q5

        EE.LD.128.USAR.IP q3, a9, 0
        EE.VMULAS.S8.QACC q6, q4
        EE.SRC.Q.LD.IP q0, a3, 16, q2, q3

        EE.LD.128.USAR.IP q1, a3, 0
        EE.VMULAS.S8.QACC q6, q2
        # EE.SRCMB.S8.QACC q7, a13, 0
        tie728_s8_vector_round_result q7, a13, a15, q5
        EE.VST.128.IP q7, a2, 16
    0:
    j dl_tie728_s8_unaligned_avg_pool2d_22c1_loop_end

2:
    loopgtz a11, 0f
    	EE.ZERO.QACC
        EE.SRC.Q.LD.IP q2, a7, 16, q0, q1

        EE.LD.128.USAR.IP q3, a7, 0
        EE.VMULAS.S8.QACC q6, q0
        EE.SRC.Q.LD.IP q4, a8, 16, q2, q3

        EE.LD.128.USAR.IP q5, a8, 0
        EE.VMULAS.S8.QACC q6, q2
        EE.SRC.Q.LD.IP q2, a9, 16, q4, q5

        EE.LD.128.USAR.IP q3, a9, 0
        EE.VMULAS.S8.QACC q6, q4
        EE.SRC.Q.LD.IP q0, a3, 16, q2, q3

        EE.LD.128.USAR.IP q1, a3, 0
        EE.VMULAS.S8.QACC q6, q2
        # EE.SRCMB.S8.QACC q7, a13, 0
        tie728_s8_vector_round_result q7, a13, a15, q5
        dl_tie728_s8_unaligned_store1 q7, a2
    0:


dl_tie728_s8_unaligned_avg_pool2d_22c1_loop_end:
	EE.ZERO.QACC
    EE.SRC.Q.LD.IP q2, a7, 16, q0, q1

    EE.LD.128.USAR.IP q3, a7, 0
    EE.VMULAS.S8.QACC q6, q0
    EE.SRC.Q.LD.IP q4, a8, 16, q2, q3

    EE.LD.128.USAR.IP q5, a8, 0
    EE.VMULAS.S8.QACC q6, q2
    EE.SRC.Q.LD.IP q2, a9, 16, q4, q5

    EE.LD.128.USAR.IP q3, a9, 0
    EE.VMULAS.S8.QACC q6, q4
    EE.SRC.Q q2, q2, q3
    EE.VMULAS.S8.QACC q6, q2
    # EE.SRCMB.S8.QACC q7, a13, 0
    tie728_s8_vector_round_result q7, a13, a15, q5
    dl_tie728_s8_unaligned_store0 q7, a2, a14

    beqz a12, dl_tie728_s8_unaligned_avg_pool2d_22c1_end

dl_tie728_s8_unaligned_avg_pool2d_22c1_remainder:
    EE.LD.128.USAR.XP q0, a3, a12
    EE.VLD.128.IP q1, a3, 0
    EE.ZERO.QACC
    EE.SRC.Q q0, q0, q1

    EE.LD.128.USAR.XP q2, a7, a12
    EE.VLD.128.IP q3, a7, 0
    EE.VMULAS.S8.QACC q6, q0
    EE.SRC.Q q2, q2, q3

    EE.LD.128.USAR.XP q4, a8, a12
    EE.VLD.128.IP q5, a8, 0
    EE.VMULAS.S8.QACC q6, q2
    EE.SRC.Q q4, q4, q5

    EE.LD.128.USAR.XP q2, a9, a12
    EE.VLD.128.IP q3, a9, 0
    EE.VMULAS.S8.QACC q6, q4
    EE.SRC.Q q2, q2, q3

    EE.VMULAS.S8.QACC q6, q2
    # EE.SRCMB.S8.QACC q7, a13, 0
    tie728_s8_vector_round_result q7, a13, a15, q5

    dl_tie728_s8_store_remainder q7, a8, a9, a10, a11, a2, a12

dl_tie728_s8_unaligned_avg_pool2d_22c1_end:

    retw




    .align 4
    .text
    .global dl_tie728_s8_avg_pool2d_hwc1
    .type	dl_tie728_s8_avg_pool2d_hwc1, @function
    .section .iram1
dl_tie728_s8_avg_pool2d_hwc1:
    .align      4
    entry       sp,     16

    l32i a5, a4, 16     # input_y_offset
    l32i a6, a4, 20     # input_x_offset
    l32i a7, a4, 4      # input_channel
    l32i a8, a4, 48     # filter_height
    l32i a9, a4, 52     # filter_width
    l32i a11, a4, 104   # c_div_x_1
    l32i a13, a4, 56    # shift

    addi a14, a4, 64
    EE.VLDBC.8 q0, a14 # avg_pool_area_inv

    srli a10, a9, 1
    addi a10, a10, -1   # filter_w / 2 - 1

    beqi a9, 1, dl_tie728_s8_avg_pool2d_h1c1 #filter_width == 1
    blti a11, 1, dl_tie728_s8_avg_pool2d_hw_small_channel

    5:
	    mov a7, a3
	    mov a14, a7
	    mov a15, a8
	    EE.ZERO.QACC
		4:
		    EE.VLD.128.XP q1, a14, a6
		    EE.VLD.128.XP q2, a14, a6
		    loopgtz a10, 0f
			    EE.VMULAS.S8.QACC.LD.XP q1, a14, a6, q0, q1
			    EE.VMULAS.S8.QACC.LD.XP q2, a14, a6, q0, q2
			0:
			bbci a9, 0, 2f
	        1:#three left
	        EE.VMULAS.S8.QACC.LD.XP q1, a14, a6, q0, q1
	        EE.VMULAS.S8.QACC q0, q2
			EE.VMULAS.S8.QACC q0, q1
	        j 3f

	        2: # two left
	        EE.VMULAS.S8.QACC q0, q1
			EE.VMULAS.S8.QACC q0, q2
	        3:
		    addi a15, a15, -1
		    add a7, a7, a5
		    mov a14, a7
	    bnez a15, 4b

	    # EE.SRCMB.S8.QACC q7, a13, 0
        tie728_s8_vector_round_result q7, a13, a15, q5
	    EE.VST.128.IP q7, a2, 16
	    addi a3, a3, 16
	    addi a11, a11, -1
    bnez a11, 5b

dl_tie728_s8_avg_pool2d_hw_small_channel:
	mov a7, a3
    mov a14, a7
    mov a15, a8
    EE.ZERO.QACC
	4:
	    EE.VLD.128.XP q1, a14, a6
	    EE.VLD.128.XP q2, a14, a6
	    loopgtz a10, 0f
		    EE.VMULAS.S8.QACC.LD.XP q1, a14, a6, q0, q1
		    EE.VMULAS.S8.QACC.LD.XP q2, a14, a6, q0, q2
		0:
		bbci a9, 0, 2f
        1:#three left
        EE.VMULAS.S8.QACC.LD.XP q1, a14, a6, q0, q1
        EE.VMULAS.S8.QACC q0, q2
		EE.VMULAS.S8.QACC q0, q1
        j 3f

        2: # two left
        EE.VMULAS.S8.QACC q0, q1
		EE.VMULAS.S8.QACC q0, q2
        3:
	    addi a15, a15, -1
	    add a7, a7, a5
	    mov a14, a7
	bnez a15, 4b
	# EE.SRCMB.S8.QACC q7, a13, 0
    tie728_s8_vector_round_result q7, a13, a15, q5

    EE.VST.128.IP q7, a2, 16
    retw

dl_tie728_s8_avg_pool2d_h1c1:
	addi a8, a8, -1
	blti a11, 1, dl_tie728_s8_max_pool2d_h1_small_channel
    1:
        mov a14, a3
        EE.ZERO.QACC
        EE.VLD.128.XP q1, a14, a5
        loopgtz a8, 0f
            EE.VMULAS.S8.QACC.LD.XP q1, a14, a5, q0, q1
        0:
        EE.VMULAS.S8.QACC q0, q1
        # EE.SRCMB.S8.QACC q7, a13, 0
        tie728_s8_vector_round_result q7, a13, a15, q5
        EE.VST.128.IP q7, a2, 16
        addi a3, a3, 16
        addi a11, a11, -1
    bnez a11, 1b

dl_tie728_s8_max_pool2d_h1_small_channel:
    mov a14, a3
    EE.ZERO.QACC
    EE.VLD.128.XP q1, a14, a5
    loopgtz a8, 0f
        EE.VMULAS.S8.QACC.LD.XP q1, a14, a5, q0, q1
    0:
    EE.VMULAS.S8.QACC q0, q1
    # EE.SRCMB.S8.QACC q7, a13, 0
    tie728_s8_vector_round_result q7, a13, a15, q5

    EE.VST.128.IP q7, a2, 16
    retw






    .align 4
    .text
    .global dl_tie728_s8_unaligned_avg_pool2d_hwc1
    .type	dl_tie728_s8_unaligned_avg_pool2d_hwc1, @function
    .section .iram1
dl_tie728_s8_unaligned_avg_pool2d_hwc1:
    .align      4
    entry       sp,     16

    # a2: int16_t *output_ptr
    # a3: int16_t *input_ptr
    # a4: void *args

    l32i a5, a4, 16     # input_y_offset
    l32i a6, a4, 20     # input_x_offset
    l32i a7, a4, 4      # input_channel
    l32i a8, a4, 48     # filter_height
    l32i a9, a4, 52     # filter_width
    l32i a11, a4, 104   # c_div_x_1
    l32i a12, a4, 60    # c_remainder
    l32i a13, a4, 56    # shift

    addi a14, a4, 64
    EE.VLDBC.8 q6, a14 # avg_pool_area_inv

    srli a10, a9, 1
    addi a10, a10, -1   # filter_w / 2 - 1

    addi a6, a6, -16

    EE.LD.128.USAR.IP q7, a2, 0 #get output_ptr sar_byte
    rur.sar_byte a15

    addi a11, a11, 1

    beqi a9, 1, dl_tie728_s8_unaligned_avg_pool2d_h1c1 #filter_width == 1
    blti a11, 1, dl_tie728_s8_unaligned_avg_pool2d_hw_small_channel


    9:
        mov a7, a3
        mov a14, a7
        mov a12, a8
        EE.ZERO.QACC
        4:
	        EE.LD.128.USAR.IP q0, a14, 16
	        EE.LD.128.USAR.XP q1, a14, a6
	        loopgtz a10, 0f
	            EE.SRC.Q.LD.IP q2, a14, 16, q0, q1
	            EE.LD.128.USAR.XP q1, a14, a6
	            EE.VMULAS.S8.QACC q6, q0

	            EE.SRC.Q.LD.IP q0, a14, 16, q2, q1
	            EE.LD.128.USAR.XP q1, a14, a6
	            EE.VMULAS.S8.QACC q6, q2
	        0:

	        bbci a9, 0, 2f
	        1:#three left
	        EE.SRC.Q.LD.IP q2, a14, 16, q0, q1
            EE.LD.128.USAR.XP q1, a14, a6
            EE.VMULAS.S8.QACC q6, q0

            EE.SRC.Q.LD.IP q0, a14, 16, q2, q1
            EE.LD.128.USAR.XP q1, a14, a6
            EE.VMULAS.S8.QACC q6, q2

	        EE.SRC.Q q0, q0, q1
	        EE.VMULAS.S8.QACC q6, q0

	        j 3f

	        2:# two left
	        EE.SRC.Q.LD.IP q2, a14, 16, q0, q1
            EE.LD.128.USAR.XP q1, a14, a6
            EE.VMULAS.S8.QACC q6, q0

	        EE.SRC.Q q2, q2, q1
	        EE.VMULAS.S8.QACC q6, q2

	        3:
	        addi a12, a12, -1
	        add a7, a7, a5
	        mov a14, a7
	    bnez a12, 4b

	    # EE.SRCMB.S8.QACC q7, a13, 0
        tie728_s8_vector_round_result q7, a13, a14, q5

        beqi a15, 0, 5f
        beqi a15, 8, 6f

        dl_tie728_s8_unaligned_store0 q7, a2, a14
        j 7f

        5:
        EE.VST.128.IP q7, a2, 16
        j 7f
        6:
        dl_tie728_s8_unaligned_store1 q7, a2

        7:
        addi a3, a3, 16
        addi a11, a11, -1
    bnez a11, 9b

dl_tie728_s8_unaligned_avg_pool2d_hw_small_channel:
	l32i a12, a4, 60    # c_remainder
	beqz a12, dl_tie728_s8_unaligned_avg_pool2d_hw_small_channel_end

    mov a7, a3
    mov a14, a7
    mov a15, a8
    addi a6, a6, 16
    sub a6, a6, a12

    EE.ZERO.QACC
    1:
        loopgtz a9, 0f
            EE.LD.128.USAR.XP q0, a14, a12
            EE.VLD.128.XP q1, a14, a6
            EE.SRC.Q q0, q0, q1
            EE.VMULAS.S8.QACC q6, q0
        0:
        addi a15, a15, -1
        add a7, a7, a5
        mov a14, a7
    bnez a15, 1b

    # EE.SRCMB.S8.QACC q7, a13, 0
    tie728_s8_vector_round_result q7, a13, a14, q5
    dl_tie728_s8_store_remainder q7, a8, a9, a10, a11, a2, a12

dl_tie728_s8_unaligned_avg_pool2d_hw_small_channel_end:
    retw

dl_tie728_s8_unaligned_avg_pool2d_h1c1:
	addi a5, a5, -16
    blti a11, 1, dl_tie728_s8_unaligned_avg_pool2d_h1_remainder

    5:
        mov a14, a3
        EE.ZERO.QACC
        loopgtz a8, 0f
            EE.LD.128.USAR.IP q0, a14, 16
            EE.VLD.128.XP q1, a14, a5
            EE.SRC.Q q0, q0, q1
            EE.VMULAS.S8.QACC q6, q0
        0:

        # EE.SRCMB.S8.QACC q7, a13, 0
        tie728_s8_vector_round_result q7, a13, a14, q5
        beqi a15, 0, 1f
        beqi a15, 8, 2f

        dl_tie728_s8_unaligned_store0 q7, a2, a9
        j 3f
        1:
        EE.VST.128.IP q7, a2, 16
        j 3f
        2:
        dl_tie728_s8_unaligned_store1 q7, a2

        3:
        addi a3, a3, 16
        addi a11, a11, -1
    bnez a11, 5b

dl_tie728_s8_unaligned_avg_pool2d_h1_remainder:
	beqz a12, dl_tie728_s8_unaligned_avg_pool2d_hwc1_end

    mov a14, a3
    addi a5, a5, 16
    sub a5, a5, a12
    EE.ZERO.QACC
    loopgtz a8, 0f
        EE.LD.128.USAR.XP q0, a14, a12
        EE.VLD.128.XP q1, a14, a5
        EE.SRC.Q q0, q0, q1
        EE.VMULAS.S8.QACC q6, q0
    0:

    # EE.SRCMB.S8.QACC q7, a13, 0
    tie728_s8_vector_round_result q7, a13, a14, q5
    dl_tie728_s8_store_remainder q7, a8, a9, a10, a11, a2, a12

dl_tie728_s8_unaligned_avg_pool2d_hwc1_end:
	retw


############################################################################################################################################################
####
#### dl_tie728_s8_avg_pool2d_hwc_sum
####
#### Accurate SIMD variant: accumulate the raw (unquantized) sum of the pooling window into QACC
#### (16 lanes x 20-bit, signed saturated) and dump the raw QACC bits to the int32_t buffer.
#### The 1/area scaling and rounding are settled by the C code afterwards, so the result is
#### bit-identical to the C reference implementation.
####
#### Requirements (checked by the caller):
####   - input_channel % 16 == 0
####   - input_ptr is 16-byte aligned (input_y/x offsets are multiples of input_channel bytes)
####   - filter_height * filter_width <= 4095 (|sum| <= 128 * area <= 2^19 - 1, no QACC saturation)
############################################################################################################################################################

    .align 4
    .text
    .global dl_tie728_s8_avg_pool2d_hwc_sum
    .type	dl_tie728_s8_avg_pool2d_hwc_sum, @function
    .section .iram1
dl_tie728_s8_avg_pool2d_hwc_sum:
    .align      4
    entry       sp,     16

    # a2: int32_t *buffer_ptr
    # a3: int8_t *input_ptr
    # a4: void *args

    l32i a5, a4, 16     # input_y_offset
    l32i a6, a4, 20     # input_x_offset
    l32i a8, a4, 48     # filter_height
    l32i a9, a4, 52     # filter_width
    l32i a11, a4, 104   # c_div_x_1

    # ones vector for raw-sum accumulation (QACC += 1 * input)
    EE.ZERO.Q q3                # q3 = 0 (compare reference)
    EE.VCMP.EQ.S8 q0, q3, q3    # q0 = -1 in all 16 lanes
    EE.VSUBS.S8 q0, q3, q0      # q0 = 1 in all 16 lanes

    srli a10, a9, 1
    addi a10, a10, -1   # filter_w / 2 - 1

    addi a11, a11, 1    # number of 16-channel blocks

    beqi a9, 1, dl_tie728_s8_avg_pool2d_hwc_sum_w1

dl_tie728_s8_avg_pool2d_hwc_sum_block_loop:
    mov a7, a3          # row base
    mov a14, a7         # running pointer
    mov a15, a8         # filter_height counter
    EE.ZERO.QACC

dl_tie728_s8_avg_pool2d_hwc_sum_row_loop:
    EE.VLD.128.XP q1, a14, a6
    EE.VLD.128.XP q2, a14, a6
    loopgtz a10, 0f
        EE.VMULAS.S8.QACC.LD.XP q1, a14, a6, q0, q1
        EE.VMULAS.S8.QACC.LD.XP q2, a14, a6, q0, q2
    0:
    bbci a9, 0, 2f
    # filter_width odd: one element left
    EE.VMULAS.S8.QACC.LD.XP q1, a14, a6, q0, q1
    EE.VMULAS.S8.QACC q0, q2
    EE.VMULAS.S8.QACC q0, q1
    j 3f

2:  # filter_width even: two elements left
    EE.VMULAS.S8.QACC q0, q1
    EE.VMULAS.S8.QACC q0, q2
3:
    addi a15, a15, -1
    add a7, a7, a5
    mov a14, a7
    bnez a15, dl_tie728_s8_avg_pool2d_hwc_sum_row_loop

    # dump the 16 x 20-bit QACC lanes to buffer_ptr. Note: EE.ST.QACC_*.L.128.IP forces the
    # address to 16-byte alignment, so L.H.32 must post-increment by 16 to keep the H.L.128
    # store at a 16-byte boundary. Layout per 64-byte block:
    #   [0,16)  QACC_L[127:0]      [16,20) QACC_L[159:128]  [20,32) gap
    #   [32,48) QACC_H[127:0]      [48,52) QACC_H[159:128]  [52,64) unused
    EE.ST.QACC_L.L.128.IP a2, 16
    EE.ST.QACC_L.H.32.IP  a2, 16
    EE.ST.QACC_H.L.128.IP a2, 16
    EE.ST.QACC_H.H.32.IP  a2, 4
    addi a2, a2, 12     # 52 -> 64 bytes per 16 int32 slots

    addi a3, a3, 16     # next 16 input channels
    addi a11, a11, -1
    bnez a11, dl_tie728_s8_avg_pool2d_hwc_sum_block_loop
    j dl_tie728_s8_avg_pool2d_hwc_sum_end

dl_tie728_s8_avg_pool2d_hwc_sum_w1:
    # filter_width == 1: one element per row
    addi a8, a8, -1

dl_tie728_s8_avg_pool2d_hwc_sum_w1_block_loop:
    mov a14, a3
    EE.ZERO.QACC
    EE.VLD.128.XP q1, a14, a5
    loopgtz a8, 0f
        EE.VMULAS.S8.QACC.LD.XP q1, a14, a5, q0, q1
    0:
    EE.VMULAS.S8.QACC q0, q1

    EE.ST.QACC_L.L.128.IP a2, 16
    EE.ST.QACC_L.H.32.IP  a2, 16
    EE.ST.QACC_H.L.128.IP a2, 16
    EE.ST.QACC_H.H.32.IP  a2, 4
    addi a2, a2, 12

    addi a3, a3, 16
    addi a11, a11, -1
    bnez a11, dl_tie728_s8_avg_pool2d_hwc_sum_w1_block_loop

dl_tie728_s8_avg_pool2d_hwc_sum_end:
    retw
