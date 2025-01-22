#include "dl_tie728_s16.S"


    .text
    .align 4
    .global dl_tie728_s16_s16_requantize_linear
    .type   dl_tie728_s16_s16_requantize_linear, @function
dl_tie728_s16_s16_requantize_linear:

    # a2: int16_t *output_ptr
    # a3: int16_t *input_ptr
    # a4: void *args

    # a5: size_div_x
    # a6: in_size_remainder
    # a7: tmp value
    # a8: output_shift
    # a9: output_scale
    # a10:
    # a11:

    # a12(not for extension instructions):
    # a13(not for extension instructions):
    # a14(not for extension instructions): tmp value
    # a15(not for extension instructions):

    entry  sp, 32

    l32i  a5, a4, 8       // size_div_x
    l32i  a6, a4, 12      // in_size_remainder
    l32i  a8, a4, 20      // output_shift
    l32i  a9, a4, 24      // output_scale

    bgez  a9, tie728_s16_s16_requantize_linear_left_shift
    beqz  a5, tie728_s16_s16_requantize_linear_right_shift_remainder

        tie728_s16_s16_requantize_linear_right_shift_loop:
            ee.ldqa.s16.128.ip  a3, 16
            addi  a5, a5, -1
            tie728_s16_vector_round_result  q0, a8, a14, q3
            ee.vst.128.ip  q0, a2, 16
            bnez  a5, tie728_s16_s16_requantize_linear_right_shift_loop

        tie728_s16_s16_requantize_linear_right_shift_remainder:
            beqz  a6, tie728_s16_s16_requantize_linear_end
            ee.ldqa.s16.128.xp  a3, a6
            srli  a6, a6, 1
            tie728_s16_vector_round_result  q0, a8, a14, q3
            dl_tie728_s16_store_remainder  q0, a6, a7, a2
            j  tie728_s16_s16_requantize_linear_end

    tie728_s16_s16_requantize_linear_left_shift:

        addi  a9, a4, 24
        ee.vldbc.16.ip  q1, a9, 0      // load output_scale
        beqz  a5, tie728_s16_s16_requantize_linear_left_shift_remainder

        tie728_s16_s16_requantize_linear_left_shift_loop:
            ee.vld.128.ip  q0, a3, 16
            ee.zero.qacc
            ee.vmulas.s16.qacc  q0, q1
            tie728_s16_vector_round_result  q0, a8, a14, q3
            ee.vst.128.ip  q0, a2, 16
            addi  a5, a5, -1
            bnez  a5, tie728_s16_s16_requantize_linear_left_shift_loop

        tie728_s16_s16_requantize_linear_left_shift_remainder:
            beqz  a6, tie728_s16_s16_requantize_linear_end
            ee.vld.128.ip  q0, a3, 16
            ee.zero.qacc
            ee.vmulas.s16.qacc  q0, q1
            tie728_s16_vector_round_result  q0, a8, a14, q3
            srli  a6, a6, 1
            dl_tie728_s16_store_remainder  q0, a6, a7, a2

    tie728_s16_s16_requantize_linear_end:
    retw



    .text
    .align 4
    .global dl_tie728_s16_s8_requantize_linear
    .type   dl_tie728_s16_s8_requantize_linear, @function
dl_tie728_s16_s8_requantize_linear:

    # a2: int16_t *output_ptr
    # a3: int8_t *input_ptr
    # a4: void *args

    # a5: size_div_x
    # a6: in_size_remainder
    # a7: out_size_remainder
    # a8: output_shift
    # a9: output_scale
    # a10:
    # a11:

    # a12(not for extension instructions): tmp value
    # a13(not for extension instructions):
    # a14(not for extension instructions): tmp value
    # a15(not for extension instructions):

    entry  sp, 32

    l32i  a5, a4, 8       // size_div_x
    l32i  a6, a4, 12      // in_size_remainder
    l32i  a7, a4, 16      // out_size_remainder
    l32i  a8, a4, 20      // output_shift
    l32i  a9, a4, 24      // output_scale

    bgez  a9, tie728_s16_s8_requantize_linear_left_shift
    beqz  a5, tie728_s16_s8_requantize_linear_right_shift_remainder

        srli  a12, a5, 1
        tie728_s16_s8_requantize_linear_right_shift_loop:
            ee.vld.128.ip  q0, a3, 16
            ee.zero.q  q1
            ee.vcmp.lt.s8  q1, q0, q1
            ee.vzip.8  q0, q1
            addi  a12, a12, -1
            ee.mov.s16.qacc  q0
            tie728_s16_vector_round_result  q0, a8, a14, q3
            ee.vst.128.ip  q0, a2, 16
            ee.mov.s16.qacc  q1
            tie728_s16_vector_round_result  q1, a8, a14, q3
            ee.vst.128.ip  q1, a2, 16
            bnez  a12, tie728_s16_s8_requantize_linear_right_shift_loop

        tie728_s16_s8_requantize_linear_right_shift_remainder:
            beqz  a6, tie728_s16_s8_requantize_linear_end
            movi  a14, 9
            blt  a6, a14, tie728_s16_s8_requantize_linear_right_shift_remainder_le8
                ee.vld.128.xp  q0, a3, a6
                ee.zero.q  q1
                ee.vcmp.lt.s8  q1, q0, q1
                ee.vzip.8  q0, q1
                srli  a7, a7, 1
                ee.mov.s16.qacc  q0
                tie728_s16_vector_round_result  q0, a8, a14, q3
                ee.vst.128.ip  q0, a2, 16
                ee.mov.s16.qacc  q1
                tie728_s16_vector_round_result  q1, a8, a14, q3
                dl_tie728_s16_store_remainder  q1, a7, a6, a2
                j  tie728_s16_s8_requantize_linear_end

            tie728_s16_s8_requantize_linear_right_shift_remainder_le8:
                ee.vld.128.xp  q0, a3, a6
                ee.zero.q  q1
                ee.vcmp.lt.s8  q1, q0, q1
                ee.vzip.8  q0, q1
                ee.mov.s16.qacc  q0
                tie728_s16_vector_round_result  q0, a8, a14, q3
                dl_tie728_s16_store_remainder  q0, a6, a7, a2
                j  tie728_s16_s8_requantize_linear_end

    tie728_s16_s8_requantize_linear_left_shift:

        addi  a9, a4, 24
        ee.vldbc.16.ip  q2, a9, 0      // load output_scale
        beqz  a5, tie728_s16_s8_requantize_linear_left_shift_remainder

        srli  a12, a5, 1
        tie728_s16_s8_requantize_linear_left_shift_loop:
            ee.vld.128.ip  q0, a3, 16
            ee.zero.q  q1
            ee.vcmp.lt.s8  q1, q0, q1
            ee.vzip.8  q0, q1
            ee.zero.qacc
            ee.vmulas.s16.qacc  q0, q2
            tie728_s16_vector_round_result  q0, a8, a14, q3
            ee.vst.128.ip  q0, a2, 16
            ee.zero.qacc
            ee.vmulas.s16.qacc  q1, q2
            tie728_s16_vector_round_result  q1, a8, a14, q3
            ee.vst.128.ip  q1, a2, 16
            addi  a12, a12, -1
            bnez  a12, tie728_s16_s8_requantize_linear_left_shift_loop

        tie728_s16_s8_requantize_linear_left_shift_remainder:
            beqz  a6, tie728_s16_s8_requantize_linear_end
            movi  a14, 9
            blt  a6, a14, tie728_s16_s8_requantize_linear_left_shift_remainder_le8
                srli  a7, a7, 1
                ee.vld.128.xp  q0, a3, a6
                ee.zero.q  q1
                ee.vcmp.lt.s8  q1, q0, q1
                ee.vzip.8  q0, q1
                ee.zero.qacc
                ee.vmulas.s16.qacc  q0, q2
                tie728_s16_vector_round_result  q0, a8, a14, q3
                ee.vst.128.ip  q0, a2, 16
                ee.zero.qacc
                ee.vmulas.s16.qacc  q1, q2
                tie728_s16_vector_round_result  q1, a8, a14, q3
                dl_tie728_s16_store_remainder  q1, a7, a6, a2
                j  tie728_s16_s8_requantize_linear_end

            tie728_s16_s8_requantize_linear_left_shift_remainder_le8:
                ee.vld.128.xp  q0, a3, a6
                ee.zero.q  q1
                ee.vcmp.lt.s8  q1, q0, q1
                ee.vzip.8  q0, q1
                ee.zero.qacc
                ee.vmulas.s16.qacc  q0, q2
                tie728_s16_vector_round_result  q0, a8, a14, q3
                dl_tie728_s16_store_remainder  q0, a6, a7, a2

    tie728_s16_s8_requantize_linear_end:
    retw
