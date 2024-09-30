#include "dl_tie728_s8.S"

    .align 4
    .text
    .global dl_tie728_s8_prelu_11c
    .type   dl_tie728_s8_prelu_11c, @function
    .section .iram1
dl_tie728_s8_prelu_11c:
    .align 4
    entry sp, 24

    # a2: int8_t *output_ptr  
    # a3: int8_t *input_ptr
    # a4: void *args
    # a5: c_div_x_1
    # a12: activation_alpha_ptr
    # a13: activation_shift
    # a14: output_shift
    # a15: output_scale


    l32i a5, a4, 100
    l32i a12, a4, 80  # activation_alpha_ptr
    l32i a13, a4, 84  # activation_shift 
    l32i a14, a4, 172 # output_shift
    l32i a15, a4, 176 # output_scale

    ee.vld.128.ip q0, a3, 16
    ee.vld.128.ip q1, a12, 16
    movi a6, 0
    s8i a6, sp, 0
    ee.vldbc.8.ip q2, sp, 0 # all 0
    s8i a15, sp, 0
    ee.vldbc.8.ip q3, sp, 0 # all output_scale


    loopgtz a5, 0f
        ee.vcmp.gt.s8 q4, q0, q2
        ee.notq q5, q4
        
        ee.zero.qacc
        # alpha * input
        ee.vmulas.s8.qacc q0, q1
        # right shift: output - alpha - input
        tie728_s8_vector_round_result q1, a13, a6, q6

        # *scale/right shift: output - input
        ee.zero.qacc
        ee.vmulas.s8.qacc q0, q3
        tie728_s8_vector_round_result q0, a14, a6, q6
        ee.andq q0, q0, q4
        ee.andq q1, q1, q5
        ee.vadds.s8.ld.incp q1, a12, q0, q0, q1

        ee.vst.128.ip q0, a2, 16
        ee.vld.128.ip q0, a3, 16
    0:

    ee.vcmp.gt.s8 q4, q0, q2
    ee.notq q5, q4
    
    ee.zero.qacc
    # alpha * input
    ee.vmulas.s8.qacc q0, q1
    # right shift: output - alpha - input
    tie728_s8_vector_round_result q1, a13, a6, q6

    # *scale/right shift: output - input
    ee.zero.qacc
    ee.vmulas.s8.qacc q0, q3
    tie728_s8_vector_round_result q0, a14, a6, q6
    ee.andq q0, q0, q4
    ee.andq q1, q1, q5
    ee.vadds.s8 q0, q0, q1

    ee.vst.128.ip q0, a2, 0

    retw
