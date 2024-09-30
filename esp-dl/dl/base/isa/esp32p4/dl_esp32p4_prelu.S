#include "dl_esp32p4_s8.S"
#include "dl_esp32p4_common.S"

    .align 2
    .text
    .global dl_esp32p4_s8_prelu_11c
    .type   dl_esp32p4_s8_prelu_11c, @function
dl_esp32p4_s8_prelu_11c:
    .align 2
    esp32p4_push_128_stacks_4r  s0, s1, s8, s9

    # a0: int8_t *output_ptr  
    # a1: int8_t *input_ptr
    # a2: void *args
    # a3: c_div_x_1
    # s8: activation_alpha_ptr
    # s9: activation_shift
    # t3: output_shift
    # t4: output_scale


    lw a3, 100(a2)
    lw s8, 80(a2)  # activation_alpha_ptr
    lw s9, 84(a2)  # activation_shift
    lw t3, 172(a2) # output_shift
    lw t4, 176(a2) # output_scale

    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, s8, 16
    sb x0, 0(sp)
    add s0, sp, x0
    esp.vldbc.8.ip q2, s0, 0 # all 0
    sb t4, 0(sp)
    add s0, sp, x0
    esp.vldbc.8.ip q3, s0, 0 # all output_scale

    add t0, a3, x0
    blez t0, 1f
    0:
        esp.vcmp.gt.s8 q4, q0, q2
        esp.notq q5, q4

        esp.vprelu.s8 q1, q0, q1, s9

        # *scale/right shift: output - input
        esp.zero.qacc
        esp.vmulas.s8.qacc q0, q3
        esp.srcmb.s8.qacc q0, t3, 1
        esp.andq q0, q0, q4
        esp.andq q1, q1, q5
        esp.vadd.s8.ld.incp q1, s8, q0, q0, q1

        esp.vst.128.ip q0, a0, 16
        esp.vld.128.ip q0, a1, 16
        addi t0, t0, -1
        bgtz t0, 0b
    1:
    
    esp.vcmp.gt.s8 q4, q0, q2
    esp.notq q5, q4
    
    esp.vprelu.s8 q1, q0, q1, s9

    # *scale/right shift: output - input
    esp.zero.qacc
    esp.vmulas.s8.qacc q0, q3
    esp.srcmb.s8.qacc q0, t3, 1
    esp.andq q0, q0, q4
    esp.andq q1, q1, q5
    esp.vadd.s8 q0, q0, q1

    esp.vst.128.ip q0, a0, 0

    esp32p4_pop_128_stacks_4r  s0, s1, s8, s9
    ret
