#include "dl_esp32p4_s16.S"
#include "dl_esp32p4_common.S"


#void dl_esp32p4_s16_sub4d_bchw_w1_8_w2_8_simdsub(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int lenght);

    .align 2
    .text
    .global dl_esp32p4_s16_sub4d_bchw_w1_8_w2_8_simdsub
    .type   dl_esp32p4_s16_sub4d_bchw_w1_8_w2_8_simdsub, @function
    #.section .iram1
dl_esp32p4_s16_sub4d_bchw_w1_8_w2_8_simdsub:
    .align 2

    # a0: int16_t *output_ptr
    # a1: int16_t *input0_ptr
    # a2: int16_t *input1_ptr
    # a3: length

    srai a3, a3, 3

    li t0, 0            
loop:
    beq t0, a3, end    
    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a2, 16
    esp.vsub.s16 q2, q0, q1
    esp.vst.128.ip q2, a0, 16
    addi t0, t0, 1      
    j loop              
end:
    ret

    
#void dl_esp32p4_s16_sub4d_bchw_w1_8_w2_1_simdsub(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int lenght);

    .align 2
    .text
    .global dl_esp32p4_s16_sub4d_bchw_w1_8_w2_1_simdsub
    .type   dl_esp32p4_s16_sub4d_bchw_w1_8_w2_1_simdsub, @function
    #.section .iram1
dl_esp32p4_s16_sub4d_bchw_w1_8_w2_1_simdsub:
    .align 2

    # a0: int16_t *output_ptr
    # a1: int16_t *input0_ptr
    # a2: int16_t *input1_ptr
    # a3: length

    srai a3, a3, 3

    li t0, 0            
loop_:
    beq t0, a3, end_    
    esp.vld.128.ip q0, a1, 16
    esp.vldbc.16.ip q1, a2, 0
    esp.vsub.s16 q2, q0, q1
    esp.vst.128.ip q2, a0, 16
    addi t0, t0, 1      
    j loop_              
end_:
    ret

#void dl_esp32p4_s16_sub4d_bchw_w1_1_w2_8_simdsub(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int lenght);

    .align 2
    .text
    .global dl_esp32p4_s16_sub4d_bchw_w1_1_w2_8_simdsub
    .type   dl_esp32p4_s16_sub4d_bchw_w1_1_w2_8_simdsub, @function
    #.section .iram1
dl_esp32p4_s16_sub4d_bchw_w1_1_w2_8_simdsub:
    .align 2

    # a0: int16_t *output_ptr
    # a1: int16_t *input0_ptr
    # a2: int16_t *input1_ptr
    # a3: length

    srai a3, a3, 3

    li t0, 0            
loop__:
    beq t0, a3, end__    
    esp.vldbc.16.ip q0, a1, 0
    esp.vld.128.ip q1, a2, 16
    esp.vsub.s16 q2, q0, q1
    esp.vst.128.ip q2, a0, 16
    addi t0, t0, 1      
    j loop__              
end__:
    ret
