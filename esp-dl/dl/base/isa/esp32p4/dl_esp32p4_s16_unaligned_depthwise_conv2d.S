#include "dl_esp32p4_s16.S"
#include "dl_esp32p4_common.S"


############################################################################################################################################################
#### 
#### esp32p4_s16_unaligned_depthwise_conv2d_33c1 series
#### 
############################################################################################################################################################
.macro esp32p4_s16_unaligned_depthwise_conv2d_3381  input_v0, input_v1, input_v2, input_back, filter_v0, filter_v1, filter_v2, input_ptr, filter_ptr, dilation_x_offset_16, dilation_y_offset_16, next_3381_16
    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_y_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v1,  \filter_ptr, 16,                \input_v0, \filter_v0
    esp.src.q.ld.ip           \input_v0,   \input_ptr,  16,                \input_v2, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_x_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v2,  \filter_ptr, 16,                \input_v1, \filter_v1
    esp.src.q.ld.ip           \input_v1,   \input_ptr,  16,                \input_v0, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_x_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v0,  \filter_ptr, 16,                \input_v2, \filter_v2
    esp.src.q.ld.ip           \input_v2,   \input_ptr,  16,                \input_v1, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_y_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v1,  \filter_ptr, 16,                \input_v0, \filter_v0
    esp.src.q.ld.ip           \input_v0,   \input_ptr,  16,                \input_v2, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_x_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v2,  \filter_ptr, 16,                \input_v1, \filter_v1
    esp.src.q.ld.ip           \input_v1,   \input_ptr,  16,                \input_v0, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_x_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v0,  \filter_ptr, 16,                \input_v2, \filter_v2
    esp.src.q.ld.ip           \input_v2,   \input_ptr,  16,                \input_v1, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \next_3381_16
    esp.vmulas.s16.qacc.ld.ip \filter_v1,  \filter_ptr, 16,                \input_v0, \filter_v0
    esp.src.q.ld.ip           \input_v0,   \input_ptr,  16,                \input_v2, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_x_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v2,  \filter_ptr, 16,                \input_v1, \filter_v1
    esp.src.q.ld.ip           \input_v1,   \input_ptr,  16,                \input_v0, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_x_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v0,  \filter_ptr, 16,                \input_v2, \filter_v2
    esp.src.q.ld.ip           \input_v2,   \input_ptr,  16,                \input_v1, \input_back
.endm



.macro esp32p4_s16_unaligned_depthwise_conv2d_3381_last  input_v0, input_v1, input_v2, input_back, filter_v0, filter_v1, input_ptr, filter_ptr, dilation_x_offset_16, dilation_y_offset_16, next_3381_16
    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_y_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v1,  \filter_ptr, 16,                \input_v0, \filter_v0
    esp.src.q.ld.ip           \input_v0,   \input_ptr,  16,                \input_v2, \input_back
    
    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_x_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v0,  \filter_ptr, 16,                \input_v1, \filter_v1
    esp.src.q.ld.ip           \input_v1,   \input_ptr,  16,                \input_v0, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_x_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v1,  \filter_ptr, 16,                \input_v2, \filter_v0
    esp.src.q.ld.ip           \input_v2,   \input_ptr,  16,                \input_v1, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_y_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v0,  \filter_ptr, 16,                \input_v0, \filter_v1
    esp.src.q.ld.ip           \input_v0,   \input_ptr,  16,                \input_v2, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_x_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v1,  \filter_ptr, 16,                \input_v1, \filter_v0
    esp.src.q.ld.ip           \input_v1,   \input_ptr,  16,                \input_v0, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \dilation_x_offset_16
    esp.vmulas.s16.qacc.ld.ip \filter_v0,  \filter_ptr, 16,                \input_v2, \filter_v1
    esp.src.q.ld.ip           \input_v2,   \input_ptr,  16,                \input_v1, \input_back

    esp.ld.128.usar.xp        \input_back, \input_ptr,  \next_3381_16
    esp.vmulas.s16.qacc.ld.ip \filter_v1,  \filter_ptr, 16,                \input_v0, \filter_v0
    esp.src.q                 \input_v2,   \input_v2,   \input_back

    esp.vmulas.s16.qacc.ld.ip \filter_v0,  \filter_ptr, 16,                \input_v1, \filter_v1
    esp.vmulas.s16.qacc       \input_v2,   \filter_v0
.endm



.macro esp32p4_s16_unaligned_depthwise_conv2d_11r1  input_v0, input_front, input_back, filter_v0, filter_front, filter_back, input_ptr, filter_ptr, c_remainder, forward
    esp.ld.128.usar.ip  \input_v0,   \input_ptr, 16
    esp.vld.128.xp      \input_back, \input_ptr, \forward
    esp.src.q           \input_v0,   \input_v0,  \input_back

    esp.ld.128.usar.xp  \filter_v0,   \filter_ptr, \c_remainder
    esp.vld.128.ip      \filter_back, \filter_ptr, 0
    esp.src.q           \filter_v0,   \filter_v0,  \filter_back
    
    esp.vmulas.s16.qacc \input_v0,    \filter_v0
.endm



.macro esp32p4_s16_unaligned_depthwise_conv2d_33r1  input_v0, input_front, input_back, filter_v0, filter_front, filter_back, input_ptr, filter_ptr, dilation_x_offset_16, dilation_y_offset_16, c_remainder  
    esp32p4_s16_unaligned_depthwise_conv2d_11r1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_remainder, \dilation_x_offset_16
    esp32p4_s16_unaligned_depthwise_conv2d_11r1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_remainder, \dilation_x_offset_16
    esp32p4_s16_unaligned_depthwise_conv2d_11r1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_remainder, \dilation_y_offset_16

    esp32p4_s16_unaligned_depthwise_conv2d_11r1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_remainder, \dilation_x_offset_16
    esp32p4_s16_unaligned_depthwise_conv2d_11r1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_remainder, \dilation_x_offset_16
    esp32p4_s16_unaligned_depthwise_conv2d_11r1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_remainder, \dilation_y_offset_16

    esp32p4_s16_unaligned_depthwise_conv2d_11r1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_remainder, \dilation_x_offset_16
    esp32p4_s16_unaligned_depthwise_conv2d_11r1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_remainder, \dilation_x_offset_16
    esp32p4_s16_unaligned_depthwise_conv2d_11r1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_remainder, \dilation_y_offset_16
.endm



.macro esp32p4_s16_unaligned_depthwise_conv2d_33c1_load_args  args, filter_ptr, dilation_x_offset, dilation_y_offset, next_hwx1, c_div_x_1, mac_shift
    # dilation_x_offset = input_channel_with_padding * dilation_x * sizeof(T)
    # dilation_y_offset = (-(filter_width - 1) * dilation_x * input_channel_with_padding + dilation_y * input_width_with_padding * input_channel_with_padding) * sizeof(T)
    # next_hwx1 = (-(filter_width - 1) * dilation_x * input_channel_with_padding - (filter_height - 1) * dilation_y * input_width_with_padding * input_channel_with_padding) * sizeof(T) + 16

    lw  \filter_ptr,         48(\args)
    lw  \dilation_x_offset, 124(\args)
    lw  \dilation_y_offset, 128(\args)
    lw  \next_hwx1,         132(\args)
    lw  \c_div_x_1,         100(\args)
    lw  \mac_shift,         64 (\args)
.endm



    .text
    .align 2
    .global dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias
    .type   dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: dilation_x_offset - 16
    # a5: dilation_y_offset - 16
    # t3: next_3381 - 16
    # t4: mac_shift
    # t5: c_div_x_1 / c_remainder
    # t6: output_sar_byte / tmp value

    # a6(not for extension instructions): tmp value
    # a7(not for extension instructions): 
    # t0(not for extension instructions): tmp value
    # t1(not for extension instructions): 
    # t2(not for extension instructions): 
    # s2(not for extension instructions): 
    # s3(not for extension instructions): 
    # s4(not for extension instructions): 
    # s5(not for extension instructions): 

    # s0: 
    # s1: 
    # s8: bias_ptr
    # s9: 
    # s10: 
    # s11:

    esp32p4_push_4_stacks_1r  s8

    esp32p4_s16_unaligned_depthwise_conv2d_33c1_load_args  a2, a3, a4, a5, t3, t5, t4
    lw  s8, 68(a2)       // bias

    addi  a4, a4, -16                           // a4: dilation_x_offset - 16
    addi  a5, a5, -16                           // a5: dilation_y_offset - 16
    addi  t3, t3, -16                           // t3: next_3381 - 16

    bltz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_c_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t6                // t6: output_sar_byte

        esp.ld.128.usar.ip  q0, a1, 16
        esp.ld.128.usar.xp  q3, a1, a4
        esp.src.q.ld.ip  q4, a3, 16, q0, q3     // q4: filter_v0; q0: input_v0

        esp.ld.128.usar.ip  q1, a1, 16
        esp.ld.128.usar.xp  q3, a1, a4
        esp.src.q.ld.ip  q2, a1, 16, q1, q3     // q2: input_v2; q1: input_v1
    
        beqz  t6, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_128b
            li  a6, 8
            beq  t6, a6, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_64b
                # esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_32b:
                beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_32b_last
                    esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_32b_loop_end
                        esp.zero.qacc
                        esp32p4_s16_conv2d_128b_vector_bias  s8
                        esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                        esp32p4_s16_128b_vector_shift_result  q3, t4
                        esp32p4_s16_32b_unaligned_vector_store  q3, a0, t6
                    esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_32b_loop_end:
                        nop
                
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_32b_last:
                    esp.zero.qacc
                    esp32p4_s16_conv2d_128b_vector_bias  s8
                    esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                    esp32p4_s16_32b_unaligned_vector_store  q3, a0, t6

                j  esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_c_remainder

            esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_64b:
                beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_64b_last
                    esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_64b_loop_end
                        esp.zero.qacc
                        esp32p4_s16_conv2d_128b_vector_bias  s8
                        esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                        esp32p4_s16_128b_vector_shift_result  q3, t4
                        esp32p4_s16_64b_unaligned_vector_store  q3, a0
                    esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_64b_loop_end:
                        nop
                
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_64b_last:
                    esp.zero.qacc
                    esp32p4_s16_conv2d_128b_vector_bias  s8
                    esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                    esp32p4_s16_64b_unaligned_vector_store  q3, a0

                j  esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_c_remainder

        esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_128b:
            beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_128b_last
                esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_128b_loop_end
                    esp.zero.qacc
                    esp32p4_s16_conv2d_128b_vector_bias  s8
                    esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_128b_loop_end:
                    esp32p4_s16_128b_aligned_vector_store  q3, a0
            
            esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_128b_last:
                esp.zero.qacc
                esp32p4_s16_conv2d_128b_vector_bias  s8
                esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                esp32p4_s16_128b_vector_shift_result  q3, t4
                esp32p4_s16_128b_aligned_vector_store  q3, a0


    esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_c_remainder:
        lw  t5, 136(a2)                         // t5: c_remainder
        beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_c_remainder_end
            esp.zero.qacc

            srli  t0, t5, 1
            esp32p4_s16_conv2d_128b_vector_bias  s8
            esp32p4_s16_unaligned_depthwise_conv2d_33r1  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t5
            esp32p4_s16_128b_vector_shift_result  q0, t4
            dl_esp32p4_s16_store_remainder  q0, t0, t6, a0
        
        esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_c_remainder_end:

    esp32p4_pop_4_stacks_1r  s8
    ret



    .text
    .align 2
    .global dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu
    .type   dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: dilation_x_offset - 16
    # a5: dilation_y_offset - 16
    # t3: next_3381 - 16
    # t4: mac_shift
    # t5: c_div_x_1 / c_remainder
    # t6: output_sar_byte / tmp value

    # a6(not for extension instructions): tmp value
    # a7(not for extension instructions): 
    # t0(not for extension instructions): tmp value
    # t1(not for extension instructions): 
    # t2(not for extension instructions): 
    # s2(not for extension instructions): 
    # s3(not for extension instructions): 
    # s4(not for extension instructions): 
    # s5(not for extension instructions): 

    # s0: activation_alpha/_address
    # s1: activation_shift
    # s8: bias_ptr
    # s9: 
    # s10: 
    # s11:

    esp32p4_push_12_stacks_3r  s0, s1, s8

    esp32p4_s16_unaligned_depthwise_conv2d_33c1_load_args  a2, a3, a4, a5, t3, t5, t4
    lw  s8, 68(a2)       // bias
    lw  s0, 76(a2)       // activation_alpha
    lw  s1, 84(a2)       // activation_shift

    addi  a4, a4, -16                           // a4: dilation_x_offset - 16
    addi  a5, a5, -16                           // a5: dilation_y_offset - 16
    addi  t3, t3, -16                           // t3: next_3381 - 16

    bltz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_c_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t6                // t6: output_sar_byte

        esp.ld.128.usar.ip  q0, a1, 16
        esp.ld.128.usar.xp  q3, a1, a4
        esp.src.q.ld.ip  q4, a3, 16, q0, q3     // q4: filter_v0; q0: input_v0

        esp.ld.128.usar.ip  q1, a1, 16
        esp.ld.128.usar.xp  q3, a1, a4
        esp.src.q.ld.ip  q2, a1, 16, q1, q3     // q2: input_v2; q1: input_v1
    
        beqz  t6, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_128b
            li  a6, 8
            beq  t6, a6, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_64b
                # esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_32b:
                beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_32b_last
                    esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_32b_loop_end
                        esp.zero.qacc
                        esp32p4_s16_conv2d_128b_vector_bias  s8
                        esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                        esp32p4_s16_128b_vector_shift_result  q3, t4
                        esp32p4_s16_128b_vector_relu  q3, s0, s1
                        esp32p4_s16_32b_unaligned_vector_store  q3, a0, t6
                    esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_32b_loop_end:
                        nop
                
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_32b_last:
                    esp.zero.qacc
                    esp32p4_s16_conv2d_128b_vector_bias  s8
                    esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                    esp32p4_s16_128b_vector_relu  q3, s0, s1
                    esp32p4_s16_32b_unaligned_vector_store  q3, a0, t6

                j  esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_c_remainder

            esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_64b:
                beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_64b_last
                    esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_64b_loop_end
                        esp.zero.qacc
                        esp32p4_s16_conv2d_128b_vector_bias  s8
                        esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                        esp32p4_s16_128b_vector_shift_result  q3, t4
                        esp32p4_s16_128b_vector_relu  q3, s0, s1
                        esp32p4_s16_64b_unaligned_vector_store  q3, a0
                    esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_64b_loop_end:
                        nop
                
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_64b_last:
                    esp.zero.qacc
                    esp32p4_s16_conv2d_128b_vector_bias  s8
                    esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                    esp32p4_s16_128b_vector_relu  q3, s0, s1
                    esp32p4_s16_64b_unaligned_vector_store  q3, a0

                j  esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_c_remainder

        esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_128b:
            beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_128b_last
                esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_128b_loop_end
                    esp.zero.qacc
                    esp32p4_s16_conv2d_128b_vector_bias  s8
                    esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                    esp32p4_s16_128b_vector_relu  q3, s0, s1
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_128b_loop_end:
                    esp32p4_s16_128b_aligned_vector_store  q3, a0
            
            esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_128b_last:
                esp.zero.qacc
                esp32p4_s16_conv2d_128b_vector_bias  s8
                esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                esp32p4_s16_128b_vector_shift_result  q3, t4
                esp32p4_s16_128b_vector_relu  q3, s0, s1
                esp32p4_s16_128b_aligned_vector_store  q3, a0


    esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_c_remainder:
        lw  t5, 136(a2)                         // t5: c_remainder
        beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_c_remainder_end
            esp.zero.qacc

            srli  t0, t5, 1
            esp32p4_s16_conv2d_128b_vector_bias  s8
            esp32p4_s16_unaligned_depthwise_conv2d_33r1  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t5
            esp32p4_s16_128b_vector_shift_result  q0, t4
            esp32p4_s16_128b_vector_relu  q0, s0, s1
            dl_esp32p4_s16_store_remainder  q0, t0, t6, a0
        
        esp32p4_s16_unaligned_depthwise_conv2d_33c1_bias_relu_c_remainder_end:

    esp32p4_pop_12_stacks_3r  s0, s1, s8
    ret



    .text
    .align 2
    .global dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1
    .type   dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: dilation_x_offset - 16
    # a5: dilation_y_offset - 16
    # t3: next_3381 - 16
    # t4: mac_shift
    # t5: c_div_x_1 / c_remainder
    # t6: output_sar_byte / tmp value

    # a6(not for extension instructions): tmp value
    # a7(not for extension instructions): 
    # t0(not for extension instructions): tmp value
    # t1(not for extension instructions): 
    # t2(not for extension instructions): 
    # s2(not for extension instructions): 
    # s3(not for extension instructions): 
    # s4(not for extension instructions): 
    # s5(not for extension instructions): 

    # s0: 
    # s1: 
    # s8: 
    # s9: 
    # s10: 
    # s11:

    esp32p4_s16_unaligned_depthwise_conv2d_33c1_load_args  a2, a3, a4, a5, t3, t5, t4

    addi  a4, a4, -16                           // a4: dilation_x_offset - 16
    addi  a5, a5, -16                           // a5: dilation_y_offset - 16
    addi  t3, t3, -16                           // t3: next_3381 - 16

    bltz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_c_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t6                // t6: output_sar_byte

        esp.ld.128.usar.ip  q0, a1, 16
        esp.ld.128.usar.xp  q3, a1, a4
        esp.src.q.ld.ip  q4, a3, 16, q0, q3     // q4: filter_v0; q0: input_v0

        esp.ld.128.usar.ip  q1, a1, 16
        esp.ld.128.usar.xp  q3, a1, a4
        esp.src.q.ld.ip  q2, a1, 16, q1, q3     // q2: input_v2; q1: input_v1
    
        beqz  t6, esp32p4_s16_unaligned_depthwise_conv2d_33c1_128b
            li  a6, 8
            beq  t6, a6, esp32p4_s16_unaligned_depthwise_conv2d_33c1_64b
                # esp32p4_s16_unaligned_depthwise_conv2d_33c1_32b:
                beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_32b_last
                    esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_32b_loop_end
                        esp.zero.qacc
                        esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                        esp32p4_s16_128b_vector_shift_result  q3, t4
                        esp32p4_s16_32b_unaligned_vector_store  q3, a0, t6
                    esp32p4_s16_unaligned_depthwise_conv2d_33c1_32b_loop_end:
                        nop
                
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_32b_last:
                    esp.zero.qacc
                    esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                    esp32p4_s16_32b_unaligned_vector_store  q3, a0, t6

                j  esp32p4_s16_unaligned_depthwise_conv2d_33c1_c_remainder

            esp32p4_s16_unaligned_depthwise_conv2d_33c1_64b:
                beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_64b_last
                    esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_64b_loop_end
                        esp.zero.qacc
                        esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                        esp32p4_s16_128b_vector_shift_result  q3, t4
                        esp32p4_s16_64b_unaligned_vector_store  q3, a0
                    esp32p4_s16_unaligned_depthwise_conv2d_33c1_64b_loop_end:
                        nop
                
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_64b_last:
                    esp.zero.qacc
                    esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                    esp32p4_s16_64b_unaligned_vector_store  q3, a0

                j  esp32p4_s16_unaligned_depthwise_conv2d_33c1_c_remainder

        esp32p4_s16_unaligned_depthwise_conv2d_33c1_128b:
            beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_128b_last
                esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_128b_loop_end
                    esp.zero.qacc
                    esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_128b_loop_end:
                    esp32p4_s16_128b_aligned_vector_store  q3, a0
            
            esp32p4_s16_unaligned_depthwise_conv2d_33c1_128b_last:
                esp.zero.qacc
                esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                esp32p4_s16_128b_vector_shift_result  q3, t4
                esp32p4_s16_128b_aligned_vector_store  q3, a0


    esp32p4_s16_unaligned_depthwise_conv2d_33c1_c_remainder:
        lw  t5, 136(a2)                         // t5: c_remainder
        beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_c_remainder_end
            esp.zero.qacc

            srli  t0, t5, 1
            esp32p4_s16_unaligned_depthwise_conv2d_33r1  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t5
            esp32p4_s16_128b_vector_shift_result  q0, t4
            dl_esp32p4_s16_store_remainder  q0, t0, t6, a0
        
        esp32p4_s16_unaligned_depthwise_conv2d_33c1_c_remainder_end:
    
    ret



    .text
    .align 2
    .global dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu
    .type   dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: dilation_x_offset - 16
    # a5: dilation_y_offset - 16
    # t3: next_3381 - 16
    # t4: mac_shift
    # t5: c_div_x_1 / c_remainder
    # t6: output_sar_byte / tmp value

    # a6(not for extension instructions): tmp value
    # a7(not for extension instructions): 
    # t0(not for extension instructions): tmp value
    # t1(not for extension instructions): 
    # t2(not for extension instructions): 
    # s2(not for extension instructions): 
    # s3(not for extension instructions): 
    # s4(not for extension instructions): 
    # s5(not for extension instructions): 

    # s0: activation_alpha/_address
    # s1: activation_shift
    # s8: 
    # s9: 
    # s10: 
    # s11:

    esp32p4_push_8_stacks_2r  s0, s1

    esp32p4_s16_unaligned_depthwise_conv2d_33c1_load_args  a2, a3, a4, a5, t3, t5, t4
    lw  s0, 76(a2)       // activation_alpha
    lw  s1, 84(a2)       // activation_shift

    addi  a4, a4, -16                           // a4: dilation_x_offset - 16
    addi  a5, a5, -16                           // a5: dilation_y_offset - 16
    addi  t3, t3, -16                           // t3: next_3381 - 16

    bltz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_c_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t6                // t6: output_sar_byte

        esp.ld.128.usar.ip  q0, a1, 16
        esp.ld.128.usar.xp  q3, a1, a4
        esp.src.q.ld.ip  q4, a3, 16, q0, q3     // q4: filter_v0; q0: input_v0

        esp.ld.128.usar.ip  q1, a1, 16
        esp.ld.128.usar.xp  q3, a1, a4
        esp.src.q.ld.ip  q2, a1, 16, q1, q3     // q2: input_v2; q1: input_v1
    
        beqz  t6, esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_128b
            li  a6, 8
            beq  t6, a6, esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_64b
                # esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_32b:
                beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_32b_last
                    esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_32b_loop_end
                        esp.zero.qacc
                        esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                        esp32p4_s16_128b_vector_shift_result  q3, t4
                        esp32p4_s16_128b_vector_relu  q3, s0, s1
                        esp32p4_s16_32b_unaligned_vector_store  q3, a0, t6
                    esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_32b_loop_end:
                        nop
                
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_32b_last:
                    esp.zero.qacc
                    esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                    esp32p4_s16_128b_vector_relu  q3, s0, s1
                    esp32p4_s16_32b_unaligned_vector_store  q3, a0, t6

                j  esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_c_remainder

            esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_64b:
                beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_64b_last
                    esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_64b_loop_end
                        esp.zero.qacc
                        esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                        esp32p4_s16_128b_vector_shift_result  q3, t4
                        esp32p4_s16_128b_vector_relu  q3, s0, s1
                        esp32p4_s16_64b_unaligned_vector_store  q3, a0
                    esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_64b_loop_end:
                        nop
                
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_64b_last:
                    esp.zero.qacc
                    esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                    esp32p4_s16_128b_vector_relu  q3, s0, s1
                    esp32p4_s16_64b_unaligned_vector_store  q3, a0

                j  esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_c_remainder

        esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_128b:
            beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_128b_last
                esp.lp.setup  0, t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_128b_loop_end
                    esp.zero.qacc
                    esp32p4_s16_unaligned_depthwise_conv2d_3381  q0, q1, q2, q3, q4, q5, q6, a1, a3, a4, a5, t3
                    esp32p4_s16_128b_vector_shift_result  q3, t4
                    esp32p4_s16_128b_vector_relu  q3, s0, s1
                esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_128b_loop_end:
                    esp32p4_s16_128b_aligned_vector_store  q3, a0
            
            esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_128b_last:
                esp.zero.qacc
                esp32p4_s16_unaligned_depthwise_conv2d_3381_last  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t3
                esp32p4_s16_128b_vector_shift_result  q3, t4
                esp32p4_s16_128b_vector_relu  q3, s0, s1
                esp32p4_s16_128b_aligned_vector_store  q3, a0


    esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_c_remainder:
        lw  t5, 136(a2)                         // t5: c_remainder
        beqz  t5, esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_c_remainder_end
            esp.zero.qacc

            srli  t0, t5, 1
            esp32p4_s16_unaligned_depthwise_conv2d_33r1  q0, q1, q2, q3, q4, q5, a1, a3, a4, a5, t5
            esp32p4_s16_128b_vector_shift_result  q0, t4
            esp32p4_s16_128b_vector_relu  q0, s0, s1
            dl_esp32p4_s16_store_remainder  q0, t0, t6, a0
        
        esp32p4_s16_unaligned_depthwise_conv2d_33c1_relu_c_remainder_end:

    esp32p4_pop_8_stacks_2r  s0, s1
    ret