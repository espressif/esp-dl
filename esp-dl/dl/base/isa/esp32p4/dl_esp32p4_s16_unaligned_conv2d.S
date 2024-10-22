#include "dl_esp32p4_s16.S"
#include "dl_esp32p4_common.S"


############################################################################################################################################################
#### 
#### esp32p4_s16_unaligned_conv2d_11cn series
#### 
############################################################################################################################################################
.macro esp32p4_s16_unaligned_conv2d_11c8  input_v, input_front, input_back, filter_v0, filter_v1, input_ptr, filter_ptr, c_div_x_1, c_remainder, temp
    # input_v:     8 input elements
    # filter_v0:    8 filter elements
    # filter_v1:    8 filter elements
    # input_ptr:    input_ptr
    # filter_ptr:   filter_ptr
    # c_div_x_1:    input_channel // 8 - 1
    
    esp.ld.128.usar.ip  \input_front, \input_ptr, 16

    bltz  \c_div_x_1, 7f
        esp.ld.128.usar.ip  \input_back, \input_ptr, 16
        esp.vld.128.ip  \filter_v0, \filter_ptr, 16
        esp.vld.128.ip  \filter_v1, \filter_ptr, 16
        
        beqz  \c_div_x_1, 8f
        esp.lp.setup  0, \c_div_x_1, 9f
            esp.src.q.qup  \input_v, \input_front, \input_back
            esp.ld.128.usar.ip  \input_back, \input_ptr, 16

            esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 0
            esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 1
            esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 2
            esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 3
            esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 4
            esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 5
            esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 6
        9:  esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 7
            
        8:
        # last entire-128b
        esp.src.q.qup  \input_v, \input_front, \input_back 
        esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 0
        esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 1
        esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 2
        esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 3
        esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 4
        esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 5
        esp.vsmulas.s16.qacc  \filter_v0, \input_v, 6
        esp.vsmulas.s16.qacc  \filter_v1, \input_v, 7
        beqz  \c_remainder, 0f # jump to c_remainder == 0
    
    # c_remainder
    7:
        esp.ld.128.usar.xp  \input_back, \input_ptr, \c_remainder
        esp.src.q.qup  \input_v, \input_front, \input_back
        esp.vld.128.ip  \filter_v0, \filter_ptr,  16

        li  \temp, 8
        blt  \c_remainder, \temp, 3f
            # remainder == 0x1__0
            esp.vld.128.ip  \filter_v1, \filter_ptr,  16

            li  \temp, 12
            blt  \c_remainder, \temp, 5f
                # remainder == 0x11_0
                li  \temp, 14
                blt  \c_remainder, \temp, 6f
                    # remainder == 0x1110, 7
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 1
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 3
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 4
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 5
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 6
                    j 0f

                6:  # remainder == 0x1100, 6
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 1
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 3
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 4
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 5
                    j 0f

            5:
                li  \temp, 10
                blt  \c_remainder, \temp, 4f
                    # remainder == 0x1010, 5
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 1
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 3
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 4
                    j 0f

                4:  # remainder == 0x1000, 4
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v, 1
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 3
                    j 0f

        3:  # remainder == 0x0__0
            li  \temp, 4
            blt  \c_remainder, \temp, 1f
                # remainder == 0x01_0
                esp.vld.128.ip  \filter_v1, \filter_ptr,  16

                li  \temp, 6
                blt  \c_remainder, \temp, 2f
                    # remainder == 0x0110, 3
                    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    j 0f

                2:  # remainder == 0x0100, 2
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    j 0f

            1:  # remainder == 0x0010, 1
                esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0

    0:
        addi  \input_ptr, \input_ptr, -16

.endm



.macro esp32p4_s16_unaligned_conv2d_11c1  input_v, input_front, input_back, filter_v, filter_front, filter_back, input_ptr, filter_ptr, c_div_x_1, c_remainder, temp, zero
    # input_v:     8 input elements
    # filter_v:    8 filter elements
    # filter_v1:    8 filter elements
    # input_ptr:    input_ptr
    # filter_ptr:   filter_ptr
    # c_div_x_1:    input_channel // 8 - 1

    esp.ld.128.usar.ip  \input_front, \input_ptr, 16
    esp.ld.128.usar.ip  \filter_front, \filter_ptr, 16

    bltz  \c_div_x_1, 7f     // input_channel < 8
        esp.ld.128.usar.ip  \input_back, \input_ptr, 16
        
        beqz  \c_div_x_1, 8f
        # esp.lp.setup  0, \c_div_x_1, (8f - 4)
        # Use the zero register as a loop counter, and the value remains zero after the loop is complete.
        mv  \zero, \c_div_x_1
        9:
            esp.src.q.qup  \input_v, \input_front, \input_back

            esp.ld.128.usar.ip  \filter_back, \filter_ptr, 16
            esp.src.q.qup  \filter_v, \filter_front, \filter_back

            esp.ld.128.usar.ip  \input_back, \input_ptr, 16
            esp.vmulas.s16.xacc  \filter_v, \input_v
            addi  \zero, \zero, -1
            bgtz  \zero, 9b
        8:
        # last entire-128b
        esp.src.q.qup  \input_v, \input_front, \input_back

        esp.ld.128.usar.ip  \filter_back, \filter_ptr, 16
        esp.src.q.qup  \filter_v, \filter_front, \filter_back

        esp.vmulas.s16.xacc  \filter_v, \input_v

        beqz  \c_remainder, 0f
    
    7:
        # c_remainder > 0
        esp.ld.128.usar.xp  \input_back, \input_ptr, \c_remainder
        esp.src.q.qup  \input_v, \input_front, \input_back

        esp.ld.128.usar.xp  \filter_back, \filter_ptr, \c_remainder
        esp.src.q.qup  \filter_v, \filter_front, \filter_back

        esp.slcxxp.2q  \input_back,  \input_v, \temp, \zero
        esp.slcxxp.2q  \filter_back, \filter_v, \temp, \zero

        esp.vmulas.s16.xacc  \filter_v, \input_v
    
    0:
        addi  \input_ptr,  \input_ptr, -16
        addi  \filter_ptr, \filter_ptr, -16
.endm



.macro esp32p4_s16_unaligned_conv2d_11cn_load_args  args, filter_ptr, c_div_x_1, n_div_x, mac_shift, c_remainder
    lw  \filter_ptr,    48(\args)       // filter
    lw  \c_div_x_1,     100(\args)      // input_channel / x - 1
    lw  \n_div_x,       96(\args)       // output_channel / x
    lw  \mac_shift,     64(\args)       // mac_shift
    lw  \c_remainder,   136(\args)      // input_channel % (vector_width / element_width) * sizeof(feature_t)
.endm



    .text
    .align 2
    .global dl_esp32p4_s16_unaligned_conv2d_11cn_bias
    .type   dl_esp32p4_s16_unaligned_conv2d_11cn_bias, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_unaligned_conv2d_11cn_bias:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 
    # t5: 
    # t6: bias_ptr

    # a6(not for extension instructions): 
    # a7(not for extension instructions): 
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): 
    # s2(not for extension instructions): 
    # s3(not for extension instructions): 
    # s4(not for extension instructions): 
    # s5(not for extension instructions): 

    # s0: output
    # s1: clamp min value
    # s8: clamp max value
    # s9: 
    # s10: 
    # s11:

    esp32p4_push_12_stacks_3r  s0, s1, s8    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value
    esp32p4_s16_unaligned_conv2d_11cn_load_args  a2, a3, t0, t1, a5, a4
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0 
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_s16_unaligned_conv2d_11cn_bias_128b
            li  t2, 8
            beq  t3, t2, esp32p4_s16_unaligned_conv2d_11cn_bias_64b
                esp32p4_s16_unaligned_conv2d_11cn_bias_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_32b_multiple_loop
                j  esp32p4_s16_unaligned_conv2d_11cn_bias_n_remainder

            esp32p4_s16_unaligned_conv2d_11cn_bias_64b:
                esp32p4_s16_unaligned_conv2d_11cn_bias_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_64b_multiple_loop
                j  esp32p4_s16_unaligned_conv2d_11cn_bias_n_remainder

        esp32p4_s16_unaligned_conv2d_11cn_bias_128b:
            esp32p4_s16_unaligned_conv2d_11cn_bias_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_128b_multiple_loop


    esp32p4_s16_unaligned_conv2d_11cn_bias_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_s16_unaligned_conv2d_11cn_bias_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                esp32p4_s16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_n_remainder_loop

        esp32p4_s16_unaligned_conv2d_11cn_bias_n_remainder_end:

    esp32p4_pop_12_stacks_3r  s0, s1, s8     // restore registers
    ret



    .text
    .align 2
    .global dl_esp32p4_s16_unaligned_conv2d_11cn_bias_relu
    .type   dl_esp32p4_s16_unaligned_conv2d_11cn_bias_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_unaligned_conv2d_11cn_bias_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 
    # t5: 
    # t6: bias_ptr

    # a6(not for extension instructions): 
    # a7(not for extension instructions): 
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): 
    # s2(not for extension instructions): 
    # s3(not for extension instructions): 
    # s4(not for extension instructions): 
    # s5(not for extension instructions): 

    # s0: output
    # s1: clamp min value
    # s8: clamp max value
    # s9: activation_alpha/_address
    # s10: activation_shift
    # s11:

    esp32p4_push_20_stacks_5r  s0, s1, s8, s9, s10    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value
    lw  s9, 76(a2)      // activation_alpha
    lw  s10, 84(a2)     // activation_shift
    esp32p4_s16_unaligned_conv2d_11cn_load_args  a2, a3, t0, t1, a5, a4
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0 
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_s16_unaligned_conv2d_11cn_bias_relu_128b
            li  t2, 8
            beq  t3, t2, esp32p4_s16_unaligned_conv2d_11cn_bias_relu_64b
                esp32p4_s16_unaligned_conv2d_11cn_bias_relu_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_relu_32b_multiple_loop
                j  esp32p4_s16_unaligned_conv2d_11cn_bias_relu_n_remainder

            esp32p4_s16_unaligned_conv2d_11cn_bias_relu_64b:
                esp32p4_s16_unaligned_conv2d_11cn_bias_relu_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_relu_64b_multiple_loop
                j  esp32p4_s16_unaligned_conv2d_11cn_bias_relu_n_remainder

        esp32p4_s16_unaligned_conv2d_11cn_bias_relu_128b:
            esp32p4_s16_unaligned_conv2d_11cn_bias_relu_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_relu_128b_multiple_loop


    esp32p4_s16_unaligned_conv2d_11cn_bias_relu_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_relu_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_s16_unaligned_conv2d_11cn_bias_relu_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                esp32p4_s16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_leakyrelu  s0, s9, s10
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_bias_relu_n_remainder_loop

        esp32p4_s16_unaligned_conv2d_11cn_bias_relu_n_remainder_end:

    esp32p4_pop_20_stacks_5r  s0, s1, s8, s9, s10     // restore registers
    ret



    .text
    .align 2
    .global dl_esp32p4_s16_unaligned_conv2d_11cn
    .type   dl_esp32p4_s16_unaligned_conv2d_11cn, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_unaligned_conv2d_11cn:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 
    # t5: 
    # t6: 

    # a6(not for extension instructions): 
    # a7(not for extension instructions): 
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): 
    # s2(not for extension instructions): 
    # s3(not for extension instructions): 
    # s4(not for extension instructions): 
    # s5(not for extension instructions): 

    # s0: output
    # s1: clamp min value
    # s8: clamp max value
    # s9: 
    # s10: 
    # s11:

    esp32p4_push_12_stacks_3r  s0, s1, s8    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value
    esp32p4_s16_unaligned_conv2d_11cn_load_args  a2, a3, t0, t1, a5, a4

    beqz  t1, esp32p4_s16_unaligned_conv2d_11cn_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0 
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_s16_unaligned_conv2d_11cn_128b
            li  t2, 8
            beq  t3, t2, esp32p4_s16_unaligned_conv2d_11cn_64b
                esp32p4_s16_unaligned_conv2d_11cn_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_32b_multiple_loop
                j  esp32p4_s16_unaligned_conv2d_11cn_n_remainder

            esp32p4_s16_unaligned_conv2d_11cn_64b:
                esp32p4_s16_unaligned_conv2d_11cn_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_64b_multiple_loop
                j  esp32p4_s16_unaligned_conv2d_11cn_n_remainder

        esp32p4_s16_unaligned_conv2d_11cn_128b:
            esp32p4_s16_unaligned_conv2d_11cn_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_128b_multiple_loop


    esp32p4_s16_unaligned_conv2d_11cn_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_s16_unaligned_conv2d_11cn_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_s16_unaligned_conv2d_11cn_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_n_remainder_loop

        esp32p4_s16_unaligned_conv2d_11cn_n_remainder_end:

    esp32p4_pop_12_stacks_3r  s0, s1, s8     // restore registers
    ret



    .text
    .align 2
    .global dl_esp32p4_s16_unaligned_conv2d_11cn_relu
    .type   dl_esp32p4_s16_unaligned_conv2d_11cn_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_unaligned_conv2d_11cn_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 
    # t5: 
    # t6: 

    # a6(not for extension instructions): 
    # a7(not for extension instructions): 
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): 
    # s2(not for extension instructions): 
    # s3(not for extension instructions): 
    # s4(not for extension instructions): 
    # s5(not for extension instructions): 

    # s0: output
    # s1: clamp min value
    # s8: clamp max value
    # s9: activation_alpha/_address
    # s10: activation_shift
    # s11:

    esp32p4_push_20_stacks_5r  s0, s1, s8, s9, s10    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value
    lw  s9, 76(a2)      // activation_alpha
    lw  s10, 84(a2)     // activation_shift
    esp32p4_s16_unaligned_conv2d_11cn_load_args  a2, a3, t0, t1, a5, a4

    beqz  t1, esp32p4_s16_unaligned_conv2d_11cn_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0 
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_s16_unaligned_conv2d_11cn_relu_128b
            li  t2, 8
            beq  t3, t2, esp32p4_s16_unaligned_conv2d_11cn_relu_64b
                esp32p4_s16_unaligned_conv2d_11cn_relu_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_relu_32b_multiple_loop
                j  esp32p4_s16_unaligned_conv2d_11cn_relu_n_remainder

            esp32p4_s16_unaligned_conv2d_11cn_relu_64b:
                esp32p4_s16_unaligned_conv2d_11cn_relu_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_relu_64b_multiple_loop
                j  esp32p4_s16_unaligned_conv2d_11cn_relu_n_remainder

        esp32p4_s16_unaligned_conv2d_11cn_relu_128b:
            esp32p4_s16_unaligned_conv2d_11cn_relu_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_relu_128b_multiple_loop


    esp32p4_s16_unaligned_conv2d_11cn_relu_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_s16_unaligned_conv2d_11cn_relu_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_s16_unaligned_conv2d_11cn_relu_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_leakyrelu  s0, s9, s10
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_s16_unaligned_conv2d_11cn_relu_n_remainder_loop

        esp32p4_s16_unaligned_conv2d_11cn_relu_n_remainder_end:

    esp32p4_pop_20_stacks_5r  s0, s1, s8, s9, s10     // restore registers
    ret