#include "dl_esp32p4_s8.S"
#include "dl_esp32p4_common.S"


    .text
    .align 2
    .global dl_esp32p4_s8_s8_requantize_linear
    .type   dl_esp32p4_s8_s8_requantize_linear, @function
    .balign 4
    .option norvc
dl_esp32p4_s8_s8_requantize_linear:

    # a0: int8_t *output_ptr
    # a1: int8_t *input_ptr
    # a2: void *args

    # a3: size_div_x / tmp value
    # a4: in_size_remainder
    # a5: tmp value
    # t3: output_shift / tmp value
    # t4: output_scale / tmp value
    # t5:
    # t6:

    # a6(not for extension instructions):
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

    lw  a3, 8(a2)       // size_div_x
    lw  a4, 12(a2)      // in_size_remainder
    lw  t3, 20(a2)      // output_shift
    lw  t4, 24(a2)      // output_scale

    bgtz  t4, esp32p4_s8_s8_requantize_linear_left_shift
    beqz  a3, esp32p4_s8_s8_requantize_linear_right_shift_remainder

        esp32p4_s8_s8_requantize_linear_right_shift_loop:
            esp.ldqa.s8.128.ip  a1, 16
            addi  a3, a3, -1
            esp32p4_s8_128b_vector_shift_result  q0, t3
            esp.vst.128.ip  q0, a0, 16
            bnez  a3, esp32p4_s8_s8_requantize_linear_right_shift_loop

        esp32p4_s8_s8_requantize_linear_right_shift_remainder:
            beqz  a4, esp32p4_s8_s8_requantize_linear_end
            esp.ldqa.s8.128.xp  a1, a4
            esp32p4_s8_128b_vector_shift_result  q0, t3
            dl_esp32p4_s8_store_remainder  q0, a3, a5, t3, t4, t0, a0, a4
            j  esp32p4_s8_s8_requantize_linear_end

    esp32p4_s8_s8_requantize_linear_left_shift:

        addi  t4, a2, 24
        esp.vldbc.16.ip  q2, t4, 0      // load output_scale
        beqz  a3, esp32p4_s8_s8_requantize_linear_left_shift_remainder

        esp32p4_s8_s8_requantize_linear_left_shift_loop:
            esp.vldext.s8.ip  q0, q1, a1, 16
            esp.zero.qacc
            esp.vmulas.s16.qacc  q0, q2
            esp32p4_s8_128b_vector_shift_result  q0, t3
            esp.zero.qacc
            esp.vmulas.s16.qacc  q1, q2
            esp32p4_s8_128b_vector_shift_result  q1, t3
            esp.vunzip.8  q0, q1
            esp.vst.128.ip  q0, a0, 16
            addi  a3, a3, -1
            bnez  a3, esp32p4_s8_s8_requantize_linear_left_shift_loop

        esp32p4_s8_s8_requantize_linear_left_shift_remainder:
            beqz  a4, esp32p4_s8_s8_requantize_linear_end
            esp.vldext.s8.xp  q0, q1, a1, a4
            esp.zero.qacc
            esp.vmulas.s16.qacc  q0, q2
            esp32p4_s8_128b_vector_shift_result  q0, t3
            esp.zero.qacc
            esp.vmulas.s16.qacc  q1, q2
            esp32p4_s8_128b_vector_shift_result  q1, t3
            esp.vunzip.8  q0, q1
            dl_esp32p4_s8_store_remainder  q0, a3, a5, t3, t4, t0, a0, a4

    esp32p4_s8_s8_requantize_linear_end:
    ret



    .text
    .align 2
    .global dl_esp32p4_s8_s16_requantize_linear
    .type   dl_esp32p4_s8_s16_requantize_linear, @function
    .balign 4
    .option norvc
dl_esp32p4_s8_s16_requantize_linear:

    # a0: int8_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: size_div_x / tmp value
    # a4: in_size_remainder / tmp value
    # a5: out_size_remainder
    # t3: output_shift / tmp value
    # t4: output_scale / tmp value
    # t5:
    # t6:

    # a6(not for extension instructions):
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

    lw  a3, 8(a2)       // size_div_x
    lw  a4, 12(a2)      // in_size_remainder
    lw  a5, 16(a2)      // out_size_remainder
    lw  t3, 20(a2)      // output_shift
    lw  t4, 24(a2)      // output_scale

    bgtz  t4, esp32p4_s8_s16_requantize_linear_left_shift
    beqz  a3, esp32p4_s8_s16_requantize_linear_right_shift_remainder

        esp32p4_s8_s16_requantize_linear_right_shift_loop:
            esp.ldqa.s16.128.ip  a1, 16
            esp32p4_s8_128b_vector_shift_result  q0, t3
            esp.ldqa.s16.128.ip  a1, 16
            addi  a3, a3, -1
            esp32p4_s8_128b_vector_shift_result  q1, t3
            esp.vunzip.8  q0, q1
            esp.vst.128.ip  q0, a0, 16
            bnez  a3, esp32p4_s8_s16_requantize_linear_right_shift_loop

        esp32p4_s8_s16_requantize_linear_right_shift_remainder:
            beqz  a5, esp32p4_s8_s16_requantize_linear_end
            li  t0, 8
            ble  a5, t0, esp32p4_s8_s16_requantize_linear_right_shift_remainder_le8
                esp.ldqa.s16.128.ip  a1, 16
                esp32p4_s8_128b_vector_shift_result  q0, t3
                esp.ldqa.s16.128.xp  a1, a4
                esp32p4_s8_128b_vector_shift_result  q1, t3
                esp.vunzip.8  q0, q1
                dl_esp32p4_s8_store_remainder  q0, a3, a4, t3, t4, t0, a0, a5
                j  esp32p4_s8_s16_requantize_linear_end

            esp32p4_s8_s16_requantize_linear_right_shift_remainder_le8:
                esp.ldqa.s16.128.xp  a1, a4
                esp.zero.q  q1
                esp32p4_s8_128b_vector_shift_result  q0, t3
                esp.vunzip.8  q0, q1
                dl_esp32p4_s8_store_remainder  q0, a3, a4, t3, t4, t0, a0, a5
                j  esp32p4_s8_s16_requantize_linear_end

    esp32p4_s8_s16_requantize_linear_left_shift:

        addi  t4, a2, 24
        esp.vldbc.16.ip  q2, t4, 0      // load output_scale
        beqz  a3, esp32p4_s8_s16_requantize_linear_left_shift_remainder

        esp32p4_s8_s16_requantize_linear_left_shift_loop:
            esp.vld.128.ip  q0, a1, 16
            esp.zero.qacc
            esp.vmulas.s16.qacc  q0, q2
            esp32p4_s8_128b_vector_shift_result  q0, t3
            esp.vld.128.ip  q1, a1, 16
            esp.zero.qacc
            esp.vmulas.s16.qacc  q1, q2
            esp32p4_s8_128b_vector_shift_result  q1, t3
            esp.vunzip.8  q0, q1
            esp.vst.128.ip  q0, a0, 16
            addi  a3, a3, -1
            bnez  a3, esp32p4_s8_s16_requantize_linear_left_shift_loop

        esp32p4_s8_s16_requantize_linear_left_shift_remainder:
            beqz  a5, esp32p4_s8_s16_requantize_linear_end
            li  t0, 8
            ble  a5, t0, esp32p4_s8_s16_requantize_linear_left_shift_remainder_le8
                esp.vld.128.ip  q0, a1, 16
                esp.zero.qacc
                esp.vmulas.s16.qacc  q0, q2
                esp32p4_s8_128b_vector_shift_result  q0, t3
                esp.vld.128.xp  q1, a1, a4
                esp.zero.qacc
                esp.vmulas.s16.qacc  q1, q2
                esp32p4_s8_128b_vector_shift_result  q1, t3
                esp.vunzip.8  q0, q1
                dl_esp32p4_s8_store_remainder  q0, a3, a4, t3, t4, t0, a0, a5
                j  esp32p4_s8_s16_requantize_linear_end

            esp32p4_s8_s16_requantize_linear_left_shift_remainder_le8:
                esp.vld.128.xp  q0, a1, a4
                esp.zero.qacc
                esp.vmulas.s16.qacc  q0, q2
                esp32p4_s8_128b_vector_shift_result  q0, t3
                esp.zero.q  q1
                esp.vunzip.8  q0, q1
                dl_esp32p4_s8_store_remainder  q0, a3, a4, t3, t4, t0, a0, a5

    esp32p4_s8_s16_requantize_linear_end:
    ret
