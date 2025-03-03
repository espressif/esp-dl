#include "dl_tie728_s8.S"



#void dl_tie728_s8_mul_w1_16_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, int lenght);

    .align 4
    .text
    .global dl_tie728_s8_mul_w1_16_w2_16
    .type   dl_tie728_s8_mul_w1_16_w2_16, @function
    #.section .iram1
dl_tie728_s8_mul_w1_16_w2_16:

    # a2: int8_t *output_ptr
    # a3: int8_t *input0_ptr
    # a4: int8_t *input1_ptr

    # a5: void *args
    # a6: c_div_x_1
    # a7: mul_shift
    # a8:
    # a9:
    # a10:
    # a11:

    # a12(not for extension instructions): tmp value
    # a13(not for extension instructions):
    # a14(not for extension instructions):
    # a15(not for extension instructions):
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

    entry  sp, 128

    l32i  a6, a5, 64
    l32i  a7, a5, 80

    ee.vld.128.ip  q0, a3, 16
    ee.vld.128.ip  q1, a4, 16
    mov  a14, a6
    blti  a14, 1, dl_tie728_s8_mul_w1_16_w2_16_loop_last

    dl_tie728_s8_mul_w1_16_w2_16_loop:
        ee.zero.qacc
        ee.vmulas.s8.qacc.ld.ip  q0, a3, 16, q0, q1
        ee.vld.128.ip  q1, a4, 16
        tie728_s8_vector_round_result  q2, a7, a12, q7
        ee.vst.128.ip  q2, a2, 16
        addi  a14, a14, -1
        bgei  a14, 1, dl_tie728_s8_mul_w1_16_w2_16_loop

    dl_tie728_s8_mul_w1_16_w2_16_loop_last:
        ee.zero.qacc
        ee.vmulas.s8.qacc  q0, q1
        tie728_s8_vector_round_result  q2, a7, a12, q7
        ee.vst.128.ip  q2, a2, 16
    retw



#void dl_tie728_s8_mul_w1_16_w2_1(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, int lenght);

    .align 4
    .text
    .global dl_tie728_s8_mul_w1_16_w2_1
    .type   dl_tie728_s8_mul_w1_16_w2_1, @function
    #.section .iram1
dl_tie728_s8_mul_w1_16_w2_1:

    # a2: int8_t *output_ptr
    # a3: int8_t *input0_ptr
    # a4: int8_t *input1_ptr

    # a5: void *args
    # a6: c_div_x_1
    # a7: mul_shift
    # a8:
    # a9:
    # a10:
    # a11:

    # a12(not for extension instructions): tmp value
    # a13(not for extension instructions):
    # a14(not for extension instructions):
    # a15(not for extension instructions):
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

    entry  sp, 128

    l32i  a6, a5, 64
    l32i  a7, a5, 80

    ee.vld.128.ip  q0, a3, 16
    ee.vldbc.8.ip  q1, a4, 0        // input1 broadcast
    mov  a14, a6
    blti  a14, 1, dl_tie728_s8_mul_w1_16_w2_1_loop_last

    dl_tie728_s8_mul_w1_16_w2_1_loop:
        ee.zero.qacc
        ee.vmulas.s8.qacc.ld.ip  q0, a3, 16, q0, q1
        tie728_s8_vector_round_result  q2, a7, a12, q7
        ee.vst.128.ip  q2, a2, 16
        addi  a14, a14, -1
        bgei  a14, 1, dl_tie728_s8_mul_w1_16_w2_1_loop

    dl_tie728_s8_mul_w1_16_w2_1_loop_last:
        ee.zero.qacc
        ee.vmulas.s8.qacc  q0, q1
        tie728_s8_vector_round_result  q2, a7, a12, q7
        ee.vst.128.ip  q2, a2, 16
    retw



#void dl_tie728_s8_mul_w1_1_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, int lenght);

    .align 4
    .text
    .global dl_tie728_s8_mul_w1_1_w2_16
    .type   dl_tie728_s8_mul_w1_1_w2_16, @function
    #.section .iram1
dl_tie728_s8_mul_w1_1_w2_16:

    # a2: int8_t *output_ptr
    # a3: int8_t *input0_ptr
    # a4: int8_t *input1_ptr

    # a5: void *args
    # a6: c_div_x_1
    # a7: mul_shift
    # a8:
    # a9:
    # a10:
    # a11:

    # a12(not for extension instructions): tmp value
    # a13(not for extension instructions):
    # a14(not for extension instructions):
    # a15(not for extension instructions):
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

    entry  sp, 128

    l32i  a6, a5, 64
    l32i  a7, a5, 80

    ee.vldbc.8.ip  q0, a3, 0       // input0 broadcast
    ee.vld.128.ip  q1, a4, 16
    mov  a14, a6
    blti  a14, 1, dl_tie728_s8_mul_w1_1_w2_16_loop_last

    dl_tie728_s8_mul_w1_1_w2_16_loop:
        ee.zero.qacc
        ee.vmulas.s8.qacc.ld.ip  q1, a4, 16, q0, q1
        tie728_s8_vector_round_result  q2, a7, a12, q7
        ee.vst.128.ip  q2, a2, 16
        addi  a14, a14, -1
        bgei  a14, 1, dl_tie728_s8_mul_w1_1_w2_16_loop

    dl_tie728_s8_mul_w1_1_w2_16_loop_last:
        ee.zero.qacc
        ee.vmulas.s8.qacc  q0, q1
        tie728_s8_vector_round_result  q2, a7, a12, q7
        ee.vst.128.ip  q2, a2, 16
    retw



    .align 4
    .text
    .global dl_tie728_s8_mul_w1_16_w2_16_unaligned
    .type   dl_tie728_s8_mul_w1_16_w2_16_unaligned, @function
    #.section .iram1
dl_tie728_s8_mul_w1_16_w2_16_unaligned:

    # a2: int8_t *output_ptr
    # a3: int8_t *input0_ptr
    # a4: int8_t *input1_ptr

    # a5: void *args
    # a6: c_div_x_1
    # a7: output sar_byte / tmp value
    # a8: tmp value
    # a9:
    # a10: c_remainder
    # a11: mul_shift

    # a12(not for extension instructions): tmp value
    # a13(not for extension instructions):
    # a14(not for extension instructions):
    # a15(not for extension instructions):
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

    entry  sp, 128

    l32i  a6, a5, 64
    l32i  a10, a5, 76
    l32i  a11, a5, 80

    // input0 exp = input1 exp = output exp

    ee.ld.128.usar.ip q5, a2, 0    // get output_ptr sar_byte
    rur.sar_byte  a7

    ee.ld.128.usar.ip q0, a3, 16
    ee.ld.128.usar.ip q3, a4, 16
    bltz a6, dl_tie728_s8_mul_w1_16_w2_16_unaligned_remainder     // channel < 16
    ee.ld.128.usar.ip q1, a3, 16

    beqz a7, dl_tie728_s8_mul_w1_16_w2_16_unaligned_128b
    movi a14, 8
    beq a7, a14, dl_tie728_s8_mul_w1_16_w2_16_unaligned_64b

        mov  a14, a6
        beqz a14, 1f
        0:
            ee.src.q.qup q2, q0, q1
            ee.ld.128.usar.ip q4, a4, 16
            ee.zero.qacc
            ee.src.q.qup q5, q3, q4
            ee.vmulas.s8.qacc  q2, q5
            tie728_s8_vector_round_result  q2, a11, a12, q7
            ee.ld.128.usar.ip q1, a3, 16
            dl_tie728_s8_unaligned_store0  q2, a2, a7
            addi a14, a14, -1
            bgei  a14, 1, 0b
        1:
        ee.src.q.qup q2, q0, q1
        ee.ld.128.usar.ip q4, a4, 16
        ee.zero.qacc
        ee.src.q.qup q5, q3, q4
        ee.vmulas.s8.qacc  q2, q5
        tie728_s8_vector_round_result  q2, a11, a12, q7
        dl_tie728_s8_unaligned_store0  q2, a2, a7
        j dl_tie728_s8_mul_w1_16_w2_16_unaligned_remainder

    // output sar = 0
    dl_tie728_s8_mul_w1_16_w2_16_unaligned_128b:
        mov  a14, a6
        beqz a14, 3f
        2:
            ee.src.q.qup q2, q0, q1
            ee.ld.128.usar.ip q4, a4, 16
            ee.zero.qacc
            ee.src.q.qup q5, q3, q4
            ee.vmulas.s8.qacc  q2, q5
            tie728_s8_vector_round_result  q2, a11, a12, q7
            ee.ld.128.usar.ip q1, a3, 16
            ee.vst.128.ip q2, a2, 16
            addi a14, a14, -1
            bgei  a14, 1, 2b
        3:
        ee.src.q.qup q2, q0, q1
        ee.ld.128.usar.ip q4, a4, 16
        ee.zero.qacc
        ee.src.q.qup q5, q3, q4
        ee.vmulas.s8.qacc  q2, q5
        tie728_s8_vector_round_result  q2, a11, a12, q7
        ee.vst.128.ip q2, a2, 16

        j dl_tie728_s8_mul_w1_16_w2_16_unaligned_remainder

    // output sar = 8
    dl_tie728_s8_mul_w1_16_w2_16_unaligned_64b:
        mov  a14, a6
        beqz a14, 5f
        4:
            ee.src.q.qup q2, q0, q1
            ee.ld.128.usar.ip q4, a4, 16
            ee.zero.qacc
            ee.src.q.qup q5, q3, q4
            ee.vmulas.s8.qacc  q2, q5
            tie728_s8_vector_round_result  q2, a11, a12, q7
            ee.ld.128.usar.ip q1, a3, 16
            dl_tie728_s8_unaligned_store1  q2, a2
            addi a14, a14, -1
            bgei  a14, 1, 4b
        5:
        ee.src.q.qup q2, q0, q1
        ee.ld.128.usar.ip q4, a4, 16
        ee.zero.qacc
        ee.src.q.qup q5, q3, q4
        ee.vmulas.s8.qacc  q2, q5
        tie728_s8_vector_round_result  q2, a11, a12, q7
        dl_tie728_s8_unaligned_store1  q2, a2
        j dl_tie728_s8_mul_w1_16_w2_16_unaligned_remainder

    dl_tie728_s8_mul_w1_16_w2_16_unaligned_remainder:
            beqz a10, dl_tie728_s8_mul_w1_16_w2_16_unaligned_end

            ee.ld.128.usar.xp q1, a3, a10
            ee.zero.qacc
            ee.src.q q2, q0, q1
            ee.ld.128.usar.xp q4, a4, a10
            ee.src.q q5, q3, q4
            ee.vmulas.s8.qacc  q2, q5
            tie728_s8_vector_round_result  q2, a11, a12, q7
            dl_tie728_s8_store_remainder  q2, a9, a11, a7, a8, a2, a10

    dl_tie728_s8_mul_w1_16_w2_16_unaligned_end:
        addi a3, a3, -16
        addi a4, a4, -16
    retw



    .align 4
    .text
    .global dl_tie728_s8_mul_w1_16_w2_1_unaligned
    .type   dl_tie728_s8_mul_w1_16_w2_1_unaligned, @function
    #.section .iram1
dl_tie728_s8_mul_w1_16_w2_1_unaligned:

    # a2: int8_t *output_ptr
    # a3: int8_t *input0_ptr
    # a4: int8_t *input1_ptr broadcast

    # a5: void *args
    # a6: c_div_x_1
    # a7: output sar_byte / tmp value
    # a8: tmp value
    # a9: tmp value
    # a10: c_remainder
    # a11: mul_shift

    # a12(not for extension instructions): tmp value
    # a13(not for extension instructions):
    # a14(not for extension instructions):
    # a15(not for extension instructions):
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

    entry  sp, 128

    l32i  a6, a5, 64
    l32i  a10, a5, 76
    l32i  a11, a5, 80

    // input0 exp = input1 exp = output exp

    ee.ld.128.usar.ip q5, a2, 0    // get output_ptr sar_byte
    rur.sar_byte  a7

    ee.vldbc.8.ip q5, a4, 0        // input1 broadcast

    ee.ld.128.usar.ip q0, a3, 16
    bltz a6, dl_tie728_s8_mul_w1_16_w2_1_unaligned_remainder  // channel < 16
    ee.ld.128.usar.ip q1, a3, 16

    beqz a7, dl_tie728_s8_mul_w1_16_w2_1_unaligned_128b
    movi a14, 8
    beq a7, a14, dl_tie728_s8_mul_w1_16_w2_1_unaligned_64b

        mov  a14, a6
        beqz a14, 1f
        0:
            ee.zero.qacc
            ee.src.q.qup q2, q0, q1
            ee.vmulas.s8.qacc  q2, q5
            tie728_s8_vector_round_result  q2, a11, a12, q7
            ee.ld.128.usar.ip q1, a3, 16
            dl_tie728_s8_unaligned_store0  q2, a2, a7
            addi a14, a14, -1
            bgei  a14, 1, 0b
        1:
        ee.zero.qacc
        ee.src.q.qup q2, q0, q1
        ee.vmulas.s8.qacc  q2, q5
        tie728_s8_vector_round_result  q2, a11, a12, q7
        dl_tie728_s8_unaligned_store0  q2, a2, a7
        j dl_tie728_s8_mul_w1_16_w2_1_unaligned_remainder

    // output sar = 0
    dl_tie728_s8_mul_w1_16_w2_1_unaligned_128b:
        mov  a14, a6
        beqz a14, 3f
        2:
            ee.zero.qacc
            ee.src.q.qup q2, q0, q1
            ee.vmulas.s8.qacc  q2, q5
            tie728_s8_vector_round_result  q2, a11, a12, q7
            ee.ld.128.usar.ip q1, a3, 16
            ee.vst.128.ip q2, a2, 16
            addi a14, a14, -1
            bgei  a14, 1, 2b
        3:
        ee.zero.qacc
        ee.src.q.qup q2, q0, q1
        ee.vmulas.s8.qacc  q2, q5
        tie728_s8_vector_round_result  q2, a11, a12, q7
        ee.vst.128.ip q2, a2, 16
        j dl_tie728_s8_mul_w1_16_w2_1_unaligned_remainder

    // output sar = 8
    dl_tie728_s8_mul_w1_16_w2_1_unaligned_64b:
        mov  a14, a6
        beqz a14, 5f
        4:
            ee.zero.qacc
            ee.src.q.qup q2, q0, q1
            ee.vmulas.s8.qacc  q2, q5
            tie728_s8_vector_round_result  q2, a11, a12, q7
            ee.ld.128.usar.ip q1, a3, 16
            dl_tie728_s8_unaligned_store1  q2, a2
            addi a14, a14, -1
            bgei  a14, 1, 4b
        5:
        ee.zero.qacc
        ee.src.q.qup q2, q0, q1
        ee.vmulas.s8.qacc  q2, q5
        tie728_s8_vector_round_result  q2, a11, a12, q7
        dl_tie728_s8_unaligned_store1  q2, a2
        j dl_tie728_s8_mul_w1_16_w2_1_unaligned_remainder

    dl_tie728_s8_mul_w1_16_w2_1_unaligned_remainder:
        beqz a10, dl_tie728_s8_mul_w1_16_w2_1_unaligned_end

        ee.ld.128.usar.xp q1, a3, a10
        ee.zero.qacc
        ee.src.q q2, q0, q1
        ee.vmulas.s8.qacc  q2, q5
        tie728_s8_vector_round_result  q2, a11, a12, q7
        dl_tie728_s8_store_remainder  q2, a9, a11, a7, a8, a2, a10

    dl_tie728_s8_mul_w1_16_w2_1_unaligned_end:
        addi a3, a3, -16
    retw



    .align 4
    .text
    .global dl_tie728_s8_mul_w1_1_w2_16_unaligned
    .type   dl_tie728_s8_mul_w1_1_w2_16_unaligned, @function
    #.section .iram1
dl_tie728_s8_mul_w1_1_w2_16_unaligned:

    # a2: int8_t *output_ptr
    # a3: int8_t *input0_ptr broadcast
    # a4: int8_t *input1_ptr

    # a5: void *args
    # a6: c_div_x_1
    # a7: output sar_byte / tmp value
    # a8: tmp value
    # a9: tmp value
    # a10: c_remainder
    # a11: mul_shift

    # a12(not for extension instructions): tmp value
    # a13(not for extension instructions):
    # a14(not for extension instructions):
    # a15(not for extension instructions):
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

    entry  sp, 128

    l32i  a6, a5, 64
    l32i  a10, a5, 76
    l32i  a11, a5, 80

    // input0 exp = input1 exp = output exp
    ee.ld.128.usar.ip q5, a2, 0    // output sar_byte
    rur.sar_byte  a7

    ee.vldbc.8.ip q5, a3, 0        // input0 broadcast

    ee.ld.128.usar.ip q0, a4, 16
    bltz a6, dl_tie728_s8_mul_w1_1_w2_16_unaligned_remainder  // channel < 16
    ee.ld.128.usar.ip q1, a4, 16

    beqz a7, dl_tie728_s8_mul_w1_1_w2_16_unaligned_128b
    movi a14, 8
    beq a7, a14, dl_tie728_s8_mul_w1_1_w2_16_unaligned_64b

        mov  a14, a6
        beqz a14, 1f
        0:
            ee.zero.qacc
            ee.src.q.qup q2, q0, q1
            ee.vmulas.s8.qacc  q5, q2
            tie728_s8_vector_round_result  q2, a11, a12, q7
            ee.ld.128.usar.ip q1, a4, 16
            dl_tie728_s8_unaligned_store0  q2, a2, a7
            addi a14, a14, -1
            bgei  a14, 1, 0b
        1:
        ee.zero.qacc
        ee.src.q.qup q2, q0, q1
        ee.vmulas.s8.qacc  q5, q2
        tie728_s8_vector_round_result  q2, a11, a12, q7
        dl_tie728_s8_unaligned_store0  q2, a2, a7
        j dl_tie728_s8_mul_w1_1_w2_16_unaligned_remainder

    // output sar = 0
    dl_tie728_s8_mul_w1_1_w2_16_unaligned_128b:
        mov  a14, a6
        beqz a14, 3f
        2:
            ee.zero.qacc
            ee.src.q.qup q2, q0, q1
            ee.vmulas.s8.qacc  q5, q2
            tie728_s8_vector_round_result  q2, a11, a12, q7
            ee.ld.128.usar.ip q1, a4, 16
            ee.vst.128.ip q2, a2, 16
            addi a14, a14, -1
            bgei  a14, 1, 2b
        3:
        ee.zero.qacc
        ee.src.q.qup q2, q0, q1
        ee.vmulas.s8.qacc  q5, q2
        tie728_s8_vector_round_result  q2, a11, a12, q7
        ee.vst.128.ip q2, a2, 16
        j dl_tie728_s8_mul_w1_1_w2_16_unaligned_remainder

    // output sar = 8
    dl_tie728_s8_mul_w1_1_w2_16_unaligned_64b:
        mov  a14, a6
        beqz a14, 5f
        4:
            ee.zero.qacc
            ee.src.q.qup q2, q0, q1
            ee.vmulas.s8.qacc  q5, q2
            tie728_s8_vector_round_result  q2, a11, a12, q7
            ee.ld.128.usar.ip q1, a4, 16
            dl_tie728_s8_unaligned_store1  q2, a2
            addi a14, a14, -1
            bgei  a14, 1, 4b
        5:
        ee.zero.qacc
        ee.src.q.qup q2, q0, q1
        ee.vmulas.s8.qacc  q5, q2
        tie728_s8_vector_round_result  q2, a11, a12, q7
        dl_tie728_s8_unaligned_store1  q2, a2
        j dl_tie728_s8_mul_w1_1_w2_16_unaligned_remainder

    dl_tie728_s8_mul_w1_1_w2_16_unaligned_remainder:
        beqz a10, dl_tie728_s8_mul_w1_1_w2_16_unaligned_end

        ee.ld.128.usar.xp q1, a4, a10
        ee.zero.qacc
        ee.src.q q2, q0, q1
        ee.vmulas.s8.qacc  q5, q2
        tie728_s8_vector_round_result  q2, a11, a12, q7
        dl_tie728_s8_store_remainder  q2, a9, a11, a7, a8, a2, a10

    dl_tie728_s8_mul_w1_1_w2_16_unaligned_end:
        addi a4, a4, -16
    retw
