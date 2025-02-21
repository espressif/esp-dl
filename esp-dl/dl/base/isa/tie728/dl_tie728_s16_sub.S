#include "dl_tie728_s16.S"



#void dl_tie728_s16_sub_w1_8_w2_8(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int lenght);

    .align 4
    .text
    .global dl_tie728_s16_sub_w1_8_w2_8
    .type   dl_tie728_s16_sub_w1_8_w2_8, @function
    #.section .iram1
dl_tie728_s16_sub_w1_8_w2_8:

    # a2: int16_t *output_ptr
    # a3: int16_t *input0_ptr
    # a4: int16_t *input1_ptr

    # a5: void *args / length
    # a6: tmp value
    # a7:
    # a8:
    # a9:
    # a10:
    # a11:

    # a12:
    # a13:
    # a14:
    # a15:

    entry  sp, 128

    l32i  a6, a5, 44
    srai a5, a6, 3

    movi a14, 0
tie728_s16_sub_w1_8_w2_8_loop:
    beq a14, a5, tie728_s16_sub_w1_8_w2_8_end
    ee.vld.128.ip q0, a3, 16
    ee.vld.128.ip q1, a4, 16
    ee.vsubs.s16 q2, q0, q1
    ee.vst.128.ip  q2, a2, 16
    addi a14, a14, 1
    j tie728_s16_sub_w1_8_w2_8_loop
tie728_s16_sub_w1_8_w2_8_end:
    retw



#void dl_tie728_s16_sub_w1_8_w2_1(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int lenght);

    .align 4
    .text
    .global dl_tie728_s16_sub_w1_8_w2_1
    .type   dl_tie728_s16_sub_w1_8_w2_1, @function
    #.section .iram1
dl_tie728_s16_sub_w1_8_w2_1:

    # a2: int16_t *output_ptr
    # a3: int16_t *input0_ptr
    # a4: int16_t *input1_ptr

    # a5: void *args / length
    # a6: tmp value
    # a7:
    # a8:
    # a9:
    # a10:
    # a11:

    # a12:
    # a13:
    # a14:
    # a15:

    entry  sp, 128

    l32i  a6, a5, 44
    srai a5, a6, 3
    ee.vldbc.16.ip q1, a4, 0

    movi a14, 0
tie728_s16_sub_w1_8_w2_1_loop:
    beq a14, a5, tie728_s16_sub_w1_8_w2_1_end
    ee.vld.128.ip q0, a3, 16
    ee.vsubs.s16 q2, q0, q1
    ee.vst.128.ip  q2, a2, 16
    addi a14, a14, 1
    j tie728_s16_sub_w1_8_w2_1_loop
tie728_s16_sub_w1_8_w2_1_end:
    retw



#void dl_tie728_s16_sub_w1_1_w2_8(int16_t *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int lenght);

    .align 4
    .text
    .global dl_tie728_s16_sub_w1_1_w2_8
    .type   dl_tie728_s16_sub_w1_1_w2_8, @function
    #.section .iram1
dl_tie728_s16_sub_w1_1_w2_8:

    # a2: int16_t *output_ptr
    # a3: int16_t *input0_ptr
    # a4: int16_t *input1_ptr

    # a5: void *args / length
    # a6: tmp value
    # a7:
    # a8:
    # a9:
    # a10:
    # a11:

    # a12:
    # a13:
    # a14:
    # a15:

    entry  sp, 128

    l32i  a6, a5, 44
    srai a5, a6, 3
    ee.vldbc.16.ip q0, a3, 0

    movi a14, 0
tie728_s16_sub_w1_1_w2_8_loop:
    beq a14, a5, tie728_s16_sub_w1_1_w2_8_end
    ee.vld.128.ip q1, a4, 16
    ee.vsubs.s16 q2, q0, q1
    ee.vst.128.ip  q2, a2, 16
    addi a14, a14, 1
    j tie728_s16_sub_w1_1_w2_8_loop
tie728_s16_sub_w1_1_w2_8_end:
    retw



    .align 4
    .text
    .global dl_tie728_s16_sub_w1_8_w2_8_unaligned
    .type   dl_tie728_s16_sub_w1_8_w2_8_unaligned, @function
    #.section .iram1
dl_tie728_s16_sub_w1_8_w2_8_unaligned:

    # a2: int16_t *output_ptr
    # a3: int16_t *input0_ptr
    # a4: int16_t *input1_ptr

    # a5: void *args
    # a6: c_div_x_1
    # a7: output sar_byte / tmp value
    # a8: tmp value
    # a9:
    # a10: c_remainder
    # a11:

    # a12:
    # a13:
    # a14: tmp value
    # a15:

    entry  sp, 128

    l32i  a6, a5, 64
    l32i  a10, a5, 76

    // input0 exp = input1 exp = output exp

    ee.ld.128.usar.ip q5, a2, 0    // get output_ptr sar_byte
    rur.sar_byte  a7

    ee.ld.128.usar.ip q0, a3, 16
    ee.ld.128.usar.ip q3, a4, 16
    bltz a6, dl_tie728_s16_sub_w1_8_w2_8_unaligned_remainder     // channel < 16
    ee.ld.128.usar.ip q1, a3, 16

    beqz a7, dl_tie728_s16_sub_w1_8_w2_8_unaligned_128b
    movi a14, 8
    beq a7, a14, dl_tie728_s16_sub_w1_8_w2_8_unaligned_64b

        mov  a14, a6
        beqz a14, 1f
        0:
            ee.src.q.qup q2, q0, q1
            ee.ld.128.usar.ip q4, a4, 16
            ee.src.q.qup q5, q3, q4
            ee.vsubs.s16 q2, q2, q5
            ee.ld.128.usar.ip q1, a3, 16
            dl_tie728_128b_unaligned_store0  q2, a2, a7
            addi a14, a14, -1
            bgei  a14, 1, 0b
        1:
        ee.src.q.qup q2, q0, q1
        ee.ld.128.usar.ip q4, a4, 16
        ee.src.q.qup q5, q3, q4
        ee.vsubs.s16 q2, q2, q5
        dl_tie728_128b_unaligned_store0  q2, a2, a7
        j dl_tie728_s16_sub_w1_8_w2_8_unaligned_remainder

    // output sar = 0
    dl_tie728_s16_sub_w1_8_w2_8_unaligned_128b:
        mov  a14, a6
        beqz a14, 3f
        2:
            ee.src.q.qup q2, q0, q1
            ee.ld.128.usar.ip q4, a4, 16
            ee.src.q.qup q5, q3, q4
            ee.vsubs.s16 q2, q2, q5
            ee.ld.128.usar.ip q1, a3, 16
            ee.vst.128.ip  q2, a2, 16
            addi a14, a14, -1
            bgei  a14, 1, 2b
        3:
        ee.src.q.qup q2, q0, q1
        ee.ld.128.usar.ip q4, a4, 16
        ee.src.q.qup q5, q3, q4
        ee.vsubs.s16 q2, q2, q5
        ee.vst.128.ip  q2, a2, 16

        j dl_tie728_s16_sub_w1_8_w2_8_unaligned_remainder

    // output sar = 8
    dl_tie728_s16_sub_w1_8_w2_8_unaligned_64b:
        mov  a14, a6
        beqz a14, 5f
        4:
            ee.src.q.qup q2, q0, q1
            ee.ld.128.usar.ip q4, a4, 16
            ee.src.q.qup q5, q3, q4
            ee.vsubs.s16 q2, q2, q5
            ee.ld.128.usar.ip q1, a3, 16
            dl_tie728_128b_unaligned_store1  q2, a2
            addi a14, a14, -1
            bgei  a14, 1, 4b
        5:
        ee.src.q.qup q2, q0, q1
        ee.ld.128.usar.ip q4, a4, 16
        ee.src.q.qup q5, q3, q4
        ee.vsubs.s16 q2, q2, q5
        dl_tie728_128b_unaligned_store1  q2, a2

        j dl_tie728_s16_sub_w1_8_w2_8_unaligned_remainder

    dl_tie728_s16_sub_w1_8_w2_8_unaligned_remainder:
            beqz a10, dl_tie728_s16_sub_w1_8_w2_8_unaligned_end

            ee.ld.128.usar.xp q1, a3, a10
            ee.src.q q2, q0, q1
            ee.ld.128.usar.xp q4, a4, a10
            ee.src.q q5, q3, q4
            ee.vsubs.s16 q2, q2, q5
            srli  a10, a10, 1
            dl_tie728_s16_store_remainder  q2, a10, a8, a2

    dl_tie728_s16_sub_w1_8_w2_8_unaligned_end:
        addi a3, a3, -16
        addi a4, a4, -16
    retw



    .align 4
    .text
    .global dl_tie728_s16_sub_w1_8_w2_1_unaligned
    .type   dl_tie728_s16_sub_w1_8_w2_1_unaligned, @function
    #.section .iram1
dl_tie728_s16_sub_w1_8_w2_1_unaligned:

    # a2: int16_t *output_ptr
    # a3: int16_t *input0_ptr
    # a4: int16_t *input1_ptr broadcast

    # a5: void *args
    # a6: c_div_x_1
    # a7: output sar_byte / tmp value
    # a8: tmp value
    # a9: tmp value
    # a10: c_remainder
    # a11:

    # a12:
    # a13:
    # a14: tmp value
    # a15:

    entry  sp, 128

    l32i  a6, a5, 64
    l32i  a10, a5, 76

    // input0 exp = input1 exp = output exp

    ee.ld.128.usar.ip q5, a2, 0    // get output_ptr sar_byte
    rur.sar_byte  a7

    ee.vldbc.16.ip q5, a4, 0       // input1 broadcast

    ee.ld.128.usar.ip q0, a3, 16
    bltz a6, dl_tie728_s16_sub_w1_8_w2_1_unaligned_remainder  // channel < 16
    ee.ld.128.usar.ip q1, a3, 16

    beqz a7, dl_tie728_s16_sub_w1_8_w2_1_unaligned_128b
    movi a14, 8
    beq a7, a14, dl_tie728_s16_sub_w1_8_w2_1_unaligned_64b

        mov  a14, a6
        beqz a14, 1f
        0:
            ee.src.q.qup q2, q0, q1
            ee.vsubs.s16 q2, q2, q5
            ee.ld.128.usar.ip q1, a3, 16
            dl_tie728_128b_unaligned_store0  q2, a2, a7
            addi a14, a14, -1
            bgei  a14, 1, 0b
        1:
        ee.src.q.qup q2, q0, q1
        ee.vsubs.s16 q2, q2, q5
        dl_tie728_128b_unaligned_store0  q2, a2, a7
        j dl_tie728_s16_sub_w1_8_w2_1_unaligned_remainder

    // output sar = 0
    dl_tie728_s16_sub_w1_8_w2_1_unaligned_128b:
        mov  a14, a6
        beqz a14, 3f
        2:
            ee.src.q.qup q2, q0, q1
            ee.vsubs.s16 q2, q2, q5
            ee.ld.128.usar.ip q1, a3, 16
            ee.vst.128.ip  q2, a2, 16
            addi a14, a14, -1
            bgei  a14, 1, 2b
        3:
        ee.src.q.qup q2, q0, q1
        ee.vsubs.s16 q2, q2, q5
        ee.vst.128.ip  q2, a2, 16
        j dl_tie728_s16_sub_w1_8_w2_1_unaligned_remainder

    // output sar = 8
    dl_tie728_s16_sub_w1_8_w2_1_unaligned_64b:
        mov  a14, a6
        beqz a14, 5f
        4:
            ee.src.q.qup q2, q0, q1
            ee.vsubs.s16 q2, q2, q5
            ee.ld.128.usar.ip q1, a3, 16
            dl_tie728_128b_unaligned_store1  q2, a2
            addi a14, a14, -1
            bgei  a14, 1, 4b
        5:
        ee.src.q.qup q2, q0, q1
        ee.vsubs.s16 q2, q2, q5
        dl_tie728_128b_unaligned_store1  q2, a2
        j dl_tie728_s16_sub_w1_8_w2_1_unaligned_remainder

    dl_tie728_s16_sub_w1_8_w2_1_unaligned_remainder:
        beqz a10, dl_tie728_s16_sub_w1_8_w2_1_unaligned_end

        ee.ld.128.usar.xp q1, a3, a10
        ee.src.q q2, q0, q1
        ee.vsubs.s16 q2, q2, q5
        srli  a10, a10, 1
        dl_tie728_s16_store_remainder  q2, a10, a8, a2

    dl_tie728_s16_sub_w1_8_w2_1_unaligned_end:
        addi a3, a3, -16
    retw



    .align 4
    .text
    .global dl_tie728_s16_sub_w1_1_w2_8_unaligned
    .type   dl_tie728_s16_sub_w1_1_w2_8_unaligned, @function
    #.section .iram1
dl_tie728_s16_sub_w1_1_w2_8_unaligned:

    # a2: int16_t *output_ptr
    # a3: int16_t *input0_ptr broadcast
    # a4: int16_t *input1_ptr

    # a5: void *args
    # a6: c_div_x_1
    # a7: output sar_byte / tmp value
    # a8: tmp value
    # a9: tmp value
    # a10: c_remainder
    # a11:

    # a12:
    # a13:
    # a14: tmp value
    # a15:

    entry  sp, 128

    l32i  a6, a5, 64
    l32i  a10, a5, 76

    // input0 exp = input1 exp = output exp
    ee.ld.128.usar.ip q5, a2, 0    // output sar_byte
    rur.sar_byte  a7

    ee.vldbc.16.ip q5, a3, 0       // input0 broadcast

    ee.ld.128.usar.ip q0, a4, 16
    bltz a6, dl_tie728_s16_sub_w1_1_w2_8_unaligned_remainder  // channel < 16
    ee.ld.128.usar.ip q1, a4, 16

    beqz a7, dl_tie728_s16_sub_w1_1_w2_8_unaligned_128b
    movi a14, 8
    beq a7, a14, dl_tie728_s16_sub_w1_1_w2_8_unaligned_64b

        mov  a14, a6
        beqz a14, 1f
        0:
            ee.src.q.qup q2, q0, q1
            ee.vsubs.s16 q2, q5, q2
            ee.ld.128.usar.ip q1, a4, 16
            dl_tie728_128b_unaligned_store0  q2, a2, a7
            addi a14, a14, -1
            bgei  a14, 1, 0b
        1:
        ee.src.q.qup q2, q0, q1
        ee.vsubs.s16 q2, q5, q2
        dl_tie728_128b_unaligned_store0  q2, a2, a7
        j dl_tie728_s16_sub_w1_1_w2_8_unaligned_remainder

    // output sar = 0
    dl_tie728_s16_sub_w1_1_w2_8_unaligned_128b:
        mov  a14, a6
        beqz a14, 3f
        2:
            ee.src.q.qup q2, q0, q1
            ee.vsubs.s16 q2, q5, q2
            ee.ld.128.usar.ip q1, a4, 16
            ee.vst.128.ip  q2, a2, 16
            addi a14, a14, -1
            bgei  a14, 1, 2b
        3:
        ee.src.q.qup q2, q0, q1
        ee.vsubs.s16 q2, q5, q2
        ee.vst.128.ip  q2, a2, 16
        j dl_tie728_s16_sub_w1_1_w2_8_unaligned_remainder

    // output sar = 8
    dl_tie728_s16_sub_w1_1_w2_8_unaligned_64b:
        mov  a14, a6
        beqz a14, 5f
        4:
            ee.src.q.qup q2, q0, q1
            ee.vsubs.s16 q2, q5, q2
            ee.ld.128.usar.ip q1, a4, 16
            dl_tie728_128b_unaligned_store1  q2, a2
            addi a14, a14, -1
            bgei  a14, 1, 4b
        5:
        ee.src.q.qup q2, q0, q1
        ee.vsubs.s16 q2, q5, q2
        dl_tie728_128b_unaligned_store1  q2, a2
        j dl_tie728_s16_sub_w1_1_w2_8_unaligned_remainder

    dl_tie728_s16_sub_w1_1_w2_8_unaligned_remainder:
        beqz a10, dl_tie728_s16_sub_w1_1_w2_8_unaligned_end

        ee.ld.128.usar.xp q1, a4, a10
        ee.src.q q2, q0, q1
        ee.vsubs.s16 q2, q5, q2
        srli  a10, a10, 1
        dl_tie728_s16_store_remainder  q2, a10, a8, a2

    dl_tie728_s16_sub_w1_1_w2_8_unaligned_end:
        addi a4, a4, -16
    retw
