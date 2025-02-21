#include "dl_esp32p4_s8.S"
#include "dl_esp32p4_common.S"



#void dl_esp32p4_s8_add_w1_16_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, int lenght);

    .align 2
    .text
    .global dl_esp32p4_s8_add_w1_16_w2_16
    .type   dl_esp32p4_s8_add_w1_16_w2_16, @function
    #.section .iram1
dl_esp32p4_s8_add_w1_16_w2_16:

    # a0: int8_t *output_ptr
    # a1: int8_t *input0_ptr
    # a2: int8_t *input1_ptr

    # a3: void *args / length
    # a4: tmp value
    # a5:
    # t3:
    # t4:
    # t5:
    # t6:

    # a6(not for extension instructions):
    # a7(not for extension instructions):
    # t0(not for extension instructions):
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

    lw a4, 44(a3)
    srai a3, a4, 4

    li t0, 0
esp32p4_s8_add_w1_16_w2_16_loop:
    beq t0, a3, esp32p4_s8_add_w1_16_w2_16_end
    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a2, 16
    esp.vadd.s8 q2, q0, q1
    esp.vst.128.ip q2, a0, 16
    addi t0, t0, 1
    j esp32p4_s8_add_w1_16_w2_16_loop
esp32p4_s8_add_w1_16_w2_16_end:
    ret



#void dl_esp32p4_s8_add_w1_16_w2_1(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, int lenght);

    .align 2
    .text
    .global dl_esp32p4_s8_add_w1_16_w2_1
    .type   dl_esp32p4_s8_add_w1_16_w2_1, @function
    #.section .iram1
dl_esp32p4_s8_add_w1_16_w2_1:

    # a0: int8_t *output_ptr
    # a1: int8_t *input0_ptr
    # a2: int8_t *input1_ptr

    # a3: void *args / length
    # a4: tmp value
    # a5:
    # t3:
    # t4:
    # t5:
    # t6:

    # a6(not for extension instructions):
    # a7(not for extension instructions):
    # t0(not for extension instructions):
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

    lw a4, 44(a3)
    srai a3, a4, 4
    esp.vldbc.8.ip q1, a2, 0

    li t0, 0
esp32p4_s8_add_w1_16_w2_1_loop:
    beq t0, a3, esp32p4_s8_add_w1_16_w2_1_end
    esp.vld.128.ip q0, a1, 16
    esp.vadd.s8 q2, q0, q1
    esp.vst.128.ip q2, a0, 16
    addi t0, t0, 1
    j esp32p4_s8_add_w1_16_w2_1_loop
esp32p4_s8_add_w1_16_w2_1_end:
    ret



#void dl_esp32p4_s8_add_w1_1_w2_16(int8_t *output_ptr, int8_t *input0_ptr, int8_t *input1_ptr, int lenght);

    .align 2
    .text
    .global dl_esp32p4_s8_add_w1_1_w2_16
    .type   dl_esp32p4_s8_add_w1_1_w2_16, @function
    #.section .iram1
dl_esp32p4_s8_add_w1_1_w2_16:

    # a0: int8_t *output_ptr
    # a1: int8_t *input0_ptr
    # a2: int8_t *input1_ptr

    # a3: void *args / length
    # a4: tmp value
    # a5:
    # t3:
    # t4:
    # t5:
    # t6:

    # a6(not for extension instructions):
    # a7(not for extension instructions):
    # t0(not for extension instructions):
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

    lw a4, 44(a3)
    srai a3, a4, 4
    esp.vldbc.8.ip q0, a1, 0

    li t0, 0
esp32p4_s8_add_w1_1_w2_16_loop:
    beq t0, a3, esp32p4_s8_add_w1_1_w2_16_end
    esp.vld.128.ip q1, a2, 16
    esp.vadd.s8 q2, q0, q1
    esp.vst.128.ip q2, a0, 16
    addi t0, t0, 1
    j esp32p4_s8_add_w1_1_w2_16_loop
esp32p4_s8_add_w1_1_w2_16_end:
    ret



    .align 2
    .text
    .global dl_esp32p4_s8_add_w1_16_w2_16_unaligned
    .type   dl_esp32p4_s8_add_w1_16_w2_16_unaligned, @function
    #.section .iram1
dl_esp32p4_s8_add_w1_16_w2_16_unaligned:

    # a0: int8_t *output_ptr
    # a1: int8_t *input0_ptr
    # a2: int8_t *input1_ptr

    # a3: void *args
    # a4: c_div_x_1
    # a5: output sar_byte / tmp value
    # t3: tmp value
    # t4:
    # t5: c_remainder
    # t6:

    # a6(not for extension instructions):
    # a7(not for extension instructions):
    # t0(not for extension instructions):
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

    lw a4, 64(a3)
    lw t5, 76(a3)

    // input0 exp = input1 exp = output exp

    esp.ld.128.usar.ip q5, a0, 0    // get output_ptr sar_byte
    esp.movx.r.sar.bytes a5

    esp.ld.128.usar.ip q0, a1, 16
    esp.ld.128.usar.ip q3, a2, 16
    bltz a4, dl_esp32p4_s8_add_w1_16_w2_16_unaligned_remainder     // channel < 16
    esp.ld.128.usar.ip q1, a1, 16

    beqz a5, dl_esp32p4_s8_add_w1_16_w2_16_unaligned_128b
    li t0, 8
    beq a5, t0, dl_esp32p4_s8_add_w1_16_w2_16_unaligned_64b

        mv  t0, a4
        beqz t0, 1f
        0:
            esp.src.q.qup q2, q0, q1
            esp.ld.128.usar.ip q4, a2, 16
            esp.src.q.qup q5, q3, q4
            esp.vadd.s8 q2, q2, q5
            esp.ld.128.usar.ip q1, a1, 16
            esp32p4_s8_32b_unaligned_vector_store q2, a0, a5
            addi t0, t0, -1
            bgtz t0, 0b
        1:
        esp.src.q.qup q2, q0, q1
        esp.ld.128.usar.ip q4, a2, 16
        esp.src.q.qup q5, q3, q4
        esp.vadd.s8 q2, q2, q5
        esp32p4_s8_32b_unaligned_vector_store q2, a0, a5
        j dl_esp32p4_s8_add_w1_16_w2_16_unaligned_remainder

    // output sar = 0
    dl_esp32p4_s8_add_w1_16_w2_16_unaligned_128b:
        mv  t0, a4
        beqz t0, 3f
        2:
            esp.src.q.qup q2, q0, q1
            esp.ld.128.usar.ip q4, a2, 16
            esp.src.q.qup q5, q3, q4
            esp.vadd.s8 q2, q2, q5
            esp.ld.128.usar.ip q1, a1, 16
            esp.vst.128.ip q2, a0, 16
            addi t0, t0, -1
            bgtz t0, 2b
        3:
        esp.src.q.qup q2, q0, q1
        esp.ld.128.usar.ip q4, a2, 16
        esp.src.q.qup q5, q3, q4
        esp.vadd.s8 q2, q2, q5
        esp.vst.128.ip q2, a0, 16

        j dl_esp32p4_s8_add_w1_16_w2_16_unaligned_remainder

    // output sar = 8
    dl_esp32p4_s8_add_w1_16_w2_16_unaligned_64b:
        mv  t0, a4
        beqz t0, 5f
        4:
            esp.src.q.qup q2, q0, q1
            esp.ld.128.usar.ip q4, a2, 16
            esp.src.q.qup q5, q3, q4
            esp.vadd.s8 q2, q2, q5
            esp.ld.128.usar.ip q1, a1, 16
            esp32p4_s8_64b_unaligned_vector_store q2, a0
            addi t0, t0, -1
            bgtz t0, 4b
        5:
        esp.src.q.qup q2, q0, q1
        esp.ld.128.usar.ip q4, a2, 16
        esp.src.q.qup q5, q3, q4
        esp.vadd.s8 q2, q2, q5
        esp32p4_s8_64b_unaligned_vector_store q2, a0
        j dl_esp32p4_s8_add_w1_16_w2_16_unaligned_remainder

    dl_esp32p4_s8_add_w1_16_w2_16_unaligned_remainder:
            beqz t5, dl_esp32p4_s8_add_w1_16_w2_16_unaligned_end

            esp.ld.128.usar.xp q1, a1, t5
            esp.src.q q2, q0, q1
            esp.ld.128.usar.xp q4, a2, t5
            esp.src.q q5, q3, q4
            esp.vadd.s8 q2, q2, q5
            dl_esp32p4_s8_store_remainder q2, t4, t6, a5, t3, t0, a0, t5

    dl_esp32p4_s8_add_w1_16_w2_16_unaligned_end:
        addi a1, a1, -16
        addi a2, a2, -16
    ret



    .align 2
    .text
    .global dl_esp32p4_s8_add_w1_16_w2_1_unaligned
    .type   dl_esp32p4_s8_add_w1_16_w2_1_unaligned, @function
    #.section .iram1
dl_esp32p4_s8_add_w1_16_w2_1_unaligned:

    # a0: int8_t *output_ptr
    # a1: int8_t *input0_ptr
    # a2: int8_t *input1_ptr broadcast

    # a3: void *args
    # a4: c_div_x_1
    # a5: output sar_byte / tmp value
    # t3: tmp value
    # t4: tmp value
    # t5: c_remainder
    # t6:

    # a6(not for extension instructions):
    # a7(not for extension instructions):
    # t0(not for extension instructions):
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

    lw a4, 64(a3)
    lw t5, 76(a3)

    // input0 exp = input1 exp = output exp

    esp.ld.128.usar.ip q5, a0, 0    // get output_ptr sar_byte
    esp.movx.r.sar.bytes a5

    esp.vldbc.8.ip q5, a2, 0        // input1 broadcast

    esp.ld.128.usar.ip q0, a1, 16
    bltz a4, dl_esp32p4_s8_add_w1_16_w2_1_unaligned_remainder  // channel < 16
    esp.ld.128.usar.ip q1, a1, 16

    beqz a5, dl_esp32p4_s8_add_w1_16_w2_1_unaligned_128b
    li t0, 8
    beq a5, t0, dl_esp32p4_s8_add_w1_16_w2_1_unaligned_64b

        mv  t0, a4
        beqz t0, 1f
        0:
            esp.src.q.qup q2, q0, q1
            esp.vadd.s8 q2, q2, q5
            esp.ld.128.usar.ip q1, a1, 16
            esp32p4_s8_32b_unaligned_vector_store q2, a0, a5
            addi t0, t0, -1
            bgtz t0, 0b
        1:
        esp.src.q.qup q2, q0, q1
        esp.vadd.s8 q2, q2, q5
        esp32p4_s8_32b_unaligned_vector_store q2, a0, a5
        j dl_esp32p4_s8_add_w1_16_w2_1_unaligned_remainder

    // output sar = 0
    dl_esp32p4_s8_add_w1_16_w2_1_unaligned_128b:
        mv  t0, a4
        beqz t0, 3f
        2:
            esp.src.q.qup q2, q0, q1
            esp.vadd.s8 q2, q2, q5
            esp.ld.128.usar.ip q1, a1, 16
            esp.vst.128.ip q2, a0, 16
            addi t0, t0, -1
            bgtz t0, 2b
        3:
        esp.src.q.qup q2, q0, q1
        esp.vadd.s8 q2, q2, q5
        esp.vst.128.ip q2, a0, 16
        j dl_esp32p4_s8_add_w1_16_w2_1_unaligned_remainder

    // output sar = 8
    dl_esp32p4_s8_add_w1_16_w2_1_unaligned_64b:
        mv  t0, a4
        beqz t0, 5f
        4:
            esp.src.q.qup q2, q0, q1
            esp.vadd.s8 q2, q2, q5
            esp.ld.128.usar.ip q1, a1, 16
            esp32p4_s8_64b_unaligned_vector_store q2, a0
            addi t0, t0, -1
            bgtz t0, 4b
        5:
        esp.src.q.qup q2, q0, q1
        esp.vadd.s8 q2, q2, q5
        esp32p4_s8_64b_unaligned_vector_store q2, a0
        j dl_esp32p4_s8_add_w1_16_w2_1_unaligned_remainder

    dl_esp32p4_s8_add_w1_16_w2_1_unaligned_remainder:
        beqz t5, dl_esp32p4_s8_add_w1_16_w2_1_unaligned_end

        esp.ld.128.usar.xp q1, a1, t5
        esp.src.q q2, q0, q1
        esp.vadd.s8 q2, q2, q5
        dl_esp32p4_s8_store_remainder q2, t4, t6, a5, t3, t0, a0, t5

    dl_esp32p4_s8_add_w1_16_w2_1_unaligned_end:
        addi a1, a1, -16
    ret



    .align 2
    .text
    .global dl_esp32p4_s8_add_w1_1_w2_16_unaligned
    .type   dl_esp32p4_s8_add_w1_1_w2_16_unaligned, @function
    #.section .iram1
dl_esp32p4_s8_add_w1_1_w2_16_unaligned:

    # a0: int8_t *output_ptr
    # a1: int8_t *input0_ptr broadcast
    # a2: int8_t *input1_ptr

    # a3: void *args
    # a4: c_div_x_1
    # a5: output sar_byte / tmp value
    # t3: tmp value
    # t4: tmp value
    # t5: c_remainder
    # t6:

    # a6(not for extension instructions):
    # a7(not for extension instructions):
    # t0(not for extension instructions):
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

    lw a4, 64(a3)
    lw t5, 76(a3)

    // input0 exp = input1 exp = output exp
    esp.ld.128.usar.ip q5, a0, 0    // output sar_byte
    esp.movx.r.sar.bytes a5

    esp.vldbc.8.ip q5, a1, 0        // input0 broadcast

    esp.ld.128.usar.ip q0, a2, 16
    bltz a4, dl_esp32p4_s8_add_w1_1_w2_16_unaligned_remainder  // channel < 16
    esp.ld.128.usar.ip q1, a2, 16

    beqz a5, dl_esp32p4_s8_add_w1_1_w2_16_unaligned_128b
    li t0, 8
    beq a5, t0, dl_esp32p4_s8_add_w1_1_w2_16_unaligned_64b

        mv  t0, a4
        beqz t0, 1f
        0:
            esp.src.q.qup q2, q0, q1
            esp.vadd.s8 q2, q5, q2
            esp.ld.128.usar.ip q1, a2, 16
            esp32p4_s8_32b_unaligned_vector_store q2, a0, a5
            addi t0, t0, -1
            bgtz t0, 0b
        1:
        esp.src.q.qup q2, q0, q1
        esp.vadd.s8 q2, q5, q2
        esp32p4_s8_32b_unaligned_vector_store q2, a0, a5
        j dl_esp32p4_s8_add_w1_1_w2_16_unaligned_remainder

    // output sar = 0
    dl_esp32p4_s8_add_w1_1_w2_16_unaligned_128b:
        mv  t0, a4
        beqz t0, 3f
        2:
            esp.src.q.qup q2, q0, q1
            esp.vadd.s8 q2, q5, q2
            esp.ld.128.usar.ip q1, a2, 16
            esp.vst.128.ip q2, a0, 16
            addi t0, t0, -1
            bgtz t0, 2b
        3:
        esp.src.q.qup q2, q0, q1
        esp.vadd.s8 q2, q5, q2
        esp.vst.128.ip q2, a0, 16
        j dl_esp32p4_s8_add_w1_1_w2_16_unaligned_remainder

    // output sar = 8
    dl_esp32p4_s8_add_w1_1_w2_16_unaligned_64b:
        mv  t0, a4
        beqz t0, 5f
        4:
            esp.src.q.qup q2, q0, q1
            esp.vadd.s8 q2, q5, q2
            esp.ld.128.usar.ip q1, a2, 16
            esp32p4_s8_64b_unaligned_vector_store q2, a0
            addi t0, t0, -1
            bgtz t0, 4b
        5:
        esp.src.q.qup q2, q0, q1
        esp.vadd.s8 q2, q5, q2
        esp32p4_s8_64b_unaligned_vector_store q2, a0
        j dl_esp32p4_s8_add_w1_1_w2_16_unaligned_remainder

    dl_esp32p4_s8_add_w1_1_w2_16_unaligned_remainder:
        beqz t5, dl_esp32p4_s8_add_w1_1_w2_16_unaligned_end

        esp.ld.128.usar.xp q1, a2, t5
        esp.src.q q2, q0, q1
        esp.vadd.s8 q2, q5, q2
        dl_esp32p4_s8_store_remainder q2, t4, t6, a5, t3, t0, a0, t5

    dl_esp32p4_s8_add_w1_1_w2_16_unaligned_end:
        addi a2, a2, -16
    ret
