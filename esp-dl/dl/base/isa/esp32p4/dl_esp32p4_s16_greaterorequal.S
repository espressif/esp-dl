#include "dl_esp32p4_s8.S"
#include "dl_esp32p4_s16.S"
#include "dl_esp32p4_common.S"



#void dl_esp32p4_s16_greaterorequal_w1_8_w2_8(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int lenght);

    .align 2
    .text
    .global dl_esp32p4_s16_greaterorequal_w1_8_w2_8
    .type   dl_esp32p4_s16_greaterorequal_w1_8_w2_8, @function
    #.section .iram1
dl_esp32p4_s16_greaterorequal_w1_8_w2_8:

    # a0: bool *output_ptr
    # a1: int16_t *input0_ptr
    # a2: int16_t *input1_ptr

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

    addi  a4, a3, 100
    esp.vldbc.16.ip  q7, a4, 0
    lw a4, 44(a3)
    srai a3, a4, 3

    li t0, 0
esp32p4_s16_greaterorequal_w1_8_w2_8_loop:
    beq t0, a3, esp32p4_s16_greaterorequal_w1_8_w2_8_end
    esp.vld.128.ip q0, a1, 16
    esp.vld.128.ip q1, a2, 16
    esp.vcmp.gt.s16 q2, q0, q1
    esp.vcmp.eq.s16 q6, q0, q1
    esp.vmax.u16  q2, q2, q6
    esp.andq  q2, q2, q7
    esp.vunzip.8  q2, q6
    esp.vst.l.64.ip  q2, a0, 8
    addi t0, t0, 1
    j esp32p4_s16_greaterorequal_w1_8_w2_8_loop
esp32p4_s16_greaterorequal_w1_8_w2_8_end:
    ret



#void dl_esp32p4_s16_greaterorequal_w1_8_w2_1(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int lenght);

    .align 2
    .text
    .global dl_esp32p4_s16_greaterorequal_w1_8_w2_1
    .type   dl_esp32p4_s16_greaterorequal_w1_8_w2_1, @function
    #.section .iram1
dl_esp32p4_s16_greaterorequal_w1_8_w2_1:

    # a0: bool *output_ptr
    # a1: int16_t *input0_ptr
    # a2: int16_t *input1_ptr

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

    addi  a4, a3, 100
    esp.vldbc.16.ip  q7, a4, 0
    lw a4, 44(a3)
    srai a3, a4, 3
    esp.vldbc.16.ip q1, a2, 0       // input1 broadcast

    li t0, 0
esp32p4_s16_greaterorequal_w1_8_w2_1_loop:
    beq t0, a3, esp32p4_s16_greaterorequal_w1_8_w2_1_end
    esp.vld.128.ip q0, a1, 16
    esp.vcmp.gt.s16 q2, q0, q1
    esp.vcmp.eq.s16 q6, q0, q1
    esp.vmax.u16  q2, q2, q6
    esp.andq  q2, q2, q7
    esp.vunzip.8  q2, q6
    esp.vst.l.64.ip  q2, a0, 8
    addi t0, t0, 1
    j esp32p4_s16_greaterorequal_w1_8_w2_1_loop
esp32p4_s16_greaterorequal_w1_8_w2_1_end:
    ret



#void dl_esp32p4_s16_greaterorequal_w1_1_w2_8(bool *output_ptr, int16_t *input0_ptr, int16_t *input1_ptr, int lenght);

    .align 2
    .text
    .global dl_esp32p4_s16_greaterorequal_w1_1_w2_8
    .type   dl_esp32p4_s16_greaterorequal_w1_1_w2_8, @function
    #.section .iram1
dl_esp32p4_s16_greaterorequal_w1_1_w2_8:

    # a0: bool *output_ptr
    # a1: int16_t *input0_ptr
    # a2: int16_t *input1_ptr

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

    addi  a4, a3, 100
    esp.vldbc.16.ip  q7, a4, 0
    lw a4, 44(a3)
    srai a3, a4, 3
    esp.vldbc.16.ip q0, a1, 0       // input0 broadcast

    li t0, 0
esp32p4_s16_greaterorequal_w1_1_w2_8_loop:
    beq t0, a3, esp32p4_s16_greaterorequal_w1_1_w2_8_end
    esp.vld.128.ip q1, a2, 16
    esp.vcmp.gt.s16 q2, q0, q1
    esp.vcmp.eq.s16 q6, q0, q1
    esp.vmax.u16  q2, q2, q6
    esp.andq  q2, q2, q7
    esp.vunzip.8  q2, q6
    esp.vst.l.64.ip  q2, a0, 8
    addi t0, t0, 1
    j esp32p4_s16_greaterorequal_w1_1_w2_8_loop
esp32p4_s16_greaterorequal_w1_1_w2_8_end:
    ret



    .align 2
    .text
    .global dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned
    .type   dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned, @function
    #.section .iram1
dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned:

    # a0: bool *output_ptr
    # a1: int16_t *input0_ptr
    # a2: int16_t *input1_ptr

    # a3: void *args
    # a4: c_div_x_1
    # a5: output sar_byte / tmp value
    # t3: tmp value
    # t4:
    # t5: c_remainder
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

    addi  t3, a3, 100
    esp.vldbc.16.ip  q7, t3, 0
    lw a4, 64(a3)
    lw t5, 76(a3)

    // input0 exp = input1 exp = output exp

    esp.ld.128.usar.ip q5, a0, 0    // get output_ptr sar_byte
    esp.movx.r.sar.bytes a5

    esp.ld.128.usar.ip q0, a1, 16
    esp.ld.128.usar.ip q3, a2, 16
    bltz a4, dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned_remainder     // channel < 16
    esp.ld.128.usar.ip q1, a1, 16

    andi  t0, a5, 7
    beqz  t0, dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned_64b
        mv  t0, a4
        beqz t0, 1f
        0:
            esp.src.q.qup q2, q0, q1
            esp.ld.128.usar.ip q4, a2, 16
            esp.src.q.qup q5, q3, q4
            esp.vcmp.gt.s16 q1, q2, q5
            esp.vcmp.eq.s16 q6, q2, q5
            esp.vmax.u16  q2, q1, q6
            esp.andq  q2, q2, q7
            esp.vunzip.8  q2, q6
            esp.ld.128.usar.ip q1, a1, 16
            esp32p4_s16_32b_unaligned_l_vector_store  q2, a0, a5
            addi t0, t0, -1
            bgtz t0, 0b
        1:
        esp.src.q.qup q2, q0, q1
        esp.ld.128.usar.ip q4, a2, 16
        esp.src.q.qup q5, q3, q4
        esp.vcmp.gt.s16 q1, q2, q5
        esp.vcmp.eq.s16 q6, q2, q5
        esp.vmax.u16  q2, q1, q6
        esp.andq  q2, q2, q7
        esp.vunzip.8  q2, q6
        esp32p4_s16_32b_unaligned_l_vector_store  q2, a0, a5
        j dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned_remainder

    // output sar = 0 or output sar = 8
    dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned_64b:
        mv  t0, a4
        beqz t0, 3f
        2:
            esp.src.q.qup q2, q0, q1
            esp.ld.128.usar.ip q4, a2, 16
            esp.src.q.qup q5, q3, q4
            esp.vcmp.gt.s16 q1, q2, q5
            esp.vcmp.eq.s16 q6, q2, q5
            esp.vmax.u16  q2, q1, q6
            esp.andq  q2, q2, q7
            esp.vunzip.8  q2, q6
            esp.ld.128.usar.ip q1, a1, 16
            // esp.vst.128.ip q2, a0, 16
            esp.vst.l.64.ip  q2, a0, 8
            addi t0, t0, -1
            bgtz t0, 2b
        3:
        esp.src.q.qup q2, q0, q1
        esp.ld.128.usar.ip q4, a2, 16
        esp.src.q.qup q5, q3, q4
        esp.vcmp.gt.s16 q1, q2, q5
        esp.vcmp.eq.s16 q6, q2, q5
        esp.vmax.u16  q2, q1, q6
        esp.andq  q2, q2, q7
        esp.vunzip.8  q2, q6
        esp.vst.l.64.ip  q2, a0, 8

        // j dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned_remainder

    dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned_remainder:
            beqz t5, dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned_end

            esp.ld.128.usar.xp q1, a1, t5
            esp.src.q q2, q0, q1
            esp.ld.128.usar.xp q4, a2, t5
            esp.src.q q5, q3, q4

            esp.vcmp.gt.s16 q1, q2, q5
            esp.vcmp.eq.s16 q6, q2, q5
            esp.vmax.u16  q2, q1, q6
            esp.andq  q2, q2, q7
            esp.vunzip.8  q2, q6
            srli  t5, t5, 1
            dl_esp32p4_s8_store_remainder q2, t4, t6, a5, t3, t0, a0, t5

    dl_esp32p4_s16_greaterorequal_w1_8_w2_8_unaligned_end:
        addi a1, a1, -16
        addi a2, a2, -16
    ret



    .align 2
    .text
    .global dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned
    .type   dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned, @function
    #.section .iram1
dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned:

    # a0: bool *output_ptr
    # a1: int16_t *input0_ptr
    # a2: int16_t *input1_ptr broadcast

    # a3: void *args
    # a4: c_div_x_1
    # a5: output sar_byte / tmp value
    # t3: tmp value
    # t4: tmp value
    # t5: c_remainder
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

    addi  t3, a3, 100
    esp.vldbc.16.ip  q7, t3, 0
    lw a4, 64(a3)
    lw t5, 76(a3)

    // input0 exp = input1 exp = output exp

    esp.ld.128.usar.ip q5, a0, 0    // get output_ptr sar_byte
    esp.movx.r.sar.bytes a5
    esp.vldbc.16.ip q5, a2, 0       // input1 broadcast

    esp.ld.128.usar.ip q0, a1, 16
    bltz a4, dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned_remainder  // channel < 16
    esp.ld.128.usar.ip q1, a1, 16

    andi  t0, a5, 7
    beqz  t0, dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned_64b
        mv  t0, a4
        beqz t0, 1f
        0:
            esp.src.q.qup q2, q0, q1
            esp.vcmp.gt.s16 q1, q2, q5
            esp.vcmp.eq.s16 q6, q2, q5
            esp.vmax.u16  q2, q1, q6
            esp.andq  q2, q2, q7
            esp.vunzip.8  q2, q6
            esp.ld.128.usar.ip q1, a1, 16
            esp32p4_s16_32b_unaligned_l_vector_store  q2, a0, a5
            addi t0, t0, -1
            bgtz t0, 0b
        1:
        esp.src.q.qup q2, q0, q1
        esp.vcmp.gt.s16 q1, q2, q5
        esp.vcmp.eq.s16 q6, q2, q5
        esp.vmax.u16  q2, q1, q6
        esp.andq  q2, q2, q7
        esp.vunzip.8  q2, q6
        esp32p4_s16_32b_unaligned_l_vector_store  q2, a0, a5
        j dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned_remainder

    // output sar = 0 or output sar = 8
    dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned_64b:
        mv  t0, a4
        beqz t0, 3f
        2:
            esp.src.q.qup q2, q0, q1
            esp.vcmp.gt.s16 q1, q2, q5
            esp.vcmp.eq.s16 q6, q2, q5
            esp.vmax.u16  q2, q1, q6
            esp.andq  q2, q2, q7
            esp.vunzip.8  q2, q6
            esp.ld.128.usar.ip q1, a1, 16
            esp.vst.l.64.ip  q2, a0, 8
            addi t0, t0, -1
            bgtz t0, 2b
        3:
        esp.src.q.qup q2, q0, q1
        esp.vcmp.gt.s16 q1, q2, q5
        esp.vcmp.eq.s16 q6, q2, q5
        esp.vmax.u16  q2, q1, q6
        esp.andq  q2, q2, q7
        esp.vunzip.8  q2, q6
        esp.vst.l.64.ip  q2, a0, 8
        // j dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned_remainder

    dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned_remainder:
        beqz t5, dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned_end

        esp.ld.128.usar.xp q1, a1, t5
        esp.src.q q2, q0, q1
        esp.vcmp.gt.s16 q1, q2, q5
        esp.vcmp.eq.s16 q6, q2, q5
        esp.vmax.u16  q2, q1, q6
        esp.andq  q2, q2, q7
        esp.vunzip.8  q2, q6
        srli  t5, t5, 1
        dl_esp32p4_s8_store_remainder q2, t4, t6, a5, t3, t0, a0, t5

    dl_esp32p4_s16_greaterorequal_w1_8_w2_1_unaligned_end:
        addi a1, a1, -16
    ret



    .align 2
    .text
    .global dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned
    .type   dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned, @function
    #.section .iram1
dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned:

    # a0: bool *output_ptr
    # a1: int16_t *input0_ptr broadcast
    # a2: int16_t *input1_ptr

    # a3: void *args
    # a4: c_div_x_1
    # a5: output sar_byte / tmp value
    # t3: tmp value
    # t4: tmp value
    # t5: c_remainder
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

    addi  t3, a3, 100
    esp.vldbc.16.ip  q7, t3, 0
    lw a4, 64(a3)
    lw t5, 76(a3)

    // input0 exp = input1 exp = output exp
    esp.ld.128.usar.ip q5, a0, 0    // output sar_byte
    esp.movx.r.sar.bytes a5
    esp.vldbc.16.ip q5, a1, 0       // input0 broadcast

    esp.ld.128.usar.ip q0, a2, 16
    bltz a4, dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned_remainder  // channel < 16
    esp.ld.128.usar.ip q1, a2, 16

    andi  t0, a5, 7
    beqz  t0, dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned_64b
        mv  t0, a4
        beqz t0, 1f
        0:
            esp.src.q.qup q2, q0, q1
            esp.vcmp.gt.s16 q1, q5, q2
            esp.vcmp.eq.s16 q6, q5, q2
            esp.vmax.u16  q2, q1, q6
            esp.andq  q2, q2, q7
            esp.vunzip.8  q2, q6
            esp.ld.128.usar.ip q1, a2, 16
            esp32p4_s16_32b_unaligned_l_vector_store  q2, a0, a5
            addi t0, t0, -1
            bgtz t0, 0b
        1:
        esp.src.q.qup q2, q0, q1
        esp.vcmp.gt.s16 q1, q5, q2
        esp.vcmp.eq.s16 q6, q5, q2
        esp.vmax.u16  q2, q1, q6
        esp.andq  q2, q2, q7
        esp.vunzip.8  q2, q6
        esp32p4_s16_32b_unaligned_l_vector_store  q2, a0, a5
        j dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned_remainder

    // output sar = 0 or output sar = 8
    dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned_64b:
        mv  t0, a4
        beqz t0, 3f
        2:
            esp.src.q.qup q2, q0, q1
            esp.vcmp.gt.s16 q1, q5, q2
            esp.vcmp.eq.s16 q6, q5, q2
            esp.vmax.u16  q2, q1, q6
            esp.andq  q2, q2, q7
            esp.vunzip.8  q2, q6
            esp.ld.128.usar.ip q1, a2, 16
            esp.vst.l.64.ip  q2, a0, 8
            addi t0, t0, -1
            bgtz t0, 2b
        3:
        esp.src.q.qup q2, q0, q1
        esp.vcmp.gt.s16 q1, q5, q2
        esp.vcmp.eq.s16 q6, q5, q2
        esp.vmax.u16  q2, q1, q6
        esp.andq  q2, q2, q7
        esp.vunzip.8  q2, q6
        esp.vst.l.64.ip  q2, a0, 8
        // j dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned_remainder

    dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned_remainder:
        beqz t5, dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned_end

        esp.ld.128.usar.xp q1, a2, t5
        esp.src.q q2, q0, q1
        esp.vcmp.gt.s16 q1, q5, q2
        esp.vcmp.eq.s16 q6, q5, q2
        esp.vmax.u16  q2, q1, q6
        esp.andq  q2, q2, q7
        esp.vunzip.8  q2, q6
        srli  t5, t5, 1
        dl_esp32p4_s8_store_remainder q2, t4, t6, a5, t3, t0, a0, t5

    dl_esp32p4_s16_greaterorequal_w1_1_w2_8_unaligned_end:
        addi a2, a2, -16
    ret
