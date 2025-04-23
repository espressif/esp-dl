#include "dl_esp32p4_s8.S"
#include "dl_esp32p4_common.S"

    .align 2
    .text
    .global dl_esp32p4_s8_resize_nearest_2x2_c1
    .type dl_esp32p4_s8_resize_nearest_2x2_c1, @function
dl_esp32p4_s8_resize_nearest_2x2_c1:
    .align 2
    esp32p4_push_128_stacks_4r  s0, s1, s8, s9
    # a0: int8_t *output_ptr
    # a1: int8_t *input_ptr
    # a2: void *args
    # a3: output_x_offset
    # a4: output_y_offset
    # a5: c_div_x
    # t3: remainder
    # t4: output_shift
    # t5: output_scale

    lw a3, 20(a2)
    lw a4, 24(a2)
    lw a5, 40(a2)
    lw t3, 44(a2)
    lw t4, 48(a2)
    lw t5, 52(a2)

    # t6 (0, 1)
    # s0 (1, 0)
    # s1 (1, 1)
    add t6, a0, a3
    add s0, a0, a4
    add s1, t6, a4

    sb t5, 0(sp)
    add s8, sp, x0
    esp.vldbc.8.ip q1, s8, 0 # all output_scale
    esp.vld.128.ip q0, a1, 16

    add t0, a5, x0
    blez t0, 1f
    0:
        esp.zero.qacc
        esp.vmulas.s8.qacc.ld.ip q0, a1, 16, q0, q1
        esp.srcmb.s8.qacc q2, t4, 1
        esp.vst.128.ip q2, a0, 16
        esp.vst.128.ip q2, t6, 16
        esp.vst.128.ip q2, s0, 16
        esp.vst.128.ip q2, s1, 16
        addi t0, t0, -1
        bgtz t0, 0b
    1:

    esp32p4_pop_128_stacks_4r  s0, s1, s8, s9
    ret



    .text
    .align 2
    .global dl_esp32p4_s8_resize_nearest_c1
    .type   dl_esp32p4_s8_resize_nearest_c1, @function
    .balign 4
    .option norvc
dl_esp32p4_s8_resize_nearest_c1:

    # a0: int8_t *output_ptr
    # a1: int8_t *input_ptr
    # a2: void *args

    # a3: c_div_x
    # a4: output_shift
    # a5: output_scale address
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

    lw  a3, 40(a2)
    lw  a4, 48(a2)
    blez  a3, dl_esp32p4_s8_resize_nearest_c1_end

    addi  a5, a2, 52
    esp.vldbc.8.ip  q1, a5, 0   // load output_scale
    esp.vld.128.ip  q0, a1, 16

    dl_esp32p4_s8_resize_nearest_c1_loop:
        esp.zero.qacc
        esp.vmulas.s8.qacc.ld.ip  q0, a1, 16, q0, q1
        esp32p4_s8_128b_vector_shift_result  q2, a4
        esp32p4_s8_128b_aligned_vector_store  q2, a0
        addi  a3, a3, -1
        bgtz  a3, dl_esp32p4_s8_resize_nearest_c1_loop

    dl_esp32p4_s8_resize_nearest_c1_end:
        ret



    .align 2
    .text
    .global dl_esp32p4_s8_unaligned_resize_nearest_2x2_c1
    .type dl_esp32p4_s8_unaligned_resize_nearest_2x2_c1, @function
dl_esp32p4_s8_unaligned_resize_nearest_2x2_c1:
    .align 2
    esp32p4_push_128_stacks_4r  s0, s1, s8, s9
    # a0: int8_t *output_ptr
    # a1: int8_t *input_ptr
    # a2: void *args
    # a3: output_x_offset
    # a4: output_y_offset
    # a5: c_div_x
    # t3: remainder
    # t4: output_shift
    # t5: output_scale


    lw a3, 20(a2)
    lw a4, 24(a2)
    lw a5, 40(a2)
    lw t3, 44(a2)
    lw t4, 48(a2)
    lw t5, 52(a2)


    # t6 (0, 1)
    # s0 (1, 0)
    # s1 (1, 1)
    add t6, a0, a3
    add s0, a0, a4
    add s1, t6, a4

    sb t5, 0(sp)
    add s8, sp, x0
    esp.vldbc.8.ip q3, s8, 0 # all output_scale
    esp.ld.128.usar.ip q0, a1, 16

    add t0, a5, x0
    blez t0, 1f
    0:
        esp.zero.qacc
        esp.ld.128.usar.ip q1, a1, 16
        esp.src.q.qup q2, q0, q1
        esp.vmulas.s8.qacc q2, q3
        esp.srcmb.s8.qacc q4, t4, 1
        esp32p4_s8_32b_unaligned_vector_store q4, a0, s8
        esp32p4_s8_32b_unaligned_vector_store q4, t6, s8
        esp32p4_s8_32b_unaligned_vector_store q4, s0, s8
        esp32p4_s8_32b_unaligned_vector_store q4, s1, s8
        addi t0, t0, -1
        bgtz t0, 0b
    1:

    bnez t3, dl_esp32p4_s8_unaligned_resize_nearest_2x2_c1_remainder
    esp32p4_pop_128_stacks_4r  s0, s1, s8, s9
    ret

dl_esp32p4_s8_unaligned_resize_nearest_2x2_c1_remainder:
    esp.zero.qacc
    esp.ld.128.usar.ip q1, a1, 16
    esp.src.q.qup q2, q0, q1
    esp.vmulas.s8.qacc q2, q3
    esp.srcmb.s8.qacc q4, t4, 1
    dl_esp32p4_s8_store_remainder q4, t4, t5, s8, s9, t0, a0, t3
    dl_esp32p4_s8_store_remainder q4, t4, t5, s8, s9, t0, t6, t3
    dl_esp32p4_s8_store_remainder q4, t4, t5, s8, s9, t0, s0, t3
    dl_esp32p4_s8_store_remainder q4, t4, t5, s8, s9, t0, s1, t3
    esp32p4_pop_128_stacks_4r  s0, s1, s8, s9
    ret



    .text
    .align 2
    .global dl_esp32p4_s8_unaligned_resize_nearest_c1
    .type   dl_esp32p4_s8_unaligned_resize_nearest_c1, @function
    .balign 4
    .option norvc
dl_esp32p4_s8_unaligned_resize_nearest_c1:

    # a0: int8_t *output_ptr
    # a1: int8_t *input_ptr
    # a2: void *args

    # a3: c_div_x / tmp value
    # a4: output_shift
    # a5: output_scale address
    # t3: c_remainder
    # t4: output_sar_byte / tmp value
    # t5: tmp value
    # t6:

    # a6(not for extension instructions): tmp value
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

    lw  a3, 40(a2)
    lw  a4, 48(a2)
    lw  t3, 44(a2)
    blez  a3, dl_esp32p4_s8_unaligned_resize_nearest_c1_remainder

    addi  a5, a2, 52
    esp.vldbc.8.ip  q3, a5, 0   // load output_scale
    esp.ld.128.usar.ip  q0, a1, 16

    esp.ld.128.usar.ip  q2, a0, 0
    esp.movx.r.sar.bytes  t4            // output_sar_byte

    beqz  t4, dl_esp32p4_s8_unaligned_resize_nearest_c1_128b_loop
    li  a6, 8
    beq  t4, a6, dl_esp32p4_s8_unaligned_resize_nearest_c1_64b_loop

    dl_esp32p4_s8_unaligned_resize_nearest_c1_32b_loop:
        esp.ld.128.usar.ip  q1, a1, 16
        esp.zero.qacc
        esp.src.q.qup  q2, q0, q1
        esp.vmulas.s8.qacc  q2, q3
        esp32p4_s8_128b_vector_shift_result  q2, a4
        esp32p4_s8_32b_unaligned_vector_store  q2, a0, t4
        addi  a3, a3, -1
        bgtz  a3, dl_esp32p4_s8_unaligned_resize_nearest_c1_32b_loop
    j  dl_esp32p4_s8_unaligned_resize_nearest_c1_remainder

    dl_esp32p4_s8_unaligned_resize_nearest_c1_64b_loop:
        esp.ld.128.usar.ip  q1, a1, 16
        esp.zero.qacc
        esp.src.q.qup  q2, q0, q1
        esp.vmulas.s8.qacc  q2, q3
        esp32p4_s8_128b_vector_shift_result  q2, a4
        esp32p4_s8_64b_unaligned_vector_store  q2, a0
        addi  a3, a3, -1
        bgtz  a3, dl_esp32p4_s8_unaligned_resize_nearest_c1_64b_loop
    j  dl_esp32p4_s8_unaligned_resize_nearest_c1_remainder

    dl_esp32p4_s8_unaligned_resize_nearest_c1_128b_loop:
        esp.ld.128.usar.ip  q1, a1, 16
        esp.zero.qacc
        esp.src.q.qup  q2, q0, q1
        esp.vmulas.s8.qacc  q2, q3
        esp32p4_s8_128b_vector_shift_result  q2, a4
        esp32p4_s8_128b_aligned_vector_store  q2, a0
        addi  a3, a3, -1
        bgtz  a3, dl_esp32p4_s8_unaligned_resize_nearest_c1_128b_loop

    dl_esp32p4_s8_unaligned_resize_nearest_c1_remainder:
        beqz  t3, dl_esp32p4_s8_unaligned_resize_nearest_c1_end

        esp.ld.128.usar.xp  q1, a1, t3
        esp.zero.qacc
        esp.src.q  q2, q0, q1
        esp.vmulas.s8.qacc  q2, q3
        esp32p4_s8_128b_vector_shift_result  q2, a4
        dl_esp32p4_s8_store_remainder  q2, a3, a5, t4, t5, a6, a0, t3

    dl_esp32p4_s8_unaligned_resize_nearest_c1_end:
        ret
