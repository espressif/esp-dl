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


############################################################################################################################################################
####
#### esp32p4_s8_resize_linear_taps
####
#### One output pixel of a 2d bilinear resize, all channels. The four bilinear
#### taps are folded into a single Q14 dot product by the caller, so this is a
#### 4-term MAC per channel with one shift at the end.
####
#### The taps have to be int16 (Q14) rather than int8 (Q7): a Q7 tap carries a
#### 2^-8 relative error, which reaches 9 LSB on int8 features and fails the
#### .espdl +-1 LSB gate, while Q14 measures 0 LSB. That forces the s16 MAC,
#### hence the vldext.s8 widening of every source vector.
####
#### Channels are done 16 at a time, in two groups of 8, because four sources
#### widened into pairs would need 8 qregs and the taps need a ninth. The
#### second group reloads the same four 16-byte lines, which are L1 hot.
####
############################################################################################################################################################

    .align 2
    .text
    .global dl_esp32p4_s8_resize_linear_taps
    .type   dl_esp32p4_s8_resize_linear_taps, @function
dl_esp32p4_s8_resize_linear_taps:
    .align 2

    # a0: int8_t *output_ptr, 16-byte aligned
    # a1: resizeLinearTapsType *, 16-byte aligned
    #     0(a1)  : int16_t w[8], Q14 taps at halfword 0..3, 127 at 4, -128 at 5
    #     16(a1) : const int8_t *p00, p10, p01, p11
    #     32(a1) : int32_t c_div_16   (channel / 16, >= 1)
    #     36(a1) : int32_t shift

    # srcmb rounds according to CFG.vxrm, and the scalar kernel this has to
    # agree with rounds half to even. Neighbouring PIE kernels leave vxrm in
    # whatever state suited them, so set it here and put it back on the way out.
    esp.movx.r.cfg  t6
    li   t0, 0xffffff0f
    and  t5, t6, t0
    ori  t5, t5, 0x60                   # vxrm = 6, round half to even
    esp.movx.w.cfg  t5

    esp.vld.128.ip  q7, a1, 0           # q7 = taps (and the two clamp bounds)
    lw   a2, 16(a1)
    lw   a3, 20(a1)
    lw   a4, 24(a1)
    lw   a5, 28(a1)
    lw   t3, 32(a1)
    lw   t4, 36(a1)

    # Both broadcasts use offset 0 and the pointer is bumped with addi. A
    # non-zero esp.vldbc.16.ip offset is not portable: before IDF 5.5.2 the
    # assembler encodes it as bytes*2 (so 2 is rejected outright, and 4 would
    # silently advance 4 bytes on newer toolchains). Offset 0 means the same
    # thing either way.
    addi t5, a1, 8
    esp.vldbc.16.ip q6, t5, 0           # q6 = 127 broadcast
    addi t5, t5, 2
    esp.vldbc.16.ip q5, t5, 0           # q5 = -128 broadcast

    0:
        # channels [0, 8): read the four sources without advancing
        esp.zero.qacc
        esp.vldext.s8.ip      q0, q1, a2, 0
        esp.vsmulas.s16.qacc  q0, q7, 0
        esp.vldext.s8.ip      q0, q1, a3, 0
        esp.vsmulas.s16.qacc  q0, q7, 1
        esp.vldext.s8.ip      q0, q1, a4, 0
        esp.vsmulas.s16.qacc  q0, q7, 2
        esp.vldext.s8.ip      q0, q1, a5, 0
        esp.vsmulas.s16.qacc  q0, q7, 3
        esp.srcmb.s16.qacc    q2, t4, 1
        esp.vmin.s16          q2, q2, q6
        esp.vmax.s16          q2, q2, q5

        # channels [8, 16): same four lines, this time advance past them
        esp.zero.qacc
        esp.vldext.s8.ip      q0, q1, a2, 16
        esp.vsmulas.s16.qacc  q1, q7, 0
        esp.vldext.s8.ip      q0, q1, a3, 16
        esp.vsmulas.s16.qacc  q1, q7, 1
        esp.vldext.s8.ip      q0, q1, a4, 16
        esp.vsmulas.s16.qacc  q1, q7, 2
        esp.vldext.s8.ip      q0, q1, a5, 16
        esp.vsmulas.s16.qacc  q1, q7, 3
        esp.srcmb.s16.qacc    q3, t4, 1
        esp.vmin.s16          q3, q3, q6
        esp.vmax.s16          q3, q3, q5

        # q2 collects the low byte of all 16 results, which is the int8 output
        esp.vunzip.8          q2, q3
        esp.vst.128.ip        q2, a0, 16

        addi t3, t3, -1
        bgtz t3, 0b

    esp.movx.w.cfg  t6
    ret
