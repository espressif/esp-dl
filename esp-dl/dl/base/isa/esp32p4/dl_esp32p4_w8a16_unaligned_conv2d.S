#include "dl_esp32p4_s16.S"
#include "dl_esp32p4_common.S"

# ESP32-P4 W8A16 unaligned conv2d: int8 filter, int16 activations.
# Handles input/output pointers that are not 16-byte aligned, and channel
# counts that are not multiples of 8. Filter layout matches the aligned
# kernels: packed [N/8][H][W][C][8] plus remainder [N%8][H][W][C].


############################################################################################################################################################
####
#### esp32p4_w8a16_unaligned_conv2d_11cn series
####
############################################################################################################################################################
.macro esp32p4_w8a16_unaligned_conv2d_11c8_a16  input_v, input_front, input_back, filter_v0, filter_v1, input_ptr, filter_ptr, c_div_x_1, c_remainder, temp
    # 8 output channels, 16-byte-aligned int8 filter packed as [C][8].
    # q5 is scratch; per-channel shift stays in q6.
    # c_div_x_1: input_channel // 8 - 1

    esp.ld.128.usar.ip  \input_front, \input_ptr, 16

    bltz  \c_div_x_1, 7f
        esp.ld.128.usar.ip  \input_back, \input_ptr, 16
        esp.vldext.s8.ip  \filter_v0, \filter_v1, \filter_ptr, 16

        beqz  \c_div_x_1, 8f
        mv  \temp, \c_div_x_1
        9:
            esp.src.q.qup  \input_v, \input_front, \input_back
            esp.ld.128.usar.ip  \input_back, \input_ptr, 16

            esp.vsmulas.s16.qacc.ld.incp  q5, \filter_ptr, \filter_v0, \input_v, 0
            esp.vsmulas.s16.qacc          \filter_v1, \input_v, 1
            esp.vext.s8                   \filter_v0, \filter_v1, q5

            esp.vsmulas.s16.qacc.ld.incp  q5, \filter_ptr, \filter_v0, \input_v, 2
            esp.vsmulas.s16.qacc          \filter_v1, \input_v, 3
            esp.vext.s8                   \filter_v0, \filter_v1, q5

            esp.vsmulas.s16.qacc.ld.incp  q5, \filter_ptr, \filter_v0, \input_v, 4
            esp.vsmulas.s16.qacc          \filter_v1, \input_v, 5
            esp.vext.s8                   \filter_v0, \filter_v1, q5

            esp.vsmulas.s16.qacc.ld.incp  q5, \filter_ptr, \filter_v0, \input_v, 6
            esp.vsmulas.s16.qacc          \filter_v1, \input_v, 7
            esp.vext.s8                   \filter_v0, \filter_v1, q5

            addi  \temp, \temp, -1
            bgtz  \temp, 9b

        8:
        # last entire 8 input channels
        esp.src.q.qup  \input_v, \input_front, \input_back
        esp.vsmulas.s16.qacc.ld.incp  q5, \filter_ptr, \filter_v0, \input_v, 0
        esp.vsmulas.s16.qacc          \filter_v1, \input_v, 1
        esp.vext.s8                   \filter_v0, \filter_v1, q5

        esp.vsmulas.s16.qacc.ld.incp  q5, \filter_ptr, \filter_v0, \input_v, 2
        esp.vsmulas.s16.qacc          \filter_v1, \input_v, 3
        esp.vext.s8                   \filter_v0, \filter_v1, q5

        esp.vsmulas.s16.qacc.ld.incp  q5, \filter_ptr, \filter_v0, \input_v, 4
        esp.vsmulas.s16.qacc          \filter_v1, \input_v, 5
        esp.vext.s8                   \filter_v0, \filter_v1, q5

        esp.vsmulas.s16.qacc          \filter_v0, \input_v, 6
        esp.vsmulas.s16.qacc          \filter_v1, \input_v, 7
        beqz  \c_remainder, 0f

    7:
        esp.ld.128.usar.xp  \input_back, \input_ptr, \c_remainder
        esp.src.q.qup  \input_v, \input_front, \input_back

        li  \temp, 8
        blt  \c_remainder, \temp, 3f
            # k = 4..7 remainder input channels
            esp.vldext.s8.ip  \filter_v0, \filter_v1, \filter_ptr, 16

            li  \temp, 12
            blt  \c_remainder, \temp, 5f
                li  \temp, 14
                blt  \c_remainder, \temp, 6f
                    # k == 7
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp.vldext.s8.ip  \filter_v0, \filter_v1, \filter_ptr, 16
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 3
                    esp.vldext.s8.ip  \filter_v0, \filter_v1, \filter_ptr, 16
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 4
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 5
                    esp.vld.l.64.ip  q5, \filter_ptr, 8
                    esp.vext.s8      \filter_v0, \filter_v1, q5
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 6
                    j 0f

                6:  # k == 6
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp.vldext.s8.ip  \filter_v0, \filter_v1, \filter_ptr, 16
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 3
                    esp.vldext.s8.ip  \filter_v0, \filter_v1, \filter_ptr, 16
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 4
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 5
                    j 0f

            5:
                li  \temp, 10
                blt  \c_remainder, \temp, 4f
                    # k == 5
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp.vldext.s8.ip  \filter_v0, \filter_v1, \filter_ptr, 16
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 3
                    esp.vld.l.64.ip  q5, \filter_ptr, 8
                    esp.vext.s8      \filter_v0, \filter_v1, q5
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 4
                    j 0f

                4:  # k == 4
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp.vldext.s8.ip  \filter_v0, \filter_v1, \filter_ptr, 16
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 3
                    j 0f

        3:  # k = 1..3
            li  \temp, 4
            blt  \c_remainder, \temp, 1f
                esp.vldext.s8.ip  \filter_v0, \filter_v1, \filter_ptr, 16

                li  \temp, 6
                blt  \c_remainder, \temp, 2f
                    # k == 3
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp.vld.l.64.ip  q5, \filter_ptr, 8
                    esp.vext.s8      \filter_v0, \filter_v1, q5
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    j 0f

                2:  # k == 2
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    j 0f

            1:  # k == 1
                esp.vld.l.64.ip  q5, \filter_ptr, 8
                esp.vext.s8      \filter_v0, \filter_v1, q5
                esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0

    0:
        addi  \input_ptr, \input_ptr, -16

.endm


# Packed [C][8] int8 is only 8-byte aligned when C is odd. vldext.s8.ip / 128-bit
# ld.incp require 16-byte alignment, so the 8-aligned path uses 64-bit halves.
.macro esp32p4_w8a16_ld_c8x2  dest0, dest1, scratch, filter_ptr
    esp.vld.l.64.ip  \scratch, \filter_ptr, 8
    esp.vld.h.64.ip  \scratch, \filter_ptr, 8
    esp.vext.s8      \dest0, \dest1, \scratch
.endm

.macro esp32p4_w8a16_ld_c8  dest0, dest1, scratch, filter_ptr
    esp.vld.l.64.ip  \scratch, \filter_ptr, 8
    esp.vext.s8      \dest0, \dest1, \scratch
.endm

.macro esp32p4_w8a16_mac_pair_ld_c8x2  dest0, dest1, scratch, filter_ptr, input_v, sel0, sel1
    esp.vsmulas.s16.qacc  \dest0, \input_v, \sel0
    esp.vld.l.64.ip  \scratch, \filter_ptr, 8
    esp.vsmulas.s16.qacc  \dest1, \input_v, \sel1
    esp.vld.h.64.ip  \scratch, \filter_ptr, 8
    esp.vext.s8      \dest0, \dest1, \scratch
.endm


.macro esp32p4_w8a16_unaligned_conv2d_11c8_a8  input_v, input_front, input_back, filter_v0, filter_v1, input_ptr, filter_ptr, c_div_x_1, c_remainder, temp
    # Filter pointer is 8-byte aligned (bit3 set).
    esp.ld.128.usar.ip  \input_front, \input_ptr, 16

    bltz  \c_div_x_1, 67f
        esp.ld.128.usar.ip  \input_back, \input_ptr, 16
        esp32p4_w8a16_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr

        beqz  \c_div_x_1, 68f
        mv  \temp, \c_div_x_1
        69:
            esp.src.q.qup  \input_v, \input_front, \input_back
            esp.ld.128.usar.ip  \input_back, \input_ptr, 16

            esp32p4_w8a16_mac_pair_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr, \input_v, 0, 1
            esp32p4_w8a16_mac_pair_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr, \input_v, 2, 3
            esp32p4_w8a16_mac_pair_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr, \input_v, 4, 5
            esp32p4_w8a16_mac_pair_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr, \input_v, 6, 7

            addi  \temp, \temp, -1
            bgtz  \temp, 69b

        68:
        esp.src.q.qup  \input_v, \input_front, \input_back
        esp32p4_w8a16_mac_pair_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr, \input_v, 0, 1
        esp32p4_w8a16_mac_pair_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr, \input_v, 2, 3
        esp32p4_w8a16_mac_pair_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr, \input_v, 4, 5
        esp.vsmulas.s16.qacc          \filter_v0, \input_v, 6
        esp.vsmulas.s16.qacc          \filter_v1, \input_v, 7
        beqz  \c_remainder, 60f

    67:
        esp.ld.128.usar.xp  \input_back, \input_ptr, \c_remainder
        esp.src.q.qup  \input_v, \input_front, \input_back

        li  \temp, 8
        blt  \c_remainder, \temp, 63f
            esp32p4_w8a16_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr

            li  \temp, 12
            blt  \c_remainder, \temp, 65f
                li  \temp, 14
                blt  \c_remainder, \temp, 66f
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp32p4_w8a16_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 3
                    esp32p4_w8a16_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 4
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 5
                    esp32p4_w8a16_ld_c8  \filter_v0, \filter_v1, q5, \filter_ptr
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 6
                    j 60f

                66:
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp32p4_w8a16_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 3
                    esp32p4_w8a16_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 4
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 5
                    j 60f

            65:
                li  \temp, 10
                blt  \c_remainder, \temp, 64f
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp32p4_w8a16_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 3
                    esp32p4_w8a16_ld_c8  \filter_v0, \filter_v1, q5, \filter_ptr
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 4
                    j 60f

                64:
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp32p4_w8a16_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 3
                    j 60f

        63:
            li  \temp, 4
            blt  \c_remainder, \temp, 61f
                esp32p4_w8a16_ld_c8x2  \filter_v0, \filter_v1, q5, \filter_ptr

                li  \temp, 6
                blt  \c_remainder, \temp, 62f
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    esp32p4_w8a16_ld_c8  \filter_v0, \filter_v1, q5, \filter_ptr
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 2
                    j 60f

                62:
                    esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0
                    esp.vsmulas.s16.qacc  \filter_v1, \input_v, 1
                    j 60f

            61:
                esp32p4_w8a16_ld_c8  \filter_v0, \filter_v1, q5, \filter_ptr
                esp.vsmulas.s16.qacc  \filter_v0, \input_v, 0

    60:
        addi  \input_ptr, \input_ptr, -16
.endm


.macro esp32p4_w8a16_unaligned_conv2d_11c8  input_v, input_front, input_back, filter_v0, filter_v1, input_ptr, filter_ptr, c_div_x_1, c_remainder, temp
    andi  \temp, \filter_ptr, 8
    bnez  \temp, 70f
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        j  80f
    70:
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
    80:
.endm



.macro esp32p4_w8a16_unaligned_conv2d_11c1  input_v, input_front, input_back, filter_v, filter_front, filter_back, input_ptr, filter_ptr, c_div_x_1, c_remainder, temp, zero
    # 1 output channel, unaligned int16 input, int8 filter laid out as [C].
    # usar.ip immediates must be multiples of 16, so each 8-byte filter group is
    # extracted with a 16-byte unaligned window and the pointer is then rewound
    # to a net +8. c_div_x_1: input_channel // 8 - 1. temp: 15 - c_remainder.

    esp.ld.128.usar.ip  \input_front, \input_ptr, 16

    bltz  \c_div_x_1, 7f
        esp.ld.128.usar.ip  \input_back, \input_ptr, 16

        beqz  \c_div_x_1, 8f
        mv  \zero, \c_div_x_1
        9:
            esp.src.q.qup  \input_v, \input_front, \input_back

            esp.ld.128.usar.ip  \filter_front, \filter_ptr, 16
            esp.ld.128.usar.ip  \filter_back, \filter_ptr, 16
            esp.src.q      \filter_v, \filter_front, \filter_back
            addi  \filter_ptr, \filter_ptr, -24
            esp.vext.s8    \filter_back, q7, \filter_v

            esp.ld.128.usar.ip  \input_back, \input_ptr, 16
            esp.vmulas.s16.xacc  \filter_back, \input_v
            addi  \zero, \zero, -1
            bgtz  \zero, 9b
        8:
        esp.src.q.qup  \input_v, \input_front, \input_back

        esp.ld.128.usar.ip  \filter_front, \filter_ptr, 16
        esp.ld.128.usar.ip  \filter_back, \filter_ptr, 16
        esp.src.q      \filter_v, \filter_front, \filter_back
        addi  \filter_ptr, \filter_ptr, -24
        esp.vext.s8    \filter_back, q7, \filter_v

        esp.vmulas.s16.xacc  \filter_back, \input_v

        beqz  \c_remainder, 0f

    7:
        esp.ld.128.usar.xp  \input_back, \input_ptr, \c_remainder
        esp.src.q.qup  \input_v, \input_front, \input_back

        srli  \zero, \c_remainder, 1
        esp.ld.128.usar.ip  \filter_front, \filter_ptr, 16
        esp.ld.128.usar.ip  \filter_back, \filter_ptr, 16
        esp.src.q      \filter_v, \filter_front, \filter_back
        addi  \filter_ptr, \filter_ptr, -32
        add   \filter_ptr, \filter_ptr, \zero
        esp.vext.s8    \filter_back, q7, \filter_v

        li  \zero, 0
        esp.slcxxp.2q  \input_back,  \input_v, \temp, \zero
        esp.slcxxp.2q  \filter_v, \filter_back, \temp, \zero

        esp.vmulas.s16.xacc  \filter_back, \input_v

    0:
        addi  \input_ptr,  \input_ptr, -16
.endm



.macro esp32p4_w8a16_unaligned_conv2d_11cn_load_args  args, filter_ptr, c_div_x_1, n_div_x, mac_shift, c_remainder
    lw  \filter_ptr,    48(\args)       // filter
    lw  \c_div_x_1,     100(\args)      // input_channel / x - 1
    lw  \n_div_x,       96(\args)       // output_channel / x
    lw  \mac_shift,     64(\args)       // mac_shift
    lw  \c_remainder,   136(\args)      // input_channel % (vector_width / element_width) * sizeof(feature_t)
.endm



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_11cn_bias
    .type   dl_esp32p4_w8a16_unaligned_conv2d_11cn_bias, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_11cn_bias:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
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
    esp32p4_w8a16_unaligned_conv2d_11cn_load_args  a2, a3, t0, t1, a5, a4
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_11cn_bias_128b
            li  t2, 8
            beq  t3, t2, esp32p4_w8a16_unaligned_conv2d_11cn_bias_64b
                esp32p4_w8a16_unaligned_conv2d_11cn_bias_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_11cn_bias_n_remainder

            esp32p4_w8a16_unaligned_conv2d_11cn_bias_64b:
                esp32p4_w8a16_unaligned_conv2d_11cn_bias_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_11cn_bias_n_remainder

        esp32p4_w8a16_unaligned_conv2d_11cn_bias_128b:
            esp32p4_w8a16_unaligned_conv2d_11cn_bias_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_11cn_bias_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_11cn_bias_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                esp32p4_w8a16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_11cn_bias_n_remainder_end:

    esp32p4_pop_12_stacks_3r  s0, s1, s8     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
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
    esp32p4_w8a16_unaligned_conv2d_11cn_load_args  a2, a3, t0, t1, a5, a4
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_128b
            li  t2, 8
            beq  t3, t2, esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_64b
                esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                esp32p4_w8a16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_leakyrelu  s0, s9, s10
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder_end:

    esp32p4_pop_20_stacks_5r  s0, s1, s8, s9, s10     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_11cn
    .type   dl_esp32p4_w8a16_unaligned_conv2d_11cn, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_11cn:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
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
    esp32p4_w8a16_unaligned_conv2d_11cn_load_args  a2, a3, t0, t1, a5, a4

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_11cn_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_11cn_128b
            li  t2, 8
            beq  t3, t2, esp32p4_w8a16_unaligned_conv2d_11cn_64b
                esp32p4_w8a16_unaligned_conv2d_11cn_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_11cn_n_remainder

            esp32p4_w8a16_unaligned_conv2d_11cn_64b:
                esp32p4_w8a16_unaligned_conv2d_11cn_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_11cn_n_remainder

        esp32p4_w8a16_unaligned_conv2d_11cn_128b:
            esp32p4_w8a16_unaligned_conv2d_11cn_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_11cn_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_11cn_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_11cn_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                esp32p4_w8a16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_11cn_n_remainder_end:

    esp32p4_pop_12_stacks_3r  s0, s1, s8     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_11cn_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_11cn_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_11cn_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
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
    esp32p4_w8a16_unaligned_conv2d_11cn_load_args  a2, a3, t0, t1, a5, a4

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_11cn_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_11cn_relu_128b
            li  t2, 8
            beq  t3, t2, esp32p4_w8a16_unaligned_conv2d_11cn_relu_64b
                esp32p4_w8a16_unaligned_conv2d_11cn_relu_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_11cn_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_11cn_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_11cn_relu_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_11cn_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_11cn_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_11cn_relu_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_relu_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_11cn_relu_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_11cn_relu_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_11cn_relu_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                esp32p4_w8a16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_leakyrelu  s0, s9, s10
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_11cn_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_11cn_relu_n_remainder_end:

    esp32p4_pop_20_stacks_5r  s0, s1, s8, s9, s10     // restore registers
    ret






############################################################################################################################################################
####
#### esp32p4_w8a16_unaligned_conv2d_33cn series
####
############################################################################################################################################################
.macro esp32p4_w8a16_unaligned_conv2d_33c8  input_v0, input_front, input_back, filter_v0, filter_v1, input_ptr, filter_ptr, c_div_x_1, c_remainder, dilation_x_offset, dilation_y_offset, temp
    # Tap stride is C*8 bytes. Even C keeps alignment; odd C flips 16-byte <->
    # 8-byte each tap. Packed [N/8][H][W][C][8] also makes later OC groups start
    # 8-byte aligned when C is odd, so the first tap must follow filter_ptr bit3.
    andi  \temp, \filter_ptr, 8
    bnez  \temp, 94f
    andi  \temp, \c_remainder, 2
    bnez  \temp, 92f

        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_y_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_y_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        j  91f

    92:
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_y_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_y_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        j  91f

    94:
        andi  \temp, \c_remainder, 2
        beqz  \temp, 93f

        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_y_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_y_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        j  91f

    93:
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_y_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_y_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
        add  \input_ptr, \input_ptr, \dilation_x_offset
        esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp

    91:
.endm



.macro esp32p4_w8a16_unaligned_conv2d_33c1  input_v0, input_front, input_back, filter_v0, filter_front, filter_back, input_ptr, filter_ptr, c_div_x_1, c_remainder, dilation_x_offset, dilation_y_offset, temp, zero
    esp32p4_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    esp32p4_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    esp32p4_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_y_offset

    esp32p4_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    esp32p4_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    esp32p4_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_y_offset

    esp32p4_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    esp32p4_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    esp32p4_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    # add \input_ptr, \input_ptr, \dilation_y_offset
.endm



.macro esp32p4_w8a16_unaligned_conv2d_hwcn_load_args  args, filter_ptr, c_div_x_1, n_div_x, mac_shift, c_remainder, dilation_x_offset, dilation_y_offset
    esp32p4_w8a16_unaligned_conv2d_11cn_load_args  \args, \filter_ptr, \c_div_x_1, \n_div_x, \mac_shift, \c_remainder
    lw  \dilation_x_offset, 108(\args)  // input dilation x offset
    lw  \dilation_y_offset, 112(\args)  // input dilation y offset
.endm



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_33cn_bias
    .type   dl_esp32p4_w8a16_unaligned_conv2d_33cn_bias, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_33cn_bias:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
    # t6: bias_ptr

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): tmp value
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
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

    esp32p4_w8a16_unaligned_conv2d_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3               //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_33cn_bias_128b
            li  a7, 8
            beq  t3, a7, esp32p4_w8a16_unaligned_conv2d_33cn_bias_64b
            esp32p4_w8a16_unaligned_conv2d_33cn_bias_32b:
                esp32p4_w8a16_unaligned_conv2d_33cn_bias_32b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_33cn_bias_n_remainder

            esp32p4_w8a16_unaligned_conv2d_33cn_bias_64b:
                esp32p4_w8a16_unaligned_conv2d_33cn_bias_64b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_33cn_bias_n_remainder

        esp32p4_w8a16_unaligned_conv2d_33cn_bias_128b:
            esp32p4_w8a16_unaligned_conv2d_33cn_bias_128b_multiple_loop:
                mv  t3, a1                      // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_33cn_bias_n_remainder:
        lw  t1, 140(a2)       # t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4       # t4: 15 - c_remainder
            li  t5,  0            # t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_33cn_bias_n_remainder_loop:
                mv  t3, a1         # t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                esp32p4_w8a16_unaligned_conv2d_33c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_33cn_bias_n_remainder_end:

    esp32p4_pop_12_stacks_3r  s0, s1, s8     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
    # t6: bias_ptr

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): tmp value
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
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

    esp32p4_w8a16_unaligned_conv2d_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3               //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_128b
            li  a7, 8
            beq  t3, a7, esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_64b
            esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_32b:
                esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_32b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_64b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_128b_multiple_loop:
                mv  t3, a1                      // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder:
        lw  t1, 140(a2)       # t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4       # t4: 15 - c_remainder
            li  t5,  0            # t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder_loop:
                mv  t3, a1         # t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                esp32p4_w8a16_unaligned_conv2d_33c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_leakyrelu  s0, s9, s10
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder_end:

    esp32p4_pop_20_stacks_5r  s0, s1, s8, s9, s10     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_33cn
    .type   dl_esp32p4_w8a16_unaligned_conv2d_33cn, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_33cn:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
    # t6:

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): tmp value
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
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

    esp32p4_w8a16_unaligned_conv2d_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_33cn_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3               //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_33cn_128b
            li  a7, 8
            beq  t3, a7, esp32p4_w8a16_unaligned_conv2d_33cn_64b
            esp32p4_w8a16_unaligned_conv2d_33cn_32b:
                esp32p4_w8a16_unaligned_conv2d_33cn_32b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_33cn_n_remainder

            esp32p4_w8a16_unaligned_conv2d_33cn_64b:
                esp32p4_w8a16_unaligned_conv2d_33cn_64b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_33cn_n_remainder

        esp32p4_w8a16_unaligned_conv2d_33cn_128b:
            esp32p4_w8a16_unaligned_conv2d_33cn_128b_multiple_loop:
                mv  t3, a1                      // t3: input_ptr
                esp.zero.qacc

                esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_33cn_n_remainder:
        lw  t1, 140(a2)       # t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_33cn_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4       # t4: 15 - c_remainder
            li  t5,  0            # t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_33cn_n_remainder_loop:
                mv  t3, a1         # t3: input_ptr
                esp.zero.xacc

                esp32p4_w8a16_unaligned_conv2d_33c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_33cn_n_remainder_end:

    esp32p4_pop_12_stacks_3r  s0, s1, s8     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_33cn_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_33cn_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_33cn_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
    # t6:

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): tmp value
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
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

    esp32p4_w8a16_unaligned_conv2d_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_33cn_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3               //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_33cn_relu_128b
            li  a7, 8
            beq  t3, a7, esp32p4_w8a16_unaligned_conv2d_33cn_relu_64b
            esp32p4_w8a16_unaligned_conv2d_33cn_relu_32b:
                esp32p4_w8a16_unaligned_conv2d_33cn_relu_32b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_33cn_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_33cn_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_33cn_relu_64b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_33cn_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_33cn_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_33cn_relu_128b_multiple_loop:
                mv  t3, a1                      // t3: input_ptr
                esp.zero.qacc

                esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_relu_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_33cn_relu_n_remainder:
        lw  t1, 140(a2)       # t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_33cn_relu_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4       # t4: 15 - c_remainder
            li  t5,  0            # t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_33cn_relu_n_remainder_loop:
                mv  t3, a1         # t3: input_ptr
                esp.zero.xacc

                esp32p4_w8a16_unaligned_conv2d_33c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, t4, t5
                esp32p4_s16_element_result  s0, a5
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_leakyrelu  s0, s9, s10
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_33cn_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_33cn_relu_n_remainder_end:

    esp32p4_pop_20_stacks_5r  s0, s1, s8, s9, s10     // restore registers
    ret






############################################################################################################################################################
####
#### esp32p4_w8a16_unaligned_conv2d_hwcn series
####
############################################################################################################################################################
.macro esp32p4_w8a16_unaligned_conv2d_hwc8  input_v0, input_front, input_back, filter_v0, filter_v1, input_ptr, filter_ptr, c_div_x_1, c_remainder, dilation_x_offset, dilation_y_offset, filter_h, filter_w, args, filter_y_offset, filter_n_offset, temp
    andi  \temp, \c_remainder, 2
    bnez  \temp, 40f

    # Even C: tap stride is 16-byte aligned, so every tap matches the first.
    andi  \temp, \filter_ptr, 8
    bnez  \temp, 41f
        lw  \filter_h, 52(\args)
        11:
            lw  \filter_w, 56(\args)
            addi  \filter_w, \filter_w, -1
            beqz  \filter_w, 12f
                10:
                    esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
                    add  \input_ptr, \input_ptr, \dilation_x_offset
                    addi  \filter_w, \filter_w, -1
                    bnez  \filter_w, 10b
            12:
            esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
            add  \filter_ptr, \filter_ptr, \filter_y_offset
            add  \input_ptr, \input_ptr, \dilation_y_offset
            addi  \filter_h, \filter_h, -1
            bnez  \filter_h, 11b
        add  \filter_ptr, \filter_ptr, \filter_n_offset
        j  49f

    41:
        lw  \filter_h, 52(\args)
        21:
            lw  \filter_w, 56(\args)
            addi  \filter_w, \filter_w, -1
            beqz  \filter_w, 22f
                20:
                    esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
                    add  \input_ptr, \input_ptr, \dilation_x_offset
                    addi  \filter_w, \filter_w, -1
                    bnez  \filter_w, 20b
            22:
            esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
            add  \filter_ptr, \filter_ptr, \filter_y_offset
            add  \input_ptr, \input_ptr, \dilation_y_offset
            addi  \filter_h, \filter_h, -1
            bnez  \filter_h, 21b
        add  \filter_ptr, \filter_ptr, \filter_n_offset
        j  49f

    40:
    # Odd C: tap stride is 8-byte aligned, so consecutive taps flip alignment.
    # Later rows / OC groups may start 8-byte aligned as well.
    lw  \filter_h, 52(\args)
    43:
        lw  \filter_w, 56(\args)
        andi  \temp, \filter_ptr, 8
        bnez  \temp, 45f
        44:
            esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
            addi  \filter_w, \filter_w, -1
            beqz  \filter_w, 48f
            add  \input_ptr, \input_ptr, \dilation_x_offset
            esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
            addi  \filter_w, \filter_w, -1
            beqz  \filter_w, 48f
            add  \input_ptr, \input_ptr, \dilation_x_offset
            j  44b
        45:
            esp32p4_w8a16_unaligned_conv2d_11c8_a8  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
            addi  \filter_w, \filter_w, -1
            beqz  \filter_w, 48f
            add  \input_ptr, \input_ptr, \dilation_x_offset
            esp32p4_w8a16_unaligned_conv2d_11c8_a16  \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp
            addi  \filter_w, \filter_w, -1
            beqz  \filter_w, 48f
            add  \input_ptr, \input_ptr, \dilation_x_offset
            j  45b
        48:
        add  \filter_ptr, \filter_ptr, \filter_y_offset
        add  \input_ptr, \input_ptr, \dilation_y_offset
        addi  \filter_h, \filter_h, -1
        bnez  \filter_h, 43b
    add  \filter_ptr, \filter_ptr, \filter_n_offset

    49:
.endm



.macro esp32p4_w8a16_unaligned_conv2d_hwc1  input_v0, input_front, input_back, filter_v0, filter_front, filter_back, input_ptr, filter_ptr, c_div_x_1, c_remainder, dilation_x_offset, dilation_y_offset, filter_h, filter_w, args, temp, zero, filter_y_offset, filter_n_offset
    lw  \filter_h, 52(\args)  # filter_height
    11:
        lw  \filter_w, 56(\args)  # filter_width
        addi  \filter_w, \filter_w, -1
        beqz  \filter_w, 12f
            10:
                esp32p4_w8a16_unaligned_conv2d_11c1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
                add  \input_ptr, \input_ptr, \dilation_x_offset

                addi  \filter_w, \filter_w, -1
                bnez  \filter_w, 10b
        12:
        esp32p4_w8a16_unaligned_conv2d_11c1  \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero

        add  \filter_ptr, \filter_ptr, \filter_y_offset
        add  \input_ptr, \input_ptr, \dilation_y_offset

        addi  \filter_h, \filter_h, -1
        bnez  \filter_h, 11b

    add  \filter_ptr, \filter_ptr, \filter_n_offset
.endm



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_hwcn_bias
    .type   dl_esp32p4_w8a16_unaligned_conv2d_hwcn_bias, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_hwcn_bias:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value / output
    # t4: 15 - c_remainder
    # t5: tmp value / zero
    # t6: bias_ptr

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): filter_h
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions): filter_w
    # s3(not for extension instructions): filter_y_offset
    # s4(not for extension instructions): filter_n_offset
    # s5(not for extension instructions):

    # s0:
    # s1: clamp min value
    # s8: clamp max value
    # s9:
    # s10:
    # s11:

    esp32p4_push_20_stacks_5r  s1, s2, s3, s4, s8
    li  s1, -32768                  // clamp min value
    li  s8, 32767                   // clamp max value

    esp32p4_w8a16_unaligned_conv2d_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        lw  s3, 60(a2)              // s3: filter_y_offset
        lw  s4, 144(a2)             // s4: filter_n_offset
        esp.movx.r.sar.bytes  t3    // t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_128b
            li  t5, 8
            beq t3, t5, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_64b
                # esp32p4_w8a16_unaligned_conv2d_hwcn_bias_32b:
                esp32p4_w8a16_unaligned_conv2d_hwcn_bias_32b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_hwcn_bias_n_remainder

            esp32p4_w8a16_unaligned_conv2d_hwcn_bias_64b:
                esp32p4_w8a16_unaligned_conv2d_hwcn_bias_64b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_hwcn_bias_n_remainder

        esp32p4_w8a16_unaligned_conv2d_hwcn_bias_128b:
            esp32p4_w8a16_unaligned_conv2d_hwcn_bias_128b_multiple_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_128b_multiple_loop

    esp32p4_w8a16_unaligned_conv2d_hwcn_bias_n_remainder:
        lw  t1, 140(a2)             // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_n_remainder_end
            lw  s3, 160(a2)         // s3: filter_y_offset_unaligned
            lw  s4, 164(a2)         // s4: filter_n_offset_unaligned
            lw  a3, 168(a2)         // a3: filter_ptr_unaligned

            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: zero

            esp32p4_w8a16_unaligned_conv2d_hwcn_bias_n_remainder_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                esp32p4_w8a16_unaligned_conv2d_hwc1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, a7, s2, a2, t4, t5, s3, s4
                esp32p4_s16_element_result  t3, a5
                esp32p4_clamp  t3, s1, s8
                esp32p4_s16_element_store  a0, t3

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_hwcn_bias_n_remainder_end:

    esp32p4_pop_20_stacks_5r  s1, s2, s3, s4, s8
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value / output
    # t4: 15 - c_remainder
    # t5: tmp value / zero
    # t6: bias_ptr

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): filter_h
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions): filter_w
    # s3(not for extension instructions): filter_y_offset
    # s4(not for extension instructions): filter_n_offset
    # s5(not for extension instructions):

    # s0:
    # s1: clamp min value
    # s8: clamp max value
    # s9: activation_alpha/_address
    # s10: activation_shift
    # s11:

    esp32p4_push_28_stacks_7r  s1, s2, s3, s4, s8, s9, s10
    li  s1, -32768                  // clamp min value
    li  s8, 32767                   // clamp max value
    lw  s9, 76(a2)      // activation_alpha
    lw  s10, 84(a2)     // activation_shift

    esp32p4_w8a16_unaligned_conv2d_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        lw  s3, 60(a2)              // s3: filter_y_offset
        lw  s4, 144(a2)             // s4: filter_n_offset
        esp.movx.r.sar.bytes  t3    // t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_128b
            li  t5, 8
            beq t3, t5, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_64b
                # esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_32b:
                esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_32b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_64b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_128b_multiple_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_128b_multiple_loop

    esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder:
        lw  t1, 140(a2)             // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder_end
            lw  s3, 160(a2)         // s3: filter_y_offset_unaligned
            lw  s4, 164(a2)         // s4: filter_n_offset_unaligned
            lw  a3, 168(a2)         // a3: filter_ptr_unaligned

            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: zero

            esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                esp32p4_w8a16_unaligned_conv2d_hwc1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, a7, s2, a2, t4, t5, s3, s4
                esp32p4_s16_element_result  t3, a5
                esp32p4_clamp  t3, s1, s8
                esp32p4_s16_element_leakyrelu  t3, s9, s10
                esp32p4_s16_element_store  a0, t3

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder_end:

    esp32p4_pop_28_stacks_7r  s1, s2, s3, s4, s8, s9, s10
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_hwcn
    .type   dl_esp32p4_w8a16_unaligned_conv2d_hwcn, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_hwcn:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value / output
    # t4: 15 - c_remainder
    # t5: tmp value / zero
    # t6:

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): filter_h
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions): filter_w
    # s3(not for extension instructions): filter_y_offset
    # s4(not for extension instructions): filter_n_offset
    # s5(not for extension instructions):

    # s0:
    # s1: clamp min value
    # s8: clamp max value
    # s9:
    # s10:
    # s11:

    esp32p4_push_20_stacks_5r  s1, s2, s3, s4, s8
    li  s1, -32768                  // clamp min value
    li  s8, 32767                   // clamp max value

    esp32p4_w8a16_unaligned_conv2d_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        lw  s3, 60(a2)              // s3: filter_y_offset
        lw  s4, 144(a2)             // s4: filter_n_offset
        esp.movx.r.sar.bytes  t3    // t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_hwcn_128b
            li  t5, 8
            beq t3, t5, esp32p4_w8a16_unaligned_conv2d_hwcn_64b
                # esp32p4_w8a16_unaligned_conv2d_hwcn_32b:
                esp32p4_w8a16_unaligned_conv2d_hwcn_32b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_hwcn_n_remainder

            esp32p4_w8a16_unaligned_conv2d_hwcn_64b:
                esp32p4_w8a16_unaligned_conv2d_hwcn_64b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_hwcn_n_remainder

        esp32p4_w8a16_unaligned_conv2d_hwcn_128b:
            esp32p4_w8a16_unaligned_conv2d_hwcn_128b_multiple_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.qacc

                esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_128b_multiple_loop

    esp32p4_w8a16_unaligned_conv2d_hwcn_n_remainder:
        lw  t1, 140(a2)             // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_n_remainder_end
            lw  s3, 160(a2)         // s3: filter_y_offset_unaligned
            lw  s4, 164(a2)         // s4: filter_n_offset_unaligned
            lw  a3, 168(a2)         // a3: filter_ptr_unaligned

            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: zero

            esp32p4_w8a16_unaligned_conv2d_hwcn_n_remainder_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.xacc

                esp32p4_w8a16_unaligned_conv2d_hwc1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, a7, s2, a2, t4, t5, s3, s4
                esp32p4_s16_element_result  t3, a5
                esp32p4_clamp  t3, s1, s8
                esp32p4_s16_element_store  a0, t3

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_hwcn_n_remainder_end:

    esp32p4_pop_20_stacks_5r  s1, s2, s3, s4, s8
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_hwcn_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_hwcn_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_hwcn_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: mac_shift
    # t3: output_sar_byte / moving input_ptr / tmp value / output
    # t4: 15 - c_remainder
    # t5: tmp value / zero
    # t6:

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): filter_h
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions): filter_w
    # s3(not for extension instructions): filter_y_offset
    # s4(not for extension instructions): filter_n_offset
    # s5(not for extension instructions):

    # s0:
    # s1: clamp min value
    # s8: clamp max value
    # s9: activation_alpha/_address
    # s10: activation_shift
    # s11:

    esp32p4_push_28_stacks_7r  s1, s2, s3, s4, s8, s9, s10
    li  s1, -32768                  // clamp min value
    li  s8, 32767                   // clamp max value
    lw  s9, 76(a2)      // activation_alpha
    lw  s10, 84(a2)     // activation_shift

    esp32p4_w8a16_unaligned_conv2d_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        lw  s3, 60(a2)              // s3: filter_y_offset
        lw  s4, 144(a2)             // s4: filter_n_offset
        esp.movx.r.sar.bytes  t3    // t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_hwcn_relu_128b
            li  t5, 8
            beq t3, t5, esp32p4_w8a16_unaligned_conv2d_hwcn_relu_64b
                # esp32p4_w8a16_unaligned_conv2d_hwcn_relu_32b:
                esp32p4_w8a16_unaligned_conv2d_hwcn_relu_32b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_hwcn_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_hwcn_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_hwcn_relu_64b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_shift_result  q0, a5
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_hwcn_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_hwcn_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_hwcn_relu_128b_multiple_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.qacc

                esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                esp32p4_s16_128b_vector_shift_result  q0, a5
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_relu_128b_multiple_loop

    esp32p4_w8a16_unaligned_conv2d_hwcn_relu_n_remainder:
        lw  t1, 140(a2)             // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_relu_n_remainder_end
            lw  s3, 160(a2)         // s3: filter_y_offset_unaligned
            lw  s4, 164(a2)         // s4: filter_n_offset_unaligned
            lw  a3, 168(a2)         // a3: filter_ptr_unaligned

            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: zero

            esp32p4_w8a16_unaligned_conv2d_hwcn_relu_n_remainder_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.xacc

                esp32p4_w8a16_unaligned_conv2d_hwc1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, a7, s2, a2, t4, t5, s3, s4
                esp32p4_s16_element_result  t3, a5
                esp32p4_clamp  t3, s1, s8
                esp32p4_s16_element_leakyrelu  t3, s9, s10
                esp32p4_s16_element_store  a0, t3

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_hwcn_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_hwcn_relu_n_remainder_end:

    esp32p4_pop_28_stacks_7r  s1, s2, s3, s4, s8, s9, s10
    ret



############################################################################################################################################################
####
#### esp32p4_w8a16_unaligned_conv2d_per_channel_11cn series
####
############################################################################################################################################################

.macro esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_load_args  args, filter_ptr, c_div_x_1, n_div_x, tie_filter_channel_factor, c_remainder
    lw  \filter_ptr,                    48(\args)       // filter
    lw  \c_div_x_1,                     100(\args)      // input_channel / x - 1
    lw  \n_div_x,                       96(\args)       // output_channel / x
    lw  \tie_filter_channel_factor,     104(\args)      // tie_filter_channel_factor
    lw  \c_remainder,                   136(\args)      // input_channel % (vector_width / element_width) * sizeof(feature_t)
.endm



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
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
    # s11: mac_shift element

    esp32p4_push_16_stacks_4r  s0, s1, s8, s11    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value
    esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_load_args  a2, a3, t0, t1, a5, a4
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_128b
            li  t2, 8
            beq  t3, t2, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_64b
                esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, s11
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_n_remainder_end:

    esp32p4_pop_16_stacks_4r  s0, s1, s8, s11     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
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
    # s11: mac_shift element

    esp32p4_push_24_stacks_6r  s0, s1, s8, s9, s10, s11    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value
    lw  s9, 76(a2)      // activation_alpha
    lw  s10, 84(a2)     // activation_shift
    esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_load_args  a2, a3, t0, t1, a5, a4
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_128b
            li  t2, 8
            beq  t3, t2, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_64b
                esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, s11
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_leakyrelu  s0, s9, s10
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_bias_relu_n_remainder_end:

    esp32p4_pop_24_stacks_6r  s0, s1, s8, s9, s10, s11     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
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
    # s11: mac_shift element

    esp32p4_push_16_stacks_4r  s0, s1, s8, s11    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value
    esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_load_args  a2, a3, t0, t1, a5, a4

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_128b
            li  t2, 8
            beq  t3, t2, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_64b
                esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, s11
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_n_remainder_end:

    esp32p4_pop_16_stacks_4r  s0, s1, s8, s11     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
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
    # s11: mac_shift element

    esp32p4_push_24_stacks_6r  s0, s1, s8, s9, s10, s11    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value
    lw  s9, 76(a2)      // activation_alpha
    lw  s10, 84(a2)     // activation_shift
    esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_load_args  a2, a3, t0, t1, a5, a4

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3     //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_128b
            li  t2, 8
            beq  t3, t2, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_64b
                esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_32b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_64b_multiple_loop:
                    mv  t3, a1     // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_128b_multiple_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.qacc

                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_11c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_n_remainder:
        lw  t1, 140(a2)              // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_n_remainder_loop:
                mv  t3, a1         // t3: input_ptr
                esp.zero.xacc

                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_11c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t4, t5
                esp32p4_s16_element_result  s0, s11
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_leakyrelu  s0, s9, s10
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_relu_n_remainder_end:

    esp32p4_pop_24_stacks_6r  s0, s1, s8, s9, s10, s11     // restore registers
    ret



############################################################################################################################################################
####
#### esp32p4_w8a16_unaligned_conv2d_per_channel_33cn series
####
############################################################################################################################################################

.macro esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_load_args  args, filter_ptr, c_div_x_1, n_div_x, tie_filter_channel_factor, c_remainder, dilation_x_offset, dilation_y_offset
    esp32p4_w8a16_unaligned_conv2d_per_channel_11cn_load_args  \args, \filter_ptr, \c_div_x_1, \n_div_x, \tie_filter_channel_factor, \c_remainder
    lw  \dilation_x_offset, 108(\args)  // input dilation x offset
    lw  \dilation_y_offset, 112(\args)  // input dilation y offset
.endm



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
    # t6: bias_ptr

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): tmp value
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions):
    # s3(not for extension instructions):
    # s4(not for extension instructions):
    # s5(not for extension instructions):

    # s0: output
    # s1: clamp min value
    # s8: clamp max value
    # s9:
    # s10:
    # s11: mac_shift element

    esp32p4_push_16_stacks_4r  s0, s1, s8, s11    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3               //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_128b
            li  a7, 8
            beq  t3, a7, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_64b
            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_32b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_32b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_64b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_128b_multiple_loop:
                mv  t3, a1                      // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_n_remainder:
        lw  t1, 140(a2)       # t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4       # t4: 15 - c_remainder
            li  t5,  0            # t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_n_remainder_loop:
                mv  t3, a1         # t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_33c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, t4, t5
                esp32p4_s16_element_result  s0, s11
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_n_remainder_end:

    esp32p4_pop_16_stacks_4r  s0, s1, s8, s11     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
    # t6: bias_ptr

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): tmp value
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions):
    # s3(not for extension instructions):
    # s4(not for extension instructions):
    # s5(not for extension instructions):

    # s0: output
    # s1: clamp min value
    # s8: clamp max value
    # s9: activation_alpha/_address
    # s10: activation_shift
    # s11: mac_shift element

    esp32p4_push_24_stacks_6r  s0, s1, s8, s9, s10, s11    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value
    lw  s9, 76(a2)      // activation_alpha
    lw  s10, 84(a2)     // activation_shift

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3               //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_128b
            li  a7, 8
            beq  t3, a7, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_64b
            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_32b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_32b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_64b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_128b_multiple_loop:
                mv  t3, a1                      // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_n_remainder:
        lw  t1, 140(a2)       # t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4       # t4: 15 - c_remainder
            li  t5,  0            # t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_n_remainder_loop:
                mv  t3, a1         # t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_33c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, t4, t5
                esp32p4_s16_element_result  s0, s11
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_leakyrelu  s0, s9, s10
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_bias_relu_n_remainder_end:

    esp32p4_pop_24_stacks_6r  s0, s1, s8, s9, s10, s11     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
    # t6:

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): tmp value
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions):
    # s3(not for extension instructions):
    # s4(not for extension instructions):
    # s5(not for extension instructions):

    # s0: output
    # s1: clamp min value
    # s8: clamp max value
    # s9:
    # s10:
    # s11: mac_shift element

    esp32p4_push_16_stacks_4r  s0, s1, s8, s11    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3               //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_128b
            li  a7, 8
            beq  t3, a7, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_64b
            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_32b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_32b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_64b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_128b_multiple_loop:
                mv  t3, a1                      // t3: input_ptr
                esp.zero.qacc

                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_n_remainder:
        lw  t1, 140(a2)       # t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4       # t4: 15 - c_remainder
            li  t5,  0            # t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_n_remainder_loop:
                mv  t3, a1         # t3: input_ptr
                esp.zero.xacc

                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_33c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, t4, t5
                esp32p4_s16_element_result  s0, s11
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_n_remainder_end:

    esp32p4_pop_16_stacks_4r  s0, s1, s8, s11     // restore registers
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value
    # t4: 15 - c_remainder
    # t5: zero
    # t6:

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): tmp value
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions):
    # s3(not for extension instructions):
    # s4(not for extension instructions):
    # s5(not for extension instructions):

    # s0: output
    # s1: clamp min value
    # s8: clamp max value
    # s9: activation_alpha/_address
    # s10: activation_shift
    # s11: mac_shift element

    esp32p4_push_24_stacks_6r  s0, s1, s8, s9, s10, s11    // push stacks
    li  s1, -32768      // clamp min value
    li  s8, 32767       // clamp max value
    lw  s9, 76(a2)      // activation_alpha
    lw  s10, 84(a2)     // activation_shift

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        esp.movx.r.sar.bytes  t3               //  t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_128b
            li  a7, 8
            beq  t3, a7, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_64b
            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_32b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_32b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_64b_multiple_loop:
                    mv  t3, a1                  // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_128b_multiple_loop:
                mv  t3, a1                      // t3: input_ptr
                esp.zero.qacc

                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_33c8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_128b_multiple_loop


    esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_n_remainder:
        lw  t1, 140(a2)       # t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_n_remainder_end
            li  t4, 15
            sub  t4, t4, a4       # t4: 15 - c_remainder
            li  t5,  0            # t5: activation_shift = zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_n_remainder_loop:
                mv  t3, a1         # t3: input_ptr
                esp.zero.xacc

                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_33c1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, t4, t5
                esp32p4_s16_element_result  s0, s11
                esp32p4_clamp  s0, s1, s8
                esp32p4_s16_element_leakyrelu  s0, s9, s10
                esp32p4_s16_element_store  a0, s0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_33cn_relu_n_remainder_end:

    esp32p4_pop_24_stacks_6r  s0, s1, s8, s9, s10, s11     // restore registers
    ret



############################################################################################################################################################
####
#### esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn series
####
############################################################################################################################################################

    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value / output
    # t4: 15 - c_remainder
    # t5: tmp value / zero
    # t6: bias_ptr

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): filter_h
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions): filter_w
    # s3(not for extension instructions): filter_y_offset
    # s4(not for extension instructions): filter_n_offset
    # s5(not for extension instructions):

    # s0:
    # s1: clamp min value
    # s8: clamp max value
    # s9:
    # s10:
    # s11: mac_shift element

    esp32p4_push_24_stacks_6r  s1, s2, s3, s4, s8, s11
    li  s1, -32768                  // clamp min value
    li  s8, 32767                   // clamp max value

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        lw  s3, 60(a2)              // s3: filter_y_offset
        lw  s4, 144(a2)             // s4: filter_n_offset
        esp.movx.r.sar.bytes  t3    // t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_128b
            li  t5, 8
            beq t3, t5, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_64b
                # esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_32b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_32b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_64b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_128b_multiple_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_128b_multiple_loop

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_n_remainder:
        lw  t1, 140(a2)             // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_n_remainder_end
            lw  s3, 160(a2)         // s3: filter_y_offset_unaligned
            lw  s4, 164(a2)         // s4: filter_n_offset_unaligned
            lw  a3, 168(a2)         // a3: filter_ptr_unaligned

            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_n_remainder_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_hwc1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, a7, s2, a2, t4, t5, s3, s4
                esp32p4_s16_element_result  t3, s11
                esp32p4_clamp  t3, s1, s8
                esp32p4_s16_element_store  a0, t3

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_n_remainder_end:

    esp32p4_pop_24_stacks_6r  s1, s2, s3, s4, s8, s11
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value / output
    # t4: 15 - c_remainder
    # t5: tmp value / zero
    # t6: bias_ptr

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): filter_h
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions): filter_w
    # s3(not for extension instructions): filter_y_offset
    # s4(not for extension instructions): filter_n_offset
    # s5(not for extension instructions):

    # s0:
    # s1: clamp min value
    # s8: clamp max value
    # s9: activation_alpha/_address
    # s10: activation_shift
    # s11: mac_shift element

    esp32p4_push_32_stacks_8r  s1, s2, s3, s4, s8, s9, s10, s11
    li  s1, -32768                  // clamp min value
    li  s8, 32767                   // clamp max value
    lw  s9, 76(a2)      // activation_alpha
    lw  s10, 84(a2)     // activation_shift

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6
    lw  t6, 68(a2)      // bias

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        lw  s3, 60(a2)              // s3: filter_y_offset
        lw  s4, 144(a2)             // s4: filter_n_offset
        esp.movx.r.sar.bytes  t3    // t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_128b
            li  t5, 8
            beq t3, t5, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_64b
                # esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_32b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_32b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_64b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp32p4_s16_conv2d_128b_vector_bias  t6
                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_128b_multiple_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.qacc

                esp32p4_s16_conv2d_128b_vector_bias  t6
                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_128b_multiple_loop

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_n_remainder:
        lw  t1, 140(a2)             // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_n_remainder_end
            lw  s3, 160(a2)         // s3: filter_y_offset_unaligned
            lw  s4, 164(a2)         // s4: filter_n_offset_unaligned
            lw  a3, 168(a2)         // a3: filter_ptr_unaligned

            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_n_remainder_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.xacc

                esp32p4_s16_conv2d_element_bias  t6
                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_hwc1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, a7, s2, a2, t4, t5, s3, s4
                esp32p4_s16_element_result  t3, s11
                esp32p4_clamp  t3, s1, s8
                esp32p4_s16_element_leakyrelu  t3, s9, s10
                esp32p4_s16_element_store  a0, t3

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_bias_relu_n_remainder_end:

    esp32p4_pop_32_stacks_8r  s1, s2, s3, s4, s8, s9, s10, s11
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value / output
    # t4: 15 - c_remainder
    # t5: tmp value / zero
    # t6:

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): filter_h
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions): filter_w
    # s3(not for extension instructions): filter_y_offset
    # s4(not for extension instructions): filter_n_offset
    # s5(not for extension instructions):

    # s0:
    # s1: clamp min value
    # s8: clamp max value
    # s9:
    # s10:
    # s11: mac_shift element

    esp32p4_push_24_stacks_6r  s1, s2, s3, s4, s8, s11
    li  s1, -32768                  // clamp min value
    li  s8, 32767                   // clamp max value

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        lw  s3, 60(a2)              // s3: filter_y_offset
        lw  s4, 144(a2)             // s4: filter_n_offset
        esp.movx.r.sar.bytes  t3    // t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_128b
            li  t5, 8
            beq t3, t5, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_64b
                # esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_32b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_32b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_64b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_128b_multiple_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.qacc

                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_128b_multiple_loop

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_n_remainder:
        lw  t1, 140(a2)             // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_n_remainder_end
            lw  s3, 160(a2)         // s3: filter_y_offset_unaligned
            lw  s4, 164(a2)         // s4: filter_n_offset_unaligned
            lw  a3, 168(a2)         // a3: filter_ptr_unaligned

            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_n_remainder_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.xacc

                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_hwc1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, a7, s2, a2, t4, t5, s3, s4
                esp32p4_s16_element_result  t3, s11
                esp32p4_clamp  t3, s1, s8
                esp32p4_s16_element_store  a0, t3

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_n_remainder_end:

    esp32p4_pop_24_stacks_6r  s1, s2, s3, s4, s8, s11
    ret



    .text
    .global dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu
    .type   dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu:

    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args

    # a3: filter_ptr
    # a4: c_remainder
    # a5: tie_filter_channel_factor
    # t3: output_sar_byte / moving input_ptr / tmp value / output
    # t4: 15 - c_remainder
    # t5: tmp value / zero
    # t6:

    # a6(not for extension instructions): dilation_y_offset
    # a7(not for extension instructions): filter_h
    # t0(not for extension instructions): c_div_x_1
    # t1(not for extension instructions): n_div_x / n_remainder
    # t2(not for extension instructions): dilation_x_offset
    # s2(not for extension instructions): filter_w
    # s3(not for extension instructions): filter_y_offset
    # s4(not for extension instructions): filter_n_offset
    # s5(not for extension instructions):

    # s0:
    # s1: clamp min value
    # s8: clamp max value
    # s9: activation_alpha/_address
    # s10: activation_shift
    # s11: mac_shift element

    esp32p4_push_32_stacks_8r  s1, s2, s3, s4, s8, s9, s10, s11
    li  s1, -32768                  // clamp min value
    li  s8, 32767                   // clamp max value
    lw  s9, 76(a2)      // activation_alpha
    lw  s10, 84(a2)     // activation_shift

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_load_args  a2, a3, t0, t1, a5, a4, t2, a6

    beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_n_remainder
        esp.ld.128.usar.ip  q0, a0, 0
        lw  s3, 60(a2)              // s3: filter_y_offset
        lw  s4, 144(a2)             // s4: filter_n_offset
        esp.movx.r.sar.bytes  t3    // t3: output_sar_byte

        beqz  t3, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_128b
            li  t5, 8
            beq t3, t5, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_64b
                # esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_32b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_32b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_32b_unaligned_vector_store  q0, a0, t3

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_32b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_n_remainder

            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_64b:
                esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_64b_multiple_loop:
                    mv  t3, a1      // t3: input_ptr
                    esp.zero.qacc

                    esp.vld.128.ip  q6, a5, 16
                    esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                    esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                    esp32p4_s16_128b_vector_relu  q0, s9, s10
                    esp32p4_s16_64b_unaligned_vector_store  q0, a0

                    addi  t1, t1, -1
                    bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_64b_multiple_loop
                j  esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_n_remainder

        esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_128b:
            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_128b_multiple_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.qacc

                esp.vld.128.ip  q6, a5, 16
                esp32p4_w8a16_unaligned_conv2d_hwc8  q0, q1, q2, q3, q4, t3, a3, t0, a4, t2, a6, a7, s2, a2, s3, s4, t5
                esp32p4_s16_128b_vector_per_channel_shift_result  q0, q6
                esp32p4_s16_128b_vector_relu  q0, s9, s10
                esp32p4_s16_128b_aligned_vector_store  q0, a0

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_128b_multiple_loop

    esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_n_remainder:
        lw  t1, 140(a2)             // t1: n_remainder
        beqz  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_n_remainder_end
            lw  s3, 160(a2)         // s3: filter_y_offset_unaligned
            lw  s4, 164(a2)         // s4: filter_n_offset_unaligned
            lw  a3, 168(a2)         // a3: filter_ptr_unaligned

            li  t4, 15
            sub  t4, t4, a4         // t4: 15 - c_remainder
            li  t5,  0              // t5: zero

            esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_n_remainder_loop:
                mv  t3, a1          // t3: input_ptr
                esp.zero.xacc

                lh  s11, 0(a5)
                addi  a5, a5, 2
                esp32p4_w8a16_unaligned_conv2d_hwc1  q0, q1, q2, q3, q4, q5, t3, a3, t0, a4, t2, a6, a7, s2, a2, t4, t5, s3, s4
                esp32p4_s16_element_result  t3, s11
                esp32p4_clamp  t3, s1, s8
                esp32p4_s16_element_leakyrelu  t3, s9, s10
                esp32p4_s16_element_store  a0, t3

                addi  t1, t1, -1
                bnez  t1, esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_n_remainder_loop

        esp32p4_w8a16_unaligned_conv2d_per_channel_hwcn_relu_n_remainder_end:

    esp32p4_pop_32_stacks_8r  s1, s2, s3, s4, s8, s9, s10, s11
    ret
