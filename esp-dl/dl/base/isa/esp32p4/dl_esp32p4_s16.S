#pragma once


############################################################################################################################################################
# result process for conv2d / depthwise_conv2d
############################################################################################################################################################
.macro esp32p4_s16_conv2d_128b_vector_bias  bias_ptr
    esp.ld.qacc.l.l.128.ip  \bias_ptr, 16
    esp.ld.qacc.l.h.128.ip  \bias_ptr, 16
    esp.ld.qacc.h.l.128.ip  \bias_ptr, 16
    esp.ld.qacc.h.h.128.ip  \bias_ptr, 16
.endm



############################################################################################################################################################
# esp32p4_s16_128b_vector series
############################################################################################################################################################
.macro esp32p4_s16_128b_vector_shift_result  output_v, mac_shift
    esp.srcmb.s16.qacc  \output_v, \mac_shift, 1
.endm



.macro esp32p4_s16_128b_aligned_vector_store  output_v, output_ptr
    esp.vst.128.ip  \output_v, \output_ptr, 16
.endm



.macro esp32p4_s16_128b_vector_relu  output_v, activation_alpha, activation_shift
    # LeakyReLU
    esp.vrelu.s16  \output_v, \activation_alpha, \activation_shift
.endm

.macro dl_esp32p4_128b_unaligned_store0 output_v, output_ptr, tmp32
    esp.movi.32.a \output_v, \tmp32, 0
    sw \tmp32, 0(\output_ptr)
    esp.movi.32.a \output_v, \tmp32, 1
    sw \tmp32, 4(\output_ptr)
    esp.movi.32.a \output_v, \tmp32, 2
    sw \tmp32, 8(\output_ptr)
    esp.movi.32.a \output_v, \tmp32, 3
    sw \tmp32, 12(\output_ptr)
    addi \output_ptr, \output_ptr, 16
.endm

.macro dl_esp32p4_128b_unaligned_store1 output_v, output_ptr
    esp.vst.l.64.ip \output_v, \output_ptr, 8
    esp.vst.h.64.ip \output_v, \output_ptr, 8
.endm

.macro dl_esp32p4_128b_last_store_data tmp_q, output_v, tmp_a, c_remainder_bytes
    beqz \c_remainder_bytes, 600f
    li \tmp_a, 15
    sub  \tmp_a, \tmp_a, \c_remainder_bytes 
    li \c_remainder_bytes, 0
    esp.slcxxp.2q \tmp_q, \output_v, \tmp_a, \c_remainder_bytes #left shift to make the rest part 0
    esp.srcxxp.2q \output_v, \tmp_q, \tmp_a, \c_remainder_bytes #right shift to lower bits
600:
.endm


.macro dl_esp32p4_s16_store_aligned_remainder remainder_data, c_remainder, tmp_a, output_ptr
607:    # remainder == 1, 0x111
    andi t0, \c_remainder, 4
    beqz t0, 603f
    andi t0, \c_remainder, 2
    beqz t0, 605f
    andi t0, \c_remainder, 1
    beqz t0, 606f
    esp.vst.l.64.ip \remainder_data, \output_ptr, 0
    esp.movi.32.a \remainder_data, \tmp_a, 2
    sw \tmp_a, 8(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 3
    sh \tmp_a, 12(\output_ptr)
    j 600f
606:    # remainder == 1, 0x110
    esp.vst.l.64.ip \remainder_data, \output_ptr, 0
    esp.movi.32.a \remainder_data, \tmp_a, 2
    sw \tmp_a, 8(\output_ptr)
    j 600f
605:    # remainder == 1, 0x101
    andi t0, \c_remainder, 1
    beqz t0, 604f
    esp.vst.l.64.ip \remainder_data, \output_ptr, 0
    esp.movi.32.a \remainder_data, \tmp_a, 2
    sh \tmp_a, 8(\output_ptr)
    j 600f
604:    # remainder == 1, 0x100
    esp.vst.l.64.ip \remainder_data, \output_ptr, 0
    j 600f
603:    # remainder == 1, 0x011
    andi t0, \c_remainder, 2
    beqz t0, 601f
    andi t0, \c_remainder, 1
    beqz t0, 602f
    esp.movi.32.a \remainder_data, \tmp_a, 0
    sw \tmp_a, 0(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 1
    sh \tmp_a, 4(\output_ptr)
    j 600f
602:    # remainder == 1, 0x010
    esp.movi.32.a \remainder_data, \tmp_a, 0
    sw \tmp_a, 0(\output_ptr)
    j 600f
601:    # remainder == 1, 0x001
    andi t0, \c_remainder, 1
    beqz t0, 600f
    esp.movi.32.a \remainder_data, \tmp_a, 0
    sh \tmp_a, 0(\output_ptr)
600:
.endm


.macro dl_esp32p4_s16_store_remainder remainder_data, c_remainder, tmp_a, output_ptr
607:    # remainder == 1, 0x111
    andi t0, \c_remainder, 4
    beqz t0, 603f
    andi t0, \c_remainder, 2
    beqz t0, 605f
    andi t0, \c_remainder, 1
    beqz t0, 606f
    esp.movi.32.a \remainder_data, \tmp_a, 0
    sw \tmp_a, 0(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 1
    sw \tmp_a, 4(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 2
    sw \tmp_a, 8(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 3
    sh \tmp_a, 12(\output_ptr)
    j 600f
606:    # remainder == 1, 0x110
    esp.movi.32.a \remainder_data, \tmp_a, 0
    sw \tmp_a, 0(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 1
    sw \tmp_a, 4(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 2
    sw \tmp_a, 8(\output_ptr)
    j 600f
605:    # remainder == 1, 0x101
    andi t0, \c_remainder, 1
    beqz t0, 604f
    esp.movi.32.a \remainder_data, \tmp_a, 0
    sw \tmp_a, 0(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 1
    sw \tmp_a, 4(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 2
    sh \tmp_a, 8(\output_ptr)
    j 600f
604:    # remainder == 1, 0x100
    esp.movi.32.a \remainder_data, \tmp_a, 0
    sw \tmp_a, 0(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 1
    sw \tmp_a, 4(\output_ptr)
    j 600f
603:    # remainder == 1, 0x011
    andi t0, \c_remainder, 2
    beqz t0, 601f
    andi t0, \c_remainder, 1
    beqz t0, 602f
    esp.movi.32.a \remainder_data, \tmp_a, 0
    sw \tmp_a, 0(\output_ptr)
    esp.movi.32.a \remainder_data, \tmp_a, 1
    sh \tmp_a, 4(\output_ptr)
    j 600f
602:    # remainder == 1, 0x010
    esp.movi.32.a \remainder_data, \tmp_a, 0
    sw \tmp_a, 0(\output_ptr)
    j 600f
601:    # remainder == 1, 0x001
    andi t0, \c_remainder, 1
    beqz t0, 600f
    esp.movi.32.a \remainder_data, \tmp_a, 0
    sh \tmp_a, 0(\output_ptr)
600:
.endm