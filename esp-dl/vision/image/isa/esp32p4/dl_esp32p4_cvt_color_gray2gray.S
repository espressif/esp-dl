#include "dl_esp32p4_color_common.S"

    .balign 4
    .text
    .global cvt_color_simd_helper_gray2gray
    .type   cvt_color_simd_helper_gray2gray, @function
cvt_color_simd_helper_gray2gray:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n

    blez a2, 1f
0:
    esp.vld.128.ip q0, a0, 16
    esp.vst.128.ip q0, a1, 16
    addi a2, a2, -1
    bnez a2, 0b
1:
    ret

    .balign 4
    .text
    .global cvt_color_simd_helper_gray2gray_qint8
    .type   cvt_color_simd_helper_gray2gray_qint8, @function
cvt_color_simd_helper_gray2gray_qint8:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n
    # a3: int *lut

    blez a2, 1f
0:
    esp.vld.128.ip q0, a0, 16
    norm_quant_1chn_8 q0, a3, q1, q2, q3
    esp.vst.128.ip q0, a1, 16
    addi a2, a2, -1
    bnez a2, 0b
1:
    ret

    .balign 4
    .text
    .global cvt_color_simd_helper_gray2gray_qint16
    .type   cvt_color_simd_helper_gray2gray_qint16, @function
cvt_color_simd_helper_gray2gray_qint16:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n
    # a3: int *lut

    blez a2, 1f
0:
    esp.vld.128.ip q1, a0, 16
    norm_quant_1chn_16 q0, q1, a3, q2, q3
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q1, a1, 16
    addi a2, a2, -1
    bnez a2, 0b
1:
    ret
