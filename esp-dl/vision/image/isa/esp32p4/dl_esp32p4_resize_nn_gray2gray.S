#include "dl_esp32p4_color_common.S"

    .balign 4
    .text
    .global resize_nn_simd_helper_gray2gray
    .type   resize_nn_simd_helper_gray2gray, @function
resize_nn_simd_helper_gray2gray:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n

    blez a3, 1f
0:
    resize_nn_load_store_1chn_128 a2, a0, a1, q0, q1, q2, a4, a5, t3, t4
    addi a3, a3, -1
    bnez a3, 0b
1:
    ret

    .balign 4
    .text
    .global resize_nn_simd_helper_gray2gray_qint8
    .type   resize_nn_simd_helper_gray2gray_qint8, @function
resize_nn_simd_helper_gray2gray_qint8:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: int *lut

    blez a3, 1f
0:
    resize_nn_load_1chn_128 q0, a0, a1, q1, q2, q3, a5, t3, t4, t5
    norm_quant_1chn_8 q0, a4, q1, q2, q3
    esp.vst.128.ip q0, a2, 16
    addi a3, a3, -1
    bnez a3, 0b
1:
    ret

    .balign 4
    .text
    .global resize_nn_simd_helper_gray2gray_qint16
    .type   resize_nn_simd_helper_gray2gray_qint16, @function
resize_nn_simd_helper_gray2gray_qint16:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: int *lut

    blez a3, 1f
0:
    resize_nn_load_1chn_128 q1, a0, a1, q0, q2, q3, a5, t3, t4, t5
    norm_quant_1chn_16 q0, q1, a4, q2, q3
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q1, a2, 16
    addi a3, a3, -1
    bnez a3, 0b
1:
    ret
