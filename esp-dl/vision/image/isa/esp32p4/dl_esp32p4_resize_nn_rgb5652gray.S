#include "dl_esp32p4_color_common.S"

.macro resize_nn_simd_helper_rgb5652gray_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n

    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, a4, a5, t3, t4
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q0, q1, a4, a5, q3, q4, q5, q6, q7
    esp.vst.128.ip q2, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm
resize_nn_simd_helper_rgb5652gray_template rgb565le2gray
resize_nn_simd_helper_rgb5652gray_template rgb565be2gray
resize_nn_simd_helper_rgb5652gray_template bgr565le2gray
resize_nn_simd_helper_rgb5652gray_template bgr565be2gray

.macro resize_nn_simd_helper_rgb5652gray_qint8_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: int *lut

    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, a5, t3, t4, t5
    esp.vunzip.8 q0, q1
    \pix_cvt q2, a4, q0, q1, a5, t3, q3, q4, q5, q6, q7
    esp.vst.128.ip q2, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm
resize_nn_simd_helper_rgb5652gray_qint8_template rgb565le2gray_qint8
resize_nn_simd_helper_rgb5652gray_qint8_template rgb565be2gray_qint8
resize_nn_simd_helper_rgb5652gray_qint8_template bgr565le2gray_qint8
resize_nn_simd_helper_rgb5652gray_qint8_template bgr565be2gray_qint8

.macro resize_nn_simd_helper_rgb5652gray_qint16_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: int *lut

    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, a5, t3, t4, t5
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q3, a4, q0, q1, a5, t3, q4, q5, q6, q7
    esp.vst.128.ip q2, a2, 16
    esp.vst.128.ip q3, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm
resize_nn_simd_helper_rgb5652gray_qint16_template rgb565le2gray_qint16
resize_nn_simd_helper_rgb5652gray_qint16_template rgb565be2gray_qint16
resize_nn_simd_helper_rgb5652gray_qint16_template bgr565le2gray_qint16
resize_nn_simd_helper_rgb5652gray_qint16_template bgr565be2gray_qint16
