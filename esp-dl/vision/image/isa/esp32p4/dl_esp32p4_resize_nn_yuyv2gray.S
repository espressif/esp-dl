#include "dl_esp32p4_color_common.S"

# byte_off is 0 for YUYV (Y at the pixel address) and 1 for UYVY.
.macro resize_nn_simd_helper_yuyv2gray_template pix_cvt, byte_off
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
    resize_nn_load_store_1chn_128 a2, a0, a1, q0, q1, q2, a4, a5, t3, t4, \byte_off
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm

resize_nn_simd_helper_yuyv2gray_template yuyv2gray, 0
resize_nn_simd_helper_yuyv2gray_template uyvy2gray, 1

.macro resize_nn_simd_helper_yuyv2gray_qint8_template pix_cvt, byte_off
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
    resize_nn_load_1chn_128 q0, a0, a1, q1, q2, q3, a5, t3, t4, t5, \byte_off
    norm_quant_1chn_8 q0, a4, q1, q2, q3
    esp.vst.128.ip q0, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm

resize_nn_simd_helper_yuyv2gray_qint8_template yuyv2gray_qint8, 0
resize_nn_simd_helper_yuyv2gray_qint8_template uyvy2gray_qint8, 1

.macro resize_nn_simd_helper_yuyv2gray_qint16_template pix_cvt, byte_off
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
    resize_nn_load_1chn_128 q1, a0, a1, q0, q2, q3, a5, t3, t4, t5, \byte_off
    norm_quant_1chn_16 q0, q1, a4, q2, q3
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q1, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm

resize_nn_simd_helper_yuyv2gray_qint16_template yuyv2gray_qint16, 0
resize_nn_simd_helper_yuyv2gray_qint16_template uyvy2gray_qint16, 1
