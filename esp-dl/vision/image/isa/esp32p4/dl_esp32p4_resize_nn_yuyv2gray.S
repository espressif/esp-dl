#include "dl_esp32p4_color_common.S"

# YUYV keeps the byte at the pixel address. UYVY keeps the next byte.
.macro resize_nn_simd_helper_yuyv2gray_template pix_cvt
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
.ifc "\pix_cvt", "yuyv2gray"
    resize_nn_load_store_1chn_128 a2, a0, a1, q0, q1, q2, a4, a5, t3, t4, 0
.endif
.ifc "\pix_cvt", "uyvy2gray"
    resize_nn_load_store_1chn_128 a2, a0, a1, q0, q1, q2, a4, a5, t3, t4, 1
.endif
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm

resize_nn_simd_helper_yuyv2gray_template yuyv2gray
resize_nn_simd_helper_yuyv2gray_template uyvy2gray

.macro resize_nn_simd_helper_yuyv2gray_qint8_template pix_cvt
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
.ifc "\pix_cvt", "yuyv2gray_qint8"
    resize_nn_load_1chn_128 q0, a0, a1, q1, q2, q3, a5, t3, t4, t5, 0
.endif
.ifc "\pix_cvt", "uyvy2gray_qint8"
    resize_nn_load_1chn_128 q0, a0, a1, q1, q2, q3, a5, t3, t4, t5, 1
.endif
    norm_quant_1chn_8 q0, a4, q1, q2, q3
    esp.vst.128.ip q0, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm

resize_nn_simd_helper_yuyv2gray_qint8_template yuyv2gray_qint8
resize_nn_simd_helper_yuyv2gray_qint8_template uyvy2gray_qint8

.macro resize_nn_simd_helper_yuyv2gray_qint16_template pix_cvt
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
.ifc "\pix_cvt", "yuyv2gray_qint16"
    resize_nn_load_1chn_128 q1, a0, a1, q0, q2, q3, a5, t3, t4, t5, 0
.endif
.ifc "\pix_cvt", "uyvy2gray_qint16"
    resize_nn_load_1chn_128 q1, a0, a1, q0, q2, q3, a5, t3, t4, t5, 1
.endif
    norm_quant_1chn_16 q0, q1, a4, q2, q3
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q1, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm

resize_nn_simd_helper_yuyv2gray_qint16_template yuyv2gray_qint16
resize_nn_simd_helper_yuyv2gray_qint16_template uyvy2gray_qint16
