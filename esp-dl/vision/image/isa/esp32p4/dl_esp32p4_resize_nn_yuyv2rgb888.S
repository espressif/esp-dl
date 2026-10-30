#include "dl_esp32p4_color_common.S"

# q0/q1/q2 are the planar output channels. Temps are a4, a5, t3-t6, then q3-q7.
.macro resize_nn_simd_helper_yuyv2rgb888_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n          (16-pixel blocks)

    blez a3, .Lend\@
    li a4, 14
.Lloop\@:
    resize_nn_\pix_cvt q0, q1, q2, a0, a1, a4, a5, t3, t4, t5, t6, q3, q4, q5, q6, q7
    esp.vzipt.8 q0, q1, q2
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q1, a2, 16
    esp.vst.128.ip q2, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm

resize_nn_simd_helper_yuyv2rgb888_template yuyv2rgb888
resize_nn_simd_helper_yuyv2rgb888_template uyvy2rgb888
resize_nn_simd_helper_yuyv2rgb888_template yuyv2bgr888
resize_nn_simd_helper_yuyv2rgb888_template uyvy2bgr888

# a4 is the LUT. Temps are a5, t3-t6, s0, then q3-q7.
.macro resize_nn_simd_helper_yuyv2rgb888_qint8_template pix_cvt
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

    addi sp, sp, -16
    sw   s0, 0(sp)
    blez a3, .Lend\@
    li a5, 14
.Lloop\@:
    resize_nn_\pix_cvt q0, q1, q2, a0, a1, a5, t3, t4, t5, t6, s0, a4, q3, q4, q5, q6, q7
    esp.vzipt.8 q0, q1, q2
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q1, a2, 16
    esp.vst.128.ip q2, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    addi sp, sp, 16
    ret
.endm

resize_nn_simd_helper_yuyv2rgb888_qint8_template yuyv2rgb888_qint8
resize_nn_simd_helper_yuyv2rgb888_qint8_template uyvy2rgb888_qint8
resize_nn_simd_helper_yuyv2rgb888_qint8_template yuyv2bgr888_qint8
resize_nn_simd_helper_yuyv2rgb888_qint8_template uyvy2bgr888_qint8

# q0/q1, q2/q3, q4/q5 are the three channels' low/high halves.
# a4 is the LUT. Temps are a5, t3-t6, s0, then q6, q7.
.macro resize_nn_simd_helper_yuyv2rgb888_qint16_template pix_cvt
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

    addi sp, sp, -16
    sw   s0, 0(sp)
    blez a3, .Lend\@
    li a5, 14
.Lloop\@:
    resize_nn_\pix_cvt q0, q1, q2, q3, q4, q5, a0, a1, a5, t3, t4, t5, t6, s0, a4, q6, q7
    esp.vzipt.16 q0, q2, q4
    esp.vzipt.16 q1, q3, q5
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q2, a2, 16
    esp.vst.128.ip q4, a2, 16
    esp.vst.128.ip q1, a2, 16
    esp.vst.128.ip q3, a2, 16
    esp.vst.128.ip q5, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    addi sp, sp, 16
    ret
.endm

resize_nn_simd_helper_yuyv2rgb888_qint16_template yuyv2rgb888_qint16
resize_nn_simd_helper_yuyv2rgb888_qint16_template uyvy2rgb888_qint16
resize_nn_simd_helper_yuyv2rgb888_qint16_template yuyv2bgr888_qint16
resize_nn_simd_helper_yuyv2rgb888_qint16_template uyvy2bgr888_qint16
