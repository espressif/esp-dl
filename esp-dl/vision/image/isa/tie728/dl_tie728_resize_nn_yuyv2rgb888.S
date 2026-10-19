#include "dl_tie728_color_common.S"

# q0/q1/q2 are the planar output channels. Temps are a6..a12, then q3-q7.
.macro resize_nn_simd_helper_yuyv2rgb888_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:
    entry sp, 96
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n          (16-pixel blocks)

    blti a5, 1, .Lend\@

    extui a12, a4, 0, 4
    bnez a12, .Ldst_unaligned\@

.Lloop\@:
    resize_nn_\pix_cvt q0, q1, q2, a2, a3, sp, a6, a7, a8, a9, a10, a11, q3, q4, q5, q6, q7
    vzipt_8 q0, q1, q2, q3
    ee.vst.128.ip q0, a4, 16
    ee.vst.128.ip q1, a4, 16
    ee.vst.128.ip q2, a4, 16
    addi a5, a5, -1
    bnez a5, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    addi a11, sp, 48
    vst_unaligned_init a4, a12, q6, q7, a11
.Lloop_unaligned\@:
    resize_nn_\pix_cvt q0, q1, q2, a2, a3, sp, a6, a7, a8, a9, a10, a11, q3, q4, q5, q6, q7
    vzipt_8 q0, q1, q2, q3
    addi a11, sp, 48
    vst_3chn_unaligned q0, q1, q2, a4, a12, q6, q7, a11
    addi a5, a5, -1
    bnez a5, .Lloop_unaligned\@
    addi a11, sp, 48
    vst_unaligned_flush a4, a12, q6, q7, a11

.Lend\@:
    retw.n
.endm

resize_nn_simd_helper_yuyv2rgb888_template yuyv2rgb888
resize_nn_simd_helper_yuyv2rgb888_template uyvy2rgb888
resize_nn_simd_helper_yuyv2rgb888_template yuyv2bgr888
resize_nn_simd_helper_yuyv2rgb888_template uyvy2bgr888

# a6 is the LUT. Temps are a7..a13, then q3-q7.
.macro resize_nn_simd_helper_yuyv2rgb888_qint8_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:
    entry sp, 96
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n          (16-pixel blocks)
    # a6: int *lut

    blti a5, 1, .Lend\@

    extui a13, a4, 0, 4
    bnez a13, .Ldst_unaligned\@

.Lloop\@:
    resize_nn_\pix_cvt q0, q1, q2, a2, a3, sp, a7, a8, a9, a10, a11, a12, a6, q3, q4, q5, q6, q7
    vzipt_8 q0, q1, q2, q3
    ee.vst.128.ip q0, a4, 16
    ee.vst.128.ip q1, a4, 16
    ee.vst.128.ip q2, a4, 16
    addi a5, a5, -1
    bnez a5, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    addi a12, sp, 48
    vst_unaligned_init a4, a13, q6, q7, a12
.Lloop_unaligned\@:
    resize_nn_\pix_cvt q0, q1, q2, a2, a3, sp, a7, a8, a9, a10, a11, a12, a6, q3, q4, q5, q6, q7
    vzipt_8 q0, q1, q2, q3
    addi a12, sp, 48
    vst_3chn_unaligned q0, q1, q2, a4, a13, q6, q7, a12
    addi a5, a5, -1
    bnez a5, .Lloop_unaligned\@
    addi a12, sp, 48
    vst_unaligned_flush a4, a13, q6, q7, a12

.Lend\@:
    retw.n
.endm

resize_nn_simd_helper_yuyv2rgb888_qint8_template yuyv2rgb888_qint8
resize_nn_simd_helper_yuyv2rgb888_qint8_template uyvy2rgb888_qint8
resize_nn_simd_helper_yuyv2rgb888_qint8_template yuyv2bgr888_qint8
resize_nn_simd_helper_yuyv2rgb888_qint8_template uyvy2bgr888_qint8

# q0/q1, q2/q3, q4/q5 are the three channels' low/high halves.
# a6 is the LUT. Temps are a7..a13, then q6, q7.
.macro resize_nn_simd_helper_yuyv2rgb888_qint16_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:
    entry sp, 96
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n          (16-pixel blocks)
    # a6: int *lut

    blti a5, 1, .Lend\@

    extui a13, a4, 0, 4
    bnez a13, .Ldst_unaligned\@

.Lloop\@:
    resize_nn_\pix_cvt q0, q1, q2, q3, q4, q5, a2, a3, sp, a7, a8, a9, a10, a11, a12, a6, q6, q7
    vzipt_16 q0, q2, q4, q6
    vzipt_16 q1, q3, q5, q6
    ee.vst.128.ip q0, a4, 16
    ee.vst.128.ip q2, a4, 16
    ee.vst.128.ip q4, a4, 16
    ee.vst.128.ip q1, a4, 16
    ee.vst.128.ip q3, a4, 16
    ee.vst.128.ip q5, a4, 16
    addi a5, a5, -1
    bnez a5, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    addi a12, sp, 48
    vst_unaligned_init a4, a13, q6, q7, a12
.Lloop_unaligned\@:
    resize_nn_\pix_cvt q0, q1, q2, q3, q4, q5, a2, a3, sp, a7, a8, a9, a10, a11, a12, a6, q6, q7
    vzipt_16 q0, q2, q4, q6
    vzipt_16 q1, q3, q5, q6
    addi a12, sp, 48
    vst_6chn_unaligned q0, q2, q4, q1, q3, q5, a4, a13, q6, q7, a12
    addi a5, a5, -1
    bnez a5, .Lloop_unaligned\@
    addi a12, sp, 48
    vst_unaligned_flush a4, a13, q6, q7, a12

.Lend\@:
    retw.n
.endm

resize_nn_simd_helper_yuyv2rgb888_qint16_template yuyv2rgb888_qint16
resize_nn_simd_helper_yuyv2rgb888_qint16_template uyvy2rgb888_qint16
resize_nn_simd_helper_yuyv2rgb888_qint16_template yuyv2bgr888_qint16
resize_nn_simd_helper_yuyv2rgb888_qint16_template uyvy2bgr888_qint16
