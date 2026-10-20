#include "dl_tie728_color_common.S"

# q0/q1/q2 are the planar channels. YUV lanes are q3-q6. q7 is the scratch.
.macro cvt_color_simd_helper_yuyv2rgb888_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:
    entry sp, 32
    # a2: uint8_t *src  (even pixel of a YUV pair)
    # a3: uint8_t *dst
    # a4: int n         (16-pixel blocks)

    blti a4, 1, .Lend\@
    movi a6, 14

    extui a7, a3, 0, 4
    bnez a7, .Ldst_unaligned\@

.Lloop\@:
    cvt_\pix_cvt q0, q1, q2, a2, q3, q4, q5, q6, a5, a6, q7
    vzipt_8 q0, q1, q2, q3
    ee.vst.128.ip q0, a3, 16
    ee.vst.128.ip q1, a3, 16
    ee.vst.128.ip q2, a3, 16
    addi a4, a4, -1
    bnez a4, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a3, a7, q6, q7, sp
.Lloop_unaligned\@:
    cvt_\pix_cvt q0, q1, q2, a2, q3, q4, q5, q6, a5, a6, q7
    vzipt_8 q0, q1, q2, q3
    vst_3chn_unaligned q0, q1, q2, a3, a7, q6, q7, sp
    addi a4, a4, -1
    bnez a4, .Lloop_unaligned\@
    vst_unaligned_flush a3, a7, q6, q7, sp

.Lend\@:
    retw.n
.endm

cvt_color_simd_helper_yuyv2rgb888_template yuyv2rgb888
cvt_color_simd_helper_yuyv2rgb888_template uyvy2rgb888
cvt_color_simd_helper_yuyv2rgb888_template yuyv2bgr888
cvt_color_simd_helper_yuyv2rgb888_template uyvy2bgr888

# a5 is the LUT. Temps are a6, a7, a8.
.macro cvt_color_simd_helper_yuyv2rgb888_qint8_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:
    entry sp, 32
    # a2: uint8_t *src  (even pixel of a YUV pair)
    # a3: uint8_t *dst
    # a4: int n         (16-pixel blocks)
    # a5: int *lut

    blti a4, 1, .Lend\@
    movi a7, 14

    extui a8, a3, 0, 4
    bnez a8, .Ldst_unaligned\@

.Lloop\@:
    cvt_\pix_cvt q0, q1, q2, a2, q3, q4, q5, q6, a5, a6, a7, q7
    vzipt_8 q0, q1, q2, q3
    ee.vst.128.ip q0, a3, 16
    ee.vst.128.ip q1, a3, 16
    ee.vst.128.ip q2, a3, 16
    addi a4, a4, -1
    bnez a4, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a3, a8, q6, q7, sp
.Lloop_unaligned\@:
    cvt_\pix_cvt q0, q1, q2, a2, q3, q4, q5, q6, a5, a6, a7, q7
    vzipt_8 q0, q1, q2, q3
    vst_3chn_unaligned q0, q1, q2, a3, a8, q6, q7, sp
    addi a4, a4, -1
    bnez a4, .Lloop_unaligned\@
    vst_unaligned_flush a3, a8, q6, q7, sp

.Lend\@:
    retw.n
.endm

cvt_color_simd_helper_yuyv2rgb888_qint8_template yuyv2rgb888_qint8
cvt_color_simd_helper_yuyv2rgb888_qint8_template uyvy2rgb888_qint8
cvt_color_simd_helper_yuyv2rgb888_qint8_template yuyv2bgr888_qint8
cvt_color_simd_helper_yuyv2rgb888_qint8_template uyvy2bgr888_qint8

# q0/q1, q2/q3, q4/q5 are the three channels' low/high halves.
# a5 is the LUT. Temps are a6, a7, a8, then q6, q7.
.macro cvt_color_simd_helper_yuyv2rgb888_qint16_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:
    entry sp, 32
    # a2: uint8_t *src  (even pixel of a YUV pair)
    # a3: uint8_t *dst
    # a4: int n         (16-pixel blocks)
    # a5: int *lut

    blti a4, 1, .Lend\@
    movi a7, 14

    extui a8, a3, 0, 4
    bnez a8, .Ldst_unaligned\@

.Lloop\@:
    cvt_\pix_cvt q0, q1, q2, q3, q4, q5, a2, a5, a6, a7, q6, q7
    vzipt_16 q0, q2, q4, q6
    vzipt_16 q1, q3, q5, q6
    ee.vst.128.ip q0, a3, 16
    ee.vst.128.ip q2, a3, 16
    ee.vst.128.ip q4, a3, 16
    ee.vst.128.ip q1, a3, 16
    ee.vst.128.ip q3, a3, 16
    ee.vst.128.ip q5, a3, 16
    addi a4, a4, -1
    bnez a4, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a3, a8, q6, q7, sp
.Lloop_unaligned\@:
    cvt_\pix_cvt q0, q1, q2, q3, q4, q5, a2, a5, a6, a7, q6, q7
    vzipt_16 q0, q2, q4, q6
    vzipt_16 q1, q3, q5, q6
    vst_6chn_unaligned q0, q2, q4, q1, q3, q5, a3, a8, q6, q7, sp
    addi a4, a4, -1
    bnez a4, .Lloop_unaligned\@
    vst_unaligned_flush a3, a8, q6, q7, sp

.Lend\@:
    retw.n
.endm

cvt_color_simd_helper_yuyv2rgb888_qint16_template yuyv2rgb888_qint16
cvt_color_simd_helper_yuyv2rgb888_qint16_template uyvy2rgb888_qint16
cvt_color_simd_helper_yuyv2rgb888_qint16_template yuyv2bgr888_qint16
cvt_color_simd_helper_yuyv2rgb888_qint16_template uyvy2bgr888_qint16
