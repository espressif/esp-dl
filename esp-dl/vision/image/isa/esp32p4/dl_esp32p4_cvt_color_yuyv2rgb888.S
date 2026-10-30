#include "dl_esp32p4_color_common.S"

# q0/q1/q2 are the planar channels. YUV lanes are q3-q6. q7 is the scratch.
.macro cvt_color_simd_helper_yuyv2rgb888_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src  (even pixel of a YUV pair)
    # a1: uint8_t *dst
    # a2: int n         (16-pixel blocks)

    blez a2, .Lend\@
    li a4, 14
.Lloop\@:
    cvt_\pix_cvt q0, q1, q2, a0, q3, q4, q5, q6, a3, a4, q7
    esp.vzipt.8 q0, q1, q2
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q2, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_yuyv2rgb888_template yuyv2rgb888
cvt_color_simd_helper_yuyv2rgb888_template uyvy2rgb888
cvt_color_simd_helper_yuyv2rgb888_template yuyv2bgr888
cvt_color_simd_helper_yuyv2rgb888_template uyvy2bgr888

# a3 is the LUT. Temps are a4, a5.
.macro cvt_color_simd_helper_yuyv2rgb888_qint8_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src  (even pixel of a YUV pair)
    # a1: uint8_t *dst
    # a2: int n         (16-pixel blocks)
    # a3: int *lut

    blez a2, .Lend\@
    li a5, 14
.Lloop\@:
    cvt_\pix_cvt q0, q1, q2, a0, q3, q4, q5, q6, a3, a4, a5, q7
    esp.vzipt.8 q0, q1, q2
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q2, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_yuyv2rgb888_qint8_template yuyv2rgb888_qint8
cvt_color_simd_helper_yuyv2rgb888_qint8_template uyvy2rgb888_qint8
cvt_color_simd_helper_yuyv2rgb888_qint8_template yuyv2bgr888_qint8
cvt_color_simd_helper_yuyv2rgb888_qint8_template uyvy2bgr888_qint8

# q0/q1, q2/q3, q4/q5 are the three channels' low/high halves.
# a3 is the LUT. Temps are a4, a5, then q6, q7.
.macro cvt_color_simd_helper_yuyv2rgb888_qint16_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src  (even pixel of a YUV pair)
    # a1: uint8_t *dst
    # a2: int n         (16-pixel blocks)
    # a3: int *lut

    blez a2, .Lend\@
    li a5, 14
.Lloop\@:
    cvt_\pix_cvt q0, q1, q2, q3, q4, q5, a0, a3, a4, a5, q6, q7
    esp.vzipt.16 q0, q2, q4
    esp.vzipt.16 q1, q3, q5
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q2, a1, 16
    esp.vst.128.ip q4, a1, 16
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q3, a1, 16
    esp.vst.128.ip q5, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_yuyv2rgb888_qint16_template yuyv2rgb888_qint16
cvt_color_simd_helper_yuyv2rgb888_qint16_template uyvy2rgb888_qint16
cvt_color_simd_helper_yuyv2rgb888_qint16_template yuyv2bgr888_qint16
cvt_color_simd_helper_yuyv2rgb888_qint16_template uyvy2bgr888_qint16
