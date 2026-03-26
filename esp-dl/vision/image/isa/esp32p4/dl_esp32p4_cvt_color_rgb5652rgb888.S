#include "dl_esp32p4_color_common.S"

.macro cvt_color_simd_helper_rgb5652rgb888_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n

    blez a2, .Lend\@
.Lloop\@:
    esp.vld.128.ip q0, a0, 16
    esp.vld.128.ip q1, a0, 16
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q3, q4, q0, q1, a3, a4
    esp.vzipt.8 q2, q3, q4
    esp.vst.128.ip q2, a1, 16
    esp.vst.128.ip q3, a1, 16
    esp.vst.128.ip q4, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb5652rgb888_template rgb565le2rgb888
cvt_color_simd_helper_rgb5652rgb888_template rgb565be2rgb888
cvt_color_simd_helper_rgb5652rgb888_template rgb565le2bgr888
cvt_color_simd_helper_rgb5652rgb888_template rgb565be2bgr888

.macro cvt_color_simd_helper_rgb5652rgb888_qint8_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n
    # a3: int *lut

    blez a2, .Lend\@
.Lloop\@:
    esp.vld.128.ip q0, a0, 16
    esp.vld.128.ip q1, a0, 16
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q3, q4, a3, q0, q1, a4, a5, q5
    esp.vzipt.8 q2, q3, q4
    esp.vst.128.ip q2, a1, 16
    esp.vst.128.ip q3, a1, 16
    esp.vst.128.ip q4, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb5652rgb888_qint8_template rgb565le2rgb888_qint8
cvt_color_simd_helper_rgb5652rgb888_qint8_template rgb565be2rgb888_qint8
cvt_color_simd_helper_rgb5652rgb888_qint8_template rgb565le2bgr888_qint8
cvt_color_simd_helper_rgb5652rgb888_qint8_template rgb565be2bgr888_qint8

.macro cvt_color_simd_helper_rgb5652rgb888_qint16_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n
    # a3: int *lut

    blez a2, .Lend\@
.Lloop\@:
    esp.vld.128.ip q0, a0, 16
    esp.vld.128.ip q1, a0, 16
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q3, q4, q5, q6, q7, a3, q0, q1, a4, a5
    esp.vzipt.16 q2, q4, q6
    esp.vzipt.16 q3, q5, q7
    esp.vst.128.ip q2, a1, 16
    esp.vst.128.ip q4, a1, 16
    esp.vst.128.ip q6, a1, 16
    esp.vst.128.ip q3, a1, 16
    esp.vst.128.ip q5, a1, 16
    esp.vst.128.ip q7, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb5652rgb888_qint16_template rgb565le2rgb888_qint16
cvt_color_simd_helper_rgb5652rgb888_qint16_template rgb565be2rgb888_qint16
cvt_color_simd_helper_rgb5652rgb888_qint16_template rgb565le2bgr888_qint16
cvt_color_simd_helper_rgb5652rgb888_qint16_template rgb565be2bgr888_qint16
