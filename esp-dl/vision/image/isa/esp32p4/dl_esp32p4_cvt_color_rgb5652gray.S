#include "dl_esp32p4_color_common.S"

.macro cvt_color_simd_helper_rgb5652gray_template pix_cvt
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
    \pix_cvt q2, q0, q1, a4, a5, q3, q4, q5, q6, q7
    esp.vst.128.ip q2, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb5652gray_template rgb565le2gray
cvt_color_simd_helper_rgb5652gray_template rgb565be2gray
cvt_color_simd_helper_rgb5652gray_template bgr565le2gray
cvt_color_simd_helper_rgb5652gray_template bgr565be2gray

.macro cvt_color_simd_helper_rgb5652gray_qint8_template pix_cvt
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
    \pix_cvt q2, a3, q0, q1, a4, a5, q3, q4, q5, q6, q7
    esp.vst.128.ip q2, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb5652gray_qint8_template rgb565le2gray_qint8
cvt_color_simd_helper_rgb5652gray_qint8_template rgb565be2gray_qint8
cvt_color_simd_helper_rgb5652gray_qint8_template bgr565le2gray_qint8
cvt_color_simd_helper_rgb5652gray_qint8_template bgr565be2gray_qint8

.macro cvt_color_simd_helper_rgb5652gray_qint16_template pix_cvt
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
    \pix_cvt q2, q3, a3, q0, q1, a4, a5, q4, q5, q6, q7
    esp.vst.128.ip q2, a1, 16
    esp.vst.128.ip q3, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb5652gray_qint16_template rgb565le2gray_qint16
cvt_color_simd_helper_rgb5652gray_qint16_template rgb565be2gray_qint16
cvt_color_simd_helper_rgb5652gray_qint16_template bgr565le2gray_qint16
cvt_color_simd_helper_rgb5652gray_qint16_template bgr565be2gray_qint16
