#include "dl_esp32p4_color_common.S"

# Scalar YUYV2Gray keeps the Y byte. YUYV stores even bytes, UYVY stores odd bytes.

.macro cvt_color_simd_helper_yuyv2gray_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n         (16-pixel blocks)

    blez a2, .Lend\@
.Lloop\@:
    esp.vld.128.ip q0, a0, 16
    esp.vld.128.ip q1, a0, 16
    esp.vunzip.8 q0, q1
.ifc "\pix_cvt", "yuyv2gray"
    esp.vst.128.ip q0, a1, 16
.endif
.ifc "\pix_cvt", "uyvy2gray"
    esp.vst.128.ip q1, a1, 16
.endif
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_yuyv2gray_template yuyv2gray
cvt_color_simd_helper_yuyv2gray_template uyvy2gray

.macro cvt_color_simd_helper_yuyv2gray_qint8_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n         (16-pixel blocks)
    # a3: int *lut

    blez a2, .Lend\@
.Lloop\@:
    esp.vld.128.ip q0, a0, 16
    esp.vld.128.ip q1, a0, 16
    esp.vunzip.8 q0, q1
.ifc "\pix_cvt", "yuyv2gray_qint8"
    norm_quant_1chn_8 q0, a3, q1, q2, q3
    esp.vst.128.ip q0, a1, 16
.endif
.ifc "\pix_cvt", "uyvy2gray_qint8"
    norm_quant_1chn_8 q1, a3, q0, q2, q3
    esp.vst.128.ip q1, a1, 16
.endif
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_yuyv2gray_qint8_template yuyv2gray_qint8
cvt_color_simd_helper_yuyv2gray_qint8_template uyvy2gray_qint8

.macro cvt_color_simd_helper_yuyv2gray_qint16_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n         (16-pixel blocks)
    # a3: int *lut

    blez a2, .Lend\@
.Lloop\@:
    esp.vld.128.ip q0, a0, 16
    esp.vld.128.ip q1, a0, 16
    esp.vunzip.8 q0, q1
.ifc "\pix_cvt", "yuyv2gray_qint16"
    norm_quant_1chn_16 q1, q0, a3, q2, q3
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q0, a1, 16
.endif
.ifc "\pix_cvt", "uyvy2gray_qint16"
    norm_quant_1chn_16 q0, q1, a3, q2, q3
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q1, a1, 16
.endif
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_yuyv2gray_qint16_template yuyv2gray_qint16
cvt_color_simd_helper_yuyv2gray_qint16_template uyvy2gray_qint16
