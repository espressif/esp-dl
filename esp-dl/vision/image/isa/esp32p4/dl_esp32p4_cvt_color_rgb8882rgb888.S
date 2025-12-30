#include "dl_esp32p4_color_common.S"

.macro cvt_color_simd_helper_rgb8882rgb888_template pix_cvt
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
    esp.vld.128.ip q2, a0, 16
.ifc "\pix_cvt", "rgb8882rgb888"
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q2, a1, 16
.endif
.ifc "\pix_cvt", "rgb8882bgr888"
    esp.vunzipt.8 q0, q1, q2
    esp.vzipt.8 q2, q1, q0
    esp.vst.128.ip q2, a1, 16
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q0, a1, 16
.endif
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb8882rgb888_template rgb8882rgb888
cvt_color_simd_helper_rgb8882rgb888_template rgb8882bgr888

.macro cvt_color_simd_helper_rgb8882rgb888_qint8_template pix_cvt
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
    esp.vld.128.ip q2, a0, 16
.ifc "\pix_cvt", "rgb8882rgb888_qint8"
    esp.vunzipt.8 q0, q1, q2
    norm_quant_3chn_8 q0, q1, q2, a3, q3, q4, q5
    esp.vzipt.8 q0, q1, q2
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q2, a1, 16
.endif
.ifc "\pix_cvt", "rgb8882bgr888_qint8"
    esp.vunzipt.8 q0, q1, q2
    norm_quant_3chn_8 q2, q1, q0, a3, q3, q4, q5
    esp.vzipt.8 q2, q1, q0
    esp.vst.128.ip q2, a1, 16
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q0, a1, 16
.endif
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb8882rgb888_qint8_template rgb8882rgb888_qint8
cvt_color_simd_helper_rgb8882rgb888_qint8_template rgb8882bgr888_qint8

.macro cvt_color_simd_helper_rgb8882rgb888_qint16_template pix_cvt
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
    esp.vld.128.ip q2, a0, 16
.ifc "\pix_cvt", "rgb8882rgb888_qint16"
    esp.vunzipt.8 q0, q1, q2
    norm_quant_3chn_16 q3, q4, q5, q0, q1, q2, a3, q6, q7
    esp.vzipt.16 q3, q5, q1
    esp.vzipt.16 q4, q0, q2
    esp.vst.128.ip q3, a1, 16
    esp.vst.128.ip q5, a1, 16
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q4, a1, 16
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q2, a1, 16
.endif
.ifc "\pix_cvt", "rgb8882bgr888_qint16"
    esp.vunzipt.8 q0, q1, q2
    norm_quant_3chn_16 q3, q4, q5, q2, q1, q0, a3, q6, q7
    esp.vzipt.16 q3, q5, q1
    esp.vzipt.16 q4, q2, q0
    esp.vst.128.ip q3, a1, 16
    esp.vst.128.ip q5, a1, 16
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q4, a1, 16
    esp.vst.128.ip q2, a1, 16
    esp.vst.128.ip q0, a1, 16
.endif
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb8882rgb888_qint16_template rgb8882rgb888_qint16
cvt_color_simd_helper_rgb8882rgb888_qint16_template rgb8882bgr888_qint16
