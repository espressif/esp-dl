#include "dl_esp32p4_color_common.S"

.macro cvt_color_simd_helper_rgb5652rgb565_template pix_cvt
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
.ifc "\pix_cvt", "rgb5652rgb565"
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q1, a1, 16
.else
.ifc "\pix_cvt", "rgb565le2rgb565be"
    esp.vunzip.8 q0, q1
    esp.vzip.8 q1, q0
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q0, a1, 16
.else
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q3, q0, q1, a3, a4
    esp.vzip.8 q2, q3
    esp.vst.128.ip q2, a1, 16
    esp.vst.128.ip q3, a1, 16
.endif
.endif
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb5652rgb565_template rgb5652rgb565
cvt_color_simd_helper_rgb5652rgb565_template rgb565le2rgb565be
cvt_color_simd_helper_rgb5652rgb565_template rgb565le2bgr565le
cvt_color_simd_helper_rgb5652rgb565_template rgb565be2bgr565be
cvt_color_simd_helper_rgb5652rgb565_template rgb565le2bgr565be
cvt_color_simd_helper_rgb5652rgb565_template rgb565be2bgr565le

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

.macro cvt_color_simd_helper_rgb8882gray_template pix_cvt
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
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, q0, q1, q2, a3, q4, q5, q6, q7
    esp.vst.128.ip q3, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm
cvt_color_simd_helper_rgb8882gray_template rgb8882gray
cvt_color_simd_helper_rgb8882gray_template bgr8882gray

.macro cvt_color_simd_helper_rgb8882gray_qint8_template pix_cvt
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
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, a3, q0, q1, q2, a4, q4, q5, q6, q7
    esp.vst.128.ip q3, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm
cvt_color_simd_helper_rgb8882gray_qint8_template rgb8882gray_qint8
cvt_color_simd_helper_rgb8882gray_qint8_template bgr8882gray_qint8

.macro cvt_color_simd_helper_rgb8882gray_qint16_template pix_cvt
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
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, q4, a3, q0, q1, q2, a4, q5, q6, q7
    esp.vst.128.ip q3, a1, 16
    esp.vst.128.ip q4, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm
cvt_color_simd_helper_rgb8882gray_qint16_template rgb8882gray_qint16
cvt_color_simd_helper_rgb8882gray_qint16_template bgr8882gray_qint16

.macro cvt_color_simd_helper_rgb8882rgb565_template pix_cvt
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
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, q4, q0, q1, q2, a3, a4
    esp.vzip.8 q3, q4
    esp.vst.128.ip q3, a1, 16
    esp.vst.128.ip q4, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm
cvt_color_simd_helper_rgb8882rgb565_template rgb8882rgb565le
cvt_color_simd_helper_rgb8882rgb565_template bgr8882rgb565le
cvt_color_simd_helper_rgb8882rgb565_template rgb8882rgb565be
cvt_color_simd_helper_rgb8882rgb565_template bgr8882rgb565be

    .balign 4
    .text
    .global cvt_color_simd_helper_gray2gray
    .type   cvt_color_simd_helper_gray2gray, @function
cvt_color_simd_helper_gray2gray:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n

    blez a2, 1f
0:
    esp.vld.128.ip q0, a0, 16
    esp.vst.128.ip q0, a1, 16
    addi a2, a2, -1
    bnez a2, 0b
1:
    ret

    .balign 4
    .text
    .global cvt_color_simd_helper_gray2gray_qint8
    .type   cvt_color_simd_helper_gray2gray_qint8, @function
cvt_color_simd_helper_gray2gray_qint8:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n
    # a3: int *lut

    blez a2, 1f
0:
    esp.vld.128.ip q0, a0, 16
    norm_quant_1chn_8 q0, a3, q1, q2, q3
    esp.vst.128.ip q0, a1, 16
    addi a2, a2, -1
    bnez a2, 0b
1:
    ret

    .balign 4
    .text
    .global cvt_color_simd_helper_gray2gray_qint16
    .type   cvt_color_simd_helper_gray2gray_qint16, @function
cvt_color_simd_helper_gray2gray_qint16:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n
    # a3: int *lut

    blez a2, 1f
0:
    esp.vld.128.ip q1, a0, 16
    norm_quant_1chn_16 q0, q1, a3, q2, q3
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q1, a1, 16
    addi a2, a2, -1
    bnez a2, 0b
1:
    ret

.macro cvt_color_simd_helper_rgb8882hsv_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n
    # a3: int *sdiv_lut
    # a4: int *hdiv_lut

    blez a2, .Lend\@
.Lloop\@:
    esp.vld.128.ip q0, a0, 16
    esp.vld.128.ip q1, a0, 16
    esp.vld.128.ip q2, a0, 16
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, q4, q5, a3, a4, q0, q1, q2, a5, t3, t4, t5, q6, q7
    esp.vzipt.8 q3, q4, q5
    esp.vst.128.ip q3, a1, 16
    esp.vst.128.ip q4, a1, 16
    esp.vst.128.ip q5, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb8882hsv_template rgb8882hsv
cvt_color_simd_helper_rgb8882hsv_template bgr8882hsv

.macro cvt_color_simd_helper_rgb5652hsv_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n
    # a3: int *sdiv_lut
    # a4: int *hdiv_lut

    blez a2, .Lend\@
.Lloop\@:
    esp.vld.128.ip q0, a0, 16
    esp.vld.128.ip q1, a0, 16
    esp.vunzip.8 q0, q1
    \pix_cvt q3, q4, q5, a3, a4, q0, q1, a5, t3, t4, t5, q2, q6, q7
    esp.vzipt.8 q3, q4, q5
    esp.vst.128.ip q3, a1, 16
    esp.vst.128.ip q4, a1, 16
    esp.vst.128.ip q5, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_rgb5652hsv_template rgb565le2hsv
cvt_color_simd_helper_rgb5652hsv_template rgb565be2hsv
cvt_color_simd_helper_rgb5652hsv_template bgr565le2hsv
cvt_color_simd_helper_rgb5652hsv_template bgr565be2hsv
