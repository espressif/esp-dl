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
