#include "dl_esp32p4_color_common.S"

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
