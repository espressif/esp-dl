#include "dl_esp32p4_color_common.S"

.macro cvt_color_simd_helper_hsv2hsv_mask_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src
    # a1: uint8_t *dst
    # a2: int n
    # a3: uint8_t *hsv_min
    # a4: uint8_t *hsv_max

    blez a2, .Lend\@
.Lloop\@:
    esp.vld.128.ip q0, a0, 16
    esp.vld.128.ip q1, a0, 16
    esp.vld.128.ip q2, a0, 16
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, a3, a4, q0, q1, q2, q4
    esp.vst.128.ip q3, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_hsv2hsv_mask_template hsv2hsv_mask0
cvt_color_simd_helper_hsv2hsv_mask_template hsv2hsv_mask1
