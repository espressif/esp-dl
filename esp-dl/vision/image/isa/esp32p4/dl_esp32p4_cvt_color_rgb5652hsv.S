#include "dl_esp32p4_color_common.S"

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

.macro cvt_color_simd_helper_rgb5652hsv_mask_template pix_cvt
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
    # a5: uint8_t *hsv_min
    # a6: uint8_t *hsv_max

    addi sp, sp, -16
    sw   s0, 0(sp)
    addi s0, a6, 0

    blez a2, .Lend\@
.Lloop\@:
    esp.vld.128.ip q0, a0, 16
    esp.vld.128.ip q1, a0, 16
    esp.vunzip.8 q0, q1
    \pix_cvt q2, a3, a4, a5, s0, q0, q1, t3, t4, t5, t6, q3, q4, q5, q6, q7
    esp.vst.128.ip q2, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    addi sp, sp, 16
    ret
.endm

cvt_color_simd_helper_rgb5652hsv_mask_template rgb565le2hsv_mask0
cvt_color_simd_helper_rgb5652hsv_mask_template rgb565be2hsv_mask0
cvt_color_simd_helper_rgb5652hsv_mask_template bgr565le2hsv_mask0
cvt_color_simd_helper_rgb5652hsv_mask_template bgr565be2hsv_mask0
cvt_color_simd_helper_rgb5652hsv_mask_template rgb565le2hsv_mask1
cvt_color_simd_helper_rgb5652hsv_mask_template rgb565be2hsv_mask1
cvt_color_simd_helper_rgb5652hsv_mask_template bgr565le2hsv_mask1
cvt_color_simd_helper_rgb5652hsv_mask_template bgr565be2hsv_mask1
