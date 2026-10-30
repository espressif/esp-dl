#include "dl_esp32p4_color_common.S"

# q0/q1/q2 are H/S/V. a3/a4 are the LUTs. Planar RGB is q3-q5. q6/q7 are temps.
.macro cvt_color_simd_helper_yuyv2hsv_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src  (even pixel of a YUV pair)
    # a1: uint8_t *dst
    # a2: int n         (16-pixel blocks)
    # a3: int *sdiv_lut
    # a4: int *hdiv_lut

    blez a2, .Lend\@
    li t3, 14
.Lloop\@:
    cvt_\pix_cvt q0, q1, q2, a0, a3, a4, a5, t3, t4, t5, t6, q3, q4, q5, q6, q7
    esp.vzipt.8 q0, q1, q2
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q1, a1, 16
    esp.vst.128.ip q2, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_yuyv2hsv_template yuyv2hsv
cvt_color_simd_helper_yuyv2hsv_template uyvy2hsv

# q0 is the mask byte. a6 is copied to t3. q4-q7 are the YUV lanes, then H/S/V/scratch.
.macro cvt_color_simd_helper_yuyv2hsv_mask_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src  (even pixel of a YUV pair)
    # a1: uint8_t *dst
    # a2: int n         (16-pixel blocks)
    # a3: int *sdiv_lut
    # a4: int *hdiv_lut
    # a5: uint8_t *hsv_min
    # a6: uint8_t *hsv_max

    addi sp, sp, -16
    sw   s0, 0(sp)
    sw   s1, 4(sp)
    addi t3, a6, 0
    blez a2, .Lend\@
    li t5, 14
.Lloop\@:
    cvt_\pix_cvt q0, a0, a3, a4, a5, t3, t4, t5, t6, s0, s1, q1, q2, q3, q4, q5, q6, q7
    esp.vst.128.ip q0, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    addi sp, sp, 16
    ret
.endm

cvt_color_simd_helper_yuyv2hsv_mask_template yuyv2hsv_mask0
cvt_color_simd_helper_yuyv2hsv_mask_template uyvy2hsv_mask0
cvt_color_simd_helper_yuyv2hsv_mask_template yuyv2hsv_mask1
cvt_color_simd_helper_yuyv2hsv_mask_template uyvy2hsv_mask1
