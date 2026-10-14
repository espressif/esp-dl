#include "dl_esp32p4_color_common.S"

# q0/q1 are the 565 halves. q2-q4 and q1 are the YUV lanes. q5-q7 are planar RGB.
.macro cvt_color_simd_helper_yuyv2rgb565_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a0: uint8_t *src  (even pixel of a YUV pair)
    # a1: uint8_t *dst
    # a2: int n         (16-pixel blocks)

    blez a2, .Lend\@
    li a4, 14
.Lloop\@:
    cvt_\pix_cvt q0, q1, a0, q2, q3, q4, q5, q6, q7, a3, a4, a5
    esp.vzip.8 q0, q1
    esp.vst.128.ip q0, a1, 16
    esp.vst.128.ip q1, a1, 16
    addi a2, a2, -1
    bnez a2, .Lloop\@
.Lend\@:
    ret
.endm

cvt_color_simd_helper_yuyv2rgb565_template yuyv2rgb565le
cvt_color_simd_helper_yuyv2rgb565_template yuyv2rgb565be
cvt_color_simd_helper_yuyv2rgb565_template yuyv2bgr565le
cvt_color_simd_helper_yuyv2rgb565_template yuyv2bgr565be
cvt_color_simd_helper_yuyv2rgb565_template uyvy2rgb565le
cvt_color_simd_helper_yuyv2rgb565_template uyvy2rgb565be
cvt_color_simd_helper_yuyv2rgb565_template uyvy2bgr565le
cvt_color_simd_helper_yuyv2rgb565_template uyvy2bgr565be
