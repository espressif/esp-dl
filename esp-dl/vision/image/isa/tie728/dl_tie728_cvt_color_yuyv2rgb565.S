#include "dl_tie728_color_common.S"

# q0/q1 are the 565 halves. q2-q4 and q1 are the YUV lanes. q5-q7 are planar RGB.
.macro cvt_color_simd_helper_yuyv2rgb565_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:
    entry sp, 48
    # a2: uint8_t *src  (even pixel of a YUV pair)
    # a3: uint8_t *dst
    # a4: int n         (16-pixel blocks)

    blti a4, 1, .Lend\@

    extui a7, a3, 0, 4
    bnez a7, .Ldst_unaligned\@

.Lloop\@:
    movi a6, 14
    cvt_\pix_cvt q0, q1, a2, q2, q3, q4, q5, q6, q7, a5, a6
    ee.vzip.8 q0, q1
    ee.vst.128.ip q0, a3, 16
    ee.vst.128.ip q1, a3, 16
    addi a4, a4, -1
    bnez a4, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a3, a7, q6, q7, sp
.Lloop_unaligned\@:
    movi a6, 14
    cvt_\pix_cvt q0, q1, a2, q2, q3, q4, q5, q6, q7, a5, a6
    ee.vzip.8 q0, q1
    vst_2chn_unaligned q0, q1, a3, a7, q6, q7, sp
    addi a4, a4, -1
    bnez a4, .Lloop_unaligned\@
    vst_unaligned_flush a3, a7, q6, q7, sp

.Lend\@:
    retw.n
.endm

cvt_color_simd_helper_yuyv2rgb565_template yuyv2rgb565le
cvt_color_simd_helper_yuyv2rgb565_template yuyv2rgb565be
cvt_color_simd_helper_yuyv2rgb565_template yuyv2bgr565le
cvt_color_simd_helper_yuyv2rgb565_template yuyv2bgr565be
cvt_color_simd_helper_yuyv2rgb565_template uyvy2rgb565le
cvt_color_simd_helper_yuyv2rgb565_template uyvy2rgb565be
cvt_color_simd_helper_yuyv2rgb565_template uyvy2bgr565le
cvt_color_simd_helper_yuyv2rgb565_template uyvy2bgr565be
