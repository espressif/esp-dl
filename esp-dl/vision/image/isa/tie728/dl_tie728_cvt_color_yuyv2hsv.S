#include "dl_tie728_color_common.S"

# q0/q1/q2 are H/S/V. a5/a6 are the LUTs. Planar RGB is q3-q5. q6/q7 are temps.
.macro cvt_color_simd_helper_yuyv2hsv_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:
    entry sp, 192
    # a2: uint8_t *src  (even pixel of a YUV pair)
    # a3: uint8_t *dst
    # a4: int n         (16-pixel blocks)
    # a5: int *sdiv_lut
    # a6: int *hdiv_lut

    blti a4, 1, .Lend\@

    extui a7, a3, 0, 4
    bnez a7, .Ldst_unaligned\@

.Lloop\@:
    movi a9, 14
    cvt_\pix_cvt q0, q1, q2, a2, a5, a6, sp, a8, a9, a10, a11, q3, q4, q5, q6, q7
    vzipt_8 q0, q1, q2, q3
    ee.vst.128.ip q0, a3, 16
    ee.vst.128.ip q1, a3, 16
    ee.vst.128.ip q2, a3, 16
    addi a4, a4, -1
    bnez a4, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    addi a12, sp, 144
    vst_unaligned_init a3, a7, q6, q7, a12
.Lloop_unaligned\@:
    movi a9, 14
    cvt_\pix_cvt q0, q1, q2, a2, a5, a6, sp, a8, a9, a10, a11, q3, q4, q5, q6, q7
    vzipt_8 q0, q1, q2, q3
    vst_3chn_unaligned q0, q1, q2, a3, a7, q6, q7, a12
    addi a4, a4, -1
    bnez a4, .Lloop_unaligned\@
    vst_unaligned_flush a3, a7, q6, q7, a12

.Lend\@:
    retw.n
.endm

cvt_color_simd_helper_yuyv2hsv_template yuyv2hsv
cvt_color_simd_helper_yuyv2hsv_template uyvy2hsv

# q0 is the mask byte. q4-q7 are the YUV lanes, then H/S/V/scratch.
.macro cvt_color_simd_helper_yuyv2hsv_mask_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:
    entry sp, 192
    # a2: uint8_t *src  (even pixel of a YUV pair)
    # a3: uint8_t *dst
    # a4: int n         (16-pixel blocks)
    # a5: int *sdiv_lut
    # a6: int *hdiv_lut
    # a7: const uint8_t *hsv_min
    # [sp + 192]: const uint8_t *hsv_max (7th argument on stack)

    blti a4, 1, .Lend\@
    l32i a8, sp, 192

    extui a9, a3, 0, 4
    bnez a9, .Ldst_unaligned\@

.Lloop\@:
    movi a11, 14
    cvt_\pix_cvt q0, a2, a5, a6, a7, a8, sp, a10, a11, a12, a13, q1, q2, q3, q4, q5, q6, q7
    ee.vst.128.ip q0, a3, 16
    addi a4, a4, -1
    bnez a4, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    addi a10, sp, 144
    vst_unaligned_init a3, a9, q6, q7, a10
.Lloop_unaligned\@:
    movi a11, 14
    cvt_\pix_cvt q0, a2, a5, a6, a7, a8, sp, a10, a11, a12, a13, q1, q2, q3, q4, q5, q6, q7
    addi a10, sp, 144
    vst_1chn_unaligned q0, a3, a9, q6, q7, a10
    addi a4, a4, -1
    bnez a4, .Lloop_unaligned\@
    addi a10, sp, 144
    vst_unaligned_flush a3, a9, q6, q7, a10

.Lend\@:
    retw.n
.endm

cvt_color_simd_helper_yuyv2hsv_mask_template yuyv2hsv_mask0
cvt_color_simd_helper_yuyv2hsv_mask_template uyvy2hsv_mask0
cvt_color_simd_helper_yuyv2hsv_mask_template yuyv2hsv_mask1
cvt_color_simd_helper_yuyv2hsv_mask_template uyvy2hsv_mask1
