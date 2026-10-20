#include "dl_tie728_color_common.S"

.macro cvt_color_simd_helper_rgb5652hsv_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n
    # a5: int *sdiv_lut
    # a6: int *hdiv_lut

    entry sp, 192
    blti a4, 1, .Lend\@

    extui a7, a3, 0, 4
    bnez a7, .Ldst_unaligned\@

    extui a7, a2, 0, 4
    bnez a7, .Lsrc_unaligned_dst_aligned\@

.Lloop\@:
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a2, 16
    ee.vunzip.8 q0, q1
    \pix_cvt q3, q4, q5, a5, a6, sp, q0, q1, a8, a9, a10, a11, q2, q6, q7
    vzipt_8 q3, q4, q5, q6
    ee.vst.128.ip q3, a3, 16
    ee.vst.128.ip q4, a3, 16
    ee.vst.128.ip q5, a3, 16
    addi a4, a4, -1
    bnez a4, .Lloop\@
    retw

.Lsrc_unaligned_dst_aligned\@:
.Lloop_src_unaligned\@:
    vld_2chn_unaligned q0, q1, a2, q2
    ee.vunzip.8 q0, q1
    \pix_cvt q3, q4, q5, a5, a6, sp, q0, q1, a8, a9, a10, a11, q2, q6, q7
    vzipt_8 q3, q4, q5, q6
    ee.vst.128.ip q3, a3, 16
    ee.vst.128.ip q4, a3, 16
    ee.vst.128.ip q5, a3, 16
    addi a4, a4, -1
    bnez a4, .Lloop_src_unaligned\@
    retw

.Ldst_unaligned\@:
    addi a12, sp, 144
    vst_unaligned_init a3, a7, q6, q7, a12
.Lloop_unaligned\@:
    vld_2chn_unaligned q0, q1, a2, q2
    ee.vunzip.8 q0, q1
    \pix_cvt q3, q4, q5, a5, a6, sp, q0, q1, a8, a9, a10, a11, q2, q6, q7
    vzipt_8 q3, q4, q5, q6
    vst_3chn_unaligned q3, q4, q5, a3, a7, q6, q7, a12
    addi a4, a4, -1
    bnez a4, .Lloop_unaligned\@
    vst_unaligned_flush a3, a7, q6, q7, a12

.Lend\@:
    retw
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

    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n
    # a5: int *sdiv_lut
    # a6: int *hdiv_lut
    # a7: const uint8_t *hsv_min
    # [sp + 192]: const uint8_t *hsv_max (7th argument on stack)

    entry sp, 192
    blti a4, 1, .Lend\@
    l32i a8, sp, 192

    extui a9, a3, 0, 4
    bnez a9, .Ldst_unaligned\@

    extui a9, a2, 0, 4
    bnez a9, .Lsrc_unaligned_dst_aligned\@

.Lloop\@:
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a2, 16
    ee.vunzip.8 q0, q1
    \pix_cvt q2, a5, a6, a7, a8, sp, q0, q1, a10, a11, a12, a13, q3, q4, q5, q6, q7
    ee.vst.128.ip q2, a3, 16
    addi a4, a4, -1
    bnez a4, .Lloop\@
    retw

.Lsrc_unaligned_dst_aligned\@:
.Lloop_src_unaligned\@:
    vld_2chn_unaligned q0, q1, a2, q2
    ee.vunzip.8 q0, q1
    \pix_cvt q2, a5, a6, a7, a8, sp, q0, q1, a10, a11, a12, a13, q3, q4, q5, q6, q7
    ee.vst.128.ip q2, a3, 16
    addi a4, a4, -1
    bnez a4, .Lloop_src_unaligned\@
    retw

.Ldst_unaligned\@:
    addi a14, sp, 144
    vst_unaligned_init a3, a9, q6, q7, a14
.Lloop_unaligned\@:
    vld_2chn_unaligned q0, q1, a2, q2
    ee.vunzip.8 q0, q1
    \pix_cvt q2, a5, a6, a7, a8, sp, q0, q1, a10, a11, a12, a13, q3, q4, q5, q6, q7
    vst_1chn_unaligned q2, a3, a9, q6, q7, a14
    addi a4, a4, -1
    bnez a4, .Lloop_unaligned\@
    vst_unaligned_flush a3, a9, q6, q7, a14

.Lend\@:
    retw
.endm

cvt_color_simd_helper_rgb5652hsv_mask_template rgb565le2hsv_mask0
cvt_color_simd_helper_rgb5652hsv_mask_template rgb565be2hsv_mask0
cvt_color_simd_helper_rgb5652hsv_mask_template bgr565le2hsv_mask0
cvt_color_simd_helper_rgb5652hsv_mask_template bgr565be2hsv_mask0
cvt_color_simd_helper_rgb5652hsv_mask_template rgb565le2hsv_mask1
cvt_color_simd_helper_rgb5652hsv_mask_template rgb565be2hsv_mask1
cvt_color_simd_helper_rgb5652hsv_mask_template bgr565le2hsv_mask1
cvt_color_simd_helper_rgb5652hsv_mask_template bgr565be2hsv_mask1
