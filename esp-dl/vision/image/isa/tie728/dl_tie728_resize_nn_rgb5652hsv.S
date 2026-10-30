#include "dl_tie728_color_common.S"

.macro resize_nn_simd_helper_rgb5652hsv_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n
    # a6: int *sdiv_lut
    # a7: int *hdiv_lut

    entry sp, 192
    blti a5, 1, .Lend\@

    extui a8, a4, 0, 4
    bnez a8, .Ldst_unaligned\@

.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a9, a10, a11, a12
    ee.vunzip.8 q0, q1
    \pix_cvt q3, q4, q5, a6, a7, sp, q0, q1, a9, a10, a11, a12, q2, q6, q7
    vzipt_8 q3, q4, q5, q6
    ee.vst.128.ip q3, a4, 16
    ee.vst.128.ip q4, a4, 16
    ee.vst.128.ip q5, a4, 16
    addi a5, a5, -1
    bnez a5, .Lloop\@
    retw

.Ldst_unaligned\@:
    addi a13, sp, 144
    vst_unaligned_init a4, a8, q6, q7, a13
.Lloop_unaligned\@:
    resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a9, a10, a11, a12
    ee.vunzip.8 q0, q1
    \pix_cvt q3, q4, q5, a6, a7, sp, q0, q1, a9, a10, a11, a12, q2, q6, q7
    vzipt_8 q3, q4, q5, q6
    vst_3chn_unaligned q3, q4, q5, a4, a8, q6, q7, a13
    addi a5, a5, -1
    bnez a5, .Lloop_unaligned\@
    vst_unaligned_flush a4, a8, q6, q7, a13

.Lend\@:
    retw
.endm

resize_nn_simd_helper_rgb5652hsv_template rgb565le2hsv
resize_nn_simd_helper_rgb5652hsv_template rgb565be2hsv
resize_nn_simd_helper_rgb5652hsv_template bgr565le2hsv
resize_nn_simd_helper_rgb5652hsv_template bgr565be2hsv

.macro resize_nn_simd_helper_rgb5652hsv_mask_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n
    # a6: int *sdiv_lut
    # a7: int *hdiv_lut
    # [sp + 192]: const uint8_t *hsv_min (7th argument)
    # [sp + 196]: const uint8_t *hsv_max (8th argument)

    entry sp, 192
    blti a5, 1, .Lend\@
    l32i a8, sp, 192
    l32i a9, sp, 196

    extui a10, a4, 0, 4
    bnez a10, .Ldst_unaligned\@

.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a11, a12, a13, a14
    ee.vunzip.8 q0, q1
    \pix_cvt q2, a6, a7, a8, a9, sp, q0, q1, a11, a12, a13, a14, q3, q4, q5, q6, q7
    ee.vst.128.ip q2, a4, 16
    addi a5, a5, -1
    bnez a5, .Lloop\@
    retw

.Ldst_unaligned\@:
    addi a15, sp, 144
    vst_unaligned_init a4, a10, q6, q7, a15
.Lloop_unaligned\@:
    resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a11, a12, a13, a14
    ee.vunzip.8 q0, q1
    \pix_cvt q2, a6, a7, a8, a9, sp, q0, q1, a11, a12, a13, a14, q3, q4, q5, q6, q7
    vst_1chn_unaligned q2, a4, a10, q6, q7, a15
    addi a5, a5, -1
    bnez a5, .Lloop_unaligned\@
    vst_unaligned_flush a4, a10, q6, q7, a15

.Lend\@:
    retw
.endm

resize_nn_simd_helper_rgb5652hsv_mask_template rgb565le2hsv_mask0
resize_nn_simd_helper_rgb5652hsv_mask_template rgb565be2hsv_mask0
resize_nn_simd_helper_rgb5652hsv_mask_template bgr565le2hsv_mask0
resize_nn_simd_helper_rgb5652hsv_mask_template bgr565be2hsv_mask0
resize_nn_simd_helper_rgb5652hsv_mask_template rgb565le2hsv_mask1
resize_nn_simd_helper_rgb5652hsv_mask_template rgb565be2hsv_mask1
resize_nn_simd_helper_rgb5652hsv_mask_template bgr565le2hsv_mask1
resize_nn_simd_helper_rgb5652hsv_mask_template bgr565be2hsv_mask1
