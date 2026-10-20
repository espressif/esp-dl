#include "dl_tie728_color_common.S"

# q0/q1/q2 are H/S/V. a6/a7 are the LUTs.
.macro resize_nn_simd_helper_yuyv2hsv_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:
    entry sp, 192
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n          (16-pixel blocks)
    # a6: int *sdiv_lut
    # a7: int *hdiv_lut

    blti a5, 1, .Lend\@

    extui a8, a4, 0, 4
    bnez a8, .Ldst_unaligned\@

.Lloop\@:
    resize_nn_\pix_cvt q0, q1, q2, a2, a3, sp, a8, a9, a10, a11, a12, a13, a6, a7, q3, q4, q5, q6, q7
    vzipt_8 q0, q1, q2, q3
    ee.vst.128.ip q0, a4, 16
    ee.vst.128.ip q1, a4, 16
    ee.vst.128.ip q2, a4, 16
    addi a5, a5, -1
    bnez a5, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    addi a9, sp, 144
    vst_unaligned_init a4, a8, q6, q7, a9
.Lloop_unaligned\@:
    s32i a8, sp, 160
    resize_nn_\pix_cvt q0, q1, q2, a2, a3, sp, a8, a9, a10, a11, a12, a13, a6, a7, q3, q4, q5, q6, q7
    vzipt_8 q0, q1, q2, q3
    l32i a8, sp, 160
    addi a9, sp, 144
    vst_3chn_unaligned q0, q1, q2, a4, a8, q6, q7, a9
    addi a5, a5, -1
    bnez a5, .Lloop_unaligned\@
    addi a9, sp, 144
    vst_unaligned_flush a4, a8, q6, q7, a9

.Lend\@:
    retw.n
.endm

resize_nn_simd_helper_yuyv2hsv_template yuyv2hsv
resize_nn_simd_helper_yuyv2hsv_template uyvy2hsv

# q0 is the mask byte.
.macro resize_nn_simd_helper_yuyv2hsv_mask_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:
    entry sp, 192
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n          (16-pixel blocks)
    # a6: int *sdiv_lut
    # a7: int *hdiv_lut
    # [sp + 192]: const uint8_t *hsv_min (7th argument)
    # [sp + 196]: const uint8_t *hsv_max (8th argument)

    blti a5, 1, .Lend\@

    extui a8, a4, 0, 4
    bnez a8, .Ldst_unaligned\@

.Lloop\@:
    resize_nn_\pix_cvt q0, a2, a3, sp, a8, a9, a10, a11, a12, a13, a6, a7, q1, q2, q3, q4, q5, q6, q7
    ee.vst.128.ip q0, a4, 16
    addi a5, a5, -1
    bnez a5, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    addi a9, sp, 144
    vst_unaligned_init a4, a8, q6, q7, a9
.Lloop_unaligned\@:
    s32i a8, sp, 160
    resize_nn_\pix_cvt q0, a2, a3, sp, a8, a9, a10, a11, a12, a13, a6, a7, q1, q2, q3, q4, q5, q6, q7
    l32i a8, sp, 160
    addi a9, sp, 144
    vst_1chn_unaligned q0, a4, a8, q6, q7, a9
    addi a5, a5, -1
    bnez a5, .Lloop_unaligned\@
    addi a9, sp, 144
    vst_unaligned_flush a4, a8, q6, q7, a9

.Lend\@:
    retw.n
.endm

resize_nn_simd_helper_yuyv2hsv_mask_template yuyv2hsv_mask0
resize_nn_simd_helper_yuyv2hsv_mask_template uyvy2hsv_mask0
resize_nn_simd_helper_yuyv2hsv_mask_template yuyv2hsv_mask1
resize_nn_simd_helper_yuyv2hsv_mask_template uyvy2hsv_mask1
