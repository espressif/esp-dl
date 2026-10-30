#include "dl_tie728_color_common.S"

# q0/q1 are the 565 halves. Temps are a6..a12, then q2-q7.
.macro resize_nn_simd_helper_yuyv2rgb565_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:
    entry sp, 96
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n          (16-pixel blocks)

    blti a5, 1, .Lend\@

    extui a12, a4, 0, 4
    bnez a12, .Ldst_unaligned\@

.Lloop\@:
    resize_nn_\pix_cvt q0, q1, a2, a3, sp, a6, a7, a8, a9, a10, a11, q2, q3, q4, q5, q6, q7
    ee.vzip.8 q0, q1
    ee.vst.128.ip q0, a4, 16
    ee.vst.128.ip q1, a4, 16
    addi a5, a5, -1
    bnez a5, .Lloop\@
    retw.n

.Ldst_unaligned\@:
    addi a11, sp, 48
    vst_unaligned_init a4, a12, q6, q7, a11
.Lloop_unaligned\@:
    resize_nn_\pix_cvt q0, q1, a2, a3, sp, a6, a7, a8, a9, a10, a11, q2, q3, q4, q5, q6, q7
    ee.vzip.8 q0, q1
    addi a11, sp, 48
    vst_2chn_unaligned q0, q1, a4, a12, q6, q7, a11
    addi a5, a5, -1
    bnez a5, .Lloop_unaligned\@
    addi a11, sp, 48
    vst_unaligned_flush a4, a12, q6, q7, a11

.Lend\@:
    retw.n
.endm

resize_nn_simd_helper_yuyv2rgb565_template yuyv2rgb565le
resize_nn_simd_helper_yuyv2rgb565_template yuyv2rgb565be
resize_nn_simd_helper_yuyv2rgb565_template yuyv2bgr565le
resize_nn_simd_helper_yuyv2rgb565_template yuyv2bgr565be
resize_nn_simd_helper_yuyv2rgb565_template uyvy2rgb565le
resize_nn_simd_helper_yuyv2rgb565_template uyvy2rgb565be
resize_nn_simd_helper_yuyv2rgb565_template uyvy2bgr565le
resize_nn_simd_helper_yuyv2rgb565_template uyvy2bgr565be
