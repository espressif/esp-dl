#include "dl_tie728_color_common.S"

.macro resize_nn_simd_helper_rgb5652rgb888_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n

    entry sp, 48
    blti a5, 1, .Lend\@
    extui a6, a4, 0, 4
    bnez a6, .Ldst_unaligned\@
    loop a5, .Lloop_end\@
        resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a7, a8, a9, a10
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, q0, q1, a7, a8
        vzipt_8 q2, q3, q4, q5
        ee.vst.128.ip q2, a4, 16
        ee.vst.128.ip q3, a4, 16
        ee.vst.128.ip q4, a4, 16
    .Lloop_end\@:
    retw
.Ldst_unaligned\@:
    vst_unaligned_init a4, a6, q7, q6
    loop a5, .Lloop_unaligned_end\@
        resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a7, a8, a9, a10
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, q0, q1, a7, a8
        vzipt_8 q2, q3, q4, q5
        vst_3chn_unaligned q2, q3, q4, a4, a6, q7, q6
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a6, q7, q6
.Lend\@:
    retw
.endm
resize_nn_simd_helper_rgb5652rgb888_template rgb565le2rgb888
resize_nn_simd_helper_rgb5652rgb888_template rgb565be2rgb888
resize_nn_simd_helper_rgb5652rgb888_template rgb565le2bgr888
resize_nn_simd_helper_rgb5652rgb888_template rgb565be2bgr888

.macro resize_nn_simd_helper_rgb5652rgb888_qint8_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n
    # a6: int *lut

    entry sp, 48
    blti a5, 1, .Lend\@
    extui a7, a4, 0, 4
    bnez a7, .Ldst_unaligned\@
    loop a5, .Lloop_end\@
        resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a8, a9, a10, a11
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, a6, q0, q1, a8, a9, q5
        vzipt_8 q2, q3, q4, q5
        ee.vst.128.ip q2, a4, 16
        ee.vst.128.ip q3, a4, 16
        ee.vst.128.ip q4, a4, 16
    .Lloop_end\@:
    retw
.Ldst_unaligned\@:
    vst_unaligned_init a4, a7, q7, q6
    loop a5, .Lloop_unaligned_end\@
        resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a8, a9, a10, a11
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, a6, q0, q1, a8, a9, q5
        vzipt_8 q2, q3, q4, q5
        vst_3chn_unaligned q2, q3, q4, a4, a7, q7, q6
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a7, q7, q6
.Lend\@:
    retw
.endm
resize_nn_simd_helper_rgb5652rgb888_qint8_template rgb565le2rgb888_qint8
resize_nn_simd_helper_rgb5652rgb888_qint8_template rgb565be2rgb888_qint8
resize_nn_simd_helper_rgb5652rgb888_qint8_template rgb565le2bgr888_qint8
resize_nn_simd_helper_rgb5652rgb888_qint8_template rgb565be2bgr888_qint8

.macro resize_nn_simd_helper_rgb5652rgb888_qint16_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n
    # a6: int *lut

    entry sp, 64
    blti a5, 1, .Lend\@
    extui a12, a4, 0, 4
    bnez a12, .Ldst_unaligned\@
    loop a5, .Lloop_end\@
        resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a8, a9, a10, a11
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, q5, q6, q7, a6, q0, q1, a8, a9
        vzipt_16 q2, q4, q6, q0
        vzipt_16 q3, q5, q7, q0
        ee.vst.128.ip q2, a4, 16
        ee.vst.128.ip q4, a4, 16
        ee.vst.128.ip q6, a4, 16
        ee.vst.128.ip q3, a4, 16
        ee.vst.128.ip q5, a4, 16
        ee.vst.128.ip q7, a4, 16
    .Lloop_end\@:
    retw
.Ldst_unaligned\@:
    addi a7, sp, 16
    vst_unaligned_init a4, a12, q0, q1, a7
    loop a5, .Lloop_unaligned_end\@
        resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a8, a9, a10, a11
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, q5, q6, q7, a6, q0, q1, a8, a9
        vzipt_16 q2, q4, q6, q0
        vzipt_16 q3, q5, q7, q0
        vst_6chn_unaligned q2, q4, q6, q3, q5, q7, a4, a12, q0, q1, a7
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a12, q0, q1, a7
.Lend\@:
    retw
.endm
resize_nn_simd_helper_rgb5652rgb888_qint16_template rgb565le2rgb888_qint16
resize_nn_simd_helper_rgb5652rgb888_qint16_template rgb565be2rgb888_qint16
resize_nn_simd_helper_rgb5652rgb888_qint16_template rgb565le2bgr888_qint16
resize_nn_simd_helper_rgb5652rgb888_qint16_template rgb565be2bgr888_qint16
