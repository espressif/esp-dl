#include "dl_tie728_color_common.S"

.macro resize_nn_simd_helper_rgb8882gray_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n

    entry sp, 64
    blti a5, 1, .Lend\@

    extui a6, a4, 0, 4
    bnez a6, .Ldst_unaligned\@

    loop a5, .Lloop_end\@
        resize_nn_load_3chn_128 q0, q1, q2, a2, a3, sp, q3, q4, q5, a7, a8, a9, a10
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q0, q1, q2, a7, q4, q5, q6, q7
        ee.vst.128.ip q3, a4, 16
    .Lloop_end\@:
    retw

.Ldst_unaligned\@:
    addi a7, sp, 16
    vst_unaligned_init a4, a6, q7, q6, a7
    loop a5, .Lloop_unaligned_end\@
        resize_nn_load_3chn_128 q0, q1, q2, a2, a3, sp, q3, q4, q5, a8, a9, a10, a11
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q0, q1, q2, a8, q4, q5, q6, q7
        vst_1chn_unaligned q3, a4, a6, q7, q6, a7
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a6, q7, q6, a7

.Lend\@:
    retw
.endm
resize_nn_simd_helper_rgb8882gray_template rgb8882gray
resize_nn_simd_helper_rgb8882gray_template bgr8882gray

.macro resize_nn_simd_helper_rgb8882gray_qint8_template pix_cvt
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
        resize_nn_load_3chn_128 q0, q1, q2, a2, a3, sp, q3, q4, q5, a8, a9, a10, a11
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, a6, q0, q1, q2, a8, q4, q5, q6, q7
        ee.vst.128.ip q3, a4, 16
    .Lloop_end\@:
    retw

.Ldst_unaligned\@:
    addi a7, sp, 16
    vst_unaligned_init a4, a12, q7, q6, a7
    loop a5, .Lloop_unaligned_end\@
        resize_nn_load_3chn_128 q0, q1, q2, a2, a3, sp, q3, q4, q5, a8, a9, a10, a11
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, a6, q0, q1, q2, a8, q4, q5, q6, q7
        vst_1chn_unaligned q3, a4, a12, q7, q6, a7
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a12, q7, q6, a7

.Lend\@:
    retw
.endm
resize_nn_simd_helper_rgb8882gray_qint8_template rgb8882gray_qint8
resize_nn_simd_helper_rgb8882gray_qint8_template bgr8882gray_qint8

.macro resize_nn_simd_helper_rgb8882gray_qint16_template pix_cvt
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
        resize_nn_load_3chn_128 q0, q1, q2, a2, a3, sp, q3, q4, q5, a8, a9, a10, a11
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q4, a6, q0, q1, q2, a8, q5, q6, q7
        ee.vst.128.ip q3, a4, 16
        ee.vst.128.ip q4, a4, 16
    .Lloop_end\@:
    retw

.Ldst_unaligned\@:
    addi a7, sp, 16
    vst_unaligned_init a4, a12, q7, q6, a7
    loop a5, .Lloop_unaligned_end\@
        resize_nn_load_3chn_128 q0, q1, q2, a2, a3, sp, q3, q4, q5, a8, a9, a10, a11
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q4, a6, q0, q1, q2, a8, q5, q6, q7
        vst_2chn_unaligned q3, q4, a4, a12, q7, q6, a7
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a12, q7, q6, a7

.Lend\@:
    retw
.endm
resize_nn_simd_helper_rgb8882gray_qint16_template rgb8882gray_qint16
resize_nn_simd_helper_rgb8882gray_qint16_template bgr8882gray_qint16
