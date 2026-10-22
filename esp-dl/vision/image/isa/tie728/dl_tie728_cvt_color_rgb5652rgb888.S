#include "dl_tie728_color_common.S"

.macro cvt_color_simd_helper_rgb5652rgb888_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n

    entry sp, 32
    blti a4, 1, .Lend\@
    extui a5, a3, 0, 4
    bnez a5, .Ldst_unaligned\@
    extui a5, a2, 0, 4
    bnez a5, .Lsrc_unaligned\@
    loop a4, .Lloop_end\@
        ee.vld.128.ip q0, a2, 16
        ee.vld.128.ip q1, a2, 16
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, q0, q1, a6, a7
        vzipt_8 q2, q3, q4, q5
        ee.vst.128.ip q2, a3, 16
        ee.vst.128.ip q3, a3, 16
        ee.vst.128.ip q4, a3, 16
    .Lloop_end\@:
    retw
.Lsrc_unaligned\@:
    loop a4, .Lloop_src_unaligned_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, q0, q1, a6, a7
        vzipt_8 q2, q3, q4, q5
        ee.vst.128.ip q2, a3, 16
        ee.vst.128.ip q3, a3, 16
        ee.vst.128.ip q4, a3, 16
    .Lloop_src_unaligned_end\@:
    retw
.Ldst_unaligned\@:
    vst_unaligned_init a3, a5, q7, q6
    loop a4, .Lloop_unaligned_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, q0, q1, a6, a7
        vzipt_8 q2, q3, q4, q5
        vst_3chn_unaligned q2, q3, q4, a3, a5, q7, q6
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a5, q7, q6
.Lend\@:
    retw
.endm

cvt_color_simd_helper_rgb5652rgb888_template rgb565le2rgb888
cvt_color_simd_helper_rgb5652rgb888_template rgb565be2rgb888
cvt_color_simd_helper_rgb5652rgb888_template rgb565le2bgr888
cvt_color_simd_helper_rgb5652rgb888_template rgb565be2bgr888

.macro cvt_color_simd_helper_rgb5652rgb888_qint8_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n
    # a5: int *lut

    entry sp, 32
    blti a4, 1, .Lend\@
    extui a6, a3, 0, 4
    bnez a6, .Ldst_unaligned\@
    loop a4, .Lloop_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, a5, q0, q1, a7, a8, q5
        vzipt_8 q2, q3, q4, q5
        ee.vst.128.ip q2, a3, 16
        ee.vst.128.ip q3, a3, 16
        ee.vst.128.ip q4, a3, 16
    .Lloop_end\@:
    retw
.Ldst_unaligned\@:
    vst_unaligned_init a3, a6, q7, q6
    loop a4, .Lloop_unaligned_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, a5, q0, q1, a7, a8, q5
        vzipt_8 q2, q3, q4, q5
        vst_3chn_unaligned q2, q3, q4, a3, a6, q7, q6
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a6, q7, q6
.Lend\@:
    retw
.endm

cvt_color_simd_helper_rgb5652rgb888_qint8_template rgb565le2rgb888_qint8
cvt_color_simd_helper_rgb5652rgb888_qint8_template rgb565be2rgb888_qint8
cvt_color_simd_helper_rgb5652rgb888_qint8_template rgb565le2bgr888_qint8
cvt_color_simd_helper_rgb5652rgb888_qint8_template rgb565be2bgr888_qint8

.macro cvt_color_simd_helper_rgb5652rgb888_qint16_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n
    # a5: int *lut

    entry sp, 32
    blti a4, 1, .Lend\@
    extui a6, a3, 0, 4
    bnez a6, .Ldst_unaligned\@
    loop a4, .Lloop_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, q5, q6, q7, a5, q0, q1, a7, a8
        vzipt_16 q2, q4, q6, q0
        vzipt_16 q3, q5, q7, q0
        ee.vst.128.ip q2, a3, 16
        ee.vst.128.ip q4, a3, 16
        ee.vst.128.ip q6, a3, 16
        ee.vst.128.ip q3, a3, 16
        ee.vst.128.ip q5, a3, 16
        ee.vst.128.ip q7, a3, 16
    .Lloop_end\@:
    retw
.Ldst_unaligned\@:
    vst_unaligned_init a3, a6, q0, q1, sp
    loop a4, .Lloop_unaligned_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q4, q5, q6, q7, a5, q0, q1, a7, a8
        vzipt_16 q2, q4, q6, q0
        vzipt_16 q3, q5, q7, q0
        vst_6chn_unaligned q2, q4, q6, q3, q5, q7, a3, a6, q0, q1, sp
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a6, q0, q1, sp
.Lend\@:
    retw
.endm

cvt_color_simd_helper_rgb5652rgb888_qint16_template rgb565le2rgb888_qint16
cvt_color_simd_helper_rgb5652rgb888_qint16_template rgb565be2rgb888_qint16
cvt_color_simd_helper_rgb5652rgb888_qint16_template rgb565le2bgr888_qint16
cvt_color_simd_helper_rgb5652rgb888_qint16_template rgb565be2bgr888_qint16
