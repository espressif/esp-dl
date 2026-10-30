#include "dl_tie728_color_common.S"

.macro cvt_color_simd_helper_rgb8882gray_template pix_cvt
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
        ee.vld.128.ip q2, a2, 16
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q0, q1, q2, a6, q4, q5, q6, q7
        ee.vst.128.ip q3, a3, 16
    .Lloop_end\@:
    retw

.Lsrc_unaligned\@:
    loop a4, .Lloop_src_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q0, q1, q2, a6, q4, q5, q6, q7
        ee.vst.128.ip q3, a3, 16
    .Lloop_src_unaligned_end\@:
    retw

.Ldst_unaligned\@:
    vst_unaligned_init a3, a5, q7, q6, sp
    loop a4, .Lloop_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q0, q1, q2, a6, q4, q5, q6, q7
        vst_1chn_unaligned q3, a3, a5, q7, q6, sp
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a5, q7, q6, sp

.Lend\@:
    retw
.endm
cvt_color_simd_helper_rgb8882gray_template rgb8882gray
cvt_color_simd_helper_rgb8882gray_template bgr8882gray

.macro cvt_color_simd_helper_rgb8882gray_qint8_template pix_cvt
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
        vld_3chn_unaligned q0, q1, q2, a2, q3
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, a5, q0, q1, q2, a7, q4, q5, q6, q7
        ee.vst.128.ip q3, a3, 16
    .Lloop_end\@:
    retw

.Ldst_unaligned\@:
    vst_unaligned_init a3, a6, q7, q6, sp
    loop a4, .Lloop_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, a5, q0, q1, q2, a7, q4, q5, q6, q7
        vst_1chn_unaligned q3, a3, a6, q7, q6, sp
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a6, q7, q6, sp

.Lend\@:
    retw
.endm
cvt_color_simd_helper_rgb8882gray_qint8_template rgb8882gray_qint8
cvt_color_simd_helper_rgb8882gray_qint8_template bgr8882gray_qint8

.macro cvt_color_simd_helper_rgb8882gray_qint16_template pix_cvt
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
        vld_3chn_unaligned q0, q1, q2, a2, q3
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q4, a5, q0, q1, q2, a7, q5, q6, q7
        ee.vst.128.ip q3, a3, 16
        ee.vst.128.ip q4, a3, 16
    .Lloop_end\@:
    retw

.Ldst_unaligned\@:
    vst_unaligned_init a3, a6, q7, q6, sp
    loop a4, .Lloop_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q4, a5, q0, q1, q2, a7, q5, q6, q7
        vst_2chn_unaligned q3, q4, a3, a6, q7, q6, sp
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a6, q7, q6, sp

.Lend\@:
    retw
.endm
cvt_color_simd_helper_rgb8882gray_qint16_template rgb8882gray_qint16
cvt_color_simd_helper_rgb8882gray_qint16_template bgr8882gray_qint16
