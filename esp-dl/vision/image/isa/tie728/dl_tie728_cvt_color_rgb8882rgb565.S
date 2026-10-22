#include "dl_tie728_color_common.S"

.macro cvt_color_simd_helper_rgb8882rgb565_template pix_cvt
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
        \pix_cvt q3, q4, q0, q1, q2, a5, a6
        ee.vzip.8 q3, q4
        ee.vst.128.ip q3, a3, 16
        ee.vst.128.ip q4, a3, 16
    .Lloop_end\@:
    retw
.Lsrc_unaligned\@:
    loop a4, .Lloop_src_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q4, q0, q1, q2, a5, a6
        ee.vzip.8 q3, q4
        ee.vst.128.ip q3, a3, 16
        ee.vst.128.ip q4, a3, 16
    .Lloop_src_unaligned_end\@:
    retw
.Ldst_unaligned\@:
    vst_unaligned_init a3, a5, q7, q6
    loop a4, .Lloop_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q4, q0, q1, q2, a6, a7
        ee.vzip.8 q3, q4
        vst_2chn_unaligned q3, q4, a3, a5, q7, q6
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a5, q7, q6
.Lend\@:
    retw
.endm

cvt_color_simd_helper_rgb8882rgb565_template rgb8882rgb565le
cvt_color_simd_helper_rgb8882rgb565_template bgr8882rgb565le
cvt_color_simd_helper_rgb8882rgb565_template rgb8882rgb565be
cvt_color_simd_helper_rgb8882rgb565_template bgr8882rgb565be
