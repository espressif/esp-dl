#include "dl_tie728_color_common.S"

.macro cvt_color_simd_helper_hsv2hsv_mask_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:

    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n
    # a5: const uint8_t *hsv_min
    # a6: const uint8_t *hsv_max

    entry sp, 32
    blti a4, 1, .Lend\@

    extui a8, a3, 0, 4
    bnez a8, .Ldst_unaligned\@

    extui a7, a2, 0, 4
    bnez a7, .Lsrc_unaligned_dst_aligned\@

    loop a4, .Lloop_end\@
        ee.vld.128.ip q0, a2, 16
        ee.vld.128.ip q1, a2, 16
        ee.vld.128.ip q2, a2, 16
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, a5, a6, q0, q1, q2, q4, q5, a7
        ee.vst.128.ip q3, a3, 16
    .Lloop_end\@:
    retw

.Lsrc_unaligned_dst_aligned\@:
    loop a4, .Lloop_src_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, a5, a6, q0, q1, q2, q4, q5, a7
        ee.vst.128.ip q3, a3, 16
    .Lloop_src_unaligned_end\@:
    retw

.Ldst_unaligned\@:
    vst_unaligned_init a3, a8, q6, q7
    loop a4, .Lloop_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, a5, a6, q0, q1, q2, q4, q5, a7
        vst_1chn_unaligned q3, a3, a8, q6, q7
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a8, q6, q7

.Lend\@:
    retw
.endm

cvt_color_simd_helper_hsv2hsv_mask_template hsv2hsv_mask0
cvt_color_simd_helper_hsv2hsv_mask_template hsv2hsv_mask1
