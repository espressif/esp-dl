#include "dl_tie728_color_common.S"

.macro cvt_color_simd_helper_rgb5652rgb565_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:
    entry sp, 32
    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n

    blti a4, 1, .Lend\@

    extui a6, a3, 0, 4
    bnez a6, .Ldst_unaligned\@

    extui a5, a2, 0, 4
    bnez a5, .Lsrc_unaligned_dst_aligned\@

    loop a4, .Lloop_aligned_end\@
        ee.vld.128.ip q0, a2, 16
        ee.vld.128.ip q1, a2, 16
.ifc "\pix_cvt", "rgb5652rgb565"
        ee.vst.128.ip q0, a3, 16
        ee.vst.128.ip q1, a3, 16
.else
.ifc "\pix_cvt", "rgb565le2rgb565be"
        ee.vunzip.8 q0, q1
        ee.vzip.8 q1, q0
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q0, a3, 16
.else
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q0, q1, a5, a6
        ee.vzip.8 q2, q3
        ee.vst.128.ip q2, a3, 16
        ee.vst.128.ip q3, a3, 16
.endif
.endif
    .Lloop_aligned_end\@:
    retw.n

.Lsrc_unaligned_dst_aligned\@:
    loop a4, .Lloop_src_unaligned_end\@
        vld_2chn_unaligned q0, q1, a2, q4
.ifc "\pix_cvt", "rgb5652rgb565"
        ee.vst.128.ip q0, a3, 16
        ee.vst.128.ip q1, a3, 16
.else
.ifc "\pix_cvt", "rgb565le2rgb565be"
        ee.vunzip.8 q0, q1
        ee.vzip.8 q1, q0
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q0, a3, 16
.else
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q0, q1, a5, a6
        ee.vzip.8 q2, q3
        ee.vst.128.ip q2, a3, 16
        ee.vst.128.ip q3, a3, 16
.endif
.endif
    .Lloop_src_unaligned_end\@:
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a3, a6, q5, q6
    loop a4, .Lloop_unaligned_end\@
        vld_2chn_unaligned q0, q1, a2, q4
.ifc "\pix_cvt", "rgb5652rgb565"
        vst_2chn_unaligned q0, q1, a3, a6, q5, q6
.else
.ifc "\pix_cvt", "rgb565le2rgb565be"
        ee.vunzip.8 q0, q1
        ee.vzip.8 q1, q0
        vst_2chn_unaligned q1, q0, a3, a6, q5, q6
.else
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q0, q1, a5, a7
        ee.vzip.8 q2, q3
        vst_2chn_unaligned q2, q3, a3, a6, q5, q6
.endif
.endif
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a6, q5, q6

.Lend\@:
    retw.n
.endm

cvt_color_simd_helper_rgb5652rgb565_template rgb5652rgb565
cvt_color_simd_helper_rgb5652rgb565_template rgb565le2rgb565be
cvt_color_simd_helper_rgb5652rgb565_template rgb565le2bgr565le
cvt_color_simd_helper_rgb5652rgb565_template rgb565be2bgr565be
cvt_color_simd_helper_rgb5652rgb565_template rgb565le2bgr565be
cvt_color_simd_helper_rgb5652rgb565_template rgb565be2bgr565le
