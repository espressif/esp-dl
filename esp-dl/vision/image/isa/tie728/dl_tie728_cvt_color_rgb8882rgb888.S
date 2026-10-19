#include "dl_tie728_color_common.S"

.macro cvt_color_simd_helper_rgb8882rgb888_template pix_cvt
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
.ifc "\pix_cvt", "rgb8882rgb888"
        ee.vst.128.ip q0, a3, 16
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q2, a3, 16
.endif
.ifc "\pix_cvt", "rgb8882bgr888"
        vunzipt_8 q0, q1, q2, q3
        vzipt_8 q2, q1, q0, q3
        ee.vst.128.ip q2, a3, 16
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q0, a3, 16
.endif
    .Lloop_end\@:
    retw
.Lsrc_unaligned\@:
    loop a4, .Lloop_src_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
.ifc "\pix_cvt", "rgb8882rgb888"
        ee.vst.128.ip q0, a3, 16
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q2, a3, 16
.endif
.ifc "\pix_cvt", "rgb8882bgr888"
        vunzipt_8 q0, q1, q2, q3
        vzipt_8 q2, q1, q0, q3
        ee.vst.128.ip q2, a3, 16
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q0, a3, 16
.endif
    .Lloop_src_unaligned_end\@:
    retw
.Ldst_unaligned\@:
    vst_unaligned_init a3, a5, q7, q6
    loop a4, .Lloop_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
.ifc "\pix_cvt", "rgb8882rgb888"
        vst_3chn_unaligned q0, q1, q2, a3, a5, q7, q6
.endif
.ifc "\pix_cvt", "rgb8882bgr888"
        vunzipt_8 q0, q1, q2, q3
        vzipt_8 q2, q1, q0, q3
        vst_3chn_unaligned q2, q1, q0, a3, a5, q7, q6
.endif
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a5, q7, q6
.Lend\@:
    retw
.endm

cvt_color_simd_helper_rgb8882rgb888_template rgb8882rgb888
cvt_color_simd_helper_rgb8882rgb888_template rgb8882bgr888

.macro cvt_color_simd_helper_rgb8882rgb888_qint8_template pix_cvt
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
.ifc "\pix_cvt", "rgb8882rgb888_qint8"
        vunzipt_8 q0, q1, q2, q3
        norm_quant_3chn_8 q0, q1, q2, a5, q3, q4, q5
        vzipt_8 q0, q1, q2, q3
        ee.vst.128.ip q0, a3, 16
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q2, a3, 16
.endif
.ifc "\pix_cvt", "rgb8882bgr888_qint8"
        vunzipt_8 q0, q1, q2, q3
        norm_quant_3chn_8 q2, q1, q0, a5, q3, q4, q5
        vzipt_8 q2, q1, q0, q3
        ee.vst.128.ip q2, a3, 16
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q0, a3, 16
.endif
    .Lloop_end\@:
    retw
.Ldst_unaligned\@:
    vst_unaligned_init a3, a6, q7, q6
    loop a4, .Lloop_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
.ifc "\pix_cvt", "rgb8882rgb888_qint8"
        vunzipt_8 q0, q1, q2, q3
        norm_quant_3chn_8 q0, q1, q2, a5, q3, q4, q5
        vzipt_8 q0, q1, q2, q3
        vst_3chn_unaligned q0, q1, q2, a3, a6, q7, q6
.endif
.ifc "\pix_cvt", "rgb8882bgr888_qint8"
        vunzipt_8 q0, q1, q2, q3
        norm_quant_3chn_8 q2, q1, q0, a5, q3, q4, q5
        vzipt_8 q2, q1, q0, q3
        vst_3chn_unaligned q2, q1, q0, a3, a6, q7, q6
.endif
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a6, q7, q6
.Lend\@:
    retw
.endm

cvt_color_simd_helper_rgb8882rgb888_qint8_template rgb8882rgb888_qint8
cvt_color_simd_helper_rgb8882rgb888_qint8_template rgb8882bgr888_qint8

.macro cvt_color_simd_helper_rgb8882rgb888_qint16_template pix_cvt
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
.ifc "\pix_cvt", "rgb8882rgb888_qint16"
        vunzipt_8 q0, q1, q2, q3
        norm_quant_3chn_16 q3, q4, q5, q0, q1, q2, a5, q6, q7
        vzipt_16 q3, q5, q1, q6
        vzipt_16 q4, q0, q2, q6
        ee.vst.128.ip q3, a3, 16
        ee.vst.128.ip q5, a3, 16
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q4, a3, 16
        ee.vst.128.ip q0, a3, 16
        ee.vst.128.ip q2, a3, 16
.endif
.ifc "\pix_cvt", "rgb8882bgr888_qint16"
        vunzipt_8 q0, q1, q2, q3
        norm_quant_3chn_16 q3, q4, q5, q2, q1, q0, a5, q6, q7
        vzipt_16 q3, q5, q1, q6
        vzipt_16 q4, q2, q0, q6
        ee.vst.128.ip q3, a3, 16
        ee.vst.128.ip q5, a3, 16
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q4, a3, 16
        ee.vst.128.ip q2, a3, 16
        ee.vst.128.ip q0, a3, 16
.endif
    .Lloop_end\@:
    retw
.Ldst_unaligned\@:
    vst_unaligned_init a3, a6, q6, q7, sp
    loop a4, .Lloop_unaligned_end\@
        vld_3chn_unaligned q0, q1, q2, a2, q3
.ifc "\pix_cvt", "rgb8882rgb888_qint16"
        vunzipt_8 q0, q1, q2, q3
        norm_quant_3chn_16 q3, q4, q5, q0, q1, q2, a5, q6, q7
        vzipt_16 q3, q5, q1, q6
        vzipt_16 q4, q0, q2, q6
        vst_6chn_unaligned q3, q5, q1, q4, q0, q2, a3, a6, q6, q7, sp
.endif
.ifc "\pix_cvt", "rgb8882bgr888_qint16"
        vunzipt_8 q0, q1, q2, q3
        norm_quant_3chn_16 q3, q4, q5, q2, q1, q0, a5, q6, q7
        vzipt_16 q3, q5, q1, q6
        vzipt_16 q4, q2, q0, q6
        vst_6chn_unaligned q3, q5, q1, q4, q2, q0, a3, a6, q6, q7, sp
.endif
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a6, q6, q7, sp
.Lend\@:
    retw
.endm

cvt_color_simd_helper_rgb8882rgb888_qint16_template rgb8882rgb888_qint16
cvt_color_simd_helper_rgb8882rgb888_qint16_template rgb8882bgr888_qint16
