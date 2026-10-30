#include "dl_tie728_color_common.S"

# Scalar YUYV2Gray keeps the Y byte. YUYV stores even bytes, UYVY stores odd bytes.

.macro cvt_color_simd_helper_yuyv2gray_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:
    entry sp, 32
    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n         (16-pixel blocks)

    blti a4, 1, .Lend\@

    extui a5, a3, 0, 4
    bnez a5, .Ldst_unaligned\@

    extui a5, a2, 0, 4
    bnez a5, .Lsrc_unaligned\@

    loop a4, .Lloop_end\@
        ee.vld.128.ip q0, a2, 16
        ee.vld.128.ip q1, a2, 16
        ee.vunzip.8 q0, q1
.ifc "\pix_cvt", "yuyv2gray"
        ee.vst.128.ip q0, a3, 16
.endif
.ifc "\pix_cvt", "uyvy2gray"
        ee.vst.128.ip q1, a3, 16
.endif
    .Lloop_end\@:
    retw.n

.Lsrc_unaligned\@:
    loop a4, .Lloop_src_unaligned_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
.ifc "\pix_cvt", "yuyv2gray"
        ee.vst.128.ip q0, a3, 16
.endif
.ifc "\pix_cvt", "uyvy2gray"
        ee.vst.128.ip q1, a3, 16
.endif
    .Lloop_src_unaligned_end\@:
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a3, a5, q7, q6
    loop a4, .Lloop_unaligned_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
.ifc "\pix_cvt", "yuyv2gray"
        vst_1chn_unaligned q0, a3, a5, q7, q6
.endif
.ifc "\pix_cvt", "uyvy2gray"
        vst_1chn_unaligned q1, a3, a5, q7, q6
.endif
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a5, q7, q6

.Lend\@:
    retw.n
.endm

cvt_color_simd_helper_yuyv2gray_template yuyv2gray
cvt_color_simd_helper_yuyv2gray_template uyvy2gray

.macro cvt_color_simd_helper_yuyv2gray_qint8_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:
    entry sp, 32
    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n         (16-pixel blocks)
    # a5: int *lut

    blti a4, 1, .Lend\@

    extui a6, a3, 0, 4
    bnez a6, .Ldst_unaligned\@

    loop a4, .Lloop_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
.ifc "\pix_cvt", "yuyv2gray_qint8"
        norm_quant_1chn_8 q0, a5, q1, q2, q3
        ee.vst.128.ip q0, a3, 16
.endif
.ifc "\pix_cvt", "uyvy2gray_qint8"
        norm_quant_1chn_8 q1, a5, q0, q2, q3
        ee.vst.128.ip q1, a3, 16
.endif
    .Lloop_end\@:
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a3, a6, q7, q6
    loop a4, .Lloop_unaligned_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
.ifc "\pix_cvt", "yuyv2gray_qint8"
        norm_quant_1chn_8 q0, a5, q1, q2, q3
        vst_1chn_unaligned q0, a3, a6, q7, q6
.endif
.ifc "\pix_cvt", "uyvy2gray_qint8"
        norm_quant_1chn_8 q1, a5, q0, q2, q3
        vst_1chn_unaligned q1, a3, a6, q7, q6
.endif
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a6, q7, q6

.Lend\@:
    retw.n
.endm

cvt_color_simd_helper_yuyv2gray_qint8_template yuyv2gray_qint8
cvt_color_simd_helper_yuyv2gray_qint8_template uyvy2gray_qint8

.macro cvt_color_simd_helper_yuyv2gray_qint16_template pix_cvt
    .balign 4
    .text
    .global cvt_color_simd_helper_\pix_cvt
    .type   cvt_color_simd_helper_\pix_cvt, @function
cvt_color_simd_helper_\pix_cvt:
    entry sp, 32
    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n         (16-pixel blocks)
    # a5: int *lut

    blti a4, 1, .Lend\@

    extui a6, a3, 0, 4
    bnez a6, .Ldst_unaligned\@

    loop a4, .Lloop_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
.ifc "\pix_cvt", "yuyv2gray_qint16"
        norm_quant_1chn_16 q1, q0, a5, q2, q3
        ee.vst.128.ip q1, a3, 16
        ee.vst.128.ip q0, a3, 16
.endif
.ifc "\pix_cvt", "uyvy2gray_qint16"
        norm_quant_1chn_16 q0, q1, a5, q2, q3
        ee.vst.128.ip q0, a3, 16
        ee.vst.128.ip q1, a3, 16
.endif
    .Lloop_end\@:
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a3, a6, q7, q6
    loop a4, .Lloop_unaligned_end\@
        vld_2chn_unaligned q0, q1, a2, q2
        ee.vunzip.8 q0, q1
.ifc "\pix_cvt", "yuyv2gray_qint16"
        norm_quant_1chn_16 q1, q0, a5, q2, q3
        vst_2chn_unaligned q1, q0, a3, a6, q7, q6
.endif
.ifc "\pix_cvt", "uyvy2gray_qint16"
        norm_quant_1chn_16 q0, q1, a5, q2, q3
        vst_2chn_unaligned q0, q1, a3, a6, q7, q6
.endif
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a3, a6, q7, q6

.Lend\@:
    retw.n
.endm

cvt_color_simd_helper_yuyv2gray_qint16_template yuyv2gray_qint16
cvt_color_simd_helper_yuyv2gray_qint16_template uyvy2gray_qint16
