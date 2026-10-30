#include "dl_tie728_color_common.S"

# YUYV keeps the byte at the pixel address. UYVY keeps the next byte.
.macro resize_nn_simd_helper_yuyv2gray_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:
    entry sp, 32
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n

    blti a5, 1, .Lend\@

    loop a5, .Lloop_end\@
.ifc "\pix_cvt", "yuyv2gray"
        resize_nn_load_store_1chn_128 a4, a2, a3, q0, q1, q2, a6, a7, a8, a9, 0
.endif
.ifc "\pix_cvt", "uyvy2gray"
        resize_nn_load_store_1chn_128 a4, a2, a3, q0, q1, q2, a6, a7, a8, a9, 1
.endif
    .Lloop_end\@:

.Lend\@:
    retw.n
.endm

resize_nn_simd_helper_yuyv2gray_template yuyv2gray
resize_nn_simd_helper_yuyv2gray_template uyvy2gray

.macro resize_nn_simd_helper_yuyv2gray_qint8_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:
    entry sp, 48
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n
    # a6: int *lut

    blti a5, 1, .Lend\@

    extui a11, a4, 0, 4
    bnez a11, .Ldst_unaligned\@

    loop a5, .Lloop_dst_aligned_end\@
.ifc "\pix_cvt", "yuyv2gray_qint8"
        resize_nn_load_1chn_128 q0, a2, a3, sp, q1, q2, q3, a7, a8, a9, a10, 0
.endif
.ifc "\pix_cvt", "uyvy2gray_qint8"
        resize_nn_load_1chn_128 q0, a2, a3, sp, q1, q2, q3, a7, a8, a9, a10, 1
.endif
        norm_quant_1chn_8 q0, a6, q1, q2, q3
        ee.vst.128.ip q0, a4, 16
    .Lloop_dst_aligned_end\@:
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a4, a11, q4, q5
    loop a5, .Lloop_unaligned_end\@
.ifc "\pix_cvt", "yuyv2gray_qint8"
        resize_nn_load_1chn_128 q0, a2, a3, sp, q1, q2, q3, a7, a8, a9, a10, 0
.endif
.ifc "\pix_cvt", "uyvy2gray_qint8"
        resize_nn_load_1chn_128 q0, a2, a3, sp, q1, q2, q3, a7, a8, a9, a10, 1
.endif
        norm_quant_1chn_8 q0, a6, q1, q2, q3
        vst_1chn_unaligned q0, a4, a11, q4, q5
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a11, q4, q5

.Lend\@:
    retw.n
.endm

resize_nn_simd_helper_yuyv2gray_qint8_template yuyv2gray_qint8
resize_nn_simd_helper_yuyv2gray_qint8_template uyvy2gray_qint8

.macro resize_nn_simd_helper_yuyv2gray_qint16_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:
    entry sp, 48
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n
    # a6: int *lut

    blti a5, 1, .Lend\@

    extui a11, a4, 0, 4
    bnez a11, .Ldst_unaligned\@

    loop a5, .Lloop_dst_aligned_end\@
.ifc "\pix_cvt", "yuyv2gray_qint16"
        resize_nn_load_1chn_128 q1, a2, a3, sp, q0, q2, q3, a7, a8, a9, a10, 0
.endif
.ifc "\pix_cvt", "uyvy2gray_qint16"
        resize_nn_load_1chn_128 q1, a2, a3, sp, q0, q2, q3, a7, a8, a9, a10, 1
.endif
        norm_quant_1chn_16 q0, q1, a6, q2, q3
        ee.vst.128.ip q0, a4, 16
        ee.vst.128.ip q1, a4, 16
    .Lloop_dst_aligned_end\@:
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a4, a11, q4, q5
    loop a5, .Lloop_unaligned_end\@
.ifc "\pix_cvt", "yuyv2gray_qint16"
        resize_nn_load_1chn_128 q1, a2, a3, sp, q0, q2, q3, a7, a8, a9, a10, 0
.endif
.ifc "\pix_cvt", "uyvy2gray_qint16"
        resize_nn_load_1chn_128 q1, a2, a3, sp, q0, q2, q3, a7, a8, a9, a10, 1
.endif
        norm_quant_1chn_16 q0, q1, a6, q2, q3
        vst_2chn_unaligned q0, q1, a4, a11, q4, q5
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a11, q4, q5

.Lend\@:
    retw.n
.endm

resize_nn_simd_helper_yuyv2gray_qint16_template yuyv2gray_qint16
resize_nn_simd_helper_yuyv2gray_qint16_template uyvy2gray_qint16
