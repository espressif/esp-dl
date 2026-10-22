#include "dl_tie728_color_common.S"

    .balign 4
    .text
    .global cvt_color_simd_helper_gray2gray
    .type   cvt_color_simd_helper_gray2gray, @function
cvt_color_simd_helper_gray2gray:
    entry sp, 32
    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n

    blti a4, 1, .Lgray_end

    extui a6, a3, 0, 4
    bnez a6, .Lgray_dst_unaligned

    extui a5, a2, 0, 4
    bnez a5, .Lgray_src_unaligned_dst_aligned

    loop a4, .Lgray_loop_aligned_end
        ee.vld.128.ip q0, a2, 16
        ee.vst.128.ip q0, a3, 16
    .Lgray_loop_aligned_end:
    retw.n

.Lgray_src_unaligned_dst_aligned:
    loop a4, .Lgray_loop_src_unaligned_end
        vld_1chn_unaligned q0, a2, q1
        ee.vst.128.ip q0, a3, 16
    .Lgray_loop_src_unaligned_end:
    retw.n

.Lgray_dst_unaligned:
    vst_unaligned_init a3, a6, q2, q3
    loop a4, .Lgray_loop_unaligned_end
        vld_1chn_unaligned q0, a2, q1
        vst_1chn_unaligned q0, a3, a6, q2, q3
    .Lgray_loop_unaligned_end:
    vst_unaligned_flush a3, a6, q2, q3

.Lgray_end:
    retw.n

    .balign 4
    .text
    .global cvt_color_simd_helper_gray2gray_qint8
    .type   cvt_color_simd_helper_gray2gray_qint8, @function
cvt_color_simd_helper_gray2gray_qint8:
    entry sp, 32
    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n
    # a5: int *lut

    blti a4, 1, .Lgray_q8_end

    extui a6, a3, 0, 4
    bnez a6, .Lgray_q8_dst_unaligned

    loop a4, .Lgray_q8_loop_dst_aligned_end
        vld_1chn_unaligned q0, a2, q1
        norm_quant_1chn_8 q0, a5, q1, q2, q3
        ee.vst.128.ip q0, a3, 16
    .Lgray_q8_loop_dst_aligned_end:
    retw.n

.Lgray_q8_dst_unaligned:
    vst_unaligned_init a3, a6, q4, q5
    loop a4, .Lgray_q8_loop_unaligned_end
        vld_1chn_unaligned q0, a2, q1
        norm_quant_1chn_8 q0, a5, q1, q2, q3
        vst_1chn_unaligned q0, a3, a6, q4, q5
    .Lgray_q8_loop_unaligned_end:
    vst_unaligned_flush a3, a6, q4, q5

.Lgray_q8_end:
    retw.n

    .balign 4
    .text
    .global cvt_color_simd_helper_gray2gray_qint16
    .type   cvt_color_simd_helper_gray2gray_qint16, @function
cvt_color_simd_helper_gray2gray_qint16:
    entry sp, 32
    # a2: uint8_t *src
    # a3: uint8_t *dst
    # a4: int n
    # a5: int *lut

    blti a4, 1, .Lgray_q16_end

    extui a6, a3, 0, 4
    bnez a6, .Lgray_q16_dst_unaligned

    loop a4, .Lgray_q16_loop_dst_aligned_end
        vld_1chn_unaligned q1, a2, q2
        norm_quant_1chn_16 q0, q1, a5, q2, q3
        ee.vst.128.ip q0, a3, 16
        ee.vst.128.ip q1, a3, 16
    .Lgray_q16_loop_dst_aligned_end:
    retw.n

.Lgray_q16_dst_unaligned:
    vst_unaligned_init a3, a6, q4, q5
    loop a4, .Lgray_q16_loop_unaligned_end
        vld_1chn_unaligned q1, a2, q2
        norm_quant_1chn_16 q0, q1, a5, q2, q3
        vst_2chn_unaligned q0, q1, a3, a6, q4, q5
    .Lgray_q16_loop_unaligned_end:
    vst_unaligned_flush a3, a6, q4, q5

.Lgray_q16_end:
    retw.n
