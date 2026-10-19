#include "dl_tie728_color_common.S"

    .balign 4
    .text
    .global resize_nn_simd_helper_gray2gray
    .type   resize_nn_simd_helper_gray2gray, @function
resize_nn_simd_helper_gray2gray:
    entry sp, 32
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n

    blti a5, 1, .Lgray_end

    loop a5, .Lgray_loop_end
        resize_nn_load_store_1chn_128 a4, a2, a3, q0, q1, q2, a6, a7, a8, a9
    .Lgray_loop_end:

.Lgray_end:
    retw.n

    .balign 4
    .text
    .global resize_nn_simd_helper_gray2gray_qint8
    .type   resize_nn_simd_helper_gray2gray_qint8, @function
resize_nn_simd_helper_gray2gray_qint8:
    entry sp, 48
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n
    # a6: int *lut

    blti a5, 1, .Lgray_q8_end

    extui a11, a4, 0, 4
    bnez a11, .Lgray_q8_dst_unaligned

    loop a5, .Lgray_q8_loop_dst_aligned_end
        resize_nn_load_1chn_128 q0, a2, a3, sp, q1, q2, q3, a7, a8, a9, a10
        norm_quant_1chn_8 q0, a6, q1, q2, q3
        ee.vst.128.ip q0, a4, 16
    .Lgray_q8_loop_dst_aligned_end:
    retw.n

.Lgray_q8_dst_unaligned:
    vst_unaligned_init a4, a11, q4, q5
    loop a5, .Lgray_q8_loop_unaligned_end
        resize_nn_load_1chn_128 q0, a2, a3, sp, q1, q2, q3, a7, a8, a9, a10
        norm_quant_1chn_8 q0, a6, q1, q2, q3
        vst_1chn_unaligned q0, a4, a11, q4, q5
    .Lgray_q8_loop_unaligned_end:
    vst_unaligned_flush a4, a11, q4, q5

.Lgray_q8_end:
    retw.n

    .balign 4
    .text
    .global resize_nn_simd_helper_gray2gray_qint16
    .type   resize_nn_simd_helper_gray2gray_qint16, @function
resize_nn_simd_helper_gray2gray_qint16:
    entry sp, 48
    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n
    # a6: int *lut

    blti a5, 1, .Lgray_q16_end

    extui a11, a4, 0, 4
    bnez a11, .Lgray_q16_dst_unaligned

    loop a5, .Lgray_q16_loop_dst_aligned_end
        resize_nn_load_1chn_128 q1, a2, a3, sp, q0, q2, q3, a7, a8, a9, a10
        norm_quant_1chn_16 q0, q1, a6, q2, q3
        ee.vst.128.ip q0, a4, 16
        ee.vst.128.ip q1, a4, 16
    .Lgray_q16_loop_dst_aligned_end:
    retw.n

.Lgray_q16_dst_unaligned:
    vst_unaligned_init a4, a11, q4, q5
    loop a5, .Lgray_q16_loop_unaligned_end
        resize_nn_load_1chn_128 q1, a2, a3, sp, q0, q2, q3, a7, a8, a9, a10
        norm_quant_1chn_16 q0, q1, a6, q2, q3
        vst_2chn_unaligned q0, q1, a4, a11, q4, q5
    .Lgray_q16_loop_unaligned_end:
    vst_unaligned_flush a4, a11, q4, q5

.Lgray_q16_end:
    retw.n
