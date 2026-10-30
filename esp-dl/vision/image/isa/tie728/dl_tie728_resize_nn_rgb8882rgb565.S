#include "dl_tie728_color_common.S"

.macro resize_nn_simd_helper_rgb8882rgb565_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n

    entry sp, 48
    blti a5, 1, .Lend\@
    extui a6, a4, 0, 4
    bnez a6, .Ldst_unaligned\@
    loop a5, .Lloop_end\@
        resize_nn_load_3chn_128 q0, q1, q2, a2, a3, sp, q3, q4, q5, a7, a8, a9, a10
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q4, q0, q1, q2, a7, a8
        ee.vzip.8 q3, q4
        ee.vst.128.ip q3, a4, 16
        ee.vst.128.ip q4, a4, 16
    .Lloop_end\@:
    retw
.Ldst_unaligned\@:
    vst_unaligned_init a4, a6, q7, q6
    loop a5, .Lloop_unaligned_end\@
        resize_nn_load_3chn_128 q0, q1, q2, a2, a3, sp, q3, q4, q5, a7, a8, a9, a10
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, q4, q0, q1, q2, a7, a8
        ee.vzip.8 q3, q4
        vst_2chn_unaligned q3, q4, a4, a6, q7, q6
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a6, q7, q6
.Lend\@:
    retw
.endm

resize_nn_simd_helper_rgb8882rgb565_template rgb8882rgb565le
resize_nn_simd_helper_rgb8882rgb565_template bgr8882rgb565le
resize_nn_simd_helper_rgb8882rgb565_template rgb8882rgb565be
resize_nn_simd_helper_rgb8882rgb565_template bgr8882rgb565be
