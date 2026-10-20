#include "dl_tie728_color_common.S"

.macro resize_nn_simd_helper_hsv2hsv_mask_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a2: uint8_t **src
    # a3: int *offsets
    # a4: uint8_t *dst
    # a5: int n
    # a6: const uint8_t *hsv_min
    # a7: const uint8_t *hsv_max

    entry sp, 48
    blti a5, 1, .Lend\@

    extui a8, a4, 0, 4
    bnez a8, .Ldst_unaligned\@

    loop a5, .Lloop_end\@
        resize_nn_load_3chn_128 q0, q1, q2, a2, a3, sp, q3, q4, q5, a9, a10, a11, a12
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, a6, a7, q0, q1, q2, q4, q5, a9
        ee.vst.128.ip q3, a4, 16
    .Lloop_end\@:
    retw

.Ldst_unaligned\@:
    vst_unaligned_init a4, a8, q6, q7
    loop a5, .Lloop_unaligned_end\@
        resize_nn_load_3chn_128 q0, q1, q2, a2, a3, sp, q3, q4, q5, a9, a10, a11, a12
        vunzipt_8 q0, q1, q2, q3
        \pix_cvt q3, a6, a7, q0, q1, q2, q4, q5, a9
        vst_1chn_unaligned q3, a4, a8, q6, q7
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a8, q6, q7

.Lend\@:
    retw
.endm

resize_nn_simd_helper_hsv2hsv_mask_template hsv2hsv_mask0
resize_nn_simd_helper_hsv2hsv_mask_template hsv2hsv_mask1
