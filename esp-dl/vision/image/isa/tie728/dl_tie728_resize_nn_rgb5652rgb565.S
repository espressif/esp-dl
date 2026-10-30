#include "dl_tie728_color_common.S"

.macro resize_nn_simd_helper_rgb5652rgb565_template pix_cvt
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

    blti a5, 1, .Lend\@

.ifc "\pix_cvt", "rgb5652rgb565"
    loop a5, .Lloop_copy_end\@
        resize_nn_load_store_2chn_128 a4, a2, a3, q2, q3, q4, a6, a7, a8, a9
    .Lloop_copy_end\@:
    retw.n
.else
    extui a6, a4, 0, 4
    bnez a6, .Ldst_unaligned\@

    loop a5, .Lloop_dst_aligned_end\@
        resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a7, a8, a9, a10
.ifc "\pix_cvt", "rgb565le2rgb565be"
        ee.vunzip.8 q0, q1
        ee.vzip.8 q1, q0
        ee.vst.128.ip q1, a4, 16
        ee.vst.128.ip q0, a4, 16
.else
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q0, q1, a7, a8
        ee.vzip.8 q2, q3
        ee.vst.128.ip q2, a4, 16
        ee.vst.128.ip q3, a4, 16
.endif
    .Lloop_dst_aligned_end\@:
    retw.n

.Ldst_unaligned\@:
    vst_unaligned_init a4, a6, q5, q6
    loop a5, .Lloop_unaligned_end\@
        resize_nn_load_2chn_128 q0, q1, a2, a3, sp, q2, q3, q4, a7, a8, a9, a10
.ifc "\pix_cvt", "rgb565le2rgb565be"
        ee.vunzip.8 q0, q1
        ee.vzip.8 q1, q0
        vst_2chn_unaligned q1, q0, a4, a6, q5, q6
.else
        ee.vunzip.8 q0, q1
        \pix_cvt q2, q3, q0, q1, a7, a8
        ee.vzip.8 q2, q3
        vst_2chn_unaligned q2, q3, a4, a6, q5, q6
.endif
    .Lloop_unaligned_end\@:
    vst_unaligned_flush a4, a6, q5, q6
.endif

.Lend\@:
    retw.n
.endm

resize_nn_simd_helper_rgb5652rgb565_template rgb5652rgb565
resize_nn_simd_helper_rgb5652rgb565_template rgb565le2rgb565be
resize_nn_simd_helper_rgb5652rgb565_template rgb565le2bgr565le
resize_nn_simd_helper_rgb5652rgb565_template rgb565be2bgr565be
resize_nn_simd_helper_rgb5652rgb565_template rgb565le2bgr565be
resize_nn_simd_helper_rgb5652rgb565_template rgb565be2bgr565le
