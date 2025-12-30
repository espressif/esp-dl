#include "dl_esp32p4_color_common.S"

.macro resize_nn_simd_helper_rgb5652rgb565_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n

    blez a3, .Lend\@
.Lloop\@:

.ifc "\pix_cvt", "rgb5652rgb565"
    resize_nn_load_store_2chn_128 a2, a0, a1, q2, q3, q4, a4, a5, t3, t4
.else
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, a4, a5, t3, t4
    .ifc "\pix_cvt", "rgb565le2rgb565be"
        esp.vunzip.8 q0, q1
        esp.vzip.8 q1, q0
        esp.vst.128.ip q1, a2, 16
        esp.vst.128.ip q0, a2, 16
    .else
        esp.vunzip.8 q0, q1
        \pix_cvt q2, q3, q0, q1, a4, a5
        esp.vzip.8 q2, q3
        esp.vst.128.ip q2, a2, 16
        esp.vst.128.ip q3, a2, 16
    .endif
.endif
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm
resize_nn_simd_helper_rgb5652rgb565_template rgb5652rgb565
resize_nn_simd_helper_rgb5652rgb565_template rgb565le2rgb565be
resize_nn_simd_helper_rgb5652rgb565_template rgb565le2bgr565le
resize_nn_simd_helper_rgb5652rgb565_template rgb565be2bgr565be
resize_nn_simd_helper_rgb5652rgb565_template rgb565le2bgr565be
resize_nn_simd_helper_rgb5652rgb565_template rgb565be2bgr565le
