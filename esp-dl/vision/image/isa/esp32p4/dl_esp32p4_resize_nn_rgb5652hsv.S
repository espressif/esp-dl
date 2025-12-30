#include "dl_esp32p4_color_common.S"

.macro resize_nn_simd_helper_rgb5652hsv_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: int *sdiv_lut
    # a5: int *hdiv_lut

    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, t3, t4, t5, t6
    esp.vunzip.8 q0, q1
    \pix_cvt q3, q4, q5, a4, a5, q0, q1, t3, t4, t5, t6, q2, q6, q7
    esp.vzipt.8 q3, q4, q5
    esp.vst.128.ip q3, a2, 16
    esp.vst.128.ip q4, a2, 16
    esp.vst.128.ip q5, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm

resize_nn_simd_helper_rgb5652hsv_template rgb565le2hsv
resize_nn_simd_helper_rgb5652hsv_template rgb565be2hsv
resize_nn_simd_helper_rgb5652hsv_template bgr565le2hsv
resize_nn_simd_helper_rgb5652hsv_template bgr565be2hsv

.macro resize_nn_simd_helper_rgb5652hsv_mask_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: int *sdiv_lut
    # a5: int *hdiv_lut
    # a6: uint8_t *hsv_min
    # a7: uint8_t *hsv_max

    addi sp, sp, -16
    sw   s0, 0(sp)
    sw   s1, 4(sp)
    addi s0, a6, 0
    addi s1, a7, 0

    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, t3, t4, t5, t6
    esp.vunzip.8 q0, q1
    \pix_cvt q2, a4, a5, s0, s1, q0, q1, t3, t4, t5, t6, q3, q4, q5, q6, q7
    esp.vst.128.ip q2, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0,  0(sp)
    lw   s1,  4(sp)
    addi sp, sp, 16
    ret
.endm

resize_nn_simd_helper_rgb5652hsv_mask_template rgb565le2hsv_mask0
resize_nn_simd_helper_rgb5652hsv_mask_template rgb565be2hsv_mask0
resize_nn_simd_helper_rgb5652hsv_mask_template bgr565le2hsv_mask0
resize_nn_simd_helper_rgb5652hsv_mask_template bgr565be2hsv_mask0
resize_nn_simd_helper_rgb5652hsv_mask_template rgb565le2hsv_mask1
resize_nn_simd_helper_rgb5652hsv_mask_template rgb565be2hsv_mask1
resize_nn_simd_helper_rgb5652hsv_mask_template bgr565le2hsv_mask1
resize_nn_simd_helper_rgb5652hsv_mask_template bgr565be2hsv_mask1
