#include "dl_esp32p4_color_common.S"

.macro resize_nn_simd_helper_hsv2hsv_mask_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: uint8_t *hsv_min
    # a5: uint8_t *hsv_max

    addi sp, sp, -16
    sw   s0, 0(sp)
    sw   s1, 4(sp)
    sw   s8, 8(sp)
    sw   s9, 12(sp)
    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_3chn_128 q0, q1, q2, a0, a1, q3, q4, q5, t3, t4, t5, t6, s0, s1, s8, s9
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, a4, a5, q0, q1, q2, q4
    esp.vst.128.ip q3, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    lw   s8, 8(sp)
    lw   s9, 12(sp)
    addi sp, sp, 16
    ret
.endm

resize_nn_simd_helper_hsv2hsv_mask_template hsv2hsv_mask0
resize_nn_simd_helper_hsv2hsv_mask_template hsv2hsv_mask1
