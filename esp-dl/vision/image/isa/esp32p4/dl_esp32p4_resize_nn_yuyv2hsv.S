#include "dl_esp32p4_color_common.S"

# q0/q1/q2 are H/S/V. a4/a5 are the LUTs.
# Temps are t3-t6, s0, s1, s8-s11, then q3-q7.
.macro resize_nn_simd_helper_yuyv2hsv_template pix_cvt
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

    addi sp, sp, -32
    sw   s0,  0(sp)
    sw   s1,  4(sp)
    sw   s8,  8(sp)
    sw   s9,  12(sp)
    sw   s10, 16(sp)
    sw   s11, 20(sp)
    blez a3, .Lend\@
    li t3, 14
.Lloop\@:
    resize_nn_\pix_cvt q0, q1, q2, a0, a1, t3, t4, t5, t6, s0, s1, a4, a5, s8, s9, s10, s11, q3, q4, q5, q6, q7
    esp.vzipt.8 q0, q1, q2
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q1, a2, 16
    esp.vst.128.ip q2, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0,  0(sp)
    lw   s1,  4(sp)
    lw   s8,  8(sp)
    lw   s9,  12(sp)
    lw   s10, 16(sp)
    lw   s11, 20(sp)
    addi sp, sp, 32
    ret
.endm

resize_nn_simd_helper_yuyv2hsv_template yuyv2hsv
resize_nn_simd_helper_yuyv2hsv_template uyvy2hsv

# q0 is the mask byte. a6/a7 are copied to s8/s9.
# a_t reuses t3-t6 after the gather.
.macro resize_nn_simd_helper_yuyv2hsv_mask_template pix_cvt
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
    sw   s8, 8(sp)
    sw   s9, 12(sp)
    addi s8, a6, 0
    addi s9, a7, 0
    blez a3, .Lend\@
.Lloop\@:
    li t3, 14
    resize_nn_\pix_cvt q0, a0, a1, t3, t4, t5, t6, s0, s1, a4, a5, s8, s9, t3, t4, t5, t6, q1, q2, q3, q4, q5, q6, q7
    esp.vst.128.ip q0, a2, 16
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

resize_nn_simd_helper_yuyv2hsv_mask_template yuyv2hsv_mask0
resize_nn_simd_helper_yuyv2hsv_mask_template uyvy2hsv_mask0
resize_nn_simd_helper_yuyv2hsv_mask_template yuyv2hsv_mask1
resize_nn_simd_helper_yuyv2hsv_mask_template uyvy2hsv_mask1
