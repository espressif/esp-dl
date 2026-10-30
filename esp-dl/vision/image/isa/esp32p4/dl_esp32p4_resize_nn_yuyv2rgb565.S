#include "dl_esp32p4_color_common.S"

# q0/q1 are the 565 halves. Temps are a4, a5, t3-t6, then q2-q7.
.macro resize_nn_simd_helper_yuyv2rgb565_template pix_cvt
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
    li a4, 14
    resize_nn_\pix_cvt q0, q1, a0, a1, a4, a5, t3, t4, t5, t6, q2, q3, q4, q5, q6, q7
    esp.vzip.8 q0, q1
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q1, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm

resize_nn_simd_helper_yuyv2rgb565_template yuyv2rgb565le
resize_nn_simd_helper_yuyv2rgb565_template yuyv2rgb565be
resize_nn_simd_helper_yuyv2rgb565_template yuyv2bgr565le
resize_nn_simd_helper_yuyv2rgb565_template yuyv2bgr565be
resize_nn_simd_helper_yuyv2rgb565_template uyvy2rgb565le
resize_nn_simd_helper_yuyv2rgb565_template uyvy2rgb565be
resize_nn_simd_helper_yuyv2rgb565_template uyvy2bgr565le
resize_nn_simd_helper_yuyv2rgb565_template uyvy2bgr565be
