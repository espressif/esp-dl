#include "dl_esp32p4_color_common.S"

.macro resize_nn_simd_helper_rgb8882rgb888_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n

    addi sp, sp, -16
    sw   s0, 0(sp)
    sw   s1, 4(sp)
    blez a3, .Lend\@
.Lloop\@:
.ifc "\pix_cvt", "rgb8882rgb888"
    resize_nn_load_store_3chn_128 a2, a0, a1, q0, q1, q2, a4, a5, t3, t4, t5, t6, s0, s1
.endif
.ifc "\pix_cvt", "rgb8882bgr888"
    resize_nn_load_3chn_128 q0, q1, q2, a0, a1, q3, q4, q5, a4, a5, t3, t4, t5, t6, s0, s1
    esp.vunzipt.8 q0, q1, q2
    esp.vzipt.8 q2, q1, q0
    esp.vst.128.ip q2, a2, 16
    esp.vst.128.ip q1, a2, 16
    esp.vst.128.ip q0, a2, 16
.endif
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    addi sp, sp, 16
    ret
.endm
resize_nn_simd_helper_rgb8882rgb888_template rgb8882rgb888
resize_nn_simd_helper_rgb8882rgb888_template rgb8882bgr888

.macro resize_nn_simd_helper_rgb8882rgb888_qint8_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: int *lut

    addi sp, sp, -16
    sw   s0, 0(sp)
    sw   s1, 4(sp)
    sw   s8, 8(sp)
    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_3chn_128 q0, q1, q2, a0, a1, q3, q4, q5, a5, t3, t4, t5, t6, s0, s1, s8
.ifc "\pix_cvt", "rgb8882rgb888_qint8"
    esp.vunzipt.8 q0, q1, q2
    norm_quant_3chn_8 q0, q1, q2, a4, q3, q4, q5
    esp.vzipt.8 q0, q1, q2
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q1, a2, 16
    esp.vst.128.ip q2, a2, 16
.endif
.ifc "\pix_cvt", "rgb8882bgr888_qint8"
    esp.vunzipt.8 q0, q1, q2
    norm_quant_3chn_8 q2, q1, q0, a4, q3, q4, q5
    esp.vzipt.8 q2, q1, q0
    esp.vst.128.ip q2, a2, 16
    esp.vst.128.ip q1, a2, 16
    esp.vst.128.ip q0, a2, 16
.endif
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    lw   s8, 8(sp)
    addi sp, sp, 16
    ret
.endm
resize_nn_simd_helper_rgb8882rgb888_qint8_template rgb8882rgb888_qint8
resize_nn_simd_helper_rgb8882rgb888_qint8_template rgb8882bgr888_qint8

.macro resize_nn_simd_helper_rgb8882rgb888_qint16_template pix_cvt
    .balign 4
    .text
    .global resize_nn_simd_helper_\pix_cvt
    .type   resize_nn_simd_helper_\pix_cvt, @function
resize_nn_simd_helper_\pix_cvt:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: int *lut

    addi sp, sp, -16
    sw   s0, 0(sp)
    sw   s1, 4(sp)
    sw   s8, 8(sp)
    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_3chn_128 q0, q1, q2, a0, a1, q3, q4, q5, a5, t3, t4, t5, t6, s0, s1, s8
.ifc "\pix_cvt", "rgb8882rgb888_qint16"
    esp.vunzipt.8 q0, q1, q2
    norm_quant_3chn_16 q3, q4, q5, q0, q1, q2, a4, q6, q7
    esp.vzipt.16 q3, q5, q1
    esp.vzipt.16 q4, q0, q2
    esp.vst.128.ip q3, a2, 16
    esp.vst.128.ip q5, a2, 16
    esp.vst.128.ip q1, a2, 16
    esp.vst.128.ip q4, a2, 16
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q2, a2, 16
.endif
.ifc "\pix_cvt", "rgb8882bgr888_qint16"
    esp.vunzipt.8 q0, q1, q2
    norm_quant_3chn_16 q3, q4, q5, q2, q1, q0, a4, q6, q7
    esp.vzipt.16 q3, q5, q1
    esp.vzipt.16 q4, q2, q0
    esp.vst.128.ip q3, a2, 16
    esp.vst.128.ip q5, a2, 16
    esp.vst.128.ip q1, a2, 16
    esp.vst.128.ip q4, a2, 16
    esp.vst.128.ip q2, a2, 16
    esp.vst.128.ip q0, a2, 16
.endif
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    lw   s8, 8(sp)
    addi sp, sp, 16
    ret
.endm
resize_nn_simd_helper_rgb8882rgb888_qint16_template rgb8882rgb888_qint16
resize_nn_simd_helper_rgb8882rgb888_qint16_template rgb8882bgr888_qint16
