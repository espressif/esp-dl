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

.macro resize_nn_simd_helper_rgb5652rgb888_template pix_cvt
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
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, a4, a5, t3, t4
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q3, q4, q0, q1, a4, a5
    esp.vzipt.8 q2, q3, q4
    esp.vst.128.ip q2, a2, 16
    esp.vst.128.ip q3, a2, 16
    esp.vst.128.ip q4, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm
resize_nn_simd_helper_rgb5652rgb888_template rgb565le2rgb888
resize_nn_simd_helper_rgb5652rgb888_template rgb565be2rgb888
resize_nn_simd_helper_rgb5652rgb888_template rgb565le2bgr888
resize_nn_simd_helper_rgb5652rgb888_template rgb565be2bgr888

.macro resize_nn_simd_helper_rgb5652rgb888_qint8_template pix_cvt
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

    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, a5, t3, t4, t5
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q3, q4, a4, q0, q1, a5, t3, q5
    esp.vzipt.8 q2, q3, q4
    esp.vst.128.ip q2, a2, 16
    esp.vst.128.ip q3, a2, 16
    esp.vst.128.ip q4, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm
resize_nn_simd_helper_rgb5652rgb888_qint8_template rgb565le2rgb888_qint8
resize_nn_simd_helper_rgb5652rgb888_qint8_template rgb565be2rgb888_qint8
resize_nn_simd_helper_rgb5652rgb888_qint8_template rgb565le2bgr888_qint8
resize_nn_simd_helper_rgb5652rgb888_qint8_template rgb565be2bgr888_qint8

.macro resize_nn_simd_helper_rgb5652rgb888_qint16_template pix_cvt
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

    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, a5, t3, t4, t5
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q3, q4, q5, q6, q7, a4, q0, q1, a5, t3
    esp.vzipt.16 q2, q4, q6
    esp.vzipt.16 q3, q5, q7
    esp.vst.128.ip q2, a2, 16
    esp.vst.128.ip q4, a2, 16
    esp.vst.128.ip q6, a2, 16
    esp.vst.128.ip q3, a2, 16
    esp.vst.128.ip q5, a2, 16
    esp.vst.128.ip q7, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm
resize_nn_simd_helper_rgb5652rgb888_qint16_template rgb565le2rgb888_qint16
resize_nn_simd_helper_rgb5652rgb888_qint16_template rgb565be2rgb888_qint16
resize_nn_simd_helper_rgb5652rgb888_qint16_template rgb565le2bgr888_qint16
resize_nn_simd_helper_rgb5652rgb888_qint16_template rgb565be2bgr888_qint16

.macro resize_nn_simd_helper_rgb5652gray_template pix_cvt
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
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, a4, a5, t3, t4
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q0, q1, a4, a5, q3, q4, q5, q6, q7
    esp.vst.128.ip q2, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm
resize_nn_simd_helper_rgb5652gray_template rgb565le2gray
resize_nn_simd_helper_rgb5652gray_template rgb565be2gray
resize_nn_simd_helper_rgb5652gray_template bgr565le2gray
resize_nn_simd_helper_rgb5652gray_template bgr565be2gray

.macro resize_nn_simd_helper_rgb5652gray_qint8_template pix_cvt
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

    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, a5, t3, t4, t5
    esp.vunzip.8 q0, q1
    \pix_cvt q2, a4, q0, q1, a5, t3, q3, q4, q5, q6, q7
    esp.vst.128.ip q2, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm
resize_nn_simd_helper_rgb5652gray_qint8_template rgb565le2gray_qint8
resize_nn_simd_helper_rgb5652gray_qint8_template rgb565be2gray_qint8
resize_nn_simd_helper_rgb5652gray_qint8_template bgr565le2gray_qint8
resize_nn_simd_helper_rgb5652gray_qint8_template bgr565be2gray_qint8

.macro resize_nn_simd_helper_rgb5652gray_qint16_template pix_cvt
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

    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_2chn_128 q0, q1, a0, a1, q2, q3, q4, a5, t3, t4, t5
    esp.vunzip.8 q0, q1
    \pix_cvt q2, q3, a4, q0, q1, a5, t3, q4, q5, q6, q7
    esp.vst.128.ip q2, a2, 16
    esp.vst.128.ip q3, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    ret
.endm
resize_nn_simd_helper_rgb5652gray_qint16_template rgb565le2gray_qint16
resize_nn_simd_helper_rgb5652gray_qint16_template rgb565be2gray_qint16
resize_nn_simd_helper_rgb5652gray_qint16_template bgr565le2gray_qint16
resize_nn_simd_helper_rgb5652gray_qint16_template bgr565be2gray_qint16

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

.macro resize_nn_simd_helper_rgb8882gray_template pix_cvt
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
    resize_nn_load_3chn_128 q0, q1, q2, a0, a1, q3, q4, q5, a4, a5, t3, t4, t5, t6, s0, s1
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, q0, q1, q2, a4, q4, q5, q6, q7
    esp.vst.128.ip q3, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    addi sp, sp, 16
    ret
.endm
resize_nn_simd_helper_rgb8882gray_template rgb8882gray
resize_nn_simd_helper_rgb8882gray_template bgr8882gray

.macro resize_nn_simd_helper_rgb8882gray_qint8_template pix_cvt
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
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, a4, q0, q1, q2, a5, q4, q5, q6, q7
    esp.vst.128.ip q3, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    lw   s8, 8(sp)
    addi sp, sp, 16
    ret
.endm
resize_nn_simd_helper_rgb8882gray_qint8_template rgb8882gray_qint8
resize_nn_simd_helper_rgb8882gray_qint8_template bgr8882gray_qint8

.macro resize_nn_simd_helper_rgb8882gray_qint16_template pix_cvt
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
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, q4, a4, q0, q1, q2, a5, q5, q6, q7
    esp.vst.128.ip q3, a2, 16
    esp.vst.128.ip q4, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    lw   s8, 8(sp)
    addi sp, sp, 16
    ret
.endm
resize_nn_simd_helper_rgb8882gray_qint16_template rgb8882gray_qint16
resize_nn_simd_helper_rgb8882gray_qint16_template bgr8882gray_qint16

.macro resize_nn_simd_helper_rgb8882rgb565_template pix_cvt
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
    resize_nn_load_3chn_128 q0, q1, q2, a0, a1, q3, q4, q5, a4, a5, t3, t4, t5, t6, s0, s1
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, q4, q0, q1, q2, a4, a5
    esp.vzip.8 q3, q4
    esp.vst.128.ip q3, a2, 16
    esp.vst.128.ip q4, a2, 16
    addi a3, a3, -1
    bnez a3, .Lloop\@
.Lend\@:
    lw   s0, 0(sp)
    lw   s1, 4(sp)
    addi sp, sp, 16
    ret
.endm
resize_nn_simd_helper_rgb8882rgb565_template rgb8882rgb565le
resize_nn_simd_helper_rgb8882rgb565_template bgr8882rgb565le
resize_nn_simd_helper_rgb8882rgb565_template rgb8882rgb565be
resize_nn_simd_helper_rgb8882rgb565_template bgr8882rgb565be

    .balign 4
    .text
    .global resize_nn_simd_helper_gray2gray
    .type   resize_nn_simd_helper_gray2gray, @function
resize_nn_simd_helper_gray2gray:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n

    blez a3, 1f
0:
    resize_nn_load_store_1chn_128 a2, a0, a1, q0, q1, q2, a4, a5, t3, t4
    addi a3, a3, -1
    bnez a3, 0b
1:
    ret

    .balign 4
    .text
    .global resize_nn_simd_helper_gray2gray_qint8
    .type   resize_nn_simd_helper_gray2gray_qint8, @function
resize_nn_simd_helper_gray2gray_qint8:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: int *lut

    blez a3, 1f
0:
    resize_nn_load_1chn_128 q0, a0, a1, q1, q2, q3, a5, t3, t4, t5
    norm_quant_1chn_8 q0, a4, q1, q2, q3
    esp.vst.128.ip q0, a2, 16
    addi a3, a3, -1
    bnez a3, 0b
1:
    ret

    .balign 4
    .text
    .global resize_nn_simd_helper_gray2gray_qint16
    .type   resize_nn_simd_helper_gray2gray_qint16, @function
resize_nn_simd_helper_gray2gray_qint16:

    # a0: uint8_t **src
    # a1: int *offsets
    # a2: uint8_t *dst
    # a3: int n
    # a4: int *lut

    blez a3, 1f
0:
    resize_nn_load_1chn_128 q1, a0, a1, q0, q2, q3, a5, t3, t4, t5
    norm_quant_1chn_16 q0, q1, a4, q2, q3
    esp.vst.128.ip q0, a2, 16
    esp.vst.128.ip q1, a2, 16
    addi a3, a3, -1
    bnez a3, 0b
1:
    ret

.macro resize_nn_simd_helper_rgb8882hsv_template pix_cvt
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

    addi sp, sp, -16
    sw   s0, 0(sp)
    sw   s1, 4(sp)
    sw   s8, 8(sp)
    sw   s9, 12(sp)
    blez a3, .Lend\@
.Lloop\@:
    resize_nn_load_3chn_128 q0, q1, q2, a0, a1, q3, q4, q5, t3, t4, t5, t6, s0, s1, s8, s9
    esp.vunzipt.8 q0, q1, q2
    \pix_cvt q3, q4, q5, a4, a5, q0, q1, q2, t3, t4, t5, t6, q6, q7
    esp.vzipt.8 q3, q4, q5
    esp.vst.128.ip q3, a2, 16
    esp.vst.128.ip q4, a2, 16
    esp.vst.128.ip q5, a2, 16
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

resize_nn_simd_helper_rgb8882hsv_template rgb8882hsv
resize_nn_simd_helper_rgb8882hsv_template bgr8882hsv

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
