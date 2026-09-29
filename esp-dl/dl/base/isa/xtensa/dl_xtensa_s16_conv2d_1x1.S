#include "dl_define_private.hpp"

#if CONFIG_XTENSA_MAC16_BOOST

############################################################################################################################################################
####
#### 1x1 conv with MAC16, int16 activations
####
############################################################################################################################################################
# void dl_xtensa_s16_conv2d_1x1(const int16_t *w, const int16_t *const *xs, int n_px, const XtensaConvParams *prm,
#                               int16_t *const *out, int out_off)
# void dl_xtensa_w8a16_conv2d_1x1(const int8_t *w, const int16_t *const *xs, int n_px, const XtensaConvParams *prm,
#                                 int16_t *const *out, int out_off)
#
# One output channel for each of the n_px pixels: out[p][out_off / 2] = requant(sum_{i < C} w[i] * x_p[i]),
# C = 4 * (loops + 1). prm: loops, init_lo, init_hi, sar, relu_lo, wq (see dl_xtensa_conv.cpp).
#
# MAC16 has a single 40-bit accumulator, so each output is its own dot product, and an m register written by ldinc can
# only be read 3 instructions later without stalling. Both are met by a staircase: input words Y_j alternate in m0/m1,
# weight words W_k in m2/m3, and the MAC stream is
#   (Y_0, W_0), (Y_1, W_0), (Y_1, W_1), (Y_2, W_1), ...
# every register being reloaded by the instruction that reads it for the last time. With Y_j = (x[2j], x[2j - 1])
# (high, low half; x[-1] = x[C] = 0, j <= C / 2) the products are
#   int16 weights W_k = (w[2k + 1], w[2k]):               hl, lh, hl, lh
#   int8 weights b0..b3 of each group of 4 expanded into
#   B = (256 * b2, 256 * b0), A = (256 * b3, 256 * b1):   hl, hh, ll, lh, with a 256x sum.
# The int8 kernel expands the row w into wq itself. The accumulator starts at the bias plus the rounding half (256x for
# int8 weights), so the output is clamps((acc >> sar) max relu_lo).
.macro dl_xtensa_conv2d_1x1 name, w8, s0, s1, s2, s3
    .section .iram1.\name, "ax", @progbits
    .global \name
    .type \name, @function
    .align 4
\name:
    entry sp, 32
    l32i a8, a5, 0
.if \w8
    l32i a13, a5, 20
    movi a14, 0xff
    slli a14, a14, 8
    slli a15, a14, 16
    or a14, a14, a15
    addi a15, a8, 1
    mov a11, a13
    loopgtz a15, 3f
    l32i a9, a2, 0
    addi a2, a2, 4
    slli a10, a9, 8
    and a9, a9, a14
    and a10, a10, a14
    s32i a10, a11, 0
    s32i a9, a11, 4
    addi a11, a11, 8
3:
    mov a2, a13
.endif
    l32i a9, a5, 4
    l32i a10, a5, 8
    l32i a11, a5, 12
    l32i a12, a5, 16
    ssr a11
1:
    l32i a13, a3, 0
    addi a3, a3, 4
    addi a14, a2, -4
    addi a13, a13, -4
    wsr a9, acclo
    wsr a10, acchi
    ldinc m0, a13
    ldinc m2, a14
    ldinc m1, a13
    ldinc m3, a14
    loopgtz a8, 2f
    mula.dd.\s0\().ldinc m0, a13, m0, m2
    mula.dd.\s1\().ldinc m2, a14, m1, m2
    mula.dd.\s2\().ldinc m1, a13, m1, m3
    mula.dd.\s3\().ldinc m3, a14, m0, m3
2:
    mula.dd.\s0\().ldinc m0, a13, m0, m2
    mula.dd.\s1 m1, m2
    mula.dd.\s2 m1, m3
    mula.dd.\s3 m0, m3
    rsr a15, acclo
    rsr a11, acchi
    l32i a13, a6, 0
    addi a6, a6, 4
    sext a11, a11, 7
    src a15, a11, a15
    add a13, a13, a7
    max a15, a15, a12
    clamps a15, a15, 15
    s16i a15, a13, 0
    addi a4, a4, -1
    bnez a4, 1b
    retw
    .size \name, .-\name
.endm

    dl_xtensa_conv2d_1x1 dl_xtensa_s16_conv2d_1x1, 0, hl, lh, hl, lh
    dl_xtensa_conv2d_1x1 dl_xtensa_w8a16_conv2d_1x1, 1, hl, hh, ll, lh

#endif
