#include "dl_define_private.hpp"

#if CONFIG_XTENSA_MAC16_BOOST

############################################################################################################################################################
####
#### depthwise conv with MAC16, int16
####
############################################################################################################################################################
# void dl_xtensa_s16_depthwise_conv2d(const int16_t *x, const uint32_t *wt, int pairs, const XtensaDwParams *prm,
#                                     int16_t *out, const int32_t *init)
#
# One output pixel, all 2 * pairs channels. MAC16 has a single 40-bit accumulator, so the channel pair (c, c + 1)
# sharing each input and weight word takes two passes over the taps, on the low and on the high halves. The taps of one
# pixel are tap_stride bytes apart and wt[pair * taps + t] = (w[t][c + 1], w[t][c]). init holds bias + rounding half
# per channel as (lo, hi); the output is clamps((acc >> sar) max relu_lo). prm: taps, tap_stride, sar, relu_lo.
.macro dl_xtensa_s16_depthwise_conv2d_pass q, init_lo, init_hi, out_off, label
    l32i a12, a7, \init_lo
    l32i a13, a7, \init_hi
    wsr a12, acclo
    wsr a13, acchi
    mov a14, a2
    mov a15, a3
    loopgtz a8, \label\()f
    l32i a12, a14, 0
    l32i a13, a15, 0
    add a14, a14, a9
    addi a15, a15, 4
    mula.aa.\q a12, a13
\label:
    rsr a12, acclo
    rsr a13, acchi
    sext a13, a13, 7
    src a12, a13, a12
    max a12, a12, a11
    clamps a12, a12, 15
    s16i a12, a6, \out_off
.endm

    .section .iram1.dl_xtensa_s16_depthwise_conv2d, "ax", @progbits
    .global dl_xtensa_s16_depthwise_conv2d
    .type dl_xtensa_s16_depthwise_conv2d, @function
    .align 4
dl_xtensa_s16_depthwise_conv2d:
    entry sp, 32
    l32i a8, a5, 0
    l32i a9, a5, 4
    l32i a10, a5, 8
    l32i a11, a5, 12
    ssr a10
1:
    dl_xtensa_s16_depthwise_conv2d_pass ll, 0, 4, 0, 2
    dl_xtensa_s16_depthwise_conv2d_pass hh, 8, 12, 2, 3
    mov a3, a15
    addi a2, a2, 4
    addi a6, a6, 4
    addi a7, a7, 16
    addi a4, a4, -1
    bnez a4, 1b
    retw
    .size dl_xtensa_s16_depthwise_conv2d, .-dl_xtensa_s16_depthwise_conv2d

#endif
