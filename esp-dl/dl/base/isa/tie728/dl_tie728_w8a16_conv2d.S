#include "dl_tie728_s16.S"


// Sign-extend 16 packed int8 in filter_v into two int16 qregs.
// zero_v must be all-zero. After vzip: filter_v = low 8, filter_hi = high 8.
.macro tie728_w8a16_sext8 filter_v filter_hi zero_v
    EE.VCMP.LT.S8 \filter_hi, \filter_v, \zero_v
    EE.VZIP.8     \filter_v, \filter_hi
.endm


############################################################################################################################################################
####
#### tie728_w8a16_conv2d_11cn series
####
############################################################################################################################################################
.macro tie728_w8a16_conv2d_11c8 input_v0 filter_v filter_hi zero_v input_ptr filter_ptr c_div_x_1
    # input_ptr  += (c_div_x_1 + 1) * 16
    # filter_ptr += (c_div_x_1 + 1) * 64
    # Software-pipelined like dl_tie728_pointwise_conv1d_qacc_i16o16k8:
    # fused vsmulas load of the next 16 int8, then vcmp.lt.s8 + vzip.8 to
    # sign-extend. Last 8-channel group is peeled so input_ptr is not over-advanced.

    EE.VLD.128.IP \input_v0, \input_ptr, 16
    EE.VLD.128.IP \filter_v, \filter_ptr, 16
    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v

    loopgtz \c_div_x_1, 0f
    EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_v, \input_v0, 0
    EE.VSMULAS.S16.QACC         \filter_hi, \input_v0, 1
    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v

    EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_v, \input_v0, 2
    EE.VSMULAS.S16.QACC         \filter_hi, \input_v0, 3
    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v

    EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_v, \input_v0, 4
    EE.VSMULAS.S16.QACC         \filter_hi, \input_v0, 5
    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v

    EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_v, \input_v0, 6
    EE.VSMULAS.S16.QACC.LD.INCP \input_v0, \input_ptr, \filter_hi, \input_v0, 7
    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v
0:
    EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_v, \input_v0, 0
    EE.VSMULAS.S16.QACC         \filter_hi, \input_v0, 1
    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v

    EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_v, \input_v0, 2
    EE.VSMULAS.S16.QACC         \filter_hi, \input_v0, 3
    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v

    EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_v, \input_v0, 4
    EE.VSMULAS.S16.QACC         \filter_hi, \input_v0, 5
    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v

    EE.VSMULAS.S16.QACC         \filter_v, \input_v0, 6
    EE.VSMULAS.S16.QACC         \filter_hi, \input_v0, 7
.endm


.macro tie728_w8a16_conv2d_11cn_load_args args filter_ptr c_div_x_1 n_rs3 mac_shift
    l32i \n_rs3,      \args,  96  // output_channel_div_8
    l32i \mac_shift,  \args,  64  // mac_shift
    l32i \filter_ptr, \args,  48  // filter
    l32i \c_div_x_1,  \args, 100  // input_channel / 8 - 1
.endm


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_11cn_bias
    .type   dl_tie728_w8a16_conv2d_11cn_bias, @function
dl_tie728_w8a16_conv2d_11cn_bias:
    .align 4
    entry sp, 16

    # a2: output_ptr
    # a3: input_ptr
    # a4: args
    # a5: filter_ptr
    # a6: c_div_x_1
    # a7: n_rs3
    # a8: mac_shift
    # a11: bias_ptr
    # a15: moving_input_ptr
    # q2: zero for vcmp.lt.s8
    tie728_w8a16_conv2d_11cn_load_args a4, a5, a6, a7, a8
    l32i a11, a4, 68
    EE.ZERO.Q q2

.Lw8a16_11cn_bias_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_s16_conv2d_128b_vector_bias a11
    tie728_w8a16_conv2d_11c8 q0, q1, q3, q2, a15, a5, a6
    tie728_s16_vector_round_result q0, a8, a15, q3
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_11cn_bias_loop
    retw


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_11cn_bias_relu
    .type   dl_tie728_w8a16_conv2d_11cn_bias_relu, @function
dl_tie728_w8a16_conv2d_11cn_bias_relu:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_11cn_load_args a4, a5, a6, a7, a8
    l32i a11, a4, 68
    l32i a12, a4, 76
    l32i a13, a4, 84
    EE.ZERO.Q q2

.Lw8a16_11cn_bias_relu_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_s16_conv2d_128b_vector_bias a11
    tie728_w8a16_conv2d_11c8 q0, q1, q3, q2, a15, a5, a6
    tie728_s16_vector_round_result q0, a8, a15, q3
    tie728_s16_conv2d_relu q0, a12, a13
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_11cn_bias_relu_loop
    retw


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_11cn
    .type   dl_tie728_w8a16_conv2d_11cn, @function
dl_tie728_w8a16_conv2d_11cn:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_11cn_load_args a4, a5, a6, a7, a8
    EE.ZERO.Q q2

.Lw8a16_11cn_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_w8a16_conv2d_11c8 q0, q1, q3, q2, a15, a5, a6
    tie728_s16_vector_round_result q0, a8, a15, q3
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_11cn_loop
    retw


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_11cn_relu
    .type   dl_tie728_w8a16_conv2d_11cn_relu, @function
dl_tie728_w8a16_conv2d_11cn_relu:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_11cn_load_args a4, a5, a6, a7, a8
    l32i a12, a4, 76
    l32i a13, a4, 84
    EE.ZERO.Q q2

.Lw8a16_11cn_relu_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_w8a16_conv2d_11c8 q0, q1, q3, q2, a15, a5, a6
    tie728_s16_vector_round_result q0, a8, a15, q3
    tie728_s16_conv2d_relu q0, a12, a13
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_11cn_relu_loop
    retw


############################################################################################################################################################
####
#### tie728_w8a16_conv2d_33cn series
####
############################################################################################################################################################
.macro tie728_w8a16_conv2d_33c8 input_v0 filter_v filter_hi zero_v input_ptr filter_ptr c_div_x_1 dilation_x_offset dilation_y_offset
    tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_y_offset

    tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_y_offset

    tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
.endm


.macro tie728_w8a16_conv2d_hwcn_load_args args filter_ptr c_div_x_1 n_rs3 mac_shift dilation_x_offset dilation_y_offset
    tie728_w8a16_conv2d_11cn_load_args \args, \filter_ptr, \c_div_x_1, \n_rs3, \mac_shift
    l32i \dilation_x_offset, \args, 108
    l32i \dilation_y_offset, \args, 112
.endm


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_33cn_bias
    .type   dl_tie728_w8a16_conv2d_33cn_bias, @function
dl_tie728_w8a16_conv2d_33cn_bias:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_hwcn_load_args a4, a5, a6, a7, a8, a9, a10
    l32i a11, a4, 68
    EE.ZERO.Q q2

.Lw8a16_33cn_bias_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_s16_conv2d_128b_vector_bias a11
    tie728_w8a16_conv2d_33c8 q0, q1, q3, q2, a15, a5, a6, a9, a10
    tie728_s16_vector_round_result q0, a8, a15, q3
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_33cn_bias_loop
    retw


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_33cn_bias_relu
    .type   dl_tie728_w8a16_conv2d_33cn_bias_relu, @function
dl_tie728_w8a16_conv2d_33cn_bias_relu:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_hwcn_load_args a4, a5, a6, a7, a8, a9, a10
    l32i a11, a4, 68
    l32i a12, a4, 76
    l32i a13, a4, 84
    EE.ZERO.Q q2

.Lw8a16_33cn_bias_relu_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_s16_conv2d_128b_vector_bias a11
    tie728_w8a16_conv2d_33c8 q0, q1, q3, q2, a15, a5, a6, a9, a10
    tie728_s16_vector_round_result q0, a8, a15, q3
    tie728_s16_conv2d_relu q0, a12, a13
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_33cn_bias_relu_loop
    retw


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_33cn
    .type   dl_tie728_w8a16_conv2d_33cn, @function
dl_tie728_w8a16_conv2d_33cn:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_hwcn_load_args a4, a5, a6, a7, a8, a9, a10
    EE.ZERO.Q q2

.Lw8a16_33cn_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_w8a16_conv2d_33c8 q0, q1, q3, q2, a15, a5, a6, a9, a10
    tie728_s16_vector_round_result q0, a8, a15, q3
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_33cn_loop
    retw


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_33cn_relu
    .type   dl_tie728_w8a16_conv2d_33cn_relu, @function
dl_tie728_w8a16_conv2d_33cn_relu:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_hwcn_load_args a4, a5, a6, a7, a8, a9, a10
    l32i a12, a4, 76
    l32i a13, a4, 84
    EE.ZERO.Q q2

.Lw8a16_33cn_relu_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_w8a16_conv2d_33c8 q0, q1, q3, q2, a15, a5, a6, a9, a10
    tie728_s16_vector_round_result q0, a8, a15, q3
    tie728_s16_conv2d_relu q0, a12, a13
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_33cn_relu_loop
    retw


############################################################################################################################################################
####
#### tie728_w8a16_conv2d_hwcn series
####
############################################################################################################################################################
.macro tie728_w8a16_conv2d_hwc8 input_v0 filter_v filter_hi zero_v input_ptr filter_ptr c_div_x_1 dilation_x_offset dilation_y_offset filter_h filter_w args filter_offset_q
    l32i \filter_h, \args, 52
    1:
        l32i \filter_w, \args, 56
        beqi \filter_w, 1, 3f
        2:
            tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
            add \input_ptr, \input_ptr, \dilation_x_offset
            addi \filter_w, \filter_w, -1
            bgei \filter_w, 2, 2b
        3:
        tie728_w8a16_conv2d_11c8 \input_v0, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1
        EE.MOVI.32.A \filter_offset_q, \filter_w, 1
        add \filter_ptr, \filter_ptr, \filter_w
        add \input_ptr, \input_ptr, \dilation_y_offset
        addi \filter_h, \filter_h, -1
        bnez \filter_h, 1b
    EE.MOVI.32.A \filter_offset_q, \filter_h, 2
    add \filter_ptr, \filter_ptr, \filter_h
.endm


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_hwcn_bias
    .type   dl_tie728_w8a16_conv2d_hwcn_bias, @function
dl_tie728_w8a16_conv2d_hwcn_bias:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_hwcn_load_args a4, a5, a6, a7, a8, a9, a10
    l32i a12, a4, 60
    l32i a11, a4, 144
    EE.MOVI.32.Q q6, a12, 1
    EE.MOVI.32.Q q6, a11, 2
    l32i a13, a4, 68
    EE.ZERO.Q q2

.Lw8a16_hwcn_bias_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_s16_conv2d_128b_vector_bias a13
    tie728_w8a16_conv2d_hwc8 q0, q1, q3, q2, a15, a5, a6, a9, a10, a11, a12, a4, q6
    tie728_s16_vector_round_result q0, a8, a15, q3
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_hwcn_bias_loop
    retw


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_hwcn_bias_relu
    .type   dl_tie728_w8a16_conv2d_hwcn_bias_relu, @function
dl_tie728_w8a16_conv2d_hwcn_bias_relu:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_hwcn_load_args a4, a5, a6, a7, a8, a9, a10
    l32i a12, a4, 60
    l32i a11, a4, 144
    EE.MOVI.32.Q q6, a12, 1
    EE.MOVI.32.Q q6, a11, 2
    l32i a13, a4, 68
    l32i a14, a4, 76
    EE.ZERO.Q q2

.Lw8a16_hwcn_bias_relu_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_s16_conv2d_128b_vector_bias a13
    tie728_w8a16_conv2d_hwc8 q0, q1, q3, q2, a15, a5, a6, a9, a10, a11, a12, a4, q6
    l32i a11, a4, 84
    tie728_s16_vector_round_result q0, a8, a15, q3
    tie728_s16_conv2d_relu q0, a14, a11
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_hwcn_bias_relu_loop
    retw


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_hwcn
    .type   dl_tie728_w8a16_conv2d_hwcn, @function
dl_tie728_w8a16_conv2d_hwcn:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_hwcn_load_args a4, a5, a6, a7, a8, a9, a10
    l32i a12, a4, 60
    l32i a11, a4, 144
    EE.MOVI.32.Q q6, a12, 1
    EE.MOVI.32.Q q6, a11, 2
    EE.ZERO.Q q2

.Lw8a16_hwcn_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_w8a16_conv2d_hwc8 q0, q1, q3, q2, a15, a5, a6, a9, a10, a11, a12, a4, q6
    tie728_s16_vector_round_result q0, a8, a15, q3
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_hwcn_loop
    retw


    .align 4
    .text
    .global dl_tie728_w8a16_conv2d_hwcn_relu
    .type   dl_tie728_w8a16_conv2d_hwcn_relu, @function
dl_tie728_w8a16_conv2d_hwcn_relu:
    .align 4
    entry sp, 16

    tie728_w8a16_conv2d_hwcn_load_args a4, a5, a6, a7, a8, a9, a10
    l32i a12, a4, 60
    l32i a11, a4, 144
    EE.MOVI.32.Q q6, a12, 1
    EE.MOVI.32.Q q6, a11, 2
    l32i a13, a4, 76
    l32i a14, a4, 84
    EE.ZERO.Q q2

.Lw8a16_hwcn_relu_loop:
    mov a15, a3
    EE.ZERO.QACC
    tie728_w8a16_conv2d_hwc8 q0, q1, q3, q2, a15, a5, a6, a9, a10, a11, a12, a4, q6
    tie728_s16_vector_round_result q0, a8, a15, q3
    tie728_s16_conv2d_relu q0, a13, a14
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    bnez a7, .Lw8a16_hwcn_relu_loop
    retw
