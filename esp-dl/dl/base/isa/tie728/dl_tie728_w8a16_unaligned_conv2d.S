#include "dl_tie728_s16_unaligned.S"
#include "dl_tie728_s16.S"

# ESP32-S3 W8A16 unaligned conv2d: int8 filter, int16 activations.
# Handles input/output pointers that are not 16-byte aligned, and channel
# counts that are not multiples of 8. Filter layout matches the aligned
# kernels: packed [N/8][H][W][C][8] plus remainder [N%8][H][W][C].
#
# When C is odd, each 8-OC tap is 8-byte aligned rather than 16-byte.
# EE.VLD.128 forces 16-byte alignment, so the a8 path uses 64-bit halves.


.macro tie728_w8a16_sext8 filter_v, filter_hi, zero_v
    EE.VCMP.LT.S8 \filter_hi, \filter_v, \zero_v
    EE.VZIP.8     \filter_v, \filter_hi
.endm


.macro tie728_w8a16_ld_c8x2 dest0, dest1, zero_v, filter_ptr
    EE.VLD.L.64.IP \dest0, \filter_ptr, 8
    EE.VLD.H.64.IP \dest0, \filter_ptr, 8
    tie728_w8a16_sext8 \dest0, \dest1, \zero_v
.endm


.macro tie728_w8a16_ld_c8 dest0, dest1, zero_v, filter_ptr
    EE.VLD.L.64.IP \dest0, \filter_ptr, 8
    tie728_w8a16_sext8 \dest0, \dest1, \zero_v
.endm

.macro tie728_w8a16_unaligned_conv2d_11c8_a16 input_v, input_front, input_back, filter_v, filter_hi, zero_v, input_ptr, filter_ptr, c_div_x_1, c_remainder, temp
    EE.LD.128.USAR.IP \input_front, \input_ptr, 16

    blti \c_div_x_1, 0, .L_rem_\@
        EE.LD.128.USAR.IP \input_back, \input_ptr, 16
        EE.VLD.128.IP \filter_v, \filter_ptr, 16
        tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v

        loopgtz \c_div_x_1, .L_last_\@
            EE.SRC.Q.QUP \input_v, \input_front, \input_back
            EE.LD.128.USAR.IP \input_back, \input_ptr, 16

            EE.VSMULAS.S16.QACC         \filter_v, \input_v, 0
            EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_hi, \input_v, 1
            tie728_w8a16_sext8          \filter_v, \filter_hi, \zero_v

            EE.VSMULAS.S16.QACC         \filter_v, \input_v, 2
            EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_hi, \input_v, 3
            tie728_w8a16_sext8          \filter_v, \filter_hi, \zero_v

            EE.VSMULAS.S16.QACC         \filter_v, \input_v, 4
            EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_hi, \input_v, 5
            tie728_w8a16_sext8          \filter_v, \filter_hi, \zero_v

            EE.VSMULAS.S16.QACC         \filter_v, \input_v, 6
            EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_hi, \input_v, 7
            tie728_w8a16_sext8          \filter_v, \filter_hi, \zero_v
        .L_last_\@:
        EE.SRC.Q.QUP \input_v, \input_front, \input_back
        EE.VSMULAS.S16.QACC         \filter_v, \input_v, 0
        EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_hi, \input_v, 1
        tie728_w8a16_sext8          \filter_v, \filter_hi, \zero_v

        EE.VSMULAS.S16.QACC         \filter_v, \input_v, 2
        EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_hi, \input_v, 3
        tie728_w8a16_sext8          \filter_v, \filter_hi, \zero_v

        EE.VSMULAS.S16.QACC         \filter_v, \input_v, 4
        EE.VSMULAS.S16.QACC.LD.INCP \filter_v, \filter_ptr, \filter_hi, \input_v, 5
        tie728_w8a16_sext8          \filter_v, \filter_hi, \zero_v

        EE.VSMULAS.S16.QACC \filter_v, \input_v, 6
        EE.VSMULAS.S16.QACC \filter_hi, \input_v, 7
        beqz \c_remainder, .L_done_\@

    .L_rem_\@:
        EE.LD.128.USAR.XP \input_back, \input_ptr, \c_remainder
        EE.SRC.Q.QUP \input_v, \input_front, \input_back

        movi \temp, 8
        blt  \c_remainder, \temp, .L_k3_\@
            EE.VLD.128.IP \filter_v, \filter_ptr, 16
            tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v

            movi \temp, 12
            blt  \c_remainder, \temp, .L_k5_\@
                movi \temp, 14
                blt  \c_remainder, \temp, .L_k6_\@
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    EE.VLD.128.IP \filter_v, \filter_ptr, 16
                    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 3
                    EE.VLD.128.IP \filter_v, \filter_ptr, 16
                    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 4
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 5
                    tie728_w8a16_ld_c8 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 6
                    j .L_done_\@

                .L_k6_\@:
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    EE.VLD.128.IP \filter_v, \filter_ptr, 16
                    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 3
                    EE.VLD.128.IP \filter_v, \filter_ptr, 16
                    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 4
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 5
                    j .L_done_\@

            .L_k5_\@:
                movi \temp, 10
                blt  \c_remainder, \temp, .L_k4_\@
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    EE.VLD.128.IP \filter_v, \filter_ptr, 16
                    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 3
                    tie728_w8a16_ld_c8 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 4
                    j .L_done_\@

                .L_k4_\@:
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    EE.VLD.128.IP \filter_v, \filter_ptr, 16
                    tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 3
                    j .L_done_\@

        .L_k3_\@:
            movi \temp, 4
            blt  \c_remainder, \temp, .L_k1_\@
                EE.VLD.128.IP \filter_v, \filter_ptr, 16
                tie728_w8a16_sext8 \filter_v, \filter_hi, \zero_v

                movi \temp, 6
                blt  \c_remainder, \temp, .L_k2_\@
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    tie728_w8a16_ld_c8 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
                    j .L_done_\@

                .L_k2_\@:
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    j .L_done_\@

            .L_k1_\@:
                tie728_w8a16_ld_c8 \filter_v, \filter_hi, \zero_v, \filter_ptr
                EE.VSMULAS.S16.QACC \filter_v, \input_v, 0

    .L_done_\@:
        addi \input_ptr, \input_ptr, -16
.endm

.macro tie728_w8a16_unaligned_conv2d_11c8_a8 input_v, input_front, input_back, filter_v, filter_hi, zero_v, input_ptr, filter_ptr, c_div_x_1, c_remainder, temp
    EE.LD.128.USAR.IP \input_front, \input_ptr, 16

    blti \c_div_x_1, 0, .L_rem_\@
        EE.LD.128.USAR.IP \input_back, \input_ptr, 16
        tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr

        loopgtz \c_div_x_1, .L_last_\@
            EE.SRC.Q.QUP \input_v, \input_front, \input_back
            EE.LD.128.USAR.IP \input_back, \input_ptr, 16

            EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
            EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
            tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr

            EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
            EE.VSMULAS.S16.QACC \filter_hi, \input_v, 3
            tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr

            EE.VSMULAS.S16.QACC \filter_v, \input_v, 4
            EE.VSMULAS.S16.QACC \filter_hi, \input_v, 5
            tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr

            EE.VSMULAS.S16.QACC \filter_v, \input_v, 6
            EE.VSMULAS.S16.QACC \filter_hi, \input_v, 7
            tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr
        .L_last_\@:
        EE.SRC.Q.QUP \input_v, \input_front, \input_back
        EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
        EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
        tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr

        EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
        EE.VSMULAS.S16.QACC \filter_hi, \input_v, 3
        tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr

        EE.VSMULAS.S16.QACC \filter_v, \input_v, 4
        EE.VSMULAS.S16.QACC \filter_hi, \input_v, 5
        tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr

        EE.VSMULAS.S16.QACC \filter_v, \input_v, 6
        EE.VSMULAS.S16.QACC \filter_hi, \input_v, 7
        beqz \c_remainder, .L_done_\@

    .L_rem_\@:
        EE.LD.128.USAR.XP \input_back, \input_ptr, \c_remainder
        EE.SRC.Q.QUP \input_v, \input_front, \input_back

        movi \temp, 8
        blt  \c_remainder, \temp, .L_k3_\@
            tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr

            movi \temp, 12
            blt  \c_remainder, \temp, .L_k5_\@
                movi \temp, 14
                blt  \c_remainder, \temp, .L_k6_\@
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 3
                    tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 4
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 5
                    tie728_w8a16_ld_c8 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 6
                    j .L_done_\@

                .L_k6_\@:
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 3
                    tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 4
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 5
                    j .L_done_\@

            .L_k5_\@:
                movi \temp, 10
                blt  \c_remainder, \temp, .L_k4_\@
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 3
                    tie728_w8a16_ld_c8 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 4
                    j .L_done_\@

                .L_k4_\@:
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 3
                    j .L_done_\@

        .L_k3_\@:
            movi \temp, 4
            blt  \c_remainder, \temp, .L_k1_\@
                tie728_w8a16_ld_c8x2 \filter_v, \filter_hi, \zero_v, \filter_ptr

                movi \temp, 6
                blt  \c_remainder, \temp, .L_k2_\@
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    tie728_w8a16_ld_c8 \filter_v, \filter_hi, \zero_v, \filter_ptr
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 2
                    j .L_done_\@

                .L_k2_\@:
                    EE.VSMULAS.S16.QACC \filter_v, \input_v, 0
                    EE.VSMULAS.S16.QACC \filter_hi, \input_v, 1
                    j .L_done_\@

            .L_k1_\@:
                tie728_w8a16_ld_c8 \filter_v, \filter_hi, \zero_v, \filter_ptr
                EE.VSMULAS.S16.QACC \filter_v, \input_v, 0

    .L_done_\@:
        addi \input_ptr, \input_ptr, -16
.endm


.macro tie728_w8a16_unaligned_conv2d_11c8 input_v, input_front, input_back, filter_v, filter_hi, zero_v, input_ptr, filter_ptr, c_div_x_1, c_remainder
    # a4 is args; save it and use it as the alignment / remainder scratch.
    s32i a4, sp, 0
    bbci \filter_ptr, 3, .L_a16_\@
        tie728_w8a16_unaligned_conv2d_11c8_a8 \input_v, \input_front, \input_back, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, a4
        j .L_done_\@
    .L_a16_\@:
        tie728_w8a16_unaligned_conv2d_11c8_a16 \input_v, \input_front, \input_back, \filter_v, \filter_hi, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, a4
    .L_done_\@:
    l32i a4, sp, 0
.endm

.macro tie728_w8a16_unaligned_conv2d_11c1 input_v, input_front, input_back, filter_v, filter_front, filter_back, input_ptr, filter_ptr, c_div_x_1, c_remainder, temp, zero
    # 1 output channel. Filter is C packed int8. USAR.IP immediates must be
    # multiples of 16, so each 8-byte group is extracted from a 16-byte window
    # and the pointer is rewound to a net +8.
    EE.LD.128.USAR.IP \input_front, \input_ptr, 16

    blti \c_div_x_1, 0, .L_rem_\@
        EE.LD.128.USAR.IP \input_back, \input_ptr, 16

        loopgtz \c_div_x_1, .L_last_\@
            EE.SRC.Q.QUP \input_v, \input_front, \input_back

            EE.LD.128.USAR.IP \filter_front, \filter_ptr, 16
            EE.LD.128.USAR.IP \filter_back, \filter_ptr, 16
            EE.SRC.Q \filter_v, \filter_front, \filter_back
            addi \filter_ptr, \filter_ptr, -24
            EE.ZERO.Q \filter_front
            EE.VCMP.LT.S8 \filter_back, \filter_v, \filter_front
            EE.VZIP.8 \filter_v, \filter_back

            EE.LD.128.USAR.IP \input_back, \input_ptr, 16
            EE.VMULAS.S16.ACCX \filter_v, \input_v
        .L_last_\@:
        EE.SRC.Q.QUP \input_v, \input_front, \input_back

        EE.LD.128.USAR.IP \filter_front, \filter_ptr, 16
        EE.LD.128.USAR.IP \filter_back, \filter_ptr, 16
        EE.SRC.Q \filter_v, \filter_front, \filter_back
        addi \filter_ptr, \filter_ptr, -24
        EE.ZERO.Q \filter_front
        EE.VCMP.LT.S8 \filter_back, \filter_v, \filter_front
        EE.VZIP.8 \filter_v, \filter_back

        EE.VMULAS.S16.ACCX \filter_v, \input_v
        beqz \c_remainder, .L_done_\@

    .L_rem_\@:
        EE.LD.128.USAR.XP \input_back, \input_ptr, \c_remainder
        EE.SRC.Q.QUP \input_v, \input_front, \input_back

        srli \zero, \c_remainder, 1
        EE.LD.128.USAR.IP \filter_front, \filter_ptr, 16
        EE.LD.128.USAR.IP \filter_back, \filter_ptr, 16
        EE.SRC.Q \filter_v, \filter_front, \filter_back
        addi \filter_ptr, \filter_ptr, -32
        add  \filter_ptr, \filter_ptr, \zero
        EE.ZERO.Q \filter_front
        EE.VCMP.LT.S8 \filter_back, \filter_v, \filter_front
        EE.VZIP.8 \filter_v, \filter_back

        movi \zero, 0
        EE.SLCXXP.2Q \input_back, \input_v, \temp, \zero
        EE.SLCXXP.2Q \filter_front, \filter_v, \temp, \zero

        EE.VMULAS.S16.ACCX \filter_v, \input_v

    .L_done_\@:
        addi \input_ptr, \input_ptr, -16
.endm

.macro tie728_w8a16_unaligned_conv2d_33c8 input_v0, input_front, input_back, filter_v0, filter_v1, zero_v, input_ptr, filter_ptr, c_div_x_1, c_remainder, dilation_x_offset, dilation_y_offset
    tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
    add \input_ptr, \input_ptr, \dilation_y_offset

    tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
    add \input_ptr, \input_ptr, \dilation_y_offset

    tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
.endm


.macro tie728_w8a16_unaligned_conv2d_33c1 input_v0, input_front, input_back, filter_v0, filter_front, filter_back, input_ptr, filter_ptr, c_div_x_1, c_remainder, dilation_x_offset, dilation_y_offset, temp, zero
    tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_y_offset

    tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_y_offset

    tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
.endm


.macro tie728_w8a16_unaligned_conv2d_hwc8 input_v0, input_front, input_back, filter_v0, filter_v1, zero_v, input_ptr, filter_ptr, c_div_x_1, c_remainder, dilation_x_offset, dilation_y_offset, filter_h, filter_w, args, filter_offset_q
    l32i \filter_h, \args, 52
    10:
        l32i \filter_w, \args, 56
        beqi \filter_w, 1, 11f
        9:
            tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
            add \input_ptr, \input_ptr, \dilation_x_offset
            addi \filter_w, \filter_w, -1
            bgei \filter_w, 2, 9b
        11:
        tie728_w8a16_unaligned_conv2d_11c8 \input_v0, \input_front, \input_back, \filter_v0, \filter_v1, \zero_v, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder
        EE.MOVI.32.A \filter_offset_q, \filter_w, 1
        add \filter_ptr, \filter_ptr, \filter_w
        add \input_ptr, \input_ptr, \dilation_y_offset
        addi \filter_h, \filter_h, -1
        bnez \filter_h, 10b
    EE.MOVI.32.A \filter_offset_q, \filter_h, 2
    add \filter_ptr, \filter_ptr, \filter_h
.endm


.macro tie728_w8a16_unaligned_conv2d_hwc1 input_v0, input_front, input_back, filter_v0, filter_front, filter_back, input_ptr, filter_ptr, c_div_x_1, c_remainder, dilation_x_offset, dilation_y_offset, filter_h, filter_w, args, temp, zero, filter_offset_q
    l32i \filter_h, \args, 52
    10:
        l32i \filter_w, \args, 56
        beqi \filter_w, 1, 11f
        9:
            tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
            add \input_ptr, \input_ptr, \dilation_x_offset
            addi \filter_w, \filter_w, -1
            bgei \filter_w, 2, 9b
        11:
        tie728_w8a16_unaligned_conv2d_11c1 \input_v0, \input_front, \input_back, \filter_v0, \filter_front, \filter_back, \input_ptr, \filter_ptr, \c_div_x_1, \c_remainder, \temp, \zero
        EE.MOVI.32.A \filter_offset_q, \filter_w, 1
        add \filter_ptr, \filter_ptr, \filter_w
        add \input_ptr, \input_ptr, \dilation_y_offset
        addi \filter_h, \filter_h, -1
        bnez \filter_h, 10b
    EE.MOVI.32.A \filter_offset_q, \filter_h, 2
    add \filter_ptr, \filter_ptr, \filter_h
.endm

############################################################################################################################################################
####
#### 11cn
####
############################################################################################################################################################

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_11cn
    .type   dl_tie728_w8a16_unaligned_conv2d_11cn, @function
dl_tie728_w8a16_unaligned_conv2d_11cn:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4,  64
    l32i  a9, a4,  96
    blti a9, 1, tie728_w8a16_unaligned_conv2d_11cn_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_11cn_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_11cn_64b
        tie728_w8a16_unaligned_conv2d_11cn_32b:
            tie728_w8a16_unaligned_conv2d_11cn_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_11cn_n_remainder

        tie728_w8a16_unaligned_conv2d_11cn_64b:
            tie728_w8a16_unaligned_conv2d_11cn_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_64b_aligned_vector_store  q0, a2

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_11cn_n_remainder

        tie728_w8a16_unaligned_conv2d_11cn_128b:
            tie728_w8a16_unaligned_conv2d_11cn_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_128b_aligned_vector_store  q0, a2

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_11cn_n_remainder:
        l32i a9, a4, 140
        beqz a9, tie728_w8a16_unaligned_conv2d_11cn_n_remainder_end
        movi a10, 15
        sub  a10, a10, a7
        movi a11,  0

        tie728_w8a16_unaligned_conv2d_11cn_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_11c1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a10, a11
            tie728_s16_element_round_result  a14, a8, a15, q0
            tie728_s16_element_store  a2, a14

            addi a9, a9, -1
            bnez a9, tie728_w8a16_unaligned_conv2d_11cn_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_11cn_n_remainder_end:

    retw

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_11cn_relu
    .type   dl_tie728_w8a16_unaligned_conv2d_11cn_relu, @function
dl_tie728_w8a16_unaligned_conv2d_11cn_relu:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4,  64
    l32i  a9, a4,  96
    movi a10,  0
    movi a11,  0
    blti a9, 1, tie728_w8a16_unaligned_conv2d_11cn_relu_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_11cn_relu_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_11cn_relu_64b
        tie728_w8a16_unaligned_conv2d_11cn_relu_32b:
            tie728_w8a16_unaligned_conv2d_11cn_relu_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_s16_conv2d_relu  q0, a10, a11
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_relu_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_11cn_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_11cn_relu_64b:
            tie728_w8a16_unaligned_conv2d_11cn_relu_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_s16_conv2d_relu  q0, a10, a11
                tie728_64b_aligned_vector_store  q0, a2

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_relu_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_11cn_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_11cn_relu_128b:
            tie728_w8a16_unaligned_conv2d_11cn_relu_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_s16_conv2d_relu  q0, a10, a11
                tie728_128b_aligned_vector_store  q0, a2

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_relu_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_11cn_relu_n_remainder:
        l32i a9, a4, 140
        beqz a9, tie728_w8a16_unaligned_conv2d_11cn_relu_n_remainder_end
        movi a10, 15
        sub  a10, a10, a7

        tie728_w8a16_unaligned_conv2d_11cn_relu_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_11c1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a10, a11
            tie728_s16_element_round_result  a14, a8, a15, q0
            tie728_s16_element_relu  a14
            tie728_s16_element_store  a2, a14

            addi a9, a9, -1
            bnez a9, tie728_w8a16_unaligned_conv2d_11cn_relu_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_11cn_relu_n_remainder_end:

    retw

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_11cn_bias
    .type   dl_tie728_w8a16_unaligned_conv2d_11cn_bias, @function
dl_tie728_w8a16_unaligned_conv2d_11cn_bias:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4,  64
    l32i  a9, a4,  96
    l32i a10, a4,  68
    blti a9, 1, tie728_w8a16_unaligned_conv2d_11cn_bias_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_11cn_bias_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_11cn_bias_64b
        tie728_w8a16_unaligned_conv2d_11cn_bias_32b:
            tie728_w8a16_unaligned_conv2d_11cn_bias_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a10
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_bias_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_11cn_bias_n_remainder

        tie728_w8a16_unaligned_conv2d_11cn_bias_64b:
            tie728_w8a16_unaligned_conv2d_11cn_bias_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a10
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_64b_aligned_vector_store  q0, a2

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_bias_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_11cn_bias_n_remainder

        tie728_w8a16_unaligned_conv2d_11cn_bias_128b:
            tie728_w8a16_unaligned_conv2d_11cn_bias_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a10
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_128b_aligned_vector_store  q0, a2

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_bias_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_11cn_bias_n_remainder:
        l32i a9, a4, 140
        beqz a9, tie728_w8a16_unaligned_conv2d_11cn_bias_n_remainder_end
        movi a11, 15
        sub  a11, a11, a7
        movi a12,  0

        tie728_w8a16_unaligned_conv2d_11cn_bias_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            tie728_s16_conv2d_element_bias  a10
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_11c1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a11, a12
            tie728_s16_element_round_result  a14, a8, a15, q0
            tie728_s16_element_store  a2, a14

            addi a9, a9, -1
            bnez a9, tie728_w8a16_unaligned_conv2d_11cn_bias_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_11cn_bias_n_remainder_end:

    retw

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_11cn_bias_relu
    .type   dl_tie728_w8a16_unaligned_conv2d_11cn_bias_relu, @function
dl_tie728_w8a16_unaligned_conv2d_11cn_bias_relu:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4,  64
    l32i  a9, a4,  96
    l32i a10, a4,  68
    movi a11,  0
    movi a12,  0
    blti a9, 1, tie728_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_11cn_bias_relu_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_11cn_bias_relu_64b
        tie728_w8a16_unaligned_conv2d_11cn_bias_relu_32b:
            tie728_w8a16_unaligned_conv2d_11cn_bias_relu_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a10
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_s16_conv2d_relu  q0, a11, a12
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_bias_relu_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_11cn_bias_relu_64b:
            tie728_w8a16_unaligned_conv2d_11cn_bias_relu_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a10
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_s16_conv2d_relu  q0, a11, a12
                tie728_64b_aligned_vector_store  q0, a2

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_bias_relu_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_11cn_bias_relu_128b:
            tie728_w8a16_unaligned_conv2d_11cn_bias_relu_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a10
                tie728_w8a16_unaligned_conv2d_11c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7
                tie728_s16_vector_round_result  q0, a8, a15, q1
                tie728_s16_conv2d_relu  q0, a11, a12
                tie728_128b_aligned_vector_store  q0, a2

                addi a9, a9, -1
                bnez a9, tie728_w8a16_unaligned_conv2d_11cn_bias_relu_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder:
        l32i a9, a4, 140
        beqz a9, tie728_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder_end
        movi a11, 15
        sub  a11, a11, a7
        movi a12,  0

        tie728_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            tie728_s16_conv2d_element_bias  a10
            tie728_w8a16_unaligned_conv2d_11c1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a11, a12
            tie728_s16_element_round_result  a14, a8, a15, q0
            tie728_s16_element_relu  a14
            tie728_s16_element_store  a2, a14

            addi a9, a9, -1
            bnez a9, tie728_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_11cn_bias_relu_n_remainder_end:

    retw

############################################################################################################################################################
####
#### 33cn
####
############################################################################################################################################################

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_33cn
    .type   dl_tie728_w8a16_unaligned_conv2d_33cn, @function
dl_tie728_w8a16_unaligned_conv2d_33cn:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4, 108
    l32i  a9, a4, 112
    l32i a10, a4,  64
    l32i a11, a4,  96
    blti a11, 1, tie728_w8a16_unaligned_conv2d_33cn_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_33cn_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_33cn_64b
        tie728_w8a16_unaligned_conv2d_33cn_32b:
            tie728_w8a16_unaligned_conv2d_33cn_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_33cn_n_remainder

        tie728_w8a16_unaligned_conv2d_33cn_64b:
            tie728_w8a16_unaligned_conv2d_33cn_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_64b_aligned_vector_store  q0, a2

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_33cn_n_remainder

        tie728_w8a16_unaligned_conv2d_33cn_128b:
            tie728_w8a16_unaligned_conv2d_33cn_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_128b_aligned_vector_store  q0, a2

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_33cn_n_remainder:
        l32i a11, a4, 140
        beqz a11, tie728_w8a16_unaligned_conv2d_33cn_n_remainder_end
        movi a12, 15
        sub  a12, a12, a7
        movi a13,  0

        tie728_w8a16_unaligned_conv2d_33cn_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_33c1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a12, a13
            tie728_s16_element_round_result  a14, a10, a15, q0
            tie728_s16_element_store  a2, a14

            addi a11, a11, -1
            bnez a11, tie728_w8a16_unaligned_conv2d_33cn_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_33cn_n_remainder_end:

    retw

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_33cn_relu
    .type   dl_tie728_w8a16_unaligned_conv2d_33cn_relu, @function
dl_tie728_w8a16_unaligned_conv2d_33cn_relu:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4, 108
    l32i  a9, a4, 112
    l32i a10, a4,  64
    l32i a11, a4,  96
    movi a13,  0
    movi a14,  0
    blti a11, 1, tie728_w8a16_unaligned_conv2d_33cn_relu_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_33cn_relu_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_33cn_relu_64b
        tie728_w8a16_unaligned_conv2d_33cn_relu_32b:
            tie728_w8a16_unaligned_conv2d_33cn_relu_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a13, a14
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_relu_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_33cn_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_33cn_relu_64b:
            tie728_w8a16_unaligned_conv2d_33cn_relu_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a13, a14
                tie728_64b_aligned_vector_store  q0, a2

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_relu_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_33cn_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_33cn_relu_128b:
            tie728_w8a16_unaligned_conv2d_33cn_relu_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a13, a14
                tie728_128b_aligned_vector_store  q0, a2

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_relu_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_33cn_relu_n_remainder:
        l32i a11, a4, 140
        beqz a11, tie728_w8a16_unaligned_conv2d_33cn_relu_n_remainder_end
        movi a12, 15
        sub  a12, a12, a7

        tie728_w8a16_unaligned_conv2d_33cn_relu_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_33c1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a12, a13
            tie728_s16_element_round_result  a14, a10, a15, q0
            tie728_s16_element_relu  a14
            tie728_s16_element_store  a2, a14

            addi a11, a11, -1
            bnez a11, tie728_w8a16_unaligned_conv2d_33cn_relu_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_33cn_relu_n_remainder_end:

    retw

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_33cn_bias
    .type   dl_tie728_w8a16_unaligned_conv2d_33cn_bias, @function
dl_tie728_w8a16_unaligned_conv2d_33cn_bias:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4, 108
    l32i  a9, a4, 112
    l32i a10, a4,  64
    l32i a11, a4,  96
    l32i a12, a4,  68
    blti a11, 1, tie728_w8a16_unaligned_conv2d_33cn_bias_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_33cn_bias_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_33cn_bias_64b
        tie728_w8a16_unaligned_conv2d_33cn_bias_32b:
            tie728_w8a16_unaligned_conv2d_33cn_bias_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a12
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_bias_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_33cn_bias_n_remainder

        tie728_w8a16_unaligned_conv2d_33cn_bias_64b:
            tie728_w8a16_unaligned_conv2d_33cn_bias_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a12
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_64b_aligned_vector_store  q0, a2

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_bias_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_33cn_bias_n_remainder

        tie728_w8a16_unaligned_conv2d_33cn_bias_128b:
            tie728_w8a16_unaligned_conv2d_33cn_bias_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a12
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_128b_aligned_vector_store  q0, a2

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_bias_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_33cn_bias_n_remainder:
        l32i a11, a4, 140
        beqz a11, tie728_w8a16_unaligned_conv2d_33cn_bias_n_remainder_end
        movi a13, 15
        sub  a13, a13, a7

        tie728_w8a16_unaligned_conv2d_33cn_bias_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            tie728_s16_conv2d_element_bias  a12
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_33c1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a13, a14
            tie728_s16_element_round_result  a14, a10, a15, q0
            tie728_s16_element_store  a2, a14

            addi a11, a11, -1
            bnez a11, tie728_w8a16_unaligned_conv2d_33cn_bias_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_33cn_bias_n_remainder_end:

    retw

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_33cn_bias_relu
    .type   dl_tie728_w8a16_unaligned_conv2d_33cn_bias_relu, @function
dl_tie728_w8a16_unaligned_conv2d_33cn_bias_relu:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4, 108
    l32i  a9, a4, 112
    l32i a10, a4,  64
    l32i a11, a4,  96
    l32i a12, a4,  68
    movi a13,  0
    movi a14,  0
    blti a11, 1, tie728_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_33cn_bias_relu_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_33cn_bias_relu_64b
        tie728_w8a16_unaligned_conv2d_33cn_bias_relu_32b:
            tie728_w8a16_unaligned_conv2d_33cn_bias_relu_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a12
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a13, a14
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_bias_relu_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_33cn_bias_relu_64b:
            tie728_w8a16_unaligned_conv2d_33cn_bias_relu_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a12
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a13, a14
                tie728_64b_aligned_vector_store  q0, a2

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_bias_relu_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_33cn_bias_relu_128b:
            tie728_w8a16_unaligned_conv2d_33cn_bias_relu_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a12
                tie728_w8a16_unaligned_conv2d_33c8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a13, a14
                tie728_128b_aligned_vector_store  q0, a2

                addi a11, a11, -1
                bnez a11, tie728_w8a16_unaligned_conv2d_33cn_bias_relu_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder:
        l32i a11, a4, 140
        beqz a11, tie728_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder_end
        movi a13, 15
        sub  a13, a13, a7

        tie728_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            tie728_s16_conv2d_element_bias  a12
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_33c1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a13, a14
            tie728_s16_element_round_result  a14, a10, a15, q0
            tie728_s16_element_relu  a14
            tie728_s16_element_store  a2, a14

            addi a11, a11, -1
            bnez a11, tie728_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_33cn_bias_relu_n_remainder_end:

    retw

############################################################################################################################################################
####
#### hwcn
####
############################################################################################################################################################

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_hwcn
    .type   dl_tie728_w8a16_unaligned_conv2d_hwcn, @function
dl_tie728_w8a16_unaligned_conv2d_hwcn:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i a12, a4, 60
    l32i a11, a4, 144
    EE.MOVI.32.Q q7, a12, 1
    EE.MOVI.32.Q q7, a11, 2
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4, 108
    l32i  a9, a4, 112
    l32i a12, a4,  96
    l32i a13, a4,  64
    blti a12, 1, tie728_w8a16_unaligned_conv2d_hwcn_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_hwcn_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_hwcn_64b
        tie728_w8a16_unaligned_conv2d_hwcn_32b:
            tie728_w8a16_unaligned_conv2d_hwcn_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                tie728_s16_vector_round_result  q0, a13, a15, q1
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_hwcn_n_remainder

        tie728_w8a16_unaligned_conv2d_hwcn_64b:
            tie728_w8a16_unaligned_conv2d_hwcn_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                tie728_s16_vector_round_result  q0, a13, a15, q1
                tie728_64b_aligned_vector_store  q0, a2

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_hwcn_n_remainder

        tie728_w8a16_unaligned_conv2d_hwcn_128b:
            tie728_w8a16_unaligned_conv2d_hwcn_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                tie728_s16_vector_round_result  q0, a13, a15, q1
                tie728_128b_aligned_vector_store  q0, a2

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_hwcn_n_remainder:
        l32i a12, a4, 140
        beqz a12, tie728_w8a16_unaligned_conv2d_hwcn_n_remainder_end
        l32i a5, a4, 160
        l32i a15, a4, 164
        EE.MOVI.32.Q q7, a5, 1
        EE.MOVI.32.Q q7, a15, 2
        l32i a5, a4, 168
        movi a13, 15
        sub  a13, a13, a7

        tie728_w8a16_unaligned_conv2d_hwcn_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_hwc1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, a13, a14, q7
            l32i a10, a4, 64
            tie728_s16_element_round_result  a14, a10, a15, q0
            tie728_s16_element_store  a2, a14

            addi a12, a12, -1
            bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_hwcn_n_remainder_end:

    retw

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_hwcn_relu
    .type   dl_tie728_w8a16_unaligned_conv2d_hwcn_relu, @function
dl_tie728_w8a16_unaligned_conv2d_hwcn_relu:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i a12, a4, 60
    l32i a11, a4, 144
    EE.MOVI.32.Q q7, a12, 1
    EE.MOVI.32.Q q7, a11, 2
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4, 108
    l32i  a9, a4, 112
    l32i a12, a4,  96
    l32i a13, a4,  64
    l32i a14, a4,  76
    blti a12, 1, tie728_w8a16_unaligned_conv2d_hwcn_relu_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_hwcn_relu_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_hwcn_relu_64b
        tie728_w8a16_unaligned_conv2d_hwcn_relu_32b:
            tie728_w8a16_unaligned_conv2d_hwcn_relu_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                l32i a10, a4, 64
                l32i a11, a4, 84
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a14, a11
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_relu_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_hwcn_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_hwcn_relu_64b:
            tie728_w8a16_unaligned_conv2d_hwcn_relu_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                l32i a10, a4, 64
                l32i a11, a4, 84
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a14, a11
                tie728_64b_aligned_vector_store  q0, a2

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_relu_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_hwcn_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_hwcn_relu_128b:
            tie728_w8a16_unaligned_conv2d_hwcn_relu_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                l32i a10, a4, 64
                l32i a11, a4, 84
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a14, a11
                tie728_128b_aligned_vector_store  q0, a2

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_relu_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_hwcn_relu_n_remainder:
        l32i a12, a4, 140
        beqz a12, tie728_w8a16_unaligned_conv2d_hwcn_relu_n_remainder_end
        l32i a5, a4, 160
        l32i a15, a4, 164
        EE.MOVI.32.Q q7, a5, 1
        EE.MOVI.32.Q q7, a15, 2
        l32i a5, a4, 168
        movi a13, 15
        sub  a13, a13, a7

        tie728_w8a16_unaligned_conv2d_hwcn_relu_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_hwc1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, a13, a14, q7
            l32i a10, a4, 64
            tie728_s16_element_round_result  a14, a10, a15, q0
            tie728_s16_element_relu  a14
            tie728_s16_element_store  a2, a14

            addi a12, a12, -1
            bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_relu_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_hwcn_relu_n_remainder_end:

    retw

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_hwcn_bias
    .type   dl_tie728_w8a16_unaligned_conv2d_hwcn_bias, @function
dl_tie728_w8a16_unaligned_conv2d_hwcn_bias:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i a12, a4, 60
    l32i a11, a4, 144
    EE.MOVI.32.Q q7, a12, 1
    EE.MOVI.32.Q q7, a11, 2
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4, 108
    l32i  a9, a4, 112
    l32i a12, a4,  96
    l32i a13, a4,  68
    blti a12, 1, tie728_w8a16_unaligned_conv2d_hwcn_bias_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_hwcn_bias_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_hwcn_bias_64b
        tie728_w8a16_unaligned_conv2d_hwcn_bias_32b:
            tie728_w8a16_unaligned_conv2d_hwcn_bias_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a13
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                l32i a10, a4, 64
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_bias_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_hwcn_bias_n_remainder

        tie728_w8a16_unaligned_conv2d_hwcn_bias_64b:
            tie728_w8a16_unaligned_conv2d_hwcn_bias_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a13
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                l32i a10, a4, 64
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_64b_aligned_vector_store  q0, a2

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_bias_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_hwcn_bias_n_remainder

        tie728_w8a16_unaligned_conv2d_hwcn_bias_128b:
            tie728_w8a16_unaligned_conv2d_hwcn_bias_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a13
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                l32i a10, a4, 64
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_128b_aligned_vector_store  q0, a2

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_bias_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_hwcn_bias_n_remainder:
        l32i a12, a4, 140
        beqz a12, tie728_w8a16_unaligned_conv2d_hwcn_bias_n_remainder_end
        l32i a5, a4, 160
        l32i a15, a4, 164
        EE.MOVI.32.Q q7, a5, 1
        EE.MOVI.32.Q q7, a15, 2
        l32i a5, a4, 168
        EE.MOVI.32.Q q6, a13, 1
        movi a13, 15
        sub  a13, a13, a7

        tie728_w8a16_unaligned_conv2d_hwcn_bias_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            EE.MOVI.32.A q6, a14, 1
            tie728_s16_conv2d_element_bias  a14
            EE.MOVI.32.Q q6, a14, 1
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_hwc1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, a13, a14, q7
            l32i a10, a4, 64
            tie728_s16_element_round_result  a14, a10, a15, q0
            tie728_s16_element_store  a2, a14

            addi a12, a12, -1
            bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_bias_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_hwcn_bias_n_remainder_end:

    retw

    .align 4
    .text
    .global dl_tie728_w8a16_unaligned_conv2d_hwcn_bias_relu
    .type   dl_tie728_w8a16_unaligned_conv2d_hwcn_bias_relu, @function
dl_tie728_w8a16_unaligned_conv2d_hwcn_bias_relu:
    .align 4
    entry sp, 128
    EE.ZERO.Q q5
    l32i a12, a4, 60
    l32i a11, a4, 144
    EE.MOVI.32.Q q7, a12, 1
    EE.MOVI.32.Q q7, a11, 2
    l32i  a5, a4,  48
    l32i  a6, a4, 100
    l32i  a7, a4, 136
    l32i  a8, a4, 108
    l32i  a9, a4, 112
    l32i a12, a4,  96
    l32i a13, a4,  68
    l32i a14, a4,  76
    blti a12, 1, tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder
    EE.LD.128.USAR.IP q0, a2, 0
    rur.sar_byte a15

        beqi a15, 0, tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_128b
        beqi a15, 8, tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_64b
        tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_32b:
            tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_32b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a13
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                l32i a10, a4, 64
                l32i a11, a4, 84
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a14, a11
                tie728_32b_aligned_vector_store  q0, a2, a15

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_32b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_64b:
            tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_64b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a13
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                l32i a10, a4, 64
                l32i a11, a4, 84
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a14, a11
                tie728_64b_aligned_vector_store  q0, a2

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_64b_multiple_loop
            j tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder

        tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_128b:
            tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_128b_multiple_loop:
                mov a15, a3
                EE.ZERO.QACC
                tie728_s16_conv2d_128b_vector_bias  a13
                tie728_w8a16_unaligned_conv2d_hwc8 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, q7
                l32i a10, a4, 64
                l32i a11, a4, 84
                tie728_s16_vector_round_result  q0, a10, a15, q1
                tie728_s16_conv2d_relu  q0, a14, a11
                tie728_128b_aligned_vector_store  q0, a2

                addi a12, a12, -1
                bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_128b_multiple_loop

    tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder:
        l32i a12, a4, 140
        beqz a12, tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder_end
        l32i a5, a4, 160
        l32i a15, a4, 164
        EE.MOVI.32.Q q7, a5, 1
        EE.MOVI.32.Q q7, a15, 2
        l32i a5, a4, 168
        EE.MOVI.32.Q q6, a13, 1
        movi a13, 15
        sub  a13, a13, a7

        tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder_loop:
            mov a15, a3
            EE.ZERO.ACCX
            EE.MOVI.32.A q6, a14, 1
            tie728_s16_conv2d_element_bias  a14
            EE.MOVI.32.Q q6, a14, 1
            movi a14,  0
            tie728_w8a16_unaligned_conv2d_hwc1 q0, q1, q2, q3, q4, q5, a15, a5, a6, a7, a8, a9, a10, a11, a4, a13, a14, q7
            l32i a10, a4, 64
            tie728_s16_element_round_result  a14, a10, a15, q0
            tie728_s16_element_relu  a14
            tie728_s16_element_store  a2, a14

            addi a12, a12, -1
            bnez a12, tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder_loop

        tie728_w8a16_unaligned_conv2d_hwcn_bias_relu_n_remainder_end:

    retw
