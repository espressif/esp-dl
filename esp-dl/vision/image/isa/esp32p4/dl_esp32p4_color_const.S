    .section .data
    .global  rgb5652rgb888_mask_tab
    .type   rgb5652rgb888_mask_tab, @object
rgb5652rgb888_mask_tab:
    # rgb565_low_byte_g_mask
    .byte 0b11100000
    # rgb565_high_byte_g_mask
    .byte 0b111
    # rgb565_high_byte_r_mask
    .byte 0b11111000

    .global  rgb8882rgb565_mask_tab
    .type   rgb8882rgb565_mask_tab, @object
rgb8882rgb565_mask_tab:
    # rgb888_5bit_mask
    .byte 0b11111000
    # rgb888_6bit_low_mask
    .byte 0b11100
    # rgb888_6bit_high_mask
    .byte 0b11100000

    .balign 16
    .global  rgb8882gray_const_tab
    .type   rgb8882gray_const_tab, @object
rgb8882gray_const_tab:
    # round_delta
    .half 16384
    .half 16384
    .half 16384
    .half 16384
    .half 16384
    .half 16384
    .half 16384
    .half 16384
    # coeff_r
    .half 9798
    # coeff_g
    .half 19235
    # coeff_b
    .half 3735
    # shift
    .half 15

    .balign 4
    .global  rgb8882hsv_const_tab
    .type   rgb8882hsv_const_tab, @object
rgb8882hsv_const_tab:
    # round_delta
    .word 2048
    # h_range
    .half 180

    .global  vsl_8_1
    .type   vsl_8_1, @object
vsl_8_1:
    .byte 1

    .global  vsl_8_2
    .type   vsl_8_2, @object
vsl_8_2:
    .byte 2

    .global  vsl_8_3
    .type   vsl_8_3, @object
vsl_8_3:
    .byte 3

    .global  vsl_8_4
    .type   vsl_8_4, @object
vsl_8_4:
    .byte 4

    .global  vsl_8_5
    .type   vsl_8_5, @object
vsl_8_5:
    .byte 5

    .global  vsl_8_6
    .type   vsl_8_6, @object
vsl_8_6:
    .byte 6

    .global  vsl_8_7
    .type   vsl_8_7, @object
vsl_8_7:
    .byte 7

    .balign 2
    .global  vsl_16_1
    .type   vsl_16_1, @object
vsl_16_1:
    .half 1

    .balign 2
    .global  vsl_16_2
    .type   vsl_16_2, @object
vsl_16_2:
    .half 2

    .balign 2
    .global  vsl_16_3
    .type   vsl_16_3, @object
vsl_16_3:
    .half 3

    .balign 2
    .global  vsl_16_4
    .type   vsl_16_4, @object
vsl_16_4:
    .half 4

    .balign 2
    .global  vsl_16_5
    .type   vsl_16_5, @object
vsl_16_5:
    .half 5

    .balign 2
    .global  vsl_16_6
    .type   vsl_16_6, @object
vsl_16_6:
    .half 6

    .balign 2
    .global  vsl_16_7
    .type   vsl_16_7, @object
vsl_16_7:
    .half 7

    .balign 2
    .global  vsl_16_8
    .type   vsl_16_8, @object
vsl_16_8:
    .half 8

    .balign 2
    .global  vsl_16_9
    .type   vsl_16_9, @object
vsl_16_9:
    .half 9

    .balign 2
    .global  vsl_16_10
    .type   vsl_16_10, @object
vsl_16_10:
    .half 10

    .balign 2
    .global  vsl_16_11
    .type   vsl_16_11, @object
vsl_16_11:
    .half 11

    .balign 2
    .global  vsl_16_12
    .type   vsl_16_12, @object
vsl_16_12:
    .half 12

    .balign 2
    .global  vsl_16_13
    .type   vsl_16_13, @object
vsl_16_13:
    .half 13

    .balign 2
    .global  vsl_16_14
    .type   vsl_16_14, @object
vsl_16_14:
    .half 14

    .balign 2
    .global  vsl_16_15
    .type   vsl_16_15, @object
vsl_16_15:
    .half 15

    .global  vsr_u8_1_mask
    .type   vsr_u8_1_mask, @object
vsr_u8_1_mask:
    .byte 0b01111111

    .global  vsr_u8_2_mask
    .type   vsr_u8_2_mask, @object
vsr_u8_2_mask:
    .byte 0b00111111

    .global  vsr_u8_3_mask
    .type   vsr_u8_3_mask, @object
vsr_u8_3_mask:
    .byte 0b00011111

    .global  vsr_u8_4_mask
    .type   vsr_u8_4_mask, @object
vsr_u8_4_mask:
    .byte 0b00001111

    .global  vsr_u8_5_mask
    .type   vsr_u8_5_mask, @object
vsr_u8_5_mask:
    .byte 0b00000111

    .global  vsr_u8_6_mask
    .type   vsr_u8_6_mask, @object
vsr_u8_6_mask:
    .byte 0b00000011

    .global  vsr_u8_7_mask
    .type   vsr_u8_7_mask, @object
vsr_u8_7_mask:
    .byte 0b00000001
