    .section .rodata

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

    .balign 2
    .global  rgb8882gray_const_tab
    .type   rgb8882gray_const_tab, @object
rgb8882gray_const_tab:
    # round_delta
    .short 16384
    # coeff_r
    .short 9798
    # coeff_g
    .short 19235
    # coeff_b
    .short 3735

    .balign 4
    .global  rgb8882hsv_const_tab
    .type   rgb8882hsv_const_tab, @object
rgb8882hsv_const_tab:
    # round_delta
    .word 2048
    # h_range
    .short 180
    # s8_bias_128
    .byte 0x80

    .balign 4
    .global  yuv2rgb888_const_tab
    .type   yuv2rgb888_const_tab, @object
yuv2rgb888_const_tab:
    # round_delta (1 << 13)
    .short 8192
    # u/v bias
    .short -128
    # CR2RI
    .short 22987
    # round_delta (1 << 13)
    .short 8192
    # CB2GI
    .short -5636
    # CR2GI
    .short -11698
    # round_delta (1 << 13)
    .short 8192
    # CB2BI
    .short 29049
    # clamp max 255
    .short 255

    .global  vsl_8_1_mask
    .type   vsl_8_1_mask, @object
vsl_8_1_mask:
    .byte 0b11111110

    .global  vsl_8_2_mask
    .type   vsl_8_2_mask, @object
vsl_8_2_mask:
    .byte 0b11111100

    .global  vsl_8_3_mask
    .type   vsl_8_3_mask, @object
vsl_8_3_mask:
    .byte 0b11111000

    .global  vsl_8_4_mask
    .type   vsl_8_4_mask, @object
vsl_8_4_mask:
    .byte 0b11110000

    .global  vsl_8_5_mask
    .type   vsl_8_5_mask, @object
vsl_8_5_mask:
    .byte 0b11100000

    .global  vsl_8_6_mask
    .type   vsl_8_6_mask, @object
vsl_8_6_mask:
    .byte 0b11000000

    .global  vsl_8_7_mask
    .type   vsl_8_7_mask, @object
vsl_8_7_mask:
    .byte 0b10000000

    .balign 2
    .global  vsl_16_1_mask
    .type   vsl_16_1_mask, @object
vsl_16_1_mask:
    .short 0xfffe

    .balign 2
    .global  vsl_16_2_mask
    .type   vsl_16_2_mask, @object
vsl_16_2_mask:
    .short 0xfffc

    .balign 2
    .global  vsl_16_3_mask
    .type   vsl_16_3_mask, @object
vsl_16_3_mask:
    .short 0xfff8

    .balign 2
    .global  vsl_16_4_mask
    .type   vsl_16_4_mask, @object
vsl_16_4_mask:
    .short 0xfff0

    .balign 2
    .global  vsl_16_5_mask
    .type   vsl_16_5_mask, @object
vsl_16_5_mask:
    .short 0xffe0

    .balign 2
    .global  vsl_16_6_mask
    .type   vsl_16_6_mask, @object
vsl_16_6_mask:
    .short 0xffc0

    .balign 2
    .global  vsl_16_7_mask
    .type   vsl_16_7_mask, @object
vsl_16_7_mask:
    .short 0xff80

    .balign 2
    .global  vsl_16_8_mask
    .type   vsl_16_8_mask, @object
vsl_16_8_mask:
    .short 0xff00

    .balign 2
    .global  vsl_16_9_mask
    .type   vsl_16_9_mask, @object
vsl_16_9_mask:
    .short 0xfe00

    .balign 2
    .global  vsl_16_10_mask
    .type   vsl_16_10_mask, @object
vsl_16_10_mask:
    .short 0xfc00

    .balign 2
    .global  vsl_16_11_mask
    .type   vsl_16_11_mask, @object
vsl_16_11_mask:
    .short 0xf800

    .balign 2
    .global  vsl_16_12_mask
    .type   vsl_16_12_mask, @object
vsl_16_12_mask:
    .short 0xf000

    .balign 2
    .global  vsl_16_13_mask
    .type   vsl_16_13_mask, @object
vsl_16_13_mask:
    .short 0xe000

    .balign 2
    .global  vsl_16_14_mask
    .type   vsl_16_14_mask, @object
vsl_16_14_mask:
    .short 0xc000

    .balign 2
    .global  vsl_16_15_mask
    .type   vsl_16_15_mask, @object
vsl_16_15_mask:
    .short 0x8000

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
