#include "dl_esp32p4_s16.S"
#include "dl_esp32p4_common.S"


############################################################################################################################################################
####
#### dl_esp32p4_s16_avg_pool2d_hwc_sum
####
############################################################################################################################################################

    .align 2
    .text
    .global dl_esp32p4_s16_avg_pool2d_hwc_sum
    .type   dl_esp32p4_s16_avg_pool2d_hwc_sum, @function
dl_esp32p4_s16_avg_pool2d_hwc_sum:
    .align      2
    esp32p4_push_128_stacks_4r  s0, s1, s8, s9

    # a0: int32_t *buffer_ptr (accumulator)
    # a1: int16_t *input_ptr
    # a2: void *args

    lw a3, 16(a2)     # input_y_offset
    lw a4, 20(a2)     # input_x_offset
    lw t3, 48(a2)     # filter_height
    lw t4, 52(a2)     # filter_width
    lw t6, 104(a2)    # c_div_x_1 (number of channel blocks)
    lw s0, 60(a2)     # c_remainder

    # Create constant vector of ones (16-bit) on stack
    addi sp, sp, -16
    mv s8, sp                  # save stack pointer to s8
    li t0, 0x00010001  # 32-bit pattern of two 0x0001 halfwords
    sw t0, 0(s8)
    sw t0, 4(s8)
    sw t0, 8(s8)
    sw t0, 12(s8)
    esp.vld.128.ip q6, s8, 16   # load 16 bytes of 0x0001 into q6 (8 int16 elements)
    addi sp, sp, 16             # restore stack

    addi t6, t6, 1

    # Main loop over channel blocks
    6:
        esp.zero.qacc

        add a5, a1, x0          # current row pointer
        add s8, a5, x0          # current position pointer
        add s9, t3, x0          # filter_height counter

        # Loop over filter_height
        5:
            # Load first 8 int16_t values
            esp.vld.128.xp q1, s8, a4
            # If filter_width == 1, handle specially
            li t0, 1
            beq t4, t0, dl_esp32p4_s16_avg_pool2d_hwc_sum_width1
            # Load second 8 int16_t values
            esp.vld.128.xp q2, s8, a4
            # Calculate (filter_width / 2) - 1
            srli t1, t4, 1
            addi t1, t1, -1
            blez t1, 1f
            0:
                esp.vmulas.s16.qacc.ld.xp q1, s8, a4, q6, q1
                esp.vmulas.s16.qacc.ld.xp q2, s8, a4, q6, q2
                addi t1, t1, -1
                bgtz t1, 0b
            1:
            # Handle remaining width
            andi t1, t4, 1
            beqz t1, 3f
            # Odd filter_width: one more element
            esp.vmulas.s16.qacc.ld.xp q1, s8, a4, q6, q1
            esp.vmulas.s16.qacc q6, q2
            esp.vmulas.s16.qacc q6, q1
            j 4f
            3:
            # Even filter_width
            esp.vmulas.s16.qacc q6, q1
            esp.vmulas.s16.qacc q6, q2
            4:
            addi s9, s9, -1
            add a5, a5, a3
            add s8, a5, x0
            bnez s9, 5b
            j 7f

    dl_esp32p4_s16_avg_pool2d_hwc_sum_width1:
            # filter_width == 1, only q1 loaded
            esp.vmulas.s16.qacc q6, q1
            addi s9, s9, -1
            add a5, a5, a3
            add s8, a5, x0
            bnez s9, 5b

        7:
        # Save qacc back to buffer_ptr
        esp32p4_save_qacc_as_32bit a0

        addi a1, a1, 16            # move input pointer to next channel block (8 int16 = 16 bytes)
        addi t6, t6, -1
    bnez t6, 6b
    esp32p4_pop_128_stacks_4r  s0, s1, s8, s9
    ret
