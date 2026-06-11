# ESP32-P4 SIMD Code Examples from esp-dl

## Table of Contents
- [Common Macros and Utilities](#common-macros-and-utilities)
- [Convolution Patterns](#convolution-patterns)
- [Elementwise Addition](#elementwise-addition)
- [Depthwise Convolution](#depthwise-convolution)
- [Fully Connected / GEMM](#fully-connected--gemm)
- [Pooling](#pooling)
- [ReLU Activation](#relu-activation)
- [Optimization Tips](#optimization-tips)

---

## Common Macros and Utilities

### Register Save/Restore
```asm
# Push 3 registers to stack (12 bytes)
.macro esp32p4_push_12_stacks_3r reg0, reg1, reg2
    addi  sp, sp, -12
    sw    \reg0, 0(sp)
    sw    \reg1, 4(sp)
    sw    \reg2, 8(sp)
.endm

.macro esp32p4_pop_12_stacks_3r reg0, reg1, reg2
    lw    \reg0, 0(sp)
    lw    \reg1, 4(sp)
    lw    \reg2, 8(sp)
    addi  sp, sp, 12
.endm
```

### QACC Bias Loading (for Conv2D)
```asm
# Load 128-bit bias into QACC
.macro esp32p4_s16_conv2d_128b_vector_bias bias_ptr
    esp.vld.128.ip  q7, \bias_ptr, 16
    esp.vmov.s16.qacc q7
.endm
```

### QACC Result Extraction with Shift
```asm
# Extract QACC result with right shift
# mac_shift controls the output precision
.macro esp32p4_s16_128b_vector_shift_result result_q, mac_shift
    # Set SAR for shift amount
    esp.movx.w.sar  \mac_shift
    # Shift QACC_H/L right by SAR, saturate to S16, store to QR
    esp.srcmb.s16.qacc  \result_q, q0, 0
.endm
```

### Per-Channel Shift
```asm
# When different output channels need different shift amounts
.macro esp32p4_s16_128b_vector_per_channel_shift_result result_q, shift_q
    esp.srcmb.s16.q.qacc  \result_q, \shift_q
.endm
```

### Aligned Vector Store
```asm
.macro esp32p4_s16_128b_aligned_vector_store data_q, dst_ptr
    esp.vst.128.ip  \data_q, \dst_ptr, 16
.endm
```

### ReLU Macro
```asm
# ReLU: result = max(0, input)
.macro esp32p4_s16_128b_vector_relu data_q, alpha_ptr, shift
    # Load zero into another QR
    esp.zero.q  q7
    # Max between data and zero
    esp.vmax.s16  \data_q, \data_q, q7
.endm
```

---

## Convolution Patterns

### 1x1 Conv2D (Pointwise) - S16, 8 channels per iteration

This is the most common conv2d pattern. Processes 8 input channels and 8 output channels per block.

```asm
# Macro: 1x1 conv2d core loop, 8 input channels (c8)
# input_ptr advances by (c_div_x_1 + 1) * 16
# filter_ptr advances to next 16 bytes
.macro esp32p4_s16_conv2d_11c8 input_v0, filter_v0, filter_v1, \
                                input_ptr, filter_ptr, c_div_x_1, tmp

    # Load first input vector (8 x int16)
    esp.vld.128.ip  \input_v0, \input_ptr, 16
    # Load first filter vectors
    esp.vld.128.ip  \filter_v0, \filter_ptr, 16
    esp.vld.128.ip  \filter_v1, \filter_ptr, 16

    beqz  \c_div_x_1, 1f

    # Loop over input channels in chunks of 8
    mv  \tmp, \c_div_x_1
    0:
        # Scalar-vector multiply-accumulate with auto filter loading
        # Each vsmulas does: qacc += filter_vN[i] * input_v0[i] for all i
        esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v0, 0
        esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v0, 1
        esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v0, 2
        esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v0, 3
        esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v0, 4
        esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v0, 5
        esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v0, 6
        # Last one loads next input vector
        esp.vsmulas.s16.qacc.ld.incp  \input_v0, \input_ptr, \filter_v1, \input_v0, 7
        # Preload next filter
        esp.vld.128.ip  \filter_v1, \filter_ptr, 16
        addi  \tmp, \tmp, -1
        bgtz  \tmp, 0b

    1:
    # Tail: 8 more vsmulas without loop overhead
    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v0, 0
    esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v0, 1
    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v0, 2
    esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v0, 3
    esp.vsmulas.s16.qacc.ld.incp  \filter_v0, \filter_ptr, \filter_v0, \input_v0, 4
    esp.vsmulas.s16.qacc.ld.incp  \filter_v1, \filter_ptr, \filter_v1, \input_v0, 5
    esp.vsmulas.s16.qacc  \filter_v0, \input_v0, 6
    esp.vsmulas.s16.qacc  \filter_v1, \input_v0, 7
.endm
```

### Function: 1x1 Conv2D with Bias + ReLU

```asm
    .text
    .global dl_esp32p4_s16_conv2d_11cn_bias_relu
    .type   dl_esp32p4_s16_conv2d_11cn_bias_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_conv2d_11cn_bias_relu:
    # a0: int16_t *output_ptr
    # a1: int16_t *input_ptr
    # a2: void *args
    # a3: int16_t *filter_ptr  (loaded from args)
    # a4: mac_shift            (loaded from args)
    # a5: bias_ptr             (loaded from args)
    # t3: activation_alpha     (loaded from args)
    # t4: activation_shift     (loaded from args)
    # t5: moving_input_ptr
    # t6: output_channel_div_8 (loop counter)

    # Load args
    lw  t6,  96(a2)     # output_channel_div_8
    lw  a4,  64(a2)     # mac_shift
    lw  a3,  48(a2)     # filter
    lw  t1,  100(a2)    # input_channel / 8 - 1
    lw  a5,  68(a2)     # bias
    lw  t3,  76(a2)     # activation_alpha
    lw  t4,  84(a2)     # activation_shift

    esp32p4_s16_conv2d_11cn_bias_relu_loop:
        mv  t5, a1          # reload input_ptr for each output channel group
        esp.zero.qacc       # clear accumulator

        # 1. Load bias into QACC
        esp.vld.128.ip  q7, a5, 16
        esp.vmov.s16.qacc q7

        # 2. Core conv2d MAC loop
        esp32p4_s16_conv2d_11c8  q0, q1, q2, t5, a3, t1, t0

        # 3. Extract result with shift
        esp32p4_s16_128b_vector_shift_result  q0, a4

        # 4. Apply ReLU
        esp32p4_s16_128b_vector_relu  q0, t3, t4

        # 5. Store result
        esp32p4_s16_128b_aligned_vector_store  q0, a0

        addi  t6, t6, -1
        bnez  t6, esp32p4_s16_conv2d_11cn_bias_relu_loop

    ret
```

### 3x3 Conv2D (Depthwise-like) using 1x1 core

Reuse the 1x1 macro for each position in the 3x3 kernel:

```asm
.macro esp32p4_s16_conv2d_33c8 input_v0, filter_v0, filter_v1, \
                               input_ptr, filter_ptr, c_div_x_1, \
                               dilation_x_offset, dilation_y_offset, tmp
    # Top-left
    esp32p4_s16_conv2d_11c8  \input_v0, \filter_v0, \filter_v1, \
                             \input_ptr, \filter_ptr, \c_div_x_1, \tmp
    add  \input_ptr, \input_ptr, \dilation_x_offset

    # Top-center
    esp32p4_s16_conv2d_11c8  \input_v0, \filter_v0, \filter_v1, \
                             \input_ptr, \filter_ptr, \c_div_x_1, \tmp
    add  \input_ptr, \input_ptr, \dilation_x_offset

    # Top-right
    esp32p4_s16_conv2d_11c8  \input_v0, \filter_v0, \filter_v1, \
                             \input_ptr, \filter_ptr, \c_div_x_1, \tmp
    add  \input_ptr, \input_ptr, \dilation_y_offset

    # Middle-left
    esp32p4_s16_conv2d_11c8  \input_v0, \filter_v0, \filter_v1, \
                             \input_ptr, \filter_ptr, \c_div_x_1, \tmp
    add  \input_ptr, \input_ptr, \dilation_x_offset

    # Middle-center
    esp32p4_s16_conv2d_11c8  \input_v0, \filter_v0, \filter_v1, \
                             \input_ptr, \filter_ptr, \c_div_x_1, \tmp
    add  \input_ptr, \input_ptr, \dilation_x_offset

    # Middle-right
    esp32p4_s16_conv2d_11c8  \input_v0, \filter_v0, \filter_v1, \
                             \input_ptr, \filter_ptr, \c_div_x_1, \tmp
    add  \input_ptr, \input_ptr, \dilation_y_offset

    # Bottom-left
    esp32p4_s16_conv2d_11c8  \input_v0, \filter_v0, \filter_v1, \
                             \input_ptr, \filter_ptr, \c_div_x_1, \tmp
    add  \input_ptr, \input_ptr, \dilation_x_offset

    # Bottom-center
    esp32p4_s16_conv2d_11c8  \input_v0, \filter_v0, \filter_v1, \
                             \input_ptr, \filter_ptr, \c_div_x_1, \tmp
    add  \input_ptr, \input_ptr, \dilation_x_offset

    # Bottom-right (no pointer advance needed)
    esp32p4_s16_conv2d_11c8  \input_v0, \filter_v0, \filter_v1, \
                             \input_ptr, \filter_ptr, \c_div_x_1, \tmp
.endm
```

---

## Elementwise Addition

### S16 Vector Addition (8 elements per iteration)

```asm
    .text
    .global dl_esp32p4_s16_add
    .type   dl_esp32p4_s16_add, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_add:
    # a0: int16_t *output_ptr
    # a1: int16_t *input0_ptr
    # a2: int16_t *input1_ptr
    # a3: int16_t *bias (unused for add)
    # a4: void *args
    # a5: mac_shift (unused)

    lw  t0, 80(a4)      # mul_shift
    lw  t1, 40(a4)      # element_num  (total elements)
    lw  t2, 88(a4)      # activation_alpha
    lw  t3, 92(a4)      # activation_shift

    # Calculate loop count: element_num / 8
    srli  t4, t1, 3     # t4 = element_num >> 3 (div by 8)
    beqz  t4, 2f        # skip if less than 8 elements

    0:
        # Load 8 x int16 from both inputs
        esp.vld.128.ip  q0, a1, 16
        esp.vld.128.ip  q1, a2, 16

        # Vector add with fused store: q2 = q0 + q1, then store
        esp.vadd.s16.st.incp  q2, q0, q1, a0, 16

        addi  t4, t4, -1
        bgtz  t4, 0b

    2:
    # Handle remaining elements (tail processing)
    andi  t4, t1, 7     # t4 = element_num % 8
    beqz  t4, 4f

    # ... tail handling with scalar code ...

    4:
    ret
```

### S16 Vector Addition with ReLU

```asm
    .text
    .global dl_esp32p4_s16_add_relu
    .type   dl_esp32p4_s16_add_relu, @function
    .balign 4
    .option norvc
dl_esp32p4_s16_add_relu:
    # Args: same as add

    lw  t0, 80(a4)
    lw  t1, 40(a4)
    lw  t2, 88(a4)
    lw  t3, 92(a4)

    srli  t4, t1, 3
    beqz  t4, 2f

    0:
        esp.vld.128.ip  q0, a1, 16
        esp.vld.128.ip  q1, a2, 16
        esp.vadd.s16  q2, q0, q1

        # ReLU: max with zero
        esp.zero.q  q7
        esp.vmax.s16  q2, q2, q7

        esp.vst.128.ip  q2, a0, 16

        addi  t4, t4, -1
        bgtz  t4, 0b

    2:
    # ... tail handling ...
    ret
```

---

## Depthwise Convolution

Depthwise conv2d processes each channel independently. Uses scalar-vector multiply-accumulate.

```asm
# Depthwise 3x3 kernel for S16
# Each QR element corresponds to one channel's spatial data

.macro esp32p4_s16_depthwise_conv2d_33s1 input_ptr, filter_ptr, output_ptr, \
                                          c_div_x_1, input_offset, filter_offset

    esp.zero.qacc

    # Row 0
    esp.vld.128.ip  q0, \input_ptr, 16
    esp.vld.128.ip  q1, \filter_ptr, 16
    esp.vmulas.s16.qacc  q0, q1

    add  \input_ptr, \input_ptr, \input_offset
    esp.vld.128.ip  q0, \input_ptr, 16
    esp.vld.128.ip  q1, \filter_ptr, 16
    esp.vmulas.s16.qacc  q0, q1

    add  \input_ptr, \input_ptr, \input_offset
    esp.vld.128.ip  q0, \input_ptr, 16
    esp.vld.128.ip  q1, \filter_ptr, 16
    esp.vmulas.s16.qacc  q0, q1

    # Row 1
    add  \input_ptr, \input_ptr, \input_offset
    esp.vld.128.ip  q0, \input_ptr, 16
    esp.vld.128.ip  q1, \filter_ptr, 16
    esp.vmulas.s16.qacc  q0, q1

    add  \input_ptr, \input_ptr, \input_offset
    esp.vld.128.ip  q0, \input_ptr, 16
    esp.vld.128.ip  q1, \filter_ptr, 16
    esp.vmulas.s16.qacc  q0, q1

    add  \input_ptr, \input_ptr, \input_offset
    esp.vld.128.ip  q0, \input_ptr, 16
    esp.vld.128.ip  q1, \filter_ptr, 16
    esp.vmulas.s16.qacc  q0, q1

    # Row 2
    add  \input_ptr, \input_ptr, \input_offset
    esp.vld.128.ip  q0, \input_ptr, 16
    esp.vld.128.ip  q1, \filter_ptr, 16
    esp.vmulas.s16.qacc  q0, q1

    add  \input_ptr, \input_ptr, \input_offset
    esp.vld.128.ip  q0, \input_ptr, 16
    esp.vld.128.ip  q1, \filter_ptr, 16
    esp.vmulas.s16.qacc  q0, q1

    add  \input_ptr, \input_ptr, \input_offset
    esp.vld.128.ip  q0, \input_ptr, 16
    esp.vld.128.ip  q1, \filter_ptr, 16
    esp.vmulas.s16.qacc  q0, q1
.endm
```

---

## Fully Connected / GEMM

GEMM is essentially a 1x1 conv2d without spatial dimensions. Use XACC for dot-product accumulation.

### Dot Product using XACC (int8)

```asm
# Compute dot product of two int8 vectors
# a0: result pointer
# a1: vector A pointer
# a2: vector B pointer
# a3: length (bytes)

    esp.zero.xacc           # clear 40-bit accumulator
    srli  t0, a3, 4         # t0 = length / 16 (128-bit blocks)
    beqz  t0, 2f

    0:
        esp.vld.128.ip  q0, a1, 16
        esp.vld.128.ip  q1, a2, 16
        esp.vmulas.s8.xacc  q0, q1   # accumulate 16 products into XACC
        addi  t0, t0, -1
        bgtz  t0, 0b

    2:
    # Extract final result from XACC
    esp.movx.r.xacc.l  a4     # read lower bits
    esp.movx.r.xacc.h  a5     # read upper bits
    # Combine a5:a4 into final 40-bit result

    # Handle remainder (scalar code)
    andi  t0, a3, 15
    # ...
```

---

## Pooling

### Max Pooling 2x2 for S16

```asm
# Max pool over 2x2 window, stride 2
# Process 8 channels simultaneously

    esp.vld.128.ip  q0, a1, 16      # row 0, col 0
    add  a3, a1, a2                  # row 1 pointer
    esp.vld.128.ip  q1, a3, 16      # row 1, col 0
    esp.vmax.s16  q0, q0, q1        # q0 = max(row0, row1)

    esp.vld.128.ip  q1, a1, 16      # row 0, col 1
    esp.vld.128.ip  q2, a3, 16      # row 1, col 1
    esp.vmax.s16  q1, q1, q2        # q1 = max(row0, row1) col1

    esp.vmax.s16  q0, q0, q1        # q0 = max over 2x2 window
    esp.vst.128.ip  q0, a0, 16      # store result
```

### Average Pooling using XACC

```asm
# Average pool over HxW window
# Sum all elements with XACC, then divide by the window size

    esp.zero.xacc

    # q7 = all-ones (each 16-bit lane = 1) so vmulas computes a plain element sum.
    # a6 points to an int16 constant {1}; broadcast it to all 8 lanes.
    esp.vldbc.16.ip  q7, a6, 0

    # Loop over window
    mv  t0, a3              # window height
    0:
        mv  t1, a4          # window width
        1:
            esp.vld.128.ip       q0, a1, 16
            esp.vmulas.s16.xacc  q0, q7   # XACC += sum(q0[i] * 1) = sum(q0[i])
            addi  t1, t1, -1
            bgtz  t1, 1b
        add  a1, a1, a5     # next row
        addi  t0, t0, -1
        bgtz  t0, 0b

    # Divide by window_size = H * W via an arithmetic right shift.
    # ESP.SRS.S.XACC takes the shift amount from t0[5:0], saturates the shifted
    # 40-bit XACC to a 32-bit signed value, and writes the result into t1.
    li  t0, \log2_window_size
    esp.srs.s.xacc  t1, t0        # t1 = sat_s32(XACC >> log2_window_size)
    # ... t1 holds the pooled average; extract/store as needed
```

---

## ReLU Activation

### PReLU (Parametric ReLU)

```asm
# PReLU: out = in > 0 ? in : in * alpha
# alpha is per-channel (loaded into q3)

    # Load data and alpha
    esp.vld.128.ip  q0, a1, 16      # input data
    esp.vld.128.ip  q3, a2, 16      # alpha values (per-channel)

    # Use PReLU instruction directly
    esp.vprelu.s16  q1, q0, q3

    esp.vst.128.ip  q1, a0, 16
```

### Leaky ReLU (using VPRELU with broadcast alpha)

```asm
# Leaky ReLU: out = in > 0 ? in : in * alpha
# alpha is a single scalar value

    # Broadcast alpha to all elements
    esp.vldbc.16.ip  q3, a2, 0

    esp.vld.128.ip  q0, a1, 16
    esp.vprelu.s16  q1, q0, q3
    esp.vst.128.ip  q1, a0, 16
```

---

## Optimization Tips

### 1. Fused Load-Compute Instructions

Always prefer fused instructions to reduce instruction count:

```asm
# Bad: 3 separate instructions
esp.vld.128.ip  q2, a1, 16
esp.vadd.s16    q3, q0, q1
esp.vst.128.ip  q3, a0, 16

# Better: fused load-compute
esp.vadd.s16.ld.incp  q3, q0, q1, a1, 16
esp.vst.128.ip        q3, a0, 16

# Even better: fused store-compute (when available)
esp.vadd.s16.st.incp  q3, q0, q1, a0, 16
```

### 2. Unroll Small Loops

For known small iteration counts, unroll manually:

```asm
# Instead of a loop with 8 iterations, just write 8 instructions:
esp.vsmulas.s16.qacc.ld.incp  q1, a3, q1, q0, 0
esp.vsmulas.s16.qacc.ld.incp  q2, a3, q2, q0, 1
esp.vsmulas.s16.qacc.ld.incp  q1, a3, q1, q0, 2
esp.vsmulas.s16.qacc.ld.incp  q2, a3, q2, q0, 3
esp.vsmulas.s16.qacc.ld.incp  q1, a3, q1, q0, 4
esp.vsmulas.s16.qacc.ld.incp  q2, a3, q2, q0, 5
esp.vsmulas.s16.qacc          q1, q0, 6
esp.vsmulas.s16.qacc          q2, q0, 7
```

### 3. Register Allocation Strategy

From esp-dl patterns, typical register allocation:

| Register | Typical Use |
|----------|-------------|
| q0 | Input data (reused frequently) |
| q1-q2 | Filter data (alternating for pipelining) |
| q3 | Per-channel shift factors / alpha |
| q5-q6 | Temporary / unaligned load buffers |
| q7 | Bias / Zero constant / Temporary |

### 4. Handle Alignment at Function Entry

```asm
# At function entry, compute alignment for store
esp.ld.128.usar.ip  q5, a0, 0
esp.movx.r.sar.bytes  t3    # t3 = output sar_byte (save for later)

# At store time, set SAR_BYTE and use spliced store
esp.movx.w.sar.bytes  t3
esp.srcq.128.st.incp  q0, a0, q5, q6
```

### 5. Use Broadcast for Scalar Operations

```asm
# Instead of loading a scalar and repeating it manually:
li  t0, 42
esp.vldbc.16.ip  q7, t0, 0     # q7 = [42, 42, 42, 42, 42, 42, 42, 42]
esp.vadd.s16     q1, q0, q7    # add 42 to all elements
```

### 6. Zero QACC Efficiently

```asm
# Always use dedicated instruction (single cycle)
esp.zero.qacc

# Never do this (wasteful):
esp.zero.q  q7
esp.vmov.s16.qacc q7    # Only if you need to load bias, otherwise use zero.qacc
```
