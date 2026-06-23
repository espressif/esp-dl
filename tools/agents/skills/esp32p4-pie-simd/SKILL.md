---
name: esp32p4-simd
description: ESP32-P4 PIE (Processor Instruction Extensions) SIMD instruction set reference and optimization guide. Use when working with ESP32-P4 custom AI/DSP SIMD instructions in assembly orintrinsic form, converting scalar code to vectorized SIMD code, or implementing neural network operators for esp-dl. Covers read/write, data exchange, arithmetic, comparison, bitwise logical, shift, and FFT-dedicated instructions with 128-bit QR vector registers.
---

# ESP32-P4 SIMD (PIE) Instruction Set

## Architecture Overview

The ESP32-P4 HP core includes a custom PIE SIMD extension supporting 128-bit vector operations on 8-bit, 16-bit, and 32-bit data elements. It integrates data transfer into arithmetic instructions and supports non-aligned 128-bit vector data access.

### Key Features
- 128-bit general-purpose vector registers (8 QR registers)
- 16 x 8-bit multipliers, 8 x 16-bit multipliers
- 256-bit accumulators (QACC_H, QACC_L) + 40-bit accumulator (XACC)
- Fused load-arithmetic and arithmetic-store instructions
- Configurable rounding and saturation modes
- Hardware misaligned access support

## Registers

### General-Purpose Registers (AR)
Only 16 of 32 RISC-V registers are available for PIE instructions:

| Registers | Description |
|-----------|-------------|
| x8-x15 (s0-s1, a0-a5) | Callee-saved and argument registers |
| x24-x31 (s8-s11, t3-t6) | Additional saved and temporary registers |
| x0-x7, x16-x23 | **NOT available** for PIE instructions |

### Vector Registers (QR)
Eight 128-bit vector registers q0-q7. Each can hold:
- 16 x 8-bit elements
- 8 x 16-bit elements
- 4 x 32-bit elements

| Register | Bits | Access | Usage |
|----------|------|--------|-------|
| q0-q7 | 128 | R/W | Vector operands and results |

### Special Registers

| Register | Bits | Access | Purpose |
|----------|------|--------|---------|
| SAR | 6 | R/W | Shift amount for multiply-shift and vector shift instructions |
| SAR_BYTE | 4 | R/W | Byte shift amount for non-aligned data handling |
| QACC_H | 256 | R/W | High 256-bit accumulator (8 x 32-bit for 8b MAC, 4 x 64-bit for 16b MAC) |
| QACC_L | 256 | R/W | Low 256-bit accumulator |
| XACC | 40 | R/W | 40-bit scalar accumulator for dot-product style accumulation |
| FFT_BIT_WIDTH | 4 | R/W | Bit width configuration for ESP.BITREV (range 3-10 bits) |
| PERF | 32 | R/W | Performance counter register |
| UA_STATE | 128 | R/W | Unaligned state register for FFT instructions |
| CFG | 32 | R/W | Configuration register (rounding mode, saturation enable, misaligned access) |

### CFG Register Fields

| Field | Bits | Description |
|-------|------|-------------|
| vxsat_en | 8 | Enable saturation status |
| vxrm | 7:4 | 4-bit rounding mode (0=FLOOR, 1=CEILING, 2=UP, 3=DOWN, 4=HALF_UP, 5=HALF_DOWN, 6=HALF_EVEN, 7=UNNECESSARY) |
| rm_exc | 3 | Exception status for UNNECESSARY mode (RO) |
| vxsat | 2 | Saturation status (RO, cleared on CFG read) |
| mis_ld | 1 | Enable hardware handle load misaligned access |
| mis_st | 0 | Enable hardware handle store misaligned access |

### SAR Usage Constraints
- **Vector shifts** (ESP.VSR.32, ESP.VSL.32): Uses lower 5 bits as shift amount
- **Multiplications** (ESP.VMUL.*, ESP.CMUL.*, ESP.FFT.AMS.*): Uses full SAR value for right-shift of intermediate results
- Set SAR via: `esp.movx.w.sar` or `esp.movx.w.cfg` with appropriate vxrm

## Instruction Categories

Instructions are organized into these categories. See `references/instructions.md` for the complete reference:

1. **Read Instructions** - Load 128-bit/64-bit/broadcast/unaligned data from memory to QR registers
2. **Write Instructions** - Store QR/accumulator data to memory
3. **Data Exchange Instructions** - Move data between AR/QR registers, zip/unzip, sign/zero extend
4. **Arithmetic Instructions** - Vector add/sub/mul, MAC operations, complex multiply, ReLU, clamping
5. **Comparison Instructions** - Vector min/max, compare equal/less-than/greater-than, saturation
6. **Bitwise Logical Instructions** - AND/OR/XOR/NOT on 128-bit QR registers
7. **Shift Instructions** - Vector shifts, spliced shifts, immediate/register-controlled shifts
8. **FFT Dedicated Instructions** - Radix-2 butterfly, complex multiply, bit-reverse, real FFT
9. **Assembly functions are declared with .balign 4 alignment**.

## Quick Instruction Reference

### Most Common Instructions (from esp-dl patterns)

| Instruction | Description |
|-------------|-------------|
| `esp.vld.128.ip qN, rs, imm` | Load 128-bit, addr += imm |
| `esp.vld.128.xp qN, rs1, rs2` | Load 128-bit, addr += rs2 |
| `esp.vldbc.16.ip qN, rs, imm` | Broadcast load 16-bit to 128-bit |
| `esp.vst.128.ip qN, rs, imm` | Store 128-bit, addr += imm |
| `esp.vadd.s16 qz, qx, qy` | Vector add 16-bit |
| `esp.vadd.s8 qz, qx, qy` | Vector add 8-bit |
| `esp.vsub.s16 qz, qx, qy` | Vector subtract 16-bit |
| `esp.vmul.s16 qz, qx, qy` | Vector multiply 16-bit (with SAR shift) |
| `esp.vmul.s8 qz, qx, qy` | Vector multiply 8-bit (with SAR shift) |
| `esp.vmulas.s16.qacc qx, qy` | Vector MAC 16-bit to QACC |
| `esp.vmulas.s8.qacc qx, qy` | Vector MAC 8-bit to QACC |
| `esp.vmulas.s16.xacc qx, qy` | Vector MAC 16-bit to XACC (dot product) |
| `esp.vsmulas.s16.qacc qx, qy, sel` | Scalar-vector MAC 16-bit to QACC |
| `esp.vmax.s16 qz, qx, qy` | Vector max 16-bit |
| `esp.vmin.s16 qz, qx, qy` | Vector min 16-bit |
| `esp.vcmp.eq.s16 qz, qx, qy` | Vector compare equal 16-bit |
| `esp.orq qz, qx, qy` | Bitwise OR 128-bit |
| `esp.andq qz, qx, qy` | Bitwise AND 128-bit |
| `esp.srcmb.s16.qacc qx, shift` | Shift QACC right and move to QR |
| `esp.zero.qacc` | Clear QACC_H and QACC_L |
| `esp.zero.xacc` | Clear XACC |
| `esp.zero.q qN` | Clear QR register |
| `esp.movx.w.sar rs` | Write SAR register |
| `esp.movx.r.sar.bytes rd` | Read SAR_BYTE |

## Optimization Workflow

When converting scalar functions to SIMD:

1. **Check data alignment**: Use 16-byte aligned data when possible (faster). Handle unaligned with `esp.ld.128.usar.ip` + `esp.src.q` / `esp.src.q.qup`
2. **Set SAR before multiply instructions**: `esp.movx.w.sar rs` to configure output shift
3. **Process in 128-bit chunks**: Loop count = total_elements / elements_per_128b (8 for 16-bit, 16 for 8-bit, 4 for 32-bit)
4. **Use fused load-arithmetic instructions** where possible to reduce instruction count:
   - `esp.vadd.s16.ld.incp qz, qx, qy, rs, imm` - add and load next
   - `esp.vmul.s16.ld.incp qz, qx, qy, rs, imm` - multiply and load next
5. **Use QACC/XACC for accumulation chains**: Initialize with `esp.zero.qacc`, accumulate with `esp.vmulas.*.qacc`, extract with `esp.srcmb.*.qacc`
6. **Use broadcast loads** for scalar operands: `esp.vldbc.16.ip qN, rs, 0`
7. **Handle remainders**: Process full 128-bit blocks in loop, handle tail elements separately

## Data Alignment Handling

### Aligned Access (16-byte boundary)
```asm
esp.vld.128.ip q0, a1, 16   # load and advance by 16
esp.vst.128.ip q0, a0, 16   # store and advance by 16
```

### Unaligned Access Pattern
```asm
# Get SAR_BYTE for output pointer
esp.ld.128.usar.ip q5, a0, 0
esp.movx.r.sar.bytes a5    # save output sar_byte

# Load unaligned data from input
esp.ld.128.usar.ip q0, a1, 16
esp.ld.128.usar.ip q1, a1, 16

# Extract aligned data from two consecutive loads
esp.src.q q2, q0, q1       # q2 = properly aligned 128-bit data
```

## Instruction Naming Convention

```
ESP.<operation>.<datatype>[.<variant>]
```

- **operation**: vadd, vsub, vmul, vmulas, vsadds, etc.
- **datatype**: s8, s16, s32 (signed); u8, u16 (unsigned)
- **variant**: ld.incp (load + addr++), st.incp (store + addr++), ld.xp (load + addr+=reg), etc.

## Label Naming Convention (REQUIRED)

For branch/loop targets, use **local labels**: either a plain number (`0:`, `1:`, referenced as `0f`/`0b`) or a `.L`-prefixed name (`.Lloop`, `.Lremainder`). Do **not** use full descriptive labels like `loop_start:` / `end_label:`.

- Numeric local labels keep tight inner loops compact and avoid name clashes.
- `.L` labels stay local to the file (not emitted into the symbol table) and clearly mark internal jump targets.

```asm
; GOOD — numeric local label
    loopgtz a6, 0f
        esp.vmin.s16.ld.incp q0, a3, q2, q0, q1
        esp.vld.128.ip       q1, a4, 16
        esp.vst.128.ip       q2, a2, 16
    0:

; GOOD — .L local label
    bgez  a9, .Lleft_shift
.Lright_shift_loop:
    ; ...
    bnez  a5, .Lright_shift_loop
.Lleft_shift:

; BAD — full descriptive global-style labels
right_shift_loop:
    ; ...
    bnez a5, right_shift_loop
```

## References

- **Full instruction listing**: See `references/instructions.md` for all instructions organized by category with syntax and semantics
- **Code examples**: See `references/examples.md` for patterns from esp-dl (conv2d, elementwise ops, depthwise conv, etc.)
