# ESP32-P4 SIMD Instruction Reference

## Table of Contents
- [Register Architecture](#register-architecture)
- [Data Overflow, Saturation, and Rounding](#data-overflow-saturation-and-rounding)
- [Read Instructions](#read-instructions)
- [Write Instructions](#write-instructions)
- [Data Exchange Instructions](#data-exchange-instructions)
- [Arithmetic Instructions](#arithmetic-instructions)
- [Comparison Instructions](#comparison-instructions)
- [Bitwise Logical Instructions](#bitwise-logical-instructions)
- [Shift Instructions](#shift-instructions)
- [FFT Dedicated Instructions](#fft-dedicated-instructions)

---

## Register Architecture

### QR (Vector) Registers
- 8 × 128-bit vector registers: `q0`–`q7`
- Also referenced as `qw`, `qx`, `qy`, `qz`, `qu`, `qv` in instruction syntax
- Each QR can be viewed as:
  - 16 × 8-bit elements (bytes)
  - 8 × 16-bit elements (half-words)
  - 4 × 32-bit elements (words)

### QACC (Quad Accumulator) Registers
- **QACC_H**: 256-bit accumulator (high half)
- **QACC_L**: 256-bit accumulator (low half)
- Together form a 512-bit accumulator for multiply-accumulate operations
- QACC_H/L are segmented based on element width:
  - S8/U8 MAC: 16 × 32-bit segments per QACC half
  - S16/U16 MAC: 4 × 64-bit segments per QACC half

### XACC (Cross Accumulator) Register
- 40-bit accumulator for dot-product (sum-of-products) operations
- `XACC[39:24]`: high 16 bits
- `XACC[23:0]`: low 24 bits (sign-extended to 32 when read)

### SAR (Shift Amount Register)
- 6-bit register (`SAR[5:0]`), controls right-shift amount for multiply/accumulate→QR moves
- Used by VMUL, CMUL, MOV.*.QACC, SRCMB instructions

### SAR_BYTE Register
- Byte-level shift amount for spliced shift instructions (SRC.Q etc.)
- Set automatically by ESP.LD.128.USAR.* instructions from address LSBs

### CFG (Configuration Register)
Control/status register accessed via `ESP.MOVX.R.CFG` / `ESP.MOVX.W.CFG`:

| Field     | Bits  | Access | Description |
|-----------|-------|--------|-------------|
| `vxsat_en`| 8     | R/W    | Enable saturation status tracking |
| `vxrm`    | 7:4   | R/W    | Rounding mode (see [Rounding Modes](#rounding-modes)) |
| `rm_exc`  | 3     | RO     | Exception flag for UNNECESSARY rounding mode |
| `vxsat`   | 2     | RO     | Saturation occurred flag (cleared on CFG read if vxsat_en=1) |
| `mis_ld`  | 1     | R/W    | Enable hardware misaligned load (0=force-align, 1=HW handle) |
| `mis_st`  | 0     | R/W    | Enable hardware misaligned store (0=force-align, 1=HW handle) |

### FFT_BIT_WIDTH Register
- 4-bit register controlling bit-reverse width (3–10 bits) for `ESP.BITREV`

### PERF (Performance Counter) Register
- 32-bit performance counter, accessed via `ESP.MOVX.R.PERF` / `ESP.MOVX.W.PERF`

---

## Data Overflow, Saturation, and Rounding

### Data Overflow Handling

When an operation result exceeds the bit-width of the destination register, two strategies are used:

1. **Saturation** (clipping): The result is clamped to the representable range.
   - Signed N-bit: clamped to `[-2^(N-1), 2^(N-1)-1]`
   - Unsigned N-bit: clamped to `[0, 2^N-1]`
   - Used by: VADD, VSUB, VSADDS, VSSUBS, VMULAS, VSMULAS, SRCMB, SRS, VCLAMP, VSAT

2. **Wraparound** (truncation): Only the lower N bits of the result are retained.
   - Used by: internal calculation results of most other instructions

### Saturation Status (vxsat)

- When `vxsat_en` is set in CFG, the `vxsat` bit records whether any saturation occurred
- `vxsat` is sticky: once set, it stays set until explicitly cleared
- Reading CFG (via `ESP.MOVX.R.CFG`) automatically clears `vxsat`

### Rounding Modes (vxrm)

The 4-bit `vxrm` field in CFG controls rounding behavior for right-shift operations:

| Mode         | vxrm | Description |
|-------------|------|-------------|
| FLOOR       | 0    | Round towards -∞ |
| CEILING     | 1    | Round towards +∞ |
| UP          | 2    | Round away from zero |
| DOWN        | 3    | Round towards zero (truncation) |
| HALF_UP     | 4    | Round to nearest; ties round up (away from zero) |
| HALF_DOWN   | 5    | Round to nearest; ties round down (towards zero) |
| HALF_EVEN   | 6    | Round to nearest; ties round to even neighbor |
| UNNECESSARY | 7    | Assert no rounding needed; `rm_exc` set if rounding would occur |

Rounding examples for common values:

| Input  | FLOOR | CEILING | UP | DOWN | HALF_UP | HALF_DOWN | HALF_EVEN |
|--------|-------|---------|----|------|---------|-----------|-----------|
| +5.5   | 5     | 6       | 6  | 5    | 6       | 5         | 6         |
| +2.5   | 2     | 3       | 3  | 2    | 3       | 2         | 2         |
| +1.6   | 1     | 2       | 2  | 1    | 2       | 2         | 2         |
| +1.1   | 1     | 2       | 2  | 1    | 1       | 1         | 1         |
| +1.0   | 1     | 1       | 1  | 1    | 1       | 1         | 1         |
| -1.0   | -1    | -1      | -1 | -1   | -1      | -1        | -1        |
| -1.1   | -2    | -1      | -2 | -1   | -1      | -1        | -1        |
| -1.6   | -2    | -1      | -2 | -1   | -2      | -2        | -2        |
| -2.5   | -3    | -2      | -3 | -2   | -3      | -2        | -2        |
| -5.5   | -6    | -5      | -6 | -5   | -6      | -5        | -6        |

### Data Alignment

| Format  | Bits  | Aligned Address LSBs |
|---------|-------|----------------------|
| 1-byte  | 8     | xxxx                |
| 2-byte  | 16    | xxx0                |
| 4-byte  | 32    | xx00                |
| 8-byte  | 64    | x000                |
| 16-byte | 128   | 0000                |

**Force alignment mode** (mis_ld=0 / mis_st=0): Low address bits are forced to 0.
**Hardware misaligned mode** (mis_ld=1 / mis_st=1): Hardware splits misaligned access into multiple aligned accesses.

---

## Read Instructions

Load data from memory into vector registers. Address post-increment variants available.

### 128-bit Vector Loads

| Instruction | Description |
|-------------|-------------|
| `ESP.VLD.128.IP qu, rs1, imm` | Load 16 bytes, rs1 += imm (imm: -128 to 112, step 16) |
| `ESP.VLD.128.XP qu, rs1, rs2` | Load 16 bytes, rs1 += rs2 |

### 64-bit Vector Loads (to high/low half of QR)

| Instruction | Description |
|-------------|-------------|
| `ESP.VLD.H.64.IP qu, rs1, imm` | Load 8 bytes to QR[127:64], rs1 += imm |
| `ESP.VLD.H.64.XP qu, rs1, rs2` | Load 8 bytes to QR[127:64], rs1 += rs2 |
| `ESP.VLD.L.64.IP qu, rs1, imm` | Load 8 bytes to QR[63:0], rs1 += imm |
| `ESP.VLD.L.64.XP qu, rs1, rs2` | Load 8 bytes to QR[63:0], rs1 += rs2 |

### Broadcast Loads (scalar to vector)

| Instruction | Description |
|-------------|-------------|
| `ESP.VLDBC.8.IP qu, rs1, imm` | Load 1 byte, broadcast to 16 bytes, rs1 += imm |
| `ESP.VLDBC.8.XP qu, rs1, rs2` | Load 1 byte, broadcast to 16 bytes, rs1 += rs2 |
| `ESP.VLDBC.16.IP qu, rs1, imm` | Load 2 bytes, broadcast to 8 × 16-bit, rs1 += imm |
| `ESP.VLDBC.16.XP qu, rs1, rs2` | Load 2 bytes, broadcast to 8 × 16-bit, rs1 += rs2 |
| `ESP.VLDBC.32.IP qu, rs1, imm` | Load 4 bytes, broadcast to 4 × 32-bit, rs1 += imm |
| `ESP.VLDBC.32.XP qu, rs1, rs2` | Load 4 bytes, broadcast to 4 × 32-bit, rs1 += rs2 |
| `ESP.VLDHBC.16.INCP qu, qz, rs1` | Load 16 bytes, broadcast each 16-bit element to 32-bit: low halves → `qu`, high halves → `qz`. rs1 += 16 |

### Unaligned 128-bit Loads

| Instruction | Description |
|-------------|-------------|
| `ESP.LD.128.USAR.IP qu, rs1, imm` | Load 16 bytes (set SAR_BYTE from addr), rs1 += imm |
| `ESP.LD.128.USAR.XP qu, rs1, rs2` | Load 16 bytes (set SAR_BYTE from addr), rs1 += rs2 |

### QACC Loads (sign/zero extended)

| Instruction | Description |
|-------------|-------------|
| `ESP.LDQA.U8.128.IP rs1, imm` | Load 16 bytes, zero-extend each 8-bit to 20-bit to QACC_H/L, rs1 += imm |
| `ESP.LDQA.U8.128.XP rs1, rs2` | Same, rs1 += rs2 |
| `ESP.LDQA.U16.128.IP rs1, imm` | Load 16 bytes, zero-extend each 16-bit to 40-bit to QACC_H/L |
| `ESP.LDQA.U16.128.XP rs1, rs2` | Same, rs1 += rs2 |
| `ESP.LDQA.S8.128.IP rs1, imm` | Load 16 bytes, sign-extend each 8-bit to 20-bit to QACC_H/L |
| `ESP.LDQA.S8.128.XP rs1, rs2` | Same, rs1 += rs2 |
| `ESP.LDQA.S16.128.IP rs1, imm` | Load 16 bytes, sign-extend each 16-bit to 40-bit to QACC_H/L |
| `ESP.LDQA.S16.128.XP rs1, rs2` | Same, rs1 += rs2 |

### QACC/XACC Direct Loads

Load data directly into QACC_H, QACC_L, XACC, or UA_STATE from memory. These instructions do NOT use a QR register — the load goes directly into the special register.

| Instruction | Description |
|-------------|-------------|
| `ESP.LD.QACC.H.H.128.IP rs1, imm` | Load 16 bytes to QACC_H[255:128], rs1 += imm (imm: -2048 to 2032, step 16) |
| `ESP.LD.QACC.H.L.128.IP rs1, imm` | Load 16 bytes to QACC_H[127:0], rs1 += imm |
| `ESP.LD.QACC.L.H.128.IP rs1, imm` | Load 16 bytes to QACC_L[255:128], rs1 += imm |
| `ESP.LD.QACC.L.L.128.IP rs1, imm` | Load 16 bytes to QACC_L[127:0], rs1 += imm |
| `ESP.LD.XACC.IP rs1, imm` | Load 8 bytes to XACC (lower 40 bits), rs1 += imm (imm: -1024 to 1016, step 8) |
| `ESP.LD.UA.STATE.IP rs1, imm` | Load 16 bytes to UA_STATE, rs1 += imm |

### Indexed/Extended Loads

| Instruction | Description |
|-------------|-------------|
| `ESP.LDXQ.32 qu, qw, rs1, sel4, sel8` | rs1 += qw[sel8*16+15:sel8*16] << 2 as index, then load 4 bytes into qu[32*sel4+31:32*sel4] |
| `ESP.VLDEXT.U8.IP qu, rs1, imm` | Vector unsigned-extend 8-bit load segments to 16-bit |
| `ESP.VLDEXT.U8.XP qu, rs1, rs2` | Same, rs1 += rs2 |
| `ESP.VLDEXT.U16.IP qu, rs1, imm` | Vector unsigned-extend 16-bit load segments to 32-bit |
| `ESP.VLDEXT.U16.XP qu, rs1, rs2` | Same, rs1 += rs2 |
| `ESP.VLDEXT.S8.IP qu, rs1, imm` | Vector signed-extend 8-bit load segments to 16-bit |
| `ESP.VLDEXT.S8.XP qu, rs1, rs2` | Same, rs1 += rs2 |
| `ESP.VLDEXT.S16.IP qu, rs1, imm` | Vector signed-extend 16-bit load segments to 32-bit |
| `ESP.VLDEXT.S16.XP qu, rs1, rs2` | Same, rs1 += rs2 |

---

## Write Instructions

Store data from vector registers or accumulators to memory.

### 128-bit Vector Stores

| Instruction | Description |
|-------------|-------------|
| `ESP.VST.128.IP qu, rs1, imm` | Store 16 bytes, rs1 += imm (imm: -128 to 112, step 16) |
| `ESP.VST.128.XP qu, rs1, rs2` | Store 16 bytes, rs1 += rs2 |

### 64-bit Vector Stores

| Instruction | Description |
|-------------|-------------|
| `ESP.VST.H.64.IP qu, rs1, imm` | Store QR[127:64] (8 bytes), rs1 += imm |
| `ESP.VST.H.64.XP qu, rs1, rs2` | Store QR[127:64], rs1 += rs2 |
| `ESP.VST.L.64.IP qu, rs1, imm` | Store QR[63:0] (8 bytes), rs1 += imm |
| `ESP.VST.L.64.XP qu, rs1, rs2` | Store QR[63:0], rs1 += rs2 |

### QACC Stores

Store QACC_H/L data directly to memory. No QR register is involved — the data flows directly from the special register to memory.

| Instruction | Description |
|-------------|-------------|
| `ESP.ST.QACC.H.H.128.IP rs1, imm` | Store QACC_H[255:128] to memory, rs1 += imm (imm: -2048 to 2032, step 16) |
| `ESP.ST.QACC.H.L.128.IP rs1, imm` | Store QACC_H[127:0] to memory, rs1 += imm |
| `ESP.ST.QACC.L.H.128.IP rs1, imm` | Store QACC_L[255:128] to memory, rs1 += imm |
| `ESP.ST.QACC.L.L.128.IP rs1, imm` | Store QACC_L[127:0] to memory, rs1 += imm |

### XACC Stores

Store XACC data directly to memory (sign-extended or zero-extended to 8 bytes). No QR register is involved.

| Instruction | Description |
|-------------|-------------|
| `ESP.ST.U.XACC.IP rs1, imm` | Zero-extend XACC[39:0] to 64-bit and store, rs1 += imm (imm: -1024 to 1016, step 8) |
| `ESP.ST.S.XACC.IP rs1, imm` | Sign-extend XACC[39:0] to 64-bit and store, rs1 += imm |

### Other Stores

| Instruction | Description |
|-------------|-------------|
| `ESP.ST.UA.STATE.IP rs1, imm` | Store UA_STATE to memory, rs1 += imm |
| `ESP.STXQ.32 qu, qw, rs1, sel4, sel8` | Store qu[32*sel4+31:32*sel4] to address rs1 + qw[sel8*16+15:sel8*16] << 2, then rs1 += qw[sel8*16+15:sel8*16] << 2 |

---

## Data Exchange Instructions

Move data between different register types and perform reordering.

### AR to QR Element Moves

| Instruction | Description |
|-------------|-------------|
| `ESP.MOVI.8.A qu, rs1, sel16` | Move QR[sel16*8+7:sel16*8] (1 byte) to AR |
| `ESP.MOVI.16.A qu, rs1, sel8` | Move QR[sel8*16+15:sel8*16] (2 bytes) to AR |
| `ESP.MOVI.32.A qu, rs1, sel4` | Move QR[sel4*32+31:sel4*32] (4 bytes) to AR |
| `ESP.MOVI.8.Q qu, rs1, sel16` | Move AR to QR[sel16*8+7:sel16*8] |
| `ESP.MOVI.16.Q qu, rs1, sel8` | Move AR to QR[sel8*16+15:sel8*16] |
| `ESP.MOVI.32.Q qu, rs1, sel4` | Move AR to QR[sel4*32+31:sel4*32] |

### Special Register Moves

| Instruction | Description |
|-------------|-------------|
| `ESP.MOVX.R.CFG rd` | Read CFG to AR. Also auto-clears vxsat bit. |
| `ESP.MOVX.W.CFG rs1` | Write AR to CFG |
| `ESP.MOVX.R.SAR.BYTES rd` | Read SAR_BYTE to AR |
| `ESP.MOVX.W.SAR.BYTES rs1` | Write AR to SAR_BYTE |
| `ESP.MOVX.R.SAR rd` | Read SAR to AR |
| `ESP.MOVX.W.SAR rs1` | Write AR to SAR |
| `ESP.MOVX.R.FFT.BIT.WIDTH rd` | Read FFT_BIT_WIDTH to AR |
| `ESP.MOVX.W.FFT.BIT.WIDTH rs1` | Write AR to FFT_BIT_WIDTH |
| `ESP.MOVX.R.PERF rd, rs1` | Read PERF counter to AR |
| `ESP.MOVX.W.PERF rs1` | Write AR to PERF counter |
| `ESP.MOVX.R.XACC.H rd` | Read XACC[39:24] (high 16 bits) to AR |
| `ESP.MOVX.R.XACC.L rd` | Read XACC[23:0] (low 24 bits, sign-extended to 32) to AR |
| `ESP.MOVX.W.XACC.H rs1` | Write AR to XACC[39:24] |
| `ESP.MOVX.W.XACC.L rs1` | Write AR to XACC[23:0] |

### QR Data Movement

| Instruction | Description |
|-------------|-------------|
| `ESP.VZIP.8 qz, qx, qy` | Zip/interleave two QR by 8-bit elements |
| `ESP.VZIP.16 qz, qx, qy` | Zip/interleave two QR by 16-bit elements |
| `ESP.VZIP.32 qz, qx, qy` | Zip/interleave two QR by 32-bit elements |
| `ESP.VUNZIP.8 qz, qx, qy` | Unzip/deinterleave two QR by 8-bit elements |
| `ESP.VUNZIP.16 qz, qx, qy` | Unzip/deinterleave two QR by 16-bit elements |
| `ESP.VUNZIP.32 qz, qx, qy` | Unzip/deinterleave two QR by 32-bit elements |
| `ESP.VZIPT.8 qz, qx, qy` | Zip three QR by 8-bit elements |
| `ESP.VZIPT.16 qz, qx, qy` | Zip three QR by 16-bit elements |
| `ESP.VUNZIPT.8 qz, qx, qy` | Unzip three QR by 8-bit elements |
| `ESP.VUNZIPT.16 qz, qx, qy` | Unzip three QR by 16-bit elements |

### Sign/Zero Extension

| Instruction | Description |
|-------------|-------------|
| `ESP.VEXT.U8 qu, qv` | Zero-extend each 8-bit element to 16-bit |
| `ESP.VEXT.S8 qu, qv` | Sign-extend each 8-bit element to 16-bit |
| `ESP.VEXT.U16 qu, qv` | Zero-extend each 16-bit element to 32-bit |
| `ESP.VEXT.S16 qu, qv` | Sign-extend each 16-bit element to 32-bit |

### QR to QACC Moves

Load a QR register's data into QACC_H/L, with sign-extension or zero-extension.

| Instruction | Description |
|-------------|-------------|
| `ESP.MOV.S8.QACC qu` | Sign-extend each of 16 × 8-bit segments in `qu` to 32-bit, write to QACC_H/L |
| `ESP.MOV.U8.QACC qu` | Zero-extend each of 16 × 8-bit segments in `qu` to 32-bit, write to QACC_H/L |
| `ESP.MOV.S16.QACC qu` | Sign-extend each of 8 × 16-bit segments in `qu` to 64-bit, write to QACC_H/L |
| `ESP.MOV.U16.QACC qu` | Zero-extend each of 8 × 16-bit segments in `qu` to 64-bit, write to QACC_H/L |

> **Note:** To extract data FROM QACC back TO a QR register, use the SRCMB instructions (see [QACC/XACC Shift and Move](#qaccxacc-shift-and-move)).

### Register Clears

| Instruction | Description |
|-------------|-------------|
| `ESP.ZERO.Q qN` | Clear QR register to zero |
| `ESP.ZERO.QACC` | Clear QACC_H and QACC_L to zero |
| `ESP.ZERO.XACC` | Clear XACC to zero |

---

## Arithmetic Instructions

**Important:** All VADD and VSUB variants (both signed and unsigned) perform **saturating** arithmetic. Results are clamped to the representable range of the destination element type.

### Vector Addition (Signed, Saturating)

| Instruction | Description |
|-------------|-------------|
| `ESP.VADD.S8 qv, qx, qy` | 16 × 8-bit signed saturating add: `min(max(qx[i]+qy[i], -2^7), 2^7-1)` |
| `ESP.VADD.S16 qv, qx, qy` | 8 × 16-bit signed saturating add: `min(max(qx[i]+qy[i], -2^15), 2^15-1)` |
| `ESP.VADD.S32 qv, qx, qy` | 4 × 32-bit signed saturating add: `min(max(qx[i]+qy[i], -2^31), 2^31-1)` |
| `ESP.VADD.S8.LD.INCP qu, rs1, qv, qx, qy` | VADD.S8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VADD.S16.LD.INCP qu, rs1, qv, qx, qy` | VADD.S16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VADD.S32.LD.INCP qu, rs1, qv, qx, qy` | VADD.S32 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VADD.S8.ST.INCP qu, rs1, qv, qx, qy` | VADD.S8 + store qu to memory, rs1 += 16 |
| `ESP.VADD.S16.ST.INCP qu, rs1, qv, qx, qy` | VADD.S16 + store qu to memory, rs1 += 16 |
| `ESP.VADD.S32.ST.INCP qu, rs1, qv, qx, qy` | VADD.S32 + store qu to memory, rs1 += 16 |

### Vector Addition (Unsigned, Saturating)

| Instruction | Description |
|-------------|-------------|
| `ESP.VADD.U8 qv, qx, qy` | 16 × 8-bit unsigned saturating add: `min(qx[i]+qy[i], 2^8-1)` |
| `ESP.VADD.U16 qv, qx, qy` | 8 × 16-bit unsigned saturating add: `min(qx[i]+qy[i], 2^16-1)` |
| `ESP.VADD.U32 qv, qx, qy` | 4 × 32-bit unsigned saturating add: `min(qx[i]+qy[i], 2^32-1)` |
| `ESP.VADD.U8.LD.INCP qu, rs1, qv, qx, qy` | VADD.U8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VADD.U16.LD.INCP qu, rs1, qv, qx, qy` | VADD.U16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VADD.U32.LD.INCP qu, rs1, qv, qx, qy` | VADD.U32 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VADD.U8.ST.INCP qu, rs1, qv, qx, qy` | VADD.U8 + store qu to memory, rs1 += 16 |
| `ESP.VADD.U16.ST.INCP qu, rs1, qv, qx, qy` | VADD.U16 + store qu to memory, rs1 += 16 |
| `ESP.VADD.U32.ST.INCP qu, rs1, qv, qx, qy` | VADD.U32 + store qu to memory, rs1 += 16 |

### Scalar Saturated Vector Addition

**Note:** VSADDS adds a **scalar** from an AR register (`rs1`) to each element of a QR vector. Different from VADD which uses two QR vectors.

| Instruction | Description |
|-------------|-------------|
| `ESP.VSADDS.S8 qv, qx, rs1` | 16 × 8-bit: `qv[i] = min(max(qx[i] + rs1[7:0], -2^7), 2^7-1)` |
| `ESP.VSADDS.S16 qv, qx, rs1` | 8 × 16-bit: `qv[i] = min(max(qx[i] + rs1[15:0], -2^15), 2^15-1)` |
| `ESP.VSADDS.U8 qv, qx, rs1` | 16 × 8-bit: `qv[i] = min(qx[i] + rs1[7:0], 2^8-1)` |
| `ESP.VSADDS.U16 qv, qx, rs1` | 8 × 16-bit: `qv[i] = min(qx[i] + rs1[15:0], 2^16-1)` |

### Vector Subtraction (Signed, Saturating)

| Instruction | Description |
|-------------|-------------|
| `ESP.VSUB.S8 qv, qx, qy` | 16 × 8-bit signed saturating sub (qx - qy) |
| `ESP.VSUB.S16 qv, qx, qy` | 8 × 16-bit signed saturating sub |
| `ESP.VSUB.S32 qv, qx, qy` | 4 × 32-bit signed saturating sub |
| `ESP.VSUB.S8.LD.INCP qu, rs1, qv, qx, qy` | VSUB.S8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VSUB.S16.LD.INCP qu, rs1, qv, qx, qy` | VSUB.S16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VSUB.S32.LD.INCP qu, rs1, qv, qx, qy` | VSUB.S32 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VSUB.S8.ST.INCP qu, rs1, qv, qx, qy` | VSUB.S8 + store qu to memory, rs1 += 16 |
| `ESP.VSUB.S16.ST.INCP qu, rs1, qv, qx, qy` | VSUB.S16 + store qu to memory, rs1 += 16 |
| `ESP.VSUB.S32.ST.INCP qu, rs1, qv, qx, qy` | VSUB.S32 + store qu to memory, rs1 += 16 |

### Vector Subtraction (Unsigned, Saturating)

| Instruction | Description |
|-------------|-------------|
| `ESP.VSUB.U8 qv, qx, qy` | 16 × 8-bit unsigned saturating sub (qx - qy): `min(qx[i]-qy[i], 2^8-1)` |
| `ESP.VSUB.U16 qv, qx, qy` | 8 × 16-bit unsigned saturating sub: `min(qx[i]-qy[i], 2^16-1)` |
| `ESP.VSUB.U32 qv, qx, qy` | 4 × 32-bit unsigned saturating sub: `min(qx[i]-qy[i], 2^32-1)` |
| `ESP.VSUB.U8.LD.INCP qu, rs1, qv, qx, qy` | VSUB.U8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VSUB.U16.LD.INCP qu, rs1, qv, qx, qy` | VSUB.U16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VSUB.U32.LD.INCP qu, rs1, qv, qx, qy` | VSUB.U32 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VSUB.U8.ST.INCP qu, rs1, qv, qx, qy` | VSUB.U8 + store qu to memory, rs1 += 16 |
| `ESP.VSUB.U16.ST.INCP qu, rs1, qv, qx, qy` | VSUB.U16 + store qu to memory, rs1 += 16 |
| `ESP.VSUB.U32.ST.INCP qu, rs1, qv, qx, qy` | VSUB.U32 + store qu to memory, rs1 += 16 |

### Scalar Saturated Vector Subtraction

**Note:** VSSUBS subtracts a **scalar** from an AR register (`rs1`) from each element of a QR vector.

| Instruction | Description |
|-------------|-------------|
| `ESP.VSSUBS.S8 qv, qx, rs1` | 16 × 8-bit: `qv[i] = min(max(qx[i] - rs1[7:0], -2^7), 2^7-1)` |
| `ESP.VSSUBS.S16 qv, qx, rs1` | 8 × 16-bit: `qv[i] = min(max(qx[i] - rs1[15:0], -2^15), 2^15-1)` |
| `ESP.VSSUBS.U8 qv, qx, rs1` | 16 × 8-bit: `qv[i] = min(qx[i] - rs1[7:0], 2^8-1)` |
| `ESP.VSSUBS.U16 qv, qx, rs1` | 8 × 16-bit: `qv[i] = min(qx[i] - rs1[15:0], 2^16-1)` |

### Vector Multiplication

Multiplies are followed by an **arithmetic** right shift of SAR bits; the lower half is kept. The rounding mode (vxrm in CFG) controls rounding during the shift.

| Instruction | Description |
|-------------|-------------|
| `ESP.VMUL.S8 qz, qx, qy` | 16 × 8-bit signed mul, result >> SAR, keep low 8 |
| `ESP.VMUL.S16 qz, qx, qy` | 8 × 16-bit signed mul, result >> SAR, keep low 16 |
| `ESP.VMUL.U8 qz, qx, qy` | 16 × 8-bit unsigned mul, result >> SAR, keep low 8 |
| `ESP.VMUL.U16 qz, qx, qy` | 8 × 16-bit unsigned mul, result >> SAR, keep low 16 |
| `ESP.VMUL.S8.LD.INCP qu, rs1, qz, qx, qy` | V.MUL.S8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMUL.S16.LD.INCP qu, rs1, qz, qx, qy` | V.MUL.S16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMUL.U8.LD.INCP qu, rs1, qz, qx, qy` | V.MUL.U8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMUL.U16.LD.INCP qu, rs1, qz, qx, qy` | V.MUL.U16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMUL.S8.ST.INCP qu, rs1, qz, qx, qy` | V.MUL.S8 + store qu to memory, rs1 += 16 |
| `ESP.VMUL.S16.ST.INCP qu, rs1, qz, qx, qy` | V.MUL.S16 + store qu to memory, rs1 += 16 |
| `ESP.VMUL.U8.ST.INCP qu, rs1, qz, qx, qy` | V.MUL.U8 + store qu to memory, rs1 += 16 |
| `ESP.VMUL.U16.ST.INCP qu, rs1, qz, qx, qy` | V.MUL.U16 + store qu to memory, rs1 += 16 |

### Extended Output Vector Multiplication

| Instruction | Description |
|-------------|-------------|
| `ESP.VMUL.S32.S16xS16 qz, qx, qy` | 8 × 16-bit signed mul, result >> SAR, produce 8 × 32-bit results (full 32-bit result kept) |
| `ESP.VMUL.S16.S8xS8 qz, qv, qx, qy` | 16 × 8-bit signed mul → 16 × 16-bit: low 8 results to `qz`, high 8 results to `qv`. Each result >> SAR before storing. |

### Vector Complex Multiplication

Operates on pairs as complex numbers (real, imag). The `sel4` immediate (0–3) selects which quadrant of the 128-bit register to operate on. Result is right-shifted by SAR.

#### Signed Complex Multiplication

| Instruction | Description |
|-------------|-------------|
| `ESP.CMUL.S16 qz, qx, qy, sel4` | 16-bit signed complex multiply (sel4 controls operand quadrants). `real = (a.re*b.re - a.im*b.im) >> SAR`, `imag = (a.re*b.im + a.im*b.re) >> SAR` (or conjugate variants per sel4) |
| `ESP.CMUL.S16.LD.INCP qu, rs1, qz, qx, qy, sel4` | CMUL.S16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.CMUL.S16.ST.INCP qu, rs1, qz, qx, qy, sel4` | CMUL.S16 + store qu to memory, rs1 += 16 |
| `ESP.CMUL.S8 qz, qx, qy, sel4` | 8-bit signed complex multiply (sel4 controls operand halves) |
| `ESP.CMUL.S8.LD.INCP qu, rs1, qz, qx, qy, sel4` | CMUL.S8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.CMUL.S8.ST.INCP qu, rs1, qz, qx, qy, sel4` | CMUL.S8 + store qu to memory, rs1 += 16 |

#### Unsigned Complex Multiplication

| Instruction | Description |
|-------------|-------------|
| `ESP.CMUL.U16 qz, qx, qy, sel4` | 16-bit unsigned complex multiply |
| `ESP.CMUL.U16.LD.INCP qu, rs1, qz, qx, qy, sel4` | CMUL.U16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.CMUL.U16.ST.INCP qu, rs1, qz, qx, qy, sel4` | CMUL.U16 + store qu to memory, rs1 += 16 |
| `ESP.CMUL.U8 qz, qx, qy, sel4` | 8-bit unsigned complex multiply |
| `ESP.CMUL.U8.LD.INCP qu, rs1, qz, qx, qy, sel4` | CMUL.U8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.CMUL.U8.ST.INCP qu, rs1, qz, qx, qy, sel4` | CMUL.U8 + store qu to memory, rs1 += 16 |

### Vector Multiply-Accumulate to QACC

Accumulates element-wise products into QACC segments. Results are **saturated** to the accumulator segment width.

**Segment widths:**
- S8/U8 MAC: 32-bit per QACC segment (16 segments per QACC half)
- S16/U16 MAC: 64-bit per QACC segment (4 segments per QACC half)

#### Signed MAC to QACC

| Instruction | Description |
|-------------|-------------|
| `ESP.VMULAS.S8.QACC qx, qy` | 16 × S8 mul, accumulate 32-bit to QACC_H/L (saturated to 32-bit signed) |
| `ESP.VMULAS.S16.QACC qx, qy` | 8 × S16 mul, accumulate 64-bit to QACC_H/L (saturated to 64-bit signed) |

#### Unsigned MAC to QACC

| Instruction | Description |
|-------------|-------------|
| `ESP.VMULAS.U8.QACC qx, qy` | 16 × U8 mul, accumulate 32-bit to QACC_H/L (saturated to 32-bit unsigned) |
| `ESP.VMULAS.U16.QACC qx, qy` | 8 × U16 mul, accumulate 64-bit to QACC_H/L (saturated to 64-bit unsigned) |

#### Fused MAC variants (signed, with memory access)

| Instruction | Description |
|-------------|-------------|
| `ESP.VMULAS.S8.QACC.LD.IP qu, rs, imm, qx, qy` | MAC S8 to QACC + load (rs += imm) |
| `ESP.VMULAS.S16.QACC.LD.IP qu, rs, imm, qx, qy` | MAC S16 to QACC + load |
| `ESP.VMULAS.S8.QACC.LD.XP qu, rs1, rs2, qx, qy` | MAC S8 to QACC + load (rs1 += rs2) |
| `ESP.VMULAS.S16.QACC.LD.XP qu, rs1, rs2, qx, qy` | MAC S16 to QACC + load (rs1 += rs2) |
| `ESP.VMULAS.S8.QACC.ST.IP qu, rs, imm, qx, qy` | MAC S8 to QACC + store |
| `ESP.VMULAS.S16.QACC.ST.IP qu, rs, imm, qx, qy` | MAC S16 to QACC + store |
| `ESP.VMULAS.S8.QACC.ST.XP qu, rs1, rs2, qx, qy` | MAC S8 to QACC + store (rs1 += rs2) |
| `ESP.VMULAS.S16.QACC.ST.XP qu, rs1, rs2, qx, qy` | MAC S16 to QACC + store (rs1 += rs2) |
| `ESP.VMULAS.S8.QACC.LDBC.INCP qu, rs, imm, qx, qy` | MAC S8 to QACC + broadcast load |
| `ESP.VMULAS.S16.QACC.LDBC.INCP qu, rs, imm, qx, qy` | MAC S16 to QACC + broadcast load |

#### Fused MAC variants (unsigned, with memory access)

| Instruction | Description |
|-------------|-------------|
| `ESP.VMULAS.U8.QACC.LD.IP qu, rs, imm, qx, qy` | MAC U8 to QACC + load |
| `ESP.VMULAS.U16.QACC.LD.IP qu, rs, imm, qx, qy` | MAC U16 to QACC + load |
| `ESP.VMULAS.U8.QACC.LD.XP qu, rs1, rs2, qx, qy` | MAC U8 to QACC + load (rs1 += rs2) |
| `ESP.VMULAS.U16.QACC.LD.XP qu, rs1, rs2, qx, qy` | MAC U16 to QACC + load (rs1 += rs2) |
| `ESP.VMULAS.U8.QACC.ST.IP qu, rs, imm, qx, qy` | MAC U8 to QACC + store |
| `ESP.VMULAS.U16.QACC.ST.IP qu, rs, imm, qx, qy` | MAC U16 to QACC + store |
| `ESP.VMULAS.U8.QACC.ST.XP qu, rs1, rs2, qx, qy` | MAC U8 to QACC + store (rs1 += rs2) |
| `ESP.VMULAS.U16.QACC.ST.XP qu, rs1, rs2, qx, qy` | MAC U16 to QACC + store (rs1 += rs2) |
| `ESP.VMULAS.U8.QACC.LDBC.INCP qu, rs, imm, qx, qy` | MAC U8 to QACC + broadcast load |
| `ESP.VMULAS.U16.QACC.LDBC.INCP qu, rs, imm, qx, qy` | MAC U16 to QACC + broadcast load |

### Vector Multiply-Accumulate to XACC

Computes the **sum** of all element-wise products (dot-product style). Result accumulated in XACC (40-bit, saturated).

| Instruction | Description |
|-------------|-------------|
| `ESP.VMULAS.S8.XACC qx, qy` | 16 × S8 mul, sum all to XACC (40-bit, saturated) |
| `ESP.VMULAS.S16.XACC qx, qy` | 8 × S16 mul, sum all to XACC (40-bit, saturated) |
| `ESP.VMULAS.U8.XACC qx, qy` | 16 × U8 mul, sum all to XACC (40-bit, saturated) |
| `ESP.VMULAS.U16.XACC qx, qy` | 8 × U16 mul, sum all to XACC (40-bit, saturated) |

**Fused variants:**

| Instruction | Description |
|-------------|-------------|
| `ESP.VMULAS.S8.XACC.LD.IP qu, rs, imm, qx, qy` | MAC to XACC + load |
| `ESP.VMULAS.S16.XACC.LD.IP qu, rs, imm, qx, qy` | MAC to XACC + load |
| `ESP.VMULAS.S8.XACC.LD.XP qu, rs1, rs2, qx, qy` | MAC to XACC + load (rs1 += rs2) |
| `ESP.VMULAS.S16.XACC.LD.XP qu, rs1, rs2, qx, qy` | MAC to XACC + load (rs1 += rs2) |
| `ESP.VMULAS.S8.XACC.ST.IP qu, rs, imm, qx, qy` | MAC to XACC + store |
| `ESP.VMULAS.S16.XACC.ST.IP qu, rs, imm, qx, qy` | MAC to XACC + store |
| `ESP.VMULAS.S8.XACC.ST.XP qu, rs1, rs2, qx, qy` | MAC to XACC + store (rs1 += rs2) |
| `ESP.VMULAS.S16.XACC.ST.XP qu, rs1, rs2, qx, qy` | MAC to XACC + store (rs1 += rs2) |

### Scalar-Vector Multiply-Accumulate to QACC

One operand is a vector (`qx`), the other is a **scalar element** selected from `qy` using `sel`. Accumulates into QACC with saturation.

#### Signed Scalar-Vector MAC

| Instruction | Description |
|-------------|-------------|
| `ESP.VSMULAS.S8.QACC qx, qy, sel16` | Select 1 of 16 bytes from qy; 16 × S8 MAC to QACC (saturated to 32-bit signed) |
| `ESP.VSMULAS.S16.QACC qx, qy, sel8` | Select 1 of 8 half-words from qy; 8 × S16 MAC to QACC (saturated to 64-bit signed) |
| `ESP.VSMULAS.S8.QACC.LD.INCP qu, rs, qx, qy, sel16` | VSMULAS.S8.QACC + load 16 bytes to qu, rs += 16 |
| `ESP.VSMULAS.S16.QACC.LD.INCP qu, rs, qx, qy, sel8` | VSMULAS.S16.QACC + load 16 bytes to qu, rs += 16 |

#### Unsigned Scalar-Vector MAC

| Instruction | Description |
|-------------|-------------|
| `ESP.VSMULAS.U8.QACC qx, qy, sel16` | Select 1 of 16 bytes from qy; 16 × U8 MAC to QACC (saturated to 32-bit unsigned) |
| `ESP.VSMULAS.U16.QACC qx, qy, sel8` | Select 1 of 8 half-words from qy; 8 × U16 MAC to QACC (saturated to 64-bit unsigned) |
| `ESP.VSMULAS.U8.QACC.LD.INCP qu, rs, qx, qy, sel16` | VSMULAS.U8.QACC + load 16 bytes to qu, rs += 16 |
| `ESP.VSMULAS.U16.QACC.LD.INCP qu, rs, qx, qy, sel8` | VSMULAS.U16.QACC + load 16 bytes to qu, rs += 16 |

### Complex Multiply-Accumulate to QACC

| Instruction | Description |
|-------------|-------------|
| `ESP.VCMULAS.S8.QACC.H qx, qy` | Complex MAC S8 to QACC_H |
| `ESP.VCMULAS.S8.QACC.L qx, qy` | Complex MAC S8 to QACC_L |
| `ESP.VCMULAS.S16.QACC.H qx, qy` | Complex MAC S16 to QACC_H |
| `ESP.VCMULAS.S16.QACC.L qx, qy` | Complex MAC S16 to QACC_L |

**Fused variants:**

| Instruction | Description |
|-------------|-------------|
| `ESP.VCMULAS.S8.QACC.H.LD.IP qu, rs, imm, qx, qy` | Complex MAC to QACC_H + load |
| `ESP.VCMULAS.S8.QACC.L.LD.IP qu, rs, imm, qx, qy` | Complex MAC to QACC_L + load |
| `ESP.VCMULAS.S16.QACC.H.LD.IP qu, rs, imm, qx, qy` | Complex MAC S16 to QACC_H + load |
| `ESP.VCMULAS.S16.QACC.L.LD.IP qu, rs, imm, qx, qy` | Complex MAC S16 to QACC_L + load |
| `ESP.VCMULAS.S8.QACC.H.LD.XP qu, rs1, rs2, qx, qy` | Complex MAC to QACC_H + load (rs1 += rs2) |
| `ESP.VCMULAS.S8.QACC.L.LD.XP qu, rs1, rs2, qx, qy` | Complex MAC to QACC_L + load (rs1 += rs2) |
| `ESP.VCMULAS.S16.QACC.H.LD.XP qu, rs1, rs2, qx, qy` | Complex MAC S16 to QACC_H + load (rs1 += rs2) |
| `ESP.VCMULAS.S16.QACC.L.LD.XP qu, rs1, rs2, qx, qy` | Complex MAC S16 to QACC_L + load (rs1 += rs2) |

### QACC/XACC Shift and Move

| Instruction | Description |
|-------------|-------------|
| `ESP.SRCMB.S8.QACC qx, imm` | Shift QACC_H/L right by `imm` per 32-bit segment, saturate to S8, move to qx |
| `ESP.SRCMB.S16.QACC qx, imm` | Shift QACC_H/L right by `imm` per 64-bit segment, saturate to S16, move to qx |
| `ESP.SRCMB.U8.QACC qx, imm` | Shift QACC_H/L right by `imm` per 32-bit segment, saturate to U8, move to qx |
| `ESP.SRCMB.U16.QACC qx, imm` | Shift QACC_H/L right by `imm` per 64-bit segment, saturate to U16, move to qx |
| `ESP.SRCMB.S8.Q.QACC qx, qy` | Same as SRCMB.S8.QACC but shift amount from QR |
| `ESP.SRCMB.S16.Q.QACC qx, qy` | Same as SRCMB.S16.QACC but shift amount from QR |
| `ESP.SRCMB.U8.Q.QACC qx, qy` | Same as SRCMB.U8.QACC but shift amount from QR |
| `ESP.SRCMB.U16.Q.QACC qx, qy` | Same as SRCMB.U16.QACC but shift amount from QR |
| `ESP.SRS.S.XACC rd, rs1` | Arithmetic right shift XACC by `rs1[5:0]`; write 40-bit result back to XACC, and write result saturated to 32-bit signed: `min(max(XACC>>rs1, -2^31), 2^31-1)` into `rd` |
| `ESP.SRS.U.XACC rd, rs1` | Unsigned counterpart: right shift XACC by `rs1[5:0]`, write 40-bit result back to XACC, write result saturated to 32-bit unsigned into `rd` |

### Activation / Other Arithmetic

| Instruction | Description |
|-------------|-------------|
| `ESP.VRELU.S8 qz, qx, qy` | 16 × S8 ReLU: `qz[i] = qx[i] > 0 ? qx[i] * qy[0] : 0` |
| `ESP.VRELU.S16 qz, qx, qy` | 8 × S16 ReLU: `qz[i] = qx[i] > 0 ? qx[i] * qy[0] : 0` |
| `ESP.VPRELU.S8 qz, qx, qy` | 16 × S8 PReLU: `qz[i] = qx[i] > 0 ? qx[i] : qx[i] * qy[i]` |
| `ESP.VPRELU.S16 qz, qx, qy` | 8 × S16 PReLU: `qz[i] = qx[i] > 0 ? qx[i] : qx[i] * qy[i]` |
| `ESP.VABS.8 qz, qx` | 16 × 8-bit absolute value |
| `ESP.VABS.16 qz, qx` | 8 × 16-bit absolute value |
| `ESP.VABS.32 qz, qx` | 4 × 32-bit absolute value |
| `ESP.SAT rsd, rs0, rs1` | Saturate `rsd` between clamp bounds derived from `rs0` and `rs1`: `min_t = max(rs1, rs0)`, `max_t = min(rs1, rs0)`, `rsd = max(min(rsd, max_t), min_t)`. rsd is both source and destination (read-modify-write). |
| `ESP.ADDX2 rd, rs1, rs2` | `rd = rs1 + (rs2 << 1)` |
| `ESP.ADDX4 rd, rs1, rs2` | `rd = rs1 + (rs2 << 2)` |
| `ESP.SUBX2 rd, rs1, rs2` | `rd = rs1 - (rs2 << 1)` |
| `ESP.SUBX4 rd, rs1, rs2` | `rd = rs1 - (rs2 << 2)` |

### Vector Clamp

**Note:** VCLAMP uses a single QR register `qx` and an immediate `sel16`, clamped to the symmetric range `[-2^sel16, 2^sel16-1]`.

| Instruction | Description |
|-------------|-------------|
| `ESP.VCLAMP.S16 qz, qx, sel16` | Clamp 8 × S16: `qz[i] = min(max(qx[i], -2^sel16), 2^sel16-1)` |

### Vector Saturation (VSAT)

**Note:** VSAT uses **two AR registers** (`rs1`, `rs2`) to define the clamp bounds. The bounds `min_t` and `max_t` are derived as `min_t = max(rs1, rs2)` and `max_t = min(rs1, rs2)` (exchanging the role of comparison to derive the actual asymmetric range). Each element in `qx` is clamped to `[min_t, max_t]`.

| Instruction | Description |
|-------------|-------------|
| `ESP.VSAT.S8 qz, qx, rs1, rs2` | Clamp each of 16 × S8 in qx to range derived from rs1/rs2 |
| `ESP.VSAT.S16 qz, qx, rs1, rs2` | Clamp each of 8 × S16 in qx to range derived from rs1/rs2 |
| `ESP.VSAT.S32 qz, qx, rs1, rs2` | Clamp each of 4 × S32 in qx to range derived from rs1/rs2 |
| `ESP.VSAT.U8 qz, qx, rs1, rs2` | Clamp each of 16 × U8 in qx to range derived from rs1/rs2 |
| `ESP.VSAT.U16 qz, qx, rs1, rs2` | Clamp each of 8 × U16 in qx to range derived from rs1/rs2 |
| `ESP.VSAT.U32 qz, qx, rs1, rs2` | Clamp each of 4 × U32 in qx to range derived from rs1/rs2 |

---

## Comparison Instructions

### Vector Maximum (Element-wise)

For VMAX, each element `qz[i] = (qx[i] >= qy[i]) ? qx[i] : qy[i]`.

#### Signed Vector Maximum

| Instruction | Description |
|-------------|-------------|
| `ESP.VMAX.S8 qz, qx, qy` | 16 × 8-bit signed element-wise max |
| `ESP.VMAX.S16 qz, qx, qy` | 8 × 16-bit signed element-wise max |
| `ESP.VMAX.S32 qz, qx, qy` | 4 × 32-bit signed element-wise max |
| `ESP.VMAX.S8.LD.INCP qu, rs1, qz, qx, qy` | VMAX.S8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMAX.S16.LD.INCP qu, rs1, qz, qx, qy` | VMAX.S16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMAX.S32.LD.INCP qu, rs1, qz, qx, qy` | VMAX.S32 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMAX.S8.ST.INCP qu, rs1, qz, qx, qy` | VMAX.S8 + store qu to memory, rs1 += 16 |
| `ESP.VMAX.S16.ST.INCP qu, rs1, qz, qx, qy` | VMAX.S16 + store qu to memory, rs1 += 16 |
| `ESP.VMAX.S32.ST.INCP qu, rs1, qz, qx, qy` | VMAX.S32 + store qu to memory, rs1 += 16 |

#### Unsigned Vector Maximum

| Instruction | Description |
|-------------|-------------|
| `ESP.VMAX.U8 qz, qx, qy` | 16 × 8-bit unsigned element-wise max |
| `ESP.VMAX.U16 qz, qx, qy` | 8 × 16-bit unsigned element-wise max |
| `ESP.VMAX.U32 qz, qx, qy` | 4 × 32-bit unsigned element-wise max |
| `ESP.VMAX.U8.LD.INCP qu, rs1, qz, qx, qy` | VMAX.U8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMAX.U16.LD.INCP qu, rs1, qz, qx, qy` | VMAX.U16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMAX.U32.LD.INCP qu, rs1, qz, qx, qy` | VMAX.U32 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMAX.U8.ST.INCP qu, rs1, qz, qx, qy` | VMAX.U8 + store qu to memory, rs1 += 16 |
| `ESP.VMAX.U16.ST.INCP qu, rs1, qz, qx, qy` | VMAX.U16 + store qu to memory, rs1 += 16 |
| `ESP.VMAX.U32.ST.INCP qu, rs1, qz, qx, qy` | VMAX.U32 + store qu to memory, rs1 += 16 |

### Scalar Maximum (Vector → AR)

Finds the maximum value across all elements of a QR register and writes it to an AR register `rd`.

| Instruction | Description |
|-------------|-------------|
| `ESP.MAX.S8.A qw, rd` | Max of 16 signed 8-bit elements → rd. `rd = {24{max_value[7]}, max_value[7:0]}` (sign-extended) |
| `ESP.MAX.S16.A qw, rd` | Max of 8 signed 16-bit elements → rd. `rd = {16{max_value[15]}, max_value[15:0]}` (sign-extended) |
| `ESP.MAX.S32.A qw, rd` | Max of 4 signed 32-bit elements → rd. `rd = max_value[31:0]` |
| `ESP.MAX.U8.A qw, rd` | Max of 16 unsigned 8-bit elements → rd. `rd = {24'b0, max_value[7:0]}` (zero-extended) |
| `ESP.MAX.U16.A qw, rd` | Max of 8 unsigned 16-bit elements → rd. `rd = {16'b0, max_value[15:0]}` (zero-extended) |
| `ESP.MAX.U32.A qw, rd` | Max of 4 unsigned 32-bit elements → rd. `rd = max_value[31:0]` |

### Vector Minimum (Element-wise)

For VMIN, each element `qz[i] = (qx[i] <= qy[i]) ? qx[i] : qy[i]`.

#### Signed Vector Minimum

| Instruction | Description |
|-------------|-------------|
| `ESP.VMIN.S8 qz, qx, qy` | 16 × 8-bit signed element-wise min |
| `ESP.VMIN.S16 qz, qx, qy` | 8 × 16-bit signed element-wise min |
| `ESP.VMIN.S32 qz, qx, qy` | 4 × 32-bit signed element-wise min |
| `ESP.VMIN.S8.LD.INCP qu, rs1, qz, qx, qy` | VMIN.S8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMIN.S16.LD.INCP qu, rs1, qz, qx, qy` | VMIN.S16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMIN.S32.LD.INCP qu, rs1, qz, qx, qy` | VMIN.S32 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMIN.S8.ST.INCP qu, rs1, qz, qx, qy` | VMIN.S8 + store qu to memory, rs1 += 16 |
| `ESP.VMIN.S16.ST.INCP qu, rs1, qz, qx, qy` | VMIN.S16 + store qu to memory, rs1 += 16 |
| `ESP.VMIN.S32.ST.INCP qu, rs1, qz, qx, qy` | VMIN.S32 + store qu to memory, rs1 += 16 |

#### Unsigned Vector Minimum

| Instruction | Description |
|-------------|-------------|
| `ESP.VMIN.U8 qz, qx, qy` | 16 × 8-bit unsigned element-wise min |
| `ESP.VMIN.U16 qz, qx, qy` | 8 × 16-bit unsigned element-wise min |
| `ESP.VMIN.U32 qz, qx, qy` | 4 × 32-bit unsigned element-wise min |
| `ESP.VMIN.U8.LD.INCP qu, rs1, qz, qx, qy` | VMIN.U8 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMIN.U16.LD.INCP qu, rs1, qz, qx, qy` | VMIN.U16 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMIN.U32.LD.INCP qu, rs1, qz, qx, qy` | VMIN.U32 + load 16 bytes to qu, rs1 += 16 |
| `ESP.VMIN.U8.ST.INCP qu, rs1, qz, qx, qy` | VMIN.U8 + store qu to memory, rs1 += 16 |
| `ESP.VMIN.U16.ST.INCP qu, rs1, qz, qx, qy` | VMIN.U16 + store qu to memory, rs1 += 16 |
| `ESP.VMIN.U32.ST.INCP qu, rs1, qz, qx, qy` | VMIN.U32 + store qu to memory, rs1 += 16 |

### Scalar Minimum (Vector → AR)

Finds the minimum value across all elements of a QR register and writes it to an AR register `rd`.

| Instruction | Description |
|-------------|-------------|
| `ESP.MIN.S8.A qw, rd` | Min of 16 signed 8-bit elements → rd. `rd = {24{min_value[7]}, min_value[7:0]}` (sign-extended) |
| `ESP.MIN.S16.A qw, rd` | Min of 8 signed 16-bit elements → rd. `rd = {16{min_value[15]}, min_value[15:0]}` (sign-extended) |
| `ESP.MIN.S32.A qw, rd` | Min of 4 signed 32-bit elements → rd. `rd = min_value[31:0]` |
| `ESP.MIN.U8.A qw, rd` | Min of 16 unsigned 8-bit elements → rd. `rd = {24'b0, min_value[7:0]}` (zero-extended) |
| `ESP.MIN.U16.A qw, rd` | Min of 8 unsigned 16-bit elements → rd. `rd = {16'b0, min_value[15:0]}` (zero-extended) |
| `ESP.MIN.U32.A qw, rd` | Min of 4 unsigned 32-bit elements → rd. `rd = min_value[31:0]` |

### Vector Compare

Result is all 1s (true) or all 0s (false) per element. The mask width matches the data width: 0xFF for 8-bit, 0xFFFF for 16-bit, 0xFFFFFFFF for 32-bit.

#### Signed Comparison

| Instruction | Description |
|-------------|-------------|
| `ESP.VCMP.EQ.S8 qz, qx, qy` | 16 × 8-bit compare equal: `qz[i] = (qx[i]==qy[i]) ? 0xFF : 0` |
| `ESP.VCMP.EQ.S16 qz, qx, qy` | 8 × 16-bit compare equal: `qz[i] = (qx[i]==qy[i]) ? 0xFFFF : 0` |
| `ESP.VCMP.EQ.S32 qz, qx, qy` | 4 × 32-bit compare equal: `qz[i] = (qx[i]==qy[i]) ? 0xFFFFFFFF : 0` |
| `ESP.VCMP.LT.S8 qz, qx, qy` | 16 × 8-bit compare less-than: `qz[i] = (qx[i] < qy[i]) ? 0xFF : 0` |
| `ESP.VCMP.LT.S16 qz, qx, qy` | 8 × 16-bit compare less-than: `qz[i] = (qx[i] < qy[i]) ? 0xFFFF : 0` |
| `ESP.VCMP.LT.S32 qz, qx, qy` | 4 × 32-bit compare less-than: `qz[i] = (qx[i] < qy[i]) ? 0xFFFFFFFF : 0` |
| `ESP.VCMP.GT.S8 qz, qx, qy` | 16 × 8-bit compare greater-than: `qz[i] = (qx[i] > qy[i]) ? 0xFF : 0` |
| `ESP.VCMP.GT.S16 qz, qx, qy` | 8 × 16-bit compare greater-than: `qz[i] = (qx[i] > qy[i]) ? 0xFFFF : 0` |
| `ESP.VCMP.GT.S32 qz, qx, qy` | 4 × 32-bit compare greater-than: `qz[i] = (qx[i] > qy[i]) ? 0xFFFFFFFF : 0` |

#### Unsigned Comparison

| Instruction | Description |
|-------------|-------------|
| `ESP.VCMP.EQ.U8 qz, qx, qy` | 16 × 8-bit unsigned compare equal: `qz[i] = (qx[i]==qy[i]) ? 0xFF : 0` |
| `ESP.VCMP.EQ.U16 qz, qx, qy` | 8 × 16-bit unsigned compare equal: `qz[i] = (qx[i]==qy[i]) ? 0xFFFF : 0` |
| `ESP.VCMP.EQ.U32 qz, qx, qy` | 4 × 32-bit unsigned compare equal: `qz[i] = (qx[i]==qy[i]) ? 0xFFFFFFFF : 0` |
| `ESP.VCMP.LT.U8 qz, qx, qy` | 16 × 8-bit unsigned less-than: `qz[i] = (qx[i] < qy[i]) ? 0xFF : 0` |
| `ESP.VCMP.LT.U16 qz, qx, qy` | 8 × 16-bit unsigned less-than: `qz[i] = (qx[i] < qy[i]) ? 0xFFFF : 0` |
| `ESP.VCMP.LT.U32 qz, qx, qy` | 4 × 32-bit unsigned less-than: `qz[i] = (qx[i] < qy[i]) ? 0xFFFFFFFF : 0` |
| `ESP.VCMP.GT.U8 qz, qx, qy` | 16 × 8-bit unsigned greater-than: `qz[i] = (qx[i] > qy[i]) ? 0xFF : 0` |
| `ESP.VCMP.GT.U16 qz, qx, qy` | 8 × 16-bit unsigned greater-than: `qz[i] = (qx[i] > qy[i]) ? 0xFFFF : 0` |
| `ESP.VCMP.GT.U32 qz, qx, qy` | 4 × 32-bit unsigned greater-than: `qz[i] = (qx[i] > qy[i]) ? 0xFFFFFFFF : 0` |

---

## Bitwise Logical Instructions

All operate on full 128-bit QR registers.

| Instruction | Description |
|-------------|-------------|
| `ESP.ORQ qz, qx, qy` | 128-bit bitwise OR: `qz = qx \| qy` |
| `ESP.XORQ qz, qx, qy` | 128-bit bitwise XOR: `qz = qx ^ qy` |
| `ESP.ANDQ qz, qx, qy` | 128-bit bitwise AND: `qz = qx & qy` |
| `ESP.NOTQ qz, qx` | 128-bit bitwise NOT: `qz = ~qx` |

---

## Shift Instructions

### Vector Shift Right (per-element by SAR)

| Instruction | Description |
|-------------|-------------|
| `ESP.VSR.U32 qz, qx` | 4 × 32-bit unsigned (logical) shift right by SAR |
| `ESP.VSR.S32 qz, qx` | 4 × 32-bit signed (arithmetic) shift right by SAR |

### Vector Shift Left (by SAR)

| Instruction | Description |
|-------------|-------------|
| `ESP.VSL.32 qz, qx` | 4 × 32-bit shift left by SAR |

### Vector Shift by Register (per-element signed shift amount)

The shift amount is **not** an immediate — it comes from a Q register (`qw`), with one signed field per element aligned to that element's lane. If an element's shift field is negative, that element is shifted right by its absolute value; otherwise it is shifted left. Bits shifted out are discarded, and vacated bits are zero-filled.

| Instruction | Description |
|-------------|-------------|
| `ESP.VSLD.8 qu, qy, qw` | 16 × 8-bit vector shift; per-lane signed amount from `qw` (negative ⇒ right shift, else left shift) |
| `ESP.VSLD.16 qu, qy, qw` | 8 × 16-bit vector shift; per-lane signed amount from `qw` (negative ⇒ right shift, else left shift) |
| `ESP.VSLD.32 qu, qy, qw` | 4 × 32-bit vector shift; per-lane signed amount from `qw` (negative ⇒ right shift, else left shift) |
| `ESP.VSRD.8 qu, qy, qw` | 16 × 8-bit vector shift; per-lane signed amount from `qw` |
| `ESP.VSRD.16 qu, qy, qw` | 8 × 16-bit vector shift; per-lane signed amount from `qw` |
| `ESP.VSRD.32 qu, qy, qw` | 4 × 32-bit vector shift; per-lane signed amount from `qw` |

Operation for `ESP.VSLD.16 qu, qy, qw` (8 lanes of 16 bits; each lane's shift amount is the low 5 bits of the corresponding 16-bit segment of `qw`, treated as signed):

```
for i in 0..7:
    lane_hi = 16*i + 15
    lane_lo = 16*i
    shamt   = qw[16*i+4 : 16*i]        # signed 5-bit field
    qu[lane_hi:lane_lo] = (shamt < 0)
        ? (qy[lane_hi:lane_lo] >> -shamt)
        : (qy[lane_hi:lane_lo] << shamt)
```

`ESP.VSLD.8`/`.32` and `ESP.VSRD.8/16/32` follow the same per-lane, sign-selects-direction pattern, just with the shift-amount field width and stride matched to the 8-bit/32-bit lane size instead of 16-bit.

### Spliced Shift Instructions (for misalignment handling)

These combine two QR registers with a byte-level shift.

| Instruction | Description |
|-------------|-------------|
| `ESP.SRC.Q qz, qx, qy` | `qz = (qy \|\| qx) >> SAR_BYTE*8`, keep 128 bits |
| `ESP.SRC.Q.qup qz, qw, qy` | `qz = (qy \|\| qw) >> SAR_BYTE*8` AND `qw = qy` (auto-update qw for pipelined unaligned loads) |
| `ESP.SRC.Q.LD.IP qu, rs, imm, qx, qy` | SRC.Q + load 16 bytes to qu, rs += imm |
| `ESP.SRC.Q.LD.XP qu, rs1, rs2, qx, qy` | SRC.Q + load 16 bytes to qu, rs1 += rs2 |
| `ESP.SLCI.2Q qz, qx, imm` | Concatenate two QR and left shift by immediate |
| `ESP.SLCXXP.2Q qz, qx, qy` | Concatenate two QR and left shift (amount from QR) |
| `ESP.SRCI.2Q qz, qx, imm` | Concatenate two QR and right shift by immediate |
| `ESP.SRCXXP.2Q qz, qx, qy` | Concatenate two QR and right shift (amount from QR) |
| `ESP.SRCQ.128.ST.INCP qz, rs, qx, qy` | SRC.Q + store qz to memory (rs += 16) |

---

## FFT Dedicated Instructions

### Radix-2 Butterfly

| Instruction | Description |
|-------------|-------------|
| `ESP.FFT.R2BF.S16 qz, qx, qy` | 4 × S16 radix-2 butterfly on 8 elements |
| `ESP.FFT.R2BF.S16.ST.INCP qz, qx, qy, rs` | Butterfly + store (rs += 16) |

### Complex Multiplication for FFT

| Instruction | Description |
|-------------|-------------|
| `ESP.FFT.CMUL.S16.LD.XP qz, qx, qy, rs` | Complex multiply + load (rs += rs2) |
| `ESP.FFT.CMUL.S16.ST.XP qz, qx, qy, rs` | Complex multiply + store (rs += rs2) |

### Bit-Reverse

| Instruction | Description |
|-------------|-------------|
| `ESP.BITREV rs1, rs2` | Bit-reverse rs2 (3–10 bits controlled by FFT_BIT_WIDTH), result in rs1 |

### Real FFT Operations

| Instruction | Description |
|-------------|-------------|
| `ESP.FFT.AMS.S16.LD.INCP.UAUP qu, rs, qx, qy` | Real FFT: multiply-subtract, load with unaligned update |
| `ESP.FFT.AMS.S16.LD.INCP qu, rs, qx, qy` | Real FFT: multiply-subtract + load |
| `ESP.FFT.AMS.S16.LD.R32.DECP qu, rs, qx, qy` | Real FFT: multiply-subtract, load 32-bit, decrement pointer |
| `ESP.FFT.AMS.S16.ST.INCP qu, rs, qx, qy` | Real FFT: multiply-subtract + store |
| `ESP.FFT.VST.R32.DECP qu, rs` | Store 32-bit real and decrement pointer |

### FFT Multiply-Subtract Pattern

The AMS instructions perform: `result = qx * qy - qx_shifted * qy_shifted` used in real FFT computation to extract the real spectrum from complex FFT output.

---

## Instruction Summary by Operand Types

| Operand Convention | Meaning |
|--------------------|---------|
| `qw`, `qx`, `qy`, `qz`, `qu`, `qv` | 128-bit QR vector registers (q0–q7) |
| `rd`, `rs`, `rs0`, `rs1`, `rs2` | 32-bit AR general-purpose registers |
| `imm` | Signed immediate offset |
| `sel4` | 2-bit element selector (0–3) for 32-bit elements |
| `sel8` | 3-bit element selector (0–7) for 16-bit elements |
| `sel16` | 4-bit element selector (0–15) for 8-bit elements |

### Naming Convention in Syntax

- **Input operands**: `qx`, `qy`, `qw` (read-only)
- **Output/destination operands**: `qz`, `qv` (write-only, or read-modify-write for MAC)
- **Load destination**: `qu` (used in fused load+compute instructions)
- **Address register**: `rs1`, `rs` (AR register holding memory address)
- **General AR**: `rd`, `rs2` or `rs0`/`rs1` for scalar operands
