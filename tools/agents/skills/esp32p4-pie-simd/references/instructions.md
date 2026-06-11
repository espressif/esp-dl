# ESP32-P4 SIMD Instruction Reference

## Table of Contents
- [Read Instructions](#read-instructions)
- [Write Instructions](#write-instructions)
- [Data Exchange Instructions](#data-exchange-instructions)
- [Arithmetic Instructions](#arithmetic-instructions)
- [Comparison Instructions](#comparison-instructions)
- [Bitwise Logical Instructions](#bitwise-logical-instructions)
- [Shift Instructions](#shift-instructions)
- [FFT Dedicated Instructions](#fft-dedicated-instructions)

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
| `ESP.VLDBC.16.IP qu, rs1, imm` | Load 2 bytes, broadcast to 8 x 16-bit, rs1 += imm |
| `ESP.VLDBC.16.XP qu, rs1, rs2` | Load 2 bytes, broadcast to 8 x 16-bit, rs1 += rs2 |
| `ESP.VLDBC.32.IP qu, rs1, imm` | Load 4 bytes, broadcast to 4 x 32-bit, rs1 += imm |
| `ESP.VLDBC.32.XP qu, rs1, rs2` | Load 4 bytes, broadcast to 4 x 32-bit, rs1 += rs2 |
| `ESP.VLDHBC.16.INCP qu, rs1` | Load high 16 bytes and broadcast (rs1 += 16) |

### Unaligned 128-bit Loads

| Instruction | Description |
|-------------|-------------|
| `ESP.LD.128.USAR.IP qu, rs1, imm` | Load 16 bytes (set SAR_BYTE from addr), rs1 += imm |
| `ESP.LD.128.USAR.XP qu, rs1, rs2` | Load 16 bytes (set SAR_BYTE from addr), rs1 += rs2 |

### QACC Loads (sign/zero extended)

| Instruction | Description |
|-------------|-------------|
| `ESP.LDQA.U8.128.IP qu, rs1, imm` | Load 16 bytes, zero-extend each to 20-bit to QACC_H/L, rs1 += imm |
| `ESP.LDQA.U8.128.XP qu, rs1, rs2` | Same, rs1 += rs2 |
| `ESP.LDQA.U16.128.IP qu, rs1, imm` | Load 16 bytes, zero-extend each 16-bit to 40-bit to QACC_H/L |
| `ESP.LDQA.U16.128.XP qu, rs1, rs2` | Same, rs1 += rs2 |
| `ESP.LDQA.S8.128.IP qu, rs1, imm` | Load 16 bytes, sign-extend each 8-bit to 20-bit to QACC_H/L |
| `ESP.LDQA.S8.128.XP qu, rs1, rs2` | Same, rs1 += rs2 |
| `ESP.LDQA.S16.128.IP qu, rs1, imm` | Load 16 bytes, sign-extend each 16-bit to 40-bit to QACC_H/L |
| `ESP.LDQA.S16.128.XP qu, rs1, rs2` | Same, rs1 += rs2 |

### QACC/XACC Direct Loads

| Instruction | Description |
|-------------|-------------|
| `ESP.LD.QACC.H.H.128.IP qu, rs1, imm` | Load 16 bytes to QACC_H[255:128], rs1 += imm |
| `ESP.LD.QACC.H.L.128.IP qu, rs1, imm` | Load 16 bytes to QACC_H[127:0], rs1 += imm |
| `ESP.LD.QACC.L.H.128.IP qu, rs1, imm` | Load 16 bytes to QACC_L[255:128], rs1 += imm |
| `ESP.LD.QACC.L.L.128.IP qu, rs1, imm` | Load 16 bytes to QACC_L[127:0], rs1 += imm |
| `ESP.LD.XACC.IP qu, rs1, imm` | Load 8 bytes to XACC, rs1 += imm |
| `ESP.LD.UA.STATE.IP qu, rs1, imm` | Load 16 bytes to UA_STATE, rs1 += imm |

### Indexed/Extended Loads

| Instruction | Description |
|-------------|-------------|
| `ESP.LDXQ.32 qu, rs1, rs2, sel4` | rs1 += qx[sel4*16+15:sel4*16] as index, then load 4 bytes |
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

| Instruction | Description |
|-------------|-------------|
| `ESP.ST.QACC.H.H.128.IP qu, rs1, imm` | Store QACC_H[255:128] to memory, rs1 += imm |
| `ESP.ST.QACC.H.L.128.IP qu, rs1, imm` | Store QACC_H[127:0] to memory, rs1 += imm |
| `ESP.ST.QACC.L.H.128.IP qu, rs1, imm` | Store QACC_L[255:128] to memory, rs1 += imm |
| `ESP.ST.QACC.L.L.128.IP qu, rs1, imm` | Store QACC_L[127:0] to memory, rs1 += imm |

### XACC Stores

| Instruction | Description |
|-------------|-------------|
| `ESP.ST.U.XACC.IP qu, rs1, imm` | Zero-extend XACC to 8 bytes and store, rs1 += imm |
| `ESP.ST.S.XACC.IP qu, rs1, imm` | Sign-extend XACC to 8 bytes and store, rs1 += imm |

### Other Stores

| Instruction | Description |
|-------------|-------------|
| `ESP.ST.UA.STATE.IP qu, rs1, imm` | Store UA_STATE to memory, rs1 += imm |
| `ESP.STXQ.32 qu, rs1, rs2, sel4` | rs1 += qx[sel4*16+15:sel4*16], then store 4 bytes |

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
| `ESP.MOVX.R.CFG rd` | Read CFG to AR |
| `ESP.MOVX.W.CFG rs1` | Write AR to CFG |
| `ESP.MOVX.R.SAR.BYTES rd` | Read SAR_BYTE to AR |
| `ESP.MOVX.W.SAR.BYTES rs1` | Write AR to SAR_BYTE |
| `ESP.MOVX.R.SAR rd` | Read SAR to AR |
| `ESP.MOVX.W.SAR rs1` | Write AR to SAR |
| `ESP.MOVX.R.FFT.BIT.WIDTH rd` | Read FFT_BIT_WIDTH to AR |
| `ESP.MOVX.W.FFT.BIT.WIDTH rs1` | Write AR to FFT_BIT_WIDTH |
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

### QACC to QR Moves

| Instruction | Description |
|-------------|-------------|
| `ESP.MOV.U8.QACC qv` | Move QACC_H/L to QR, each 32-bit segment >> SAR then zero-extended to 8-bit |
| `ESP.MOV.S8.QACC qv` | Same with sign-extension to 8-bit |
| `ESP.MOV.U16.QACC qv` | Move QACC_H/L to QR, each 64-bit segment >> SAR then zero-extended to 16-bit |
| `ESP.MOV.S16.QACC qv` | Same with sign-extension to 16-bit |

### Register Clears

| Instruction | Description |
|-------------|-------------|
| `ESP.ZERO.Q qN` | Clear QR register to zero |
| `ESP.ZERO.QACC` | Clear QACC_H and QACC_L to zero |
| `ESP.ZERO.XACC` | Clear XACC to zero |

---

## Arithmetic Instructions

### Vector Addition

| Instruction | Description |
|-------------|-------------|
| `ESP.VADD.S8 qz, qx, qy` | 16 x 8-bit signed add |
| `ESP.VADD.S16 qz, qx, qy` | 8 x 16-bit signed add |
| `ESP.VADD.S32 qz, qx, qy` | 4 x 32-bit signed add |
| `ESP.VADD.S8.LD.INCP qz, qx, qy, rs, imm` | Add S8 and load next (rs += imm) |
| `ESP.VADD.S16.LD.INCP qz, qx, qy, rs, imm` | Add S16 and load next |
| `ESP.VADD.S32.LD.INCP qz, qx, qy, rs, imm` | Add S32 and load next |
| `ESP.VADD.S8.ST.INCP qz, qx, qy, rs, imm` | Add S8 and store (rs += imm) |
| `ESP.VADD.S16.ST.INCP qz, qx, qy, rs, imm` | Add S16 and store |
| `ESP.VADD.S32.ST.INCP qz, qx, qy, rs, imm` | Add S32 and store |

### Saturated Vector Addition

| Instruction | Description |
|-------------|-------------|
| `ESP.VSADDS.S8 qz, qx, qy` | 16 x 8-bit signed add with saturation |
| `ESP.VSADDS.S16 qz, qx, qy` | 8 x 16-bit signed add with saturation |
| `ESP.VSADDS.U8 qz, qx, qy` | 16 x 8-bit unsigned add with saturation |
| `ESP.VSADDS.U16 qz, qx, qy` | 8 x 16-bit unsigned add with saturation |

### Vector Subtraction

| Instruction | Description |
|-------------|-------------|
| `ESP.VSUB.S8 qz, qx, qy` | 16 x 8-bit signed subtract (qx - qy) |
| `ESP.VSUB.S16 qz, qx, qy` | 8 x 16-bit signed subtract |
| `ESP.VSUB.S32 qz, qx, qy` | 4 x 32-bit signed subtract |
| `ESP.VSUB.S8.LD.INCP qz, qx, qy, rs, imm` | Sub S8 and load next |
| `ESP.VSUB.S16.LD.INCP qz, qx, qy, rs, imm` | Sub S16 and load next |
| `ESP.VSUB.S32.LD.INCP qz, qx, qy, rs, imm` | Sub S32 and load next |
| `ESP.VSUB.S8.ST.INCP qz, qx, qy, rs, imm` | Sub S8 and store |
| `ESP.VSUB.S16.ST.INCP qz, qx, qy, rs, imm` | Sub S16 and store |
| `ESP.VSUB.S32.ST.INCP qz, qx, qy, rs, imm` | Sub S32 and store |

### Saturated Vector Subtraction

| Instruction | Description |
|-------------|-------------|
| `ESP.VSSUBS.S8 qz, qx, qy` | 16 x 8-bit signed subtract with saturation |
| `ESP.VSSUBS.S16 qz, qx, qy` | 8 x 16-bit signed subtract with saturation |
| `ESP.VSSUBS.U8 qz, qx, qy` | 16 x 8-bit unsigned subtract with saturation |
| `ESP.VSSUBS.U16 qz, qx, qy` | 8 x 16-bit unsigned subtract with saturation |

### Vector Multiplication

Multiplies are followed by a right shift of SAR bits, keeping the lower half.

| Instruction | Description |
|-------------|-------------|
| `ESP.VMUL.S8 qz, qx, qy` | 16 x 8-bit signed multiply, result >> SAR, keep low 8 |
| `ESP.VMUL.S16 qz, qx, qy` | 8 x 16-bit signed multiply, result >> SAR, keep low 16 |
| `ESP.VMUL.U8 qz, qx, qy` | 16 x 8-bit unsigned multiply, result >> SAR, keep low 8 |
| `ESP.VMUL.U16 qz, qx, qy` | 8 x 16-bit unsigned multiply, result >> SAR, keep low 16 |
| `ESP.VMUL.S8.LD.INCP qz, qx, qy, rs, imm` | Mul S8 and load next |
| `ESP.VMUL.S16.LD.INCP qz, qx, qy, rs, imm` | Mul S16 and load next |
| `ESP.VMUL.U8.LD.INCP qz, qx, qy, rs, imm` | Mul U8 and load next |
| `ESP.VMUL.U16.LD.INCP qz, qx, qy, rs, imm` | Mul U16 and load next |
| `ESP.VMUL.S8.ST.INCP qz, qx, qy, rs, imm` | Mul S8 and store |
| `ESP.VMUL.S16.ST.INCP qz, qx, qy, rs, imm` | Mul S16 and store |
| `ESP.VMUL.U8.ST.INCP qz, qx, qy, rs, imm` | Mul U8 and store |
| `ESP.VMUL.U16.ST.INCP qz, qx, qy, rs, imm` | Mul U16 and store |
| `ESP.VMUL.S32.S16xS16 qz, qx, qy` | 8 x 16-bit multiply, result >> SAR, produce 8 x 32-bit results |

### Vector Complex Multiplication (S16)

Operates on pairs of 16-bit values as complex numbers (real, imag).

| Instruction | Description |
|-------------|-------------|
| `ESP.CMUL.S16 qz, qx, qy` | 4 x 16-bit complex multiply (8 elements as 4 complex pairs) |
| `ESP.CMUL.S16.LD.INCP qz, qx, qy, rs, imm` | Complex mul and load next |
| `ESP.CMUL.S16.ST.INCP qz, qx, qy, rs, imm` | Complex mul and store |
| `ESP.CMUL.S8 qz, qx, qy` | 8 x 8-bit complex multiply |
| `ESP.CMUL.S8.LD.INCP qz, qx, qy, rs, imm` | Complex mul S8 and load next |
| `ESP.CMUL.S8.ST.INCP qz, qx, qy, rs, imm` | Complex mul S8 and store |

### Vector Multiply-Accumulate to QACC

| Instruction | Description |
|-------------|-------------|
| `ESP.VMULAS.S8.QACC qx, qy` | 16 x S8 mul, accumulate 32-bit to QACC_H/L (saturated) |
| `ESP.VMULAS.S16.QACC qx, qy` | 8 x S16 mul, accumulate 64-bit to QACC_H/L (saturated) |
| `ESP.VMULAS.U8.QACC qx, qy` | 16 x U8 mul, accumulate 32-bit to QACC_H/L (saturated) |
| `ESP.VMULAS.U16.QACC qx, qy` | 8 x U16 mul, accumulate 64-bit to QACC_H/L (saturated) |

**Fused variants with memory access:**
| Instruction | Description |
|-------------|-------------|
| `ESP.VMULAS.S8.QACC.LD.IP qu, rs, imm, qx, qy` | MAC S8 to QACC and load |
| `ESP.VMULAS.S16.QACC.LD.IP qu, rs, imm, qx, qy` | MAC S16 to QACC and load |
| `ESP.VMULAS.S8.QACC.LD.XP qu, rs1, rs2, qx, qy` | MAC S8 to QACC and load (addr += reg) |
| `ESP.VMULAS.S16.QACC.LD.XP qu, rs1, rs2, qx, qy` | MAC S16 to QACC and load (addr += reg) |
| `ESP.VMULAS.S8.QACC.ST.IP qu, rs, imm, qx, qy` | MAC S8 to QACC and store |
| `ESP.VMULAS.S16.QACC.ST.IP qu, rs, imm, qx, qy` | MAC S16 to QACC and store |
| `ESP.VMULAS.S8.QACC.ST.XP qu, rs1, rs2, qx, qy` | MAC S8 to QACC and store (addr += reg) |
| `ESP.VMULAS.S16.QACC.ST.XP qu, rs1, rs2, qx, qy` | MAC S16 to QACC and store (addr += reg) |
| `ESP.VMULAS.S8.QACC.LDBC.INCP qu, rs, imm, qx, qy` | MAC S8 to QACC and broadcast load |
| `ESP.VMULAS.S16.QACC.LDBC.INCP qu, rs, imm, qx, qy` | MAC S16 to QACC and broadcast load |
| `ESP.VMULAS.U8.QACC.LD.IP qu, rs, imm, qx, qy` | MAC U8 to QACC and load |
| `ESP.VMULAS.U16.QACC.LD.IP qu, rs, imm, qx, qy` | MAC U16 to QACC and load |
| `ESP.VMULAS.U8.QACC.LD.XP qu, rs1, rs2, qx, qy` | MAC U8 to QACC and load (addr += reg) |
| `ESP.VMULAS.U16.QACC.LD.XP qu, rs1, rs2, qx, qy` | MAC U16 to QACC and load (addr += reg) |
| `ESP.VMULAS.U8.QACC.ST.IP qu, rs, imm, qx, qy` | MAC U8 to QACC and store |
| `ESP.VMULAS.U16.QACC.ST.IP qu, rs, imm, qx, qy` | MAC U16 to QACC and store |
| `ESP.VMULAS.U8.QACC.ST.XP qu, rs1, rs2, qx, qy` | MAC U8 to QACC and store (addr += reg) |
| `ESP.VMULAS.U16.QACC.ST.XP qu, rs1, rs2, qx, qy` | MAC U16 to QACC and store (addr += reg) |

### Vector Multiply-Accumulate to XACC

XACC = sum of all element-wise products (dot-product style).

| Instruction | Description |
|-------------|-------------|
| `ESP.VMULAS.S8.XACC qx, qy` | 16 x S8 mul, sum all to XACC (40-bit, saturated) |
| `ESP.VMULAS.S16.XACC qx, qy` | 8 x S16 mul, sum all to XACC (40-bit, saturated) |
| `ESP.VMULAS.U8.XACC qx, qy` | 16 x U8 mul, sum all to XACC |
| `ESP.VMULAS.U16.XACC qx, qy` | 8 x U16 mul, sum all to XACC |

**Fused variants:**
| Instruction | Description |
|-------------|-------------|
| `ESP.VMULAS.S8.XACC.LD.IP qu, rs, imm, qx, qy` | MAC to XACC and load |
| `ESP.VMULAS.S16.XACC.LD.IP qu, rs, imm, qx, qy` | MAC to XACC and load |
| `ESP.VMULAS.S8.XACC.LD.XP qu, rs1, rs2, qx, qy` | MAC to XACC and load (addr += reg) |
| `ESP.VMULAS.S16.XACC.LD.XP qu, rs1, rs2, qx, qy` | MAC to XACC and load (addr += reg) |
| `ESP.VMULAS.S8.XACC.ST.IP qu, rs, imm, qx, qy` | MAC to XACC and store |
| `ESP.VMULAS.S16.XACC.ST.IP qu, rs, imm, qx, qy` | MAC to XACC and store |
| `ESP.VMULAS.S8.XACC.ST.XP qu, rs1, rs2, qx, qy` | MAC to XACC and store (addr += reg) |
| `ESP.VMULAS.S16.XACC.ST.XP qu, rs1, rs2, qx, qy` | MAC to XACC and store (addr += reg) |

### Scalar-Vector Multiply-Accumulate to QACC

One operand is a vector, the other is selected as a scalar from the vector using `sel`.

| Instruction | Description |
|-------------|-------------|
| `ESP.VSMULAS.S8.QACC qx, qy, sel4` | Scalar-vector MAC S8 to QACC (select byte from qy) |
| `ESP.VSMULAS.S16.QACC qx, qy, sel4` | Scalar-vector MAC S16 to QACC (select 16-bit from qy) |
| `ESP.VSMULAS.S8.QACC.LD.INCP qu, rs, qx, qy, sel4` | Scalar-vector MAC S8 and load |
| `ESP.VSMULAS.S16.QACC.LD.INCP qu, rs, qx, qy, sel4` | Scalar-vector MAC S16 and load |

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
| `ESP.VCMULAS.S8.QACC.H.LD.IP qu, rs, imm, qx, qy` | Complex MAC to QACC_H and load |
| `ESP.VCMULAS.S8.QACC.L.LD.IP qu, rs, imm, qx, qy` | Complex MAC to QACC_L and load |
| `ESP.VCMULAS.S16.QACC.H.LD.IP qu, rs, imm, qx, qy` | Complex MAC S16 to QACC_H and load |
| `ESP.VCMULAS.S16.QACC.L.LD.IP qu, rs, imm, qx, qy` | Complex MAC S16 to QACC_L and load |
| `ESP.VCMULAS.S8.QACC.H.LD.XP qu, rs1, rs2, qx, qy` | Complex MAC to QACC_H, load (addr += reg) |
| `ESP.VCMULAS.S8.QACC.L.LD.XP qu, rs1, rs2, qx, qy` | Complex MAC to QACC_L, load (addr += reg) |
| `ESP.VCMULAS.S16.QACC.H.LD.XP qu, rs1, rs2, qx, qy` | Complex MAC S16 to QACC_H, load (addr += reg) |
| `ESP.VCMULAS.S16.QACC.L.LD.XP qu, rs1, rs2, qx, qy` | Complex MAC S16 to QACC_L, load (addr += reg) |

### QACC/XACC Shift and Move

| Instruction | Description |
|-------------|-------------|
| `ESP.SRCMB.S8.QACC qx, imm` | Shift QACC_H/L right by imm per 32-bit segment, saturate to S8, move to qx |
| `ESP.SRCMB.S16.QACC qx, imm` | Shift QACC_H/L right by imm per 64-bit segment, saturate to S16, move to qx |
| `ESP.SRCMB.U8.QACC qx, imm` | Shift QACC_H/L right by imm per 32-bit segment, saturate to U8, move to qx |
| `ESP.SRCMB.U16.QACC qx, imm` | Shift QACC_H/L right by imm per 64-bit segment, saturate to U16, move to qx |
| `ESP.SRCMB.S8.Q.QACC qx, qy` | Same as SRCMB.S8.QACC but shift amount from QR |
| `ESP.SRCMB.S16.Q.QACC qx, qy` | Same as SRCMB.S16.QACC but shift amount from QR |
| `ESP.SRCMB.U8.Q.QACC qx, qy` | Same as SRCMB.U8.QACC but shift amount from QR |
| `ESP.SRCMB.U16.Q.QACC qx, qy` | Same as SRCMB.U16.QACC but shift amount from QR |
| `ESP.SRS.S.XACC rd, rs1` | Arithmetic right shift XACC by `rs1[5:0]`; write the 40-bit shifted result back to XACC, and write the result saturated to a 32-bit signed value `min(max(XACC>>rs1, -2^31), 2^31-1)` into general register `rd` |
| `ESP.SRS.U.XACC rd, rs1` | Unsigned counterpart of `ESP.SRS.S.XACC`: right shift XACC by `rs1[5:0]`, write the 40-bit result back to XACC, and write the result saturated to a 32-bit value into `rd` |

### Activation / Other Arithmetic

| Instruction | Description |
|-------------|-------------|
| `ESP.VRELU.S8 qz, qx, qy` | 16 x S8 ReLU: qz[i] = qx[i] > 0 ? qx[i] * qy[0] : 0 |
| `ESP.VRELU.S16 qz, qx, qy` | 8 x S16 ReLU: qz[i] = qx[i] > 0 ? qx[i] * qy[0] : 0 |
| `ESP.VPRELU.S8 qz, qx, qy` | 16 x S8 PReLU: qz[i] = qx[i] > 0 ? qx[i] : qx[i] * qy[i] |
| `ESP.VPRELU.S16 qz, qx, qy` | 8 x S16 PReLU: qz[i] = qx[i] > 0 ? qx[i] : qx[i] * qy[i] |
| `ESP.VABS.8 qz, qx` | 16 x 8-bit absolute value |
| `ESP.VABS.16 qz, qx` | 8 x 16-bit absolute value |
| `ESP.VABS.32 qz, qx` | 4 x 32-bit absolute value |
| `ESP.SAT rs1, rs2` | Saturate rs2 between [rs1, rs1] range |
| `ESP.ADDX2 rd, rs1, rs2` | rd = rs1 + (rs2 << 1) |
| `ESP.ADDX4 rd, rs1, rs2` | rd = rs1 + (rs2 << 2) |
| `ESP.SUBX2 rd, rs1, rs2` | rd = rs1 - (rs2 << 1) |
| `ESP.SUBX4 rd, rs1, rs2` | rd = rs1 - (rs2 << 2) |

---

## Comparison Instructions

### Vector Maximum

| Instruction | Description |
|-------------|-------------|
| `ESP.VMAX.S8 qz, qx, qy` | 16 x 8-bit element-wise max |
| `ESP.VMAX.S16 qz, qx, qy` | 8 x 16-bit element-wise max |
| `ESP.VMAX.S32 qz, qx, qy` | 4 x 32-bit element-wise max |
| `ESP.VMAX.S8.LD.INCP qz, qx, qy, rs, imm` | Max S8 and load next |
| `ESP.VMAX.S16.LD.INCP qz, qx, qy, rs, imm` | Max S16 and load next |
| `ESP.VMAX.S32.LD.INCP qz, qx, qy, rs, imm` | Max S32 and load next |
| `ESP.VMAX.S8.ST.INCP qz, qx, qy, rs, imm` | Max S8 and store |
| `ESP.VMAX.S16.ST.INCP qz, qx, qy, rs, imm` | Max S16 and store |
| `ESP.VMAX.S32.ST.INCP qz, qx, qy, rs, imm` | Max S32 and store |
| `ESP.MAX.S16.A rd, rs1` | Get max 16-bit value from QR and store to AR |
| `ESP.MAX.S32.A rd, rs1` | Get max 32-bit value from QR and store to AR |

### Vector Minimum

| Instruction | Description |
|-------------|-------------|
| `ESP.VMIN.S8 qz, qx, qy` | 16 x 8-bit element-wise min |
| `ESP.VMIN.S16 qz, qx, qy` | 8 x 16-bit element-wise min |
| `ESP.VMIN.S32 qz, qx, qy` | 4 x 32-bit element-wise min |
| `ESP.VMIN.S8.LD.INCP qz, qx, qy, rs, imm` | Min S8 and load next |
| `ESP.VMIN.S16.LD.INCP qz, qx, qy, rs, imm` | Min S16 and load next |
| `ESP.VMIN.S32.LD.INCP qz, qx, qy, rs, imm` | Min S32 and load next |
| `ESP.VMIN.S8.ST.INCP qz, qx, qy, rs, imm` | Min S8 and store |
| `ESP.VMIN.S16.ST.INCP qz, qx, qy, rs, imm` | Min S16 and store |
| `ESP.VMIN.S32.ST.INCP qz, qx, qy, rs, imm` | Min S32 and store |
| `ESP.MIN.S16.A rd, rs1` | Get min 16-bit value from QR and store to AR |
| `ESP.MIN.S32.A rd, rs1` | Get min 32-bit value from QR and store to AR |

### Vector Compare

Result is all 1s (true) or all 0s (false) per element.

| Instruction | Description |
|-------------|-------------|
| `ESP.VCMP.EQ.S8 qz, qx, qy` | 16 x 8-bit compare equal |
| `ESP.VCMP.EQ.S16 qz, qx, qy` | 8 x 16-bit compare equal |
| `ESP.VCMP.EQ.S32 qz, qx, qy` | 4 x 32-bit compare equal |
| `ESP.VCMP.LT.S8 qz, qx, qy` | 16 x 8-bit compare less-than (qx < qy) |
| `ESP.VCMP.LT.S16 qz, qx, qy` | 8 x 16-bit compare less-than |
| `ESP.VCMP.LT.S32 qz, qx, qy` | 4 x 32-bit compare less-than |
| `ESP.VCMP.GT.S8 qz, qx, qy` | 16 x 8-bit compare greater-than (qx > qy) |
| `ESP.VCMP.GT.S16 qz, qx, qy` | 8 x 16-bit compare greater-than |
| `ESP.VCMP.GT.S32 qz, qx, qy` | 4 x 32-bit compare greater-than |

### Saturation

| Instruction | Description |
|-------------|-------------|
| `ESP.VCLAMP.S16 qz, qx, qy` | Clamp 8 x S16: qz[i] = min(max(qx[i], qy_low[i]), qy_high[i]) |
| `ESP.VSAT.S8 qz, qx` | Saturate each element in qx to S8 range |
| `ESP.VSAT.S16 qz, qx` | Saturate each element in qx to S16 range |
| `ESP.VSAT.S32 qz, qx` | Saturate each element in qx to S32 range |
| `ESP.VSAT.U8 qz, qx` | Saturate each element in qx to U8 range |
| `ESP.VSAT.U16 qz, qx` | Saturate each element in qx to U16 range |
| `ESP.VSAT.U32 qz, qx` | Saturate each element in qx to U32 range |

---

## Bitwise Logical Instructions

All operate on full 128-bit QR registers.

| Instruction | Description |
|-------------|-------------|
| `ESP.ORQ qz, qx, qy` | 128-bit bitwise OR: qz = qx \| qy |
| `ESP.XORQ qz, qx, qy` | 128-bit bitwise XOR: qz = qx ^ qy |
| `ESP.ANDQ qz, qx, qy` | 128-bit bitwise AND: qz = qx & qy |
| `ESP.NOTQ qz, qx` | 128-bit bitwise NOT: qz = ~qx |

---

## Shift Instructions

### Vector Shift Right (per-element)

| Instruction | Description |
|-------------|-------------|
| `ESP.VSR.U32 qz, qx` | 4 x 32-bit unsigned shift right by SAR |
| `ESP.VSR.S32 qz, qx` | 4 x 32-bit signed shift right by SAR |

### Vector Shift Left

| Instruction | Description |
|-------------|-------------|
| `ESP.VSL.32 qz, qx` | 4 x 32-bit shift left by SAR |

### Vector Shift by Immediate

| Instruction | Description |
|-------------|-------------|
| `ESP.VSRD.8 qz, qx, imm` | 16 x 8-bit shift right by immediate (1-16) |
| `ESP.VSRD.16 qz, qx, imm` | 8 x 16-bit shift right by immediate (1-16) |
| `ESP.VSRD.32 qz, qx, imm` | 4 x 32-bit shift right by immediate (1-16) |
| `ESP.VSLD.8 qz, qx, imm` | 16 x 8-bit shift left by immediate (1-16) |
| `ESP.VSLD.16 qz, qx, imm` | 8 x 16-bit shift left by immediate (1-16) |
| `ESP.VSLD.32 qz, qx, imm` | 4 x 32-bit shift left by immediate (1-16) |

### Spliced Shift Instructions (for misalignment handling)

These combine two QR registers with a byte-level shift.

| Instruction | Description |
|-------------|-------------|
| `ESP.SRC.Q qz, qx, qy` | qz = (qx \|\| qy) >> SAR_BYTE*8, keep 128 bits |
| `ESP.SRC.Q.LD.IP qu, rs, imm, qx, qy` | SRC.Q and load next |
| `ESP.SRC.Q.LD.XP qu, rs1, rs2, qx, qy` | SRC.Q and load (addr += reg) |
| `ESP.SLCI.2Q qz, qx, imm` | Concatenate and left shift immediate |
| `ESP.SLCXXP.2Q qz, qx, qy` | Concatenate two QR and shift |
| `ESP.SRCI.2Q qz, qx, imm` | Concatenate and right shift immediate |
| `ESP.SRCXXP.2Q qz, qx, qy` | Concatenate two QR and right shift |
| `ESP.SRCQ.128.ST.INCP qz, rs, qx, qy` | SRC.Q and store (rs += 16) |

---

## FFT Dedicated Instructions

### Radix-2 Butterfly

| Instruction | Description |
|-------------|-------------|
| `ESP.FFT.R2BF.S16 qz, qx, qy` | 4 x S16 radix-2 butterfly on 8 elements |
| `ESP.FFT.R2BF.S16.ST.INCP qz, qx, qy, rs` | Butterfly and store (rs += 16) |

### Complex Multiplication for FFT

| Instruction | Description |
|-------------|-------------|
| `ESP.FFT.CMUL.S16.LD.XP qz, qx, qy, rs` | Complex multiply and load (addr += reg) |
| `ESP.FFT.CMUL.S16.ST.XP qz, qx, qy, rs` | Complex multiply and store (addr += reg) |

### Bit-Reverse

| Instruction | Description |
|-------------|-------------|
| `ESP.BITREV rs1, rs2` | Bit-reverse rs2 (3-10 bits controlled by FFT_BIT_WIDTH), result in rs1 |

### Real FFT Operations

| Instruction | Description |
|-------------|-------------|
| `ESP.FFT.AMS.S16.LD.INCP.UAUP qu, rs, qx, qy` | Real FFT: multiply-subtract, load with unaligned update |
| `ESP.FFT.AMS.S16.LD.INCP qu, rs, qx, qy` | Real FFT: multiply-subtract and load |
| `ESP.FFT.AMS.S16.LD.R32.DECP qu, rs, qx, qy` | Real FFT: multiply-subtract, load 32-bit, decrement pointer |
| `ESP.FFT.AMS.S16.ST.INCP qu, rs, qx, qy` | Real FFT: multiply-subtract and store |
| `ESP.FFT.VST.R32.DECP qu, rs` | Store 32-bit real and decrement pointer |

### FFT Multiply-Subtract Pattern

The AMS instructions perform: `result = qx * qy - qx_shifted * qy_shifted` used in real FFT computation to extract the real spectrum from complex FFT output.
