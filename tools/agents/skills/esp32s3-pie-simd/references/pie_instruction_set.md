# Chapter 1 Processor Instruction Extensions (PIE)

## 1.1 Overview

ESP32-S3 provides a set of Processor Instruction Extensions (PIE) for accelerating AI and DSP algorithms. These instructions are designed based on Tensilica Instruction Extension (TIE) language.

### 1.1.1 Features

| Feature | Description |
|---------|-------------|
| 128-bit General-purpose Registers | 8 QR registers supporting SIMD operations |
| Vector Operations | 8/16/32-bit vector multiplication, addition, subtraction, accumulation, shifting, comparison |
| Data and Computation Integration | Arithmetic instructions can perform data transfer simultaneously |
| Unaligned Data Support | Supports unaligned 128-bit vector data access |
| Saturated Operations | Supports saturation to prevent overflow |

### 1.1.2 Internal Structure

```
+-------------------+--------------------------------+-------------------+
|   Address Unit    |   8x 128-bit QR Registers      |       ALU         |
| (8/16/32/64/      |       (q0-q7)                  |  - 16x 8-bit      |
|  128-bit aligned) |                                |    multipliers    |
|                   |                                |  - 8x 16-bit      |
|                   |                                |    multipliers    |
+-------------------+--------------------------------+-------------------+
|  QACC_H (160-bit) |  QACC_L (160-bit) |  ACCX (40-bit)                    |
+-------------------+-------------------+-------------------+
```

## 1.2 Registers

### 1.2.1 General-purpose Registers

| Register | Count | Bit Width | Type | Description |
|----------|-------|-----------|------|-------------|
| AR | 16 | 32-bit | R/W | Xtensa native general-purpose registers |
| FR | 16 | 32-bit | R/W | Floating-point general-purpose registers |
| QR | 8 | 128-bit | R/W | PIE vector registers (q0-q7) |

**QR Register Data Formats:**
- 16 x 8-bit data segments
- 8 x 16-bit data segments
- 4 x 32-bit data segments

### 1.2.2 Special Registers

| Register | Bit Width | Access | Description |
|----------|-----------|--------|-------------|
| SAR | 6-bit | R/W | Shift amount register (in bits) |
| SAR_BYTE | 4-bit | R/W | Byte shift amount register |
| ACCX | 40-bit | R/W | Accumulator (accumulates all multiplication results) |
| QACC_H | 160-bit | R/W | High accumulator (8/16-bit segmented accumulation) |
| QACC_L | 160-bit | R/W | Low accumulator (8/16-bit segmented accumulation) |
| FFT_BIT_WIDTH | 4-bit | R/W | FFT bit-reversal width control |
| UA_STATE | 128-bit | R/W | Unaligned data state register |

### 1.2.3 Special Register Access Instructions

```asm
; Native special registers (SAR)
RSR.SAR  ar    ; Read SAR to AR register
WSR.SAR  ar    ; Write AR to SAR register
XSR.SAR  ar    ; Exchange AR and SAR values

; User-defined special registers
RUR.ACCX_0     ar    ; Read ACCX low 32 bits
RUR.ACCX_1     ar    ; Read ACCX high 8 bits
WUR.ACCX_0     ar    ; Write ACCX low 32 bits
WUR.ACCX_1     ar    ; Write ACCX high 8 bits
RUR.SAR_BYTE   ar    ; Read SAR_BYTE
WUR.SAR_BYTE   ar    ; Write SAR_BYTE
RUR.QACC_H_0   ar    ; Read QACC_H segment 0 (31:0)
RUR.QACC_H_1   ar    ; Read QACC_H segment 1 (63:32)
RUR.QACC_H_2   ar    ; Read QACC_H segment 2 (95:64)
RUR.QACC_H_3   ar    ; Read QACC_H segment 3 (127:96)
RUR.QACC_H_4   ar    ; Read QACC_H segment 4 (159:128)
; ... Similar access for QACC_L
```

## 1.3 Data Formats and Alignment

### 1.3.1 Data Formats

| Data Format | Length | Alignment Requirement | Address Low Bits |
|-------------|--------|----------------------|------------------|
| 1-byte | 8-bit | Any | xxxx |
| 2-byte | 16-bit | 2-byte | xxx0 |
| 4-byte | 32-bit | 4-byte | xx00 |
| 8-byte | 64-bit | 8-byte | x000 |
| 16-byte | 128-bit | 16-byte | 0000 |

### 1.3.2 Memory Alignment Requirements

**Important:** PIE instructions force alignment of access addresses, low bits are forced to 0:
- 128-bit access: low 4 bits forced to 0
- 64-bit access: low 3 bits forced to 0
- 32-bit access: low 2 bits forced to 0
- 16-bit access: low 1 bit forced to 0

### 1.3.3 Unaligned Data Processing

```asm
; Typical process for handling unaligned 128-bit data
; Assume a8 stores unaligned address

EE.LD.128.USAR.IP  q0, a8, 16    ; Load aligned data, save offset to SAR_BYTE
EE.VLD.128.IP      q1, a8, 16    ; Load next aligned data
EE.SRC.Q           q2, q0, q1    ; Shift and concatenate based on SAR_BYTE, result in q2
```

## 1.4 Instruction Set


### 1.4.1 Load Instructions

#### EE.VLD.128.IP - Vector Load 128-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `0001 imm16[5:4] qu[2:1] 1010 qu[0] imm16[3:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.VLD.128.IP qu, as, -512..496` |
| **Description** | Loads 128 bits from memory to register qu. Forces lower 4 bits of address in as to 0. After access, as is incremented by 6-bit sign-extended constant left-shifted by 4. |

**Operation:**
```
qu[127:0] = load128({as[31:4], 4{0}})
as[31:0] = as[31:0] + {20{imm16[7]}, imm16[7:0], 4{0}}
```

---

#### EE.VLD.128.XP - Vector Load 128-bit with Register Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qu[2:1] 1101 qu[0] 110 ad[3:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.VLD.128.XP qu, as, ad` |
| **Description** | Loads 128 bits from memory to register qu. Forces lower 4 bits of address in as to 0. After access, as is incremented by value in register ad. |

**Operation:**
```
qu[127:0] = load128({as[31:4], 4{0}})
as[31:0] = as[31:0] + ad[31:0]
```

---

#### EE.LD.128.USAR.IP - Load 128-bit with Unaligned Support and Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `0001 imm16[5:4] qu[2:1] 1110 qu[0] imm16[3:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.LD.128.USAR.IP qu, as, -512..496` |
| **Description** | Loads 128 bits from memory to register qu. Saves the offset (lower 4 bits of as) to SAR_BYTE. Forces lower 4 bits of address to 0. After access, as is incremented. |

**Operation:**
```
SAR_BYTE[3:0] = as[3:0]
qu[127:0] = load128({as[31:4], 4{0}})
as[31:0] = as[31:0] + {20{imm16[7]}, imm16[7:0], 4{0}}
```

---

#### EE.LD.128.USAR.XP - Load 128-bit with Unaligned Support and Register Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qu[2:1] 1101 qu[0] 111 ad[3:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.LD.128.USAR.XP qu, as, ad` |
| **Description** | Loads 128 bits from memory to register qu. Saves the offset to SAR_BYTE. Forces lower 4 bits of address to 0. After access, as is incremented by ad. |

**Operation:**
```
SAR_BYTE[3:0] = as[3:0]
qu[127:0] = load128({as[31:4], 4{0}})
as[31:0] = as[31:0] + ad[31:0]
```

---

#### EE.VLD.H.64.IP - Vector Load High 64-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `1 imm8[7] qu[2:1] 1011 qu[0] imm8[6:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.VLD.H.64.IP qu, as, -1024..1016` |
| **Description** | Loads upper 64 bits from memory to register qu. Forces lower 3 bits of address to 0. After access, as is incremented by 8-bit sign-extended constant left-shifted by 3. |

**Operation:**
```
qu[127:64] = load64({as[31:3], 3{0}})
as[31:0] = as[31:0] + {21{imm8[7]}, imm8[7:0], 3{0}}
```

---

#### EE.VLD.L.64.IP - Vector Load Low 64-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `1 imm8[7] qu[2:1] 0100 qu[0] imm8[6:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.VLD.L.64.IP qu, as, -1024..1016` |
| **Description** | Loads lower 64 bits from memory to register qu. Forces lower 3 bits of address to 0. After access, as is incremented. |

**Operation:**
```
qu[63:0] = load64({as[31:3], 3{0}})
as[31:0] = as[31:0] + {21{imm8[7]}, imm8[7:0], 3{0}}
```

---

### 1.4.2 Broadcast Load Instructions

#### EE.VLDBC.8 - Vector Load Broadcast 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `100 qu[2] 0110 qu[1] 0 qu[0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.VLDBC.8 qu, as` |
| **Description** | Loads 8-bit data from memory and broadcasts it to sixteen 8-bit data segments in register qu. After access, as is incremented by 1. |

**Operation:**
```
qu[127:0] = {16{load8(as[31:0])}}
as[31:0] = as[31:0] + 1
```

---

#### EE.VLDBC.8.IP - Vector Load Broadcast 8-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `100 qu[2] 0110 qu[1] 1 qu[0] imm1[0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.VLDBC.8.IP qu, as, 0..1` |
| **Description** | Loads 8-bit data and broadcasts to 16 segments. After access, as is incremented by imm1. |

---

#### EE.VLDBC.16 - Vector Load Broadcast 16-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `100 qu[2] 0110 qu[1] 0 qu[0] as[3:0] 0100` (with different encoding) |
| **Assembler Syntax** | `EE.VLDBC.16 qu, as` |
| **Description** | Loads 16-bit data from memory and broadcasts it to eight 16-bit data segments in register qu. After access, as is incremented by 2. |

**Operation:**
```
qu[127:0] = {8{load16({as[31:1], 1{0}})}}
as[31:0] = as[31:0] + 2
```

---

#### EE.VLDBC.32 - Vector Load Broadcast 32-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `100 qu[2] 0110 qu[1] 0 qu[0] as[3:0] 0100` (with different encoding) |
| **Assembler Syntax** | `EE.VLDBC.32 qu, as` |
| **Description** | Loads 32-bit data from memory and broadcasts it to four 32-bit data segments in register qu. After access, as is incremented by 4. |

**Operation:**
```
qu[127:0] = {4{load32({as[31:2], 2{0}})}}
as[31:0] = as[31:0] + 4
```

---

### 1.4.3 QACC Load Instructions

#### EE.LDQA.S8.128.IP - Load QACC Signed 8-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `0001 imm16[5:4] 0000 1001 imm16[3:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.LDQA.S8.128.IP as, -512..496` |
| **Description** | Loads 128 bits from memory, divides into sixteen 8-bit segments. Sign-extends each segment to 20 bits and stores in QACC_H and QACC_L. After access, as is incremented. |

**Operation:**
```
temp[127:0] = load128({as[31:4], 4{0}})
QACC_L[ 19: 0] = {12{temp[7]}}, temp[7:0]}
QACC_L[ 39:20] = {12{temp[15]}}, temp[15:8]}
...
QACC_H[139:120] = {12{temp[119]}}, temp[119:112]}
QACC_H[159:140] = {12{temp[127]}}, temp[127:120]}
as[31:0] = as[31:0] + {20{imm16[7]}, imm16[7:0], 4{0}}
```

---

#### EE.LDQA.S16.128.IP - Load QACC Signed 16-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `0001 imm16[5:4] 0000 1010 imm16[3:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.LDQA.S16.128.IP as, -512..496` |
| **Description** | Loads 128 bits, divides into eight 16-bit segments. Sign-extends each segment to 40 bits and stores in QACC_H and QACC_L. |

---

#### EE.LDQA.U8.128.IP - Load QACC Unsigned 8-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `0001 imm16[5:4] 0000 1011 imm16[3:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.LDQA.U8.128.IP as, -512..496` |
| **Description** | Loads 128 bits, divides into sixteen 8-bit segments. Zero-extends each segment to 20 bits and stores in QACC. |

---

#### EE.LDQA.U16.128.IP - Load QACC Unsigned 16-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `0001 imm16[5:4] 0000 1100 imm16[3:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.LDQA.U16.128.IP as, -512..496` |
| **Description** | Loads 128 bits, divides into eight 16-bit segments. Zero-extends each segment to 40 bits and stores in QACC. |

---

### 1.4.4 Store Instructions

#### EE.VST.128.IP - Vector Store 128-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `1 imm16[7] qv[2:1] 1010 qv[0] imm16[6:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.VST.128.IP qv, as, -2048..2032` |
| **Description** | Forces lower 4 bits of address to 0 and stores 128 bits from register qv to memory. After access, as is incremented by 8-bit sign-extended constant left-shifted by 4. |

**Operation:**
```
qv[127:0] => store128({as[31:4], 4{0}})
as[31:0] = as[31:0] + {20{imm16[7]}, imm16[7:0], 4{0}}
```

---

#### EE.VST.128.XP - Vector Store 128-bit with Register Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qv[2:1] 1101 qv[0] 111 ad[3:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.VST.128.XP qv, as, ad` |
| **Description** | Forces lower 4 bits of address to 0 and stores 128 bits from register qv to memory. After access, as is incremented by value in register ad. |

---

#### EE.VST.H.64.IP - Vector Store High 64-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `1 imm8[7] qv[2:1] 1011 qv[0] imm8[6:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.VST.H.64.IP qv, as, -1024..1016` |
| **Description** | Forces lower 3 bits of address to 0 and stores upper 64 bits from register qv to memory. After access, as is incremented. |

**Operation:**
```
qv[127:64] => store64({as[31:3], 3{0}})
as[31:0] = as[31:0] + {21{imm8[7]}, imm8[7:0], 3{0}}
```

---

#### EE.VST.L.64.IP - Vector Store Low 64-bit with Immediate Post-increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `1 imm8[7] qv[2:1] 0100 qv[0] imm8[6:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.VST.L.64.IP qv, as, -1024..1016` |
| **Description** | Forces lower 3 bits of address to 0 and stores lower 64 bits from register qv to memory. After access, as is incremented. |

---

### 1.4.5 Arithmetic Instructions

#### EE.VADDS.S8 - Vector Add Signed 8-bit with Saturation

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 0 qy[1:0] qx[2:0] 11000100` |
| **Assembler Syntax** | `EE.VADDS.S8 qa, qx, qy` |
| **Description** | Performs vector addition on sixteen 8-bit data segments. Results are saturated to signed 8-bit range and written to qa. |

**Operation:**
```
qa[ 7: 0] = min(max(qx[ 7: 0] + qy[ 7: 0], -2^7), 2^7-1)
qa[ 15: 8] = min(max(qx[ 15: 8] + qy[ 15: 8], -2^7), 2^7-1)
...
qa[127:120] = min(max(qx[127:120] + qy[127:120], -2^7), 2^7-1)
```

---

#### EE.VADDS.S16 - Vector Add Signed 16-bit with Saturation

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 0 qy[1:0] qx[2:0] 11010100` |
| **Assembler Syntax** | `EE.VADDS.S16 qa, qx, qy` |
| **Description** | Performs vector addition on eight 16-bit data segments. Results are saturated to signed 16-bit range. |

**Operation:**
```
qa[ 15: 0] = min(max(qx[ 15: 0] + qy[ 15: 0], -2^15), 2^15-1)
qa[ 31:16] = min(max(qx[ 31:16] + qy[ 31:16], -2^15), 2^15-1)
...
qa[127:112] = min(max(qx[127:112] + qy[127:112], -2^15), 2^15-1)
```

---

#### EE.VADDS.S32 - Vector Add Signed 32-bit with Saturation

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 0 qy[1:0] qx[2:0] 11100100` |
| **Assembler Syntax** | `EE.VADDS.S32 qa, qx, qy` |
| **Description** | Performs vector addition on four 32-bit data segments. Results are saturated to signed 32-bit range. |

---

#### EE.VADDS.S8.LD.INCP - Vector Add Signed 8-bit with Load and Increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `111000 qu[2:1] qy[0] 000 qu[0] qa[2:0] qx[1:0] qy[2:1] 1100 as[3:0] 111 qx[2]` |
| **Assembler Syntax** | `EE.VADDS.S8.LD.INCP qu, as, qa, qx, qy` |
| **Description** | Performs vector addition on 8-bit data. During operation, loads 16 bytes from memory to qu. After access, as is incremented by 16. |

---

#### EE.VADDS.S16.LD.INCP - Vector Add Signed 16-bit with Load and Increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `111000 qu[2:1] qy[0] 000 qu[0] qa[2:0] qx[1:0] qy[2:1] 1101 as[3:0] 111 qx[2]` |
| **Assembler Syntax** | `EE.VADDS.S16.LD.INCP qu, as, qa, qx, qy` |
| **Description** | Performs vector addition on 16-bit data with simultaneous load. |

---

#### EE.VADDS.S32.LD.INCP - Vector Add Signed 32-bit with Load and Increment

| Field | Description |
|-------|-------------|
| **Instruction Word** | `111000 qu[2:1] qy[0] 000 qu[0] qa[2:0] qx[1:0] qy[2:1] 1110 as[3:0] 111 qx[2]` |
| **Assembler Syntax** | `EE.VADDS.S32.LD.INCP qu, as, qa, qx, qy` |
| **Description** | Performs vector addition on 32-bit data with simultaneous load. |

---

#### EE.VSUBS.S8 - Vector Subtract Signed 8-bit with Saturation

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 1 qy[1:0] qx[2:0] 11110100` |
| **Assembler Syntax** | `EE.VSUBS.S8 qa, qx, qy` |
| **Description** | Performs vector subtraction on sixteen 8-bit data segments. qx is subtrahend, qy is minuend. Results are saturated. |

**Operation:**
```
qa[ 7: 0] = min(max(qx[ 7: 0] - qy[ 7: 0], -2^7), 2^7-1)
qa[ 15: 8] = min(max(qx[ 15: 8] - qy[ 15: 8], -2^7), 2^7-1)
...
qa[127:120] = min(max(qx[127:120] - qy[127:120], -2^7), 2^7-1)
```

---

#### EE.VSUBS.S16 - Vector Subtract Signed 16-bit with Saturation

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 1 qy[1:0] qx[2:0] 11010100` |
| **Assembler Syntax** | `EE.VSUBS.S16 qa, qx, qy` |
| **Description** | Performs vector subtraction on eight 16-bit data segments. |

---

#### EE.VSUBS.S32 - Vector Subtract Signed 32-bit with Saturation

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 1 qy[1:0] qx[2:0] 11100100` |
| **Assembler Syntax** | `EE.VSUBS.S32 qa, qx, qy` |
| **Description** | Performs vector subtraction on four 32-bit data segments. |

---

### 1.4.6 Multiplication Instructions

#### EE.VMUL.S8 - Vector Multiply Signed 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qz[2:1] 1100 qz[0] qy[2] 0 qy[1:0] qx[2:0] 01000100` |
| **Assembler Syntax** | `EE.VMUL.S8 qz, qx, qy` |
| **Description** | Performs signed multiplication on sixteen 8-bit data segments. Results are right-shifted by SAR bits and stored in qz. |

**Operation:**
```
qz[ 7: 0] = (qx[ 7: 0] * qy[ 7: 0]) >> SAR[5:0]
qz[ 15: 8] = (qx[ 15: 8] * qy[ 15: 8]) >> SAR[5:0]
...
qz[127:120] = (qx[127:120] * qy[127:120]) >> SAR[5:0]
```

---

#### EE.VMUL.S16 - Vector Multiply Signed 16-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qz[2:1] 1100 qz[0] qy[2] 0 qy[1:0] qx[2:0] 01010100` |
| **Assembler Syntax** | `EE.VMUL.S16 qz, qx, qy` |
| **Description** | Performs signed multiplication on eight 16-bit data segments. Results are right-shifted by SAR bits. |

---

#### EE.VMUL.U8 - Vector Multiply Unsigned 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qz[2:1] 1100 qz[0] qy[2] 0 qy[1:0] qx[2:0] 01100100` |
| **Assembler Syntax** | `EE.VMUL.U8 qz, qx, qy` |
| **Description** | Performs unsigned multiplication on sixteen 8-bit data segments. |

---

#### EE.VMUL.U16 - Vector Multiply Unsigned 16-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qz[2:1] 1100 qz[0] qy[2] 0 qy[1:0] qx[2:0] 01110100` |
| **Assembler Syntax** | `EE.VMUL.U16 qz, qx, qy` |
| **Description** | Performs unsigned multiplication on eight 16-bit data segments. |

---

### 1.4.7 Complex Multiplication Instructions

#### EE.CMUL.S16 - Complex Multiply Signed 16-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qz[2:1] 1100 qz[0] qy[2] 0 qy[1:0] qx[2:0] sel4[1:0] 0100` |
| **Assembler Syntax** | `EE.CMUL.S16 qz, qx, qy, sel4` |
| **Description** | Selects one of four 32-bit complex data segments from qy based on sel4. Performs complex multiplication with qx and stores result in qz. |

**Operation:**
```
temp[31:0] = qy[sel4*32+31:sel4*32]
qz[ 15: 0] = qx[ 15: 0] * temp[15:0] - qx[ 31:16] * temp[31:16]
qz[ 31:16] = qx[ 15: 0] * temp[31:16] + qx[ 31:16] * temp[15:0]
...
qz[127:112] = qx[111:96] * temp[15:0] - qx[127:112] * temp[31:16]
```

---

### 1.4.8 Multiply-Accumulate Instructions (ACCX)

#### EE.VMULAS.S8.ACCX - Vector Multiply-Accumulate Signed 8-bit to ACCX

| Field | Description |
|-------|-------------|
| **Instruction Word** | `000110100 qy[2] 0 qy[1:0] qx[2:0] 11000100` |
| **Assembler Syntax** | `EE.VMULAS.S8.ACCX qx, qy` |
| **Description** | Divides qx and qy into sixteen 8-bit segments. Performs signed multiply-accumulate on all segments. Accumulated result is added to ACCX and saturated. |

**Operation:**
```
add0[15:0] = qx[ 7: 0] * qy[ 7: 0]
add1[15:0] = qx[ 15: 8] * qy[ 15: 8]
...
add15[15:0] = qx[127:120] * qy[127:120]
sum[40:0] = ACCX[39:0] + add0[15:0] + add1[15:0] + ... + add15[15:0]
ACCX[39:0] = min(max(sum[40:0], -2^39), 2^39-1)
```

---

#### EE.VMULAS.S16.ACCX - Vector Multiply-Accumulate Signed 16-bit to ACCX

| Field | Description |
|-------|-------------|
| **Instruction Word** | `000110100 qy[2] 0 qy[1:0] qx[2:0] 10000100` |
| **Assembler Syntax** | `EE.VMULAS.S16.ACCX qx, qy` |
| **Description** | Divides qx and qy into eight 16-bit segments. Performs signed multiply-accumulate to ACCX. |

---

#### EE.VMULAS.U8.ACCX - Vector Multiply-Accumulate Unsigned 8-bit to ACCX

| Field | Description |
|-------|-------------|
| **Instruction Word** | `000010100 qy[2] 0 qy[1:0] qx[2:0] 11000100` |
| **Assembler Syntax** | `EE.VMULAS.U8.ACCX qx, qy` |
| **Description** | Performs unsigned multiply-accumulate on sixteen 8-bit segments to ACCX. |

---

#### EE.VMULAS.U16.ACCX - Vector Multiply-Accumulate Unsigned 16-bit to ACCX

| Field | Description |
|-------|-------------|
| **Instruction Word** | `000010100 qy[2] 0 qy[1:0] qx[2:0] 10000100` |
| **Assembler Syntax** | `EE.VMULAS.U16.ACCX qx, qy` |
| **Description** | Performs unsigned multiply-accumulate on eight 16-bit segments to ACCX. |

---

### 1.4.9 Multiply-Accumulate Instructions (QACC)

#### EE.VMULAS.S8.QACC - Vector Multiply-Accumulate Signed 8-bit to QACC

| Field | Description |
|-------|-------------|
| **Instruction Word** | `000110100 qy[2] 1 qy[1:0] qx[2:0] 11000100` |
| **Assembler Syntax** | `EE.VMULAS.S8.QACC qx, qy` |
| **Description** | Divides qx and qy into sixteen 8-bit segments. Signed multiplication results are added to corresponding 20-bit segments in QACC_H and QACC_L. Results are saturated to 20-bit signed numbers. |

**Operation:**
```
QACC_L[ 19: 0] = min(max(QACC_L[ 19: 0] + qx[ 7: 0] * qy[ 7: 0], -2^19), 2^19-1)
QACC_L[ 39:20] = min(max(QACC_L[ 39:20] + qx[ 15: 8] * qy[ 15: 8], -2^19), 2^19-1)
...
QACC_H[159:140] = min(max(QACC_H[159:140] + qx[127:120] * qy[127:120], -2^19), 2^19-1)
```

---

#### EE.VMULAS.S16.QACC - Vector Multiply-Accumulate Signed 16-bit to QACC

| Field | Description |
|-------|-------------|
| **Instruction Word** | `000110100 qy[2] 1 qy[1:0] qx[2:0] 10000100` |
| **Assembler Syntax** | `EE.VMULAS.S16.QACC qx, qy` |
| **Description** | Divides qx and qy into eight 16-bit segments. Results are added to corresponding 40-bit segments in QACC and saturated to 40-bit signed numbers. |

---

#### EE.VMULAS.U8.QACC - Vector Multiply-Accumulate Unsigned 8-bit to QACC

| Field | Description |
|-------|-------------|
| **Instruction Word** | `000010100 qy[2] 1 qy[1:0] qx[2:0] 11000100` |
| **Assembler Syntax** | `EE.VMULAS.U8.QACC qx, qy` |
| **Description** | Performs unsigned multiply-accumulate to QACC with 20-bit saturation. |

---

#### EE.VMULAS.U16.QACC - Vector Multiply-Accumulate Unsigned 16-bit to QACC

| Field | Description |
|-------|-------------|
| **Instruction Word** | `000010100 qy[2] 1 qy[1:0] qx[2:0] 10000100` |
| **Assembler Syntax** | `EE.VMULAS.U16.QACC qx, qy` |
| **Description** | Performs unsigned multiply-accumulate to QACC with 40-bit saturation. |

---

### 1.4.10 Comparison Instructions

#### EE.VMAX.S8 - Vector Maximum Signed 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 0 qy[1:0] qx[2:0] 10010100` |
| **Assembler Syntax** | `EE.VMAX.S8 qa, qx, qy` |
| **Description** | Compares sixteen 8-bit segments and stores maximum values in qa. |

**Operation:**
```
qa[ 7: 0] = max(qx[ 7: 0], qy[ 7: 0])
qa[ 15: 8] = max(qx[ 15: 8], qy[ 15: 8])
...
qa[127:120] = max(qx[127:120], qy[127:120])
```

---

#### EE.VMAX.S16 - Vector Maximum Signed 16-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 0 qy[1:0] qx[2:0] 10100100` |
| **Assembler Syntax** | `EE.VMAX.S16 qa, qx, qy` |
| **Description** | Compares eight 16-bit segments and stores maximum values. |

---

#### EE.VMAX.S32 - Vector Maximum Signed 32-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 0 qy[1:0] qx[2:0] 10110100` |
| **Assembler Syntax** | `EE.VMAX.S32 qa, qx, qy` |
| **Description** | Compares four 32-bit segments and stores maximum values. |

---

#### EE.VMIN.S8 - Vector Minimum Signed 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 1 qy[1:0] qx[2:0] 10010100` |
| **Assembler Syntax** | `EE.VMIN.S8 qa, qx, qy` |
| **Description** | Compares sixteen 8-bit segments and stores minimum values in qa. |

---

#### EE.VMIN.S16 - Vector Minimum Signed 16-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 1 qy[1:0] qx[2:0] 10100100` |
| **Assembler Syntax** | `EE.VMIN.S16 qa, qx, qy` |
| **Description** | Compares eight 16-bit segments and stores minimum values. |

---

#### EE.VMIN.S32 - Vector Minimum Signed 32-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 1 qy[1:0] qx[2:0] 10110100` |
| **Assembler Syntax** | `EE.VMIN.S32 qa, qx, qy` |
| **Description** | Compares four 32-bit segments and stores minimum values. |

---

#### EE.VCMP.EQ.S8 - Vector Compare Equal Signed 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 0 qy[1:0] qx[2:0] 11001100` |
| **Assembler Syntax** | `EE.VCMP.EQ.S8 qa, qx, qy` |
| **Description** | Compares sixteen 8-bit segments for equality. Result is 0xFF if equal, 0 otherwise. |

**Operation:**
```
qa[ 7: 0] = (qx[ 7: 0] == qy[ 7: 0]) ? 0xFF : 0
qa[ 15: 8] = (qx[ 15: 8] == qy[ 15: 8]) ? 0xFF : 0
...
qa[127:120] = (qx[127:120] == qy[127:120]) ? 0xFF : 0
```

---

#### EE.VCMP.LT.S8 - Vector Compare Less Than Signed 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 0 qy[1:0] qx[2:0] 11011100` |
| **Assembler Syntax** | `EE.VCMP.LT.S8 qa, qx, qy` |
| **Description** | Compares if qx < qy. Result is 0xFF if true, 0 otherwise. |

---

#### EE.VCMP.GT.S8 - Vector Compare Greater Than Signed 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qa[2:1] 1110 qa[0] qy[2] 1 qy[1:0] qx[2:0] 11001100` |
| **Assembler Syntax** | `EE.VCMP.GT.S8 qa, qx, qy` |
| **Description** | Compares if qx > qy. Result is 0xFF if true, 0 otherwise. |

---

### 1.4.11 Bitwise Instructions

#### EE.ANDQ - Bitwise AND

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qa[2:1] 1101 qa[0] 000 qy[2:1] 01 qx[2:1] qy[0] qx[0] 0100` |
| **Assembler Syntax** | `EE.ANDQ qa, qx, qy` |
| **Description** | Performs bitwise AND operation on qx and qy, stores result in qa. |

**Operation:**
```
qa = qx & qy
```

---

#### EE.ORQ - Bitwise OR

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qa[2:1] 1101 qa[0] 001 qy[2:1] 01 qx[2:1] qy[0] qx[0] 0100` |
| **Assembler Syntax** | `EE.ORQ qa, qx, qy` |
| **Description** | Performs bitwise OR operation on qx and qy. |

**Operation:**
```
qa = qx | qy
```

---

#### EE.XORQ - Bitwise XOR

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qa[2:1] 1101 qa[0] 011 qy[2:1] 01 qx[2:1] qy[0] qx[0] 0100` |
| **Assembler Syntax** | `EE.XORQ qa, qx, qy` |
| **Description** | Performs bitwise XOR operation on qx and qy. |

**Operation:**
```
qa = qx ^ qy
```

---

#### EE.NOTQ - Bitwise NOT

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qa[2:1] 1101 qa[0] 010 1111 01 qx[2:1] 0 qx[0] 0100` |
| **Assembler Syntax** | `EE.NOTQ qa, qx` |
| **Description** | Performs bitwise NOT operation on qx, stores result in qa. |

**Operation:**
```
qa = ~qx
```

---

### 1.4.12 Shift Instructions

#### EE.VSL.32 - Vector Shift Left 32-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qs[2:1] 1101 qs[0] 01111110 qa[2:0] 0100` |
| **Assembler Syntax** | `EE.VSL.32 qa, qs` |
| **Description** | Performs left shift on four 32-bit segments in qs. Shift amount is value in SAR[5:0]. Lower bits are padded with 0. |

**Operation:**
```
qa[ 31: 0] = qs[ 31: 0] << SAR[5:0]
qa[ 63:32] = qs[ 63:32] << SAR[5:0]
qa[ 95:64] = qs[ 95:64] << SAR[5:0]
qa[127:96] = qs[127:96] << SAR[5:0]
```

---

#### EE.VSR.32 - Vector Shift Right 32-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qs[2:1] 1101 qs[0] 01111111 qa[2:0] 0100` |
| **Assembler Syntax** | `EE.VSR.32 qa, qs` |
| **Description** | Performs arithmetic right shift on four 32-bit segments. Higher bits are padded with sign bit. |

**Operation:**
```
qa[ 31: 0] = qs[ 31: 0] >> SAR[5:0]
qa[ 63:32] = qs[ 63:32] >> SAR[5:0]
qa[ 95:64] = qs[ 95:64] >> SAR[5:0]
qa[127:96] = qs[127:96] >> SAR[5:0]
```

---

#### EE.SRC.Q - Shift Right and Concatenate Q registers

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qa[2:1] 1101 qa[0] 100 qs0[2:1] 01 qs1[2:1] 0 qs1[0] qs0[0] 0100` |
| **Assembler Syntax** | `EE.SRC.Q qa, qs0, qs1` |
| **Description** | Concatenates qs1 and qs0 (256 bits total), shifts right by SAR_BYTE*8 bits, stores lower 128 bits in qa. |

**Operation:**
```
qa[127:0] = {qs1[127:0], qs0[127:0]} >> {SAR_BYTE[3:0] << 3}
```

---

### 1.4.13 Data Movement Instructions

#### MV.QR - Move Q Register

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qu[2:1] 1111 qu[0] 000 qs[2:1] 000 qs[0] 00100` |
| **Assembler Syntax** | `MV.QR qu, qs` |
| **Description** | Moves value from source QR register qs to target QR register qu. |

**Operation:**
```
qu = qs
```

---

#### EE.ZERO.Q - Zero Q Register

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qa[2:1] 1101 qa[0] 111111110100100` |
| **Assembler Syntax** | `EE.ZERO.Q qa` |
| **Description** | Clears value in register qa to 0. |

**Operation:**
```
qa = 0
```

---

#### EE.ZERO.ACCX - Zero ACCX

| Field | Description |
|-------|-------------|
| **Instruction Word** | `001001010000100000000100` |
| **Assembler Syntax** | `EE.ZERO.ACCX` |
| **Description** | Clears value in special register ACCX to 0. |

**Operation:**
```
ACCX = 0
```

---

#### EE.ZERO.QACC - Zero QACC

| Field | Description |
|-------|-------------|
| **Instruction Word** | `001001010000100001000100` |
| **Assembler Syntax** | `EE.ZERO.QACC` |
| **Description** | Clears values in special registers QACC_L and QACC_H to 0. |

**Operation:**
```
QACC_L = 0
QACC_H = 0
```

---

### 1.4.14 ReLU Instructions

#### EE.VRELU.S8 - Vector ReLU Signed 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qs[2:1] 1101 qs[0] 101 ax[3:0] ay[3:0] 0100` |
| **Assembler Syntax** | `EE.VRELU.S8 qs, ax, ay` |
| **Description** | Divides qs into sixteen 8-bit segments. If segment value <= 0, multiplies by lower 8 bits of ax and right-shifts by lower 5 bits of ay. Otherwise, value remains unchanged. |

**Operation:**
```
qs[ 7: 0] = (qs[ 7: 0] <= 0) ? (qs[ 7: 0] * ax[7:0]) >> ay[4:0] : qs[ 7: 0]
qs[ 15: 8] = (qs[ 15: 8] <= 0) ? (qs[ 15: 8] * ax[7:0]) >> ay[4:0] : qs[ 15: 8]
...
qs[127:120] = (qs[127:120] <= 0) ? (qs[127:120] * ax[7:0]) >> ay[4:0] : qs[127:120]
```

---

#### EE.VRELU.S16 - Vector ReLU Signed 16-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qs[2:1] 1101 qs[0] 001 ax[3:0] ay[3:0] 0100` |
| **Assembler Syntax** | `EE.VRELU.S16 qs, ax, ay` |
| **Description** | Divides qs into eight 16-bit segments. If segment value <= 0, multiplies by lower 16 bits of ax and right-shifts by lower 6 bits of ay. |

---

#### EE.VPRELU.S8 - Vector PReLU Signed 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qz[2:1] 1100 qz[0] qy[2] 1 qy[1:0] qx[2:0] ay[3:0] 0100` |
| **Assembler Syntax** | `EE.VPRELU.S8 qz, qx, qy, ay` |
| **Description** | Divides qx into sixteen 8-bit segments. If segment value <= 0, multiplies by corresponding segment in qy and right-shifts by lower 5 bits of ay. Result stored in qz. |

**Operation:**
```
qz[ 7: 0] = (qx[ 7: 0] <= 0) ? (qx[ 7: 0] * qy[ 7: 0]) >> ay[4:0] : qx[ 7: 0]
qz[ 15: 8] = (qx[ 15: 8] <= 0) ? (qx[ 15: 8] * qy[ 15: 8]) >> ay[4:0] : qx[ 15: 8]
...
qz[127:120] = (qx[127:120] <= 0) ? (qx[127:120] * qy[127:120]) >> ay[4:0] : qx[127:120]
```

---

#### EE.VPRELU.S16 - Vector PReLU Signed 16-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `10 qz[2:1] 1100 qz[0] qy[2] 0 qy[1:0] qx[2:0] ay[3:0] 0100` |
| **Assembler Syntax** | `EE.VPRELU.S16 qz, qx, qy, ay` |
| **Description** | Divides qx into eight 16-bit segments. If segment value <= 0, multiplies by corresponding segment in qy and right-shifts by lower 6 bits of ay. |

---

### 1.4.15 Zip/Unzip Instructions

#### EE.VZIP.8 - Vector Zip 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qs1[2:1] 1100 qs1[0] qs0[2:0] 001111010100` |
| **Assembler Syntax** | `EE.VZIP.8 qs0, qs1` |
| **Description** | Implements zip algorithm on 8-bit vector data. Interleaves data from qs0 and qs1. |

---

#### EE.VZIP.16 - Vector Zip 16-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qs1[2:1] 1100 qs1[0] qs0[2:0] 001110110100` |
| **Assembler Syntax** | `EE.VZIP.16 qs0, qs1` |
| **Description** | Implements zip algorithm on 16-bit vector data. |

---

#### EE.VZIP.32 - Vector Zip 32-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qs1[2:1] 1100 qs1[0] qs0[2:0] 001111000100` |
| **Assembler Syntax** | `EE.VZIP.32 qs0, qs1` |
| **Description** | Implements zip algorithm on 32-bit vector data. |

---

#### EE.VUNZIP.8 - Vector Unzip 8-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qs1[2:1] 1100 qs1[0] qs0[2:0] 001110100100` |
| **Assembler Syntax** | `EE.VUNZIP.8 qs0, qs1` |
| **Description** | Implements unzip algorithm on 8-bit vector data. Deinterleaves data. |

---

#### EE.VUNZIP.16 - Vector Unzip 16-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qs1[2:1] 1100 qs1[0] qs0[2:0] 001110000100` |
| **Assembler Syntax** | `EE.VUNZIP.16 qs0, qs1` |
| **Description** | Implements unzip algorithm on 16-bit vector data. |

---

#### EE.VUNZIP.32 - Vector Unzip 32-bit

| Field | Description |
|-------|-------------|
| **Instruction Word** | `11 qs1[2:1] 1100 qs1[0] qs0[2:0] 001110010100` |
| **Assembler Syntax** | `EE.VUNZIP.32 qs0, qs1` |
| **Description** | Implements unzip algorithm on 32-bit vector data. |

---

### 1.4.16 GPIO Instructions

#### EE.WR_MASK_GPIO_OUT - Write Masked GPIO Output

| Field | Description |
|-------|-------------|
| **Instruction Word** | `011100100100 ax[3:0] as[3:0] 0100` |
| **Assembler Syntax** | `EE.WR_MASK_GPIO_OUT as, ax` |
| **Description** | Dedicated CPU GPIO instruction to set specified bits in GPIO_OUT. Lower 8 bits of ax store mask, lower 8 bits of as store assignment content. |

**Operation:**
```
GPIO_OUT[7:0] = (GPIO_OUT[7:0] & ~ax[7:0]) | (as[7:0] & ax[7:0])
```

---

#### EE.SET_BIT_GPIO_OUT - Set Bit GPIO Output

| Field | Description |
|-------|-------------|
| **Instruction Word** | `011100100000 imm8[7:0] 0100` |
| **Assembler Syntax** | `EE.SET_BIT_GPIO_OUT imm8` |
| **Description** | Sets specified bit in GPIO_OUT. |

**Operation:**
```
GPIO_OUT[imm8] = 1
```

---

#### EE.CLR_BIT_GPIO_OUT - Clear Bit GPIO Output

| Field | Description |
|-------|-------------|
| **Instruction Word** | `011100100001 imm8[7:0] 0100` |
| **Assembler Syntax** | `EE.CLR_BIT_GPIO_OUT imm8` |
| **Description** | Clears specified bit in GPIO_OUT. |

**Operation:**
```
GPIO_OUT[imm8] = 0
```

---

#### EE.GET_GPIO_IN - Get GPIO Input

| Field | Description |
|-------|-------------|
| **Instruction Word** | `0111001000100000 au[3:0] 0100` |
| **Assembler Syntax** | `EE.GET_GPIO_IN au` |
| **Description** | Reads GPIO_IN to lower 8 bits of au. |

**Operation:**
```
au[7:0] = GPIO_IN[7:0]
```

---

## 1.5 Instruction Quick Reference

### Load Instructions

| Instruction | Description |
|-------------|-------------|
| EE.VLD.128.IP | Vector Load 128-bit with Immediate Post-increment |
| EE.VLD.128.XP | Vector Load 128-bit with Register Post-increment |
| EE.LD.128.USAR.IP | Load 128-bit with Unaligned Support |
| EE.LD.128.USAR.XP | Load 128-bit with Unaligned Support (Register) |
| EE.VLD.H.64.IP | Vector Load High 64-bit |
| EE.VLD.L.64.IP | Vector Load Low 64-bit |
| EE.VLDBC.8 | Vector Load Broadcast 8-bit |
| EE.VLDBC.16 | Vector Load Broadcast 16-bit |
| EE.VLDBC.32 | Vector Load Broadcast 32-bit |

### Store Instructions

| Instruction | Description |
|-------------|-------------|
| EE.VST.128.IP | Vector Store 128-bit with Immediate Post-increment |
| EE.VST.128.XP | Vector Store 128-bit with Register Post-increment |
| EE.VST.H.64.IP | Vector Store High 64-bit |
| EE.VST.L.64.IP | Vector Store Low 64-bit |

### Arithmetic Instructions

| Instruction | Description |
|-------------|-------------|
| EE.VADDS.S8 | Vector Add Signed 8-bit with Saturation |
| EE.VADDS.S16 | Vector Add Signed 16-bit with Saturation |
| EE.VADDS.S32 | Vector Add Signed 32-bit with Saturation |
| EE.VSUBS.S8 | Vector Subtract Signed 8-bit with Saturation |
| EE.VSUBS.S16 | Vector Subtract Signed 16-bit with Saturation |
| EE.VSUBS.S32 | Vector Subtract Signed 32-bit with Saturation |

### Multiplication Instructions

| Instruction | Description |
|-------------|-------------|
| EE.VMUL.S8 | Vector Multiply Signed 8-bit |
| EE.VMUL.S16 | Vector Multiply Signed 16-bit |
| EE.VMUL.U8 | Vector Multiply Unsigned 8-bit |
| EE.VMUL.U16 | Vector Multiply Unsigned 16-bit |
| EE.CMUL.S16 | Complex Multiply Signed 16-bit |

### Multiply-Accumulate Instructions

| Instruction | Description |
|-------------|-------------|
| EE.VMULAS.S8.ACCX | MAC Signed 8-bit to ACCX |
| EE.VMULAS.S16.ACCX | MAC Signed 16-bit to ACCX |
| EE.VMULAS.U8.ACCX | MAC Unsigned 8-bit to ACCX |
| EE.VMULAS.U16.ACCX | MAC Unsigned 16-bit to ACCX |
| EE.VMULAS.S8.QACC | MAC Signed 8-bit to QACC |
| EE.VMULAS.S16.QACC | MAC Signed 16-bit to QACC |
| EE.VMULAS.U8.QACC | MAC Unsigned 8-bit to QACC |
| EE.VMULAS.U16.QACC | MAC Unsigned 16-bit to QACC |

### Comparison Instructions

| Instruction | Description |
|-------------|-------------|
| EE.VMAX.S8 | Vector Maximum Signed 8-bit |
| EE.VMAX.S16 | Vector Maximum Signed 16-bit |
| EE.VMAX.S32 | Vector Maximum Signed 32-bit |
| EE.VMIN.S8 | Vector Minimum Signed 8-bit |
| EE.VMIN.S16 | Vector Minimum Signed 16-bit |
| EE.VMIN.S32 | Vector Minimum Signed 32-bit |
| EE.VCMP.EQ.S8 | Vector Compare Equal |
| EE.VCMP.LT.S8 | Vector Compare Less Than |
| EE.VCMP.GT.S8 | Vector Compare Greater Than |

### Bitwise Instructions

| Instruction | Description |
|-------------|-------------|
| EE.ANDQ | Bitwise AND |
| EE.ORQ | Bitwise OR |
| EE.XORQ | Bitwise XOR |
| EE.NOTQ | Bitwise NOT |

### Shift Instructions

| Instruction | Description |
|-------------|-------------|
| EE.VSL.32 | Vector Shift Left 32-bit |
| EE.VSR.32 | Vector Shift Right 32-bit |
| EE.SRC.Q | Shift Right and Concatenate |

### Data Movement Instructions

| Instruction | Description |
|-------------|-------------|
| MV.QR | Move Q Register |
| EE.ZERO.Q | Zero Q Register |
| EE.ZERO.ACCX | Zero ACCX |
| EE.ZERO.QACC | Zero QACC |

### ReLU Instructions

| Instruction | Description |
|-------------|-------------|
| EE.VRELU.S8 | Vector ReLU Signed 8-bit |
| EE.VRELU.S16 | Vector ReLU Signed 16-bit |
| EE.VPRELU.S8 | Vector PReLU Signed 8-bit |
| EE.VPRELU.S16 | Vector PReLU Signed 16-bit |

### Zip/Unzip Instructions

| Instruction | Description |
|-------------|-------------|
| EE.VZIP.8 | Vector Zip 8-bit |
| EE.VZIP.16 | Vector Zip 16-bit |
| EE.VZIP.32 | Vector Zip 32-bit |
| EE.VUNZIP.8 | Vector Unzip 8-bit |
| EE.VUNZIP.16 | Vector Unzip 16-bit |
| EE.VUNZIP.32 | Vector Unzip 32-bit |

### GPIO Instructions

| Instruction | Description |
|-------------|-------------|
| EE.WR_MASK_GPIO_OUT | Write Masked GPIO Output |
| EE.SET_BIT_GPIO_OUT | Set Bit GPIO Output |
| EE.CLR_BIT_GPIO_OUT | Clear Bit GPIO Output |
| EE.GET_GPIO_IN | Get GPIO Input |

---

## 1.6 Notes

### Data Alignment

1. **Forced Alignment:** PIE instructions force alignment of access addresses. Ensure data is properly aligned to avoid performance loss.
2. **16-byte Alignment Declaration:** Use `aligned(16)` attribute or `heap_caps_aligned_alloc` for memory allocation.
3. **Unaligned Processing:** Use `EE.LD.128.USAR.*` + `EE.SRC.Q` combination for unaligned data.

### Overflow Handling

1. **Saturated Operations:** `EE.VADDS.*`, `EE.VSUBS.*` instructions automatically perform saturation.
2. **Wraparound:** Instructions without explicit saturation use wraparound mode.
3. **Accumulator Saturation:** `EE.SRCMB.*.QACC` instructions saturate during extraction.

### Pipeline Hazards

| Hazard Type | Description | Solution |
|-------------|-------------|----------|
| Data Hazard | Instruction depends on previous result | Insert NOP or reorder instructions |
| Resource Hazard | Multiple instructions use same hardware | Avoid issuing conflicting instructions |
| Control Hazard | Branch causes pipeline flush | Reduce branches or use conditional execution |

### Register Usage Convention

```asm
; Recommended register allocation
; a2-a7: Function arguments/temporary registers
; a8-a15: Address registers (as/ad/au/av/ax/ay)
; q0-q7: Vector data registers

; Typical loop structure
entry   sp, 128
l32i    a6, a5, 44        ; Load length
srai    a5, a6, 3         ; Length/8 (16-bit elements)
loopgtz a5, end_loop      ; Loop
    ee.vld.128.ip   q0, a3, 16    ; Load input0
    ee.vld.128.ip   q1, a4, 16    ; Load input1
    ee.vadds.s16    q2, q0, q1    ; Vector addition
    ee.vst.128.ip   q2, a2, 16    ; Store result
end_loop:
retw
```

---

**Document Version:** 1.0  
**Applicable Chip:** ESP32-S3  
**Last Updated:** 2026-04-02
