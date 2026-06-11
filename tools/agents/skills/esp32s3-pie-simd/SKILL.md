---
name: ESP32-S3 PIE SIMD Optimization
description: >
  ESP32-S3 Processor Instruction Extensions (PIE) SIMD reference for accelerating C code.
  Use when converting scalar C loops into PIE SIMD assembly (EE.* instructions).
  Covers 128-bit QR registers, vector arithmetic, MAC, non-aligned data, FFT, and common patterns.
  Target chip: ESP32-S3 (Xtensa LX7 with TIE extensions).
---

# ESP32-S3 PIE SIMD Quick Reference

## REGISTERS

### General Purpose
| Register | Count | Width | Use |
|----------|-------|-------|-----|
| AR (a0-a15) | 16 | 32-bit | Address/data, a1=sp, a2-a7=args, a8-a15=addr operands |
| QR (q0-q7) | 8 | 128-bit | SIMD vectors: 16×8-bit, 8×16-bit, or 4×32-bit lanes |
| FR (f0-f15) | 16 | 32-bit | IEEE754 float |

### Special Registers (PIE-specific)
| Register | Width | Access via | Purpose |
|----------|-------|-----------|---------|
| SAR | 6-bit | RSR/WUR.SAR | Mul result right-shift amount (bits) |
| SAR_BYTE | 4-bit | RUR/WUR.SAR_BYTE | Byte shift amount for SRC.Q |
| ACCX | 40-bit | RUR/WUR ACCX_0,ACCX_1 | Scalar accumulator (sum of all lane products) |
| QACC_H | 160-bit | RUR/WUR QACC_H_0..4 | Per-lane accumulator high (8×20-bit or 4×40-bit) |
| QACC_L | 160-bit | RUR/WUR QACC_L_0..4 | Per-lane accumulator low |
| FFT_BIT_WIDTH | 4-bit | WUR.FFT_BIT_WIDTH | Bit-reversal width (0→3bit ... 7→10bit) |
| UA_STATE | 128-bit | RUR/WUR UA_STATE | Unaligned load persistent state |

### Special Register Access
```asm
WSR.SAR aN          ; write SAR
RSR.SAR aN          ; read SAR
WUR.ACCX_0 aN       ; write ACCX[31:0]
WUR.ACCX_1 aN       ; write ACCX[39:32]  (lower 8 bits of aN used)
RUR.ACCX_0 aN       ; read ACCX[31:0]
RUR.ACCX_1 aN       ; read ACCX[39:32] → aN[7:0]
WUR.SAR_BYTE aN      ; write SAR_BYTE
RUR.SAR_BYTE aN      ; read SAR_BYTE
WUR.QACC_H_0 aN      ; write QACC_H segment 0 (bits 31:0)
RUR.QACC_H_0 aN      ; read QACC_H segment 0 (bits 31:0)
; QACC_H segments: _0(31:0) _1(63:32) _2(95:64) _3(127:96) _4(159:128)
; QACC_L same segment layout
EE.ZERO.ACCX         ; ACCX = 0
EE.ZERO.QACC         ; QACC_H = 0, QACC_L = 0
EE.ZERO.Q qN         ; qN = 0
```

---

## ALIGNMENT RULES (CRITICAL)

PIE loads/stores **force** address low bits to 0:
- 128-bit access: low 4 bits forced to 0 (16-byte aligned)
- 64-bit access: low 3 bits forced to 0 (8-byte aligned)
- 32-bit access: low 2 bits forced to 0 (4-byte aligned)
- 16-bit access: low 1 bit forced to 0 (2-byte aligned)

**Always declare buffers with `__attribute__((aligned(16)))` or use `heap_caps_aligned_alloc(16, size, MALLOC_CAP_DEFAULT)`.**

### Unaligned Load Pattern (3 instructions)
```asm
EE.LD.128.USAR.IP  q0, aN, 16    ; load aligned chunk, save offset→SAR_BYTE, aN+=16
EE.VLD.128.IP      q1, aN, 16    ; load next aligned chunk, aN+=16
EE.SRC.Q           q2, q0, q1    ; {q1,q0} >> SAR_BYTE*8 → q2 (lower 128 bits)
```

---

## INSTRUCTION CLASSIFICATION BY OPERATION

### 1. LOAD (Memory → QR / QACC)
```
EE.VLD.128.IP       qu, as, imm16    ; qu=load128(aligned(as)), as+=sext(imm16)*16
EE.VLD.128.XP       qu, as, ad       ; qu=load128(aligned(as)), as+=ad
EE.LD.128.USAR.IP   qu, as, imm16    ; SAR_BYTE=as[3:0], qu=load128(aligned(as)), as+=imm16
EE.LD.128.USAR.XP   qu, as, ad       ; SAR_BYTE=as[3:0], qu=load128(aligned(as)), as+=ad
EE.VLD.H.64.IP      qu, as, imm8     ; qu[127:64]=load64(aligned(as)), as+=imm8*8
EE.VLD.L.64.IP      qu, as, imm8     ; qu[63:0]=load64(aligned(as)), as+=imm8*8
EE.VLDBC.8          qu, as           ; qu = broadcast(load8(as)), as+=1
EE.VLDBC.16         qu, as           ; qu = broadcast(load16(aligned(as))), as+=2
EE.VLDBC.32         qu, as           ; qu = broadcast(load32(aligned(as))), as+=4
EE.LDQA.S8.128.IP   as, imm16        ; sign-extend 16×8bit→16×20bit into QACC_H+L, as+=imm16
EE.LDQA.S16.128.IP  as, imm16        ; sign-extend 8×16bit→8×40bit into QACC_H+L, as+=imm16
; IP = Immediate Post-increment, XP = register post-increment
; imm16 for 128-bit: step = sext(imm16[7:0])*16, range -2048..2032
```

### 2. STORE (QR → Memory)
```
EE.VST.128.IP       qv, as, imm16    ; store128(qv, aligned(as)), as+=sext(imm16)*16
EE.VST.128.XP       qv, as, ad       ; store128(qv, aligned(as)), as+=ad
EE.VST.H.64.IP      qv, as, imm8     ; store64(qv[127:64], aligned(as)), as+=imm8*8
EE.VST.L.64.IP      qv, as, imm8     ; store64(qv[63:0], aligned(as)), as+=imm8*8
```

### 3. VECTOR ARITHMETIC (lane-wise, saturated)
```
EE.VADDS.S8         qa, qx, qy       ; qa[i]=saturate_s8(qx[i]+qy[i])  for i in 0..15
EE.VADDS.S16        qa, qx, qy       ; qa[i]=saturate_s16(qx[i]+qy[i]) for i in 0..7
EE.VADDS.S32        qa, qx, qy       ; qa[i]=saturate_s32(qx[i]+qy[i]) for i in 0..3
EE.VSUBS.S8         qa, qx, qy       ; qa[i]=saturate_s8(qx[i]-qy[i])
EE.VSUBS.S16        qa, qx, qy       ; qa[i]=saturate_s16(qx[i]-qy[i])
EE.VSUBS.S32        qa, qx, qy       ; qa[i]=saturate_s32(qx[i]-qy[i])
; Combined Load+Arithmetic (qu=load result, qa=arithmetic result)
EE.VADDS.S8.LD.INCP  qu, as, qa, qx, qy  ; as+=16
EE.VSUBS.S8.LD.INCP  qu, as, qa, qx, qy
; Combined Store+Arithmetic (store qv, compute qa)
EE.VADDS.S8.ST.INCP  qv, as, qa, qx, qy  ; store qv→aligned(as), as+=16
```

### 4. VECTOR MULTIPLY (lane-wise, SAR right-shift)
```
EE.VMUL.S8          qz, qx, qy       ; qz[i]=(s8*qx[i] * s8*qy[i]) >> SAR[5:0]  for i=0..15
EE.VMUL.S16         qz, qx, qy       ; qz[i]=(s16*qx[i] * s16*qy[i]) >> SAR[5:0] for i=0..7
EE.VMUL.U8          qz, qx, qy       ; unsigned variant
EE.VMUL.U16         qz, qx, qy       ; unsigned variant
; SET SAR BEFORE MULTIPLY:
;   movi aN, shift_amount
;   wsr.sar aN
```

### 5. COMPLEX MULTIPLY (16-bit, 4 selectable 32-bit complex pairs from qy)
```
EE.CMUL.S16         qz, qx, qy, sel4
; sel4 selects which 32-bit segment of qy: 0→[31:0], 1→[63:32], 2→[95:64], 3→[127:96]
; qz[2*i+1:2*i] = complex_mul(qx[2*i+1:2*i], qy[sel4*32+31:sel4*32]) >> SAR
```

### 6. MULTIPLY-ACCUMULATE

**ACCX mode (scalar sum of all lane products):**
```
EE.VMULAS.S8.ACCX    qx, qy    ; ACCX += Σ(qx[i] * qy[i]) for i=0..15, saturated to 40-bit
EE.VMULAS.S16.ACCX   qx, qy    ; ACCX += Σ(qx[i] * qy[i]) for i=0..7
EE.VMULAS.U8.ACCX    qx, qy    ; unsigned
EE.VMULAS.U16.ACCX   qx, qy    ; unsigned
; Read raw result: RUR.ACCX_0 aN (low 32), RUR.ACCX_1 aN (high 8)
EE.SRS.ACCX rd, rs, sel        ; arithmetic right shift ACCX by rs[5:0], write 40-bit
                               ;   shifted result back to ACCX, and write it saturated
                               ;   to a 32-bit signed value min(max(ACCX>>rs,-2^31),2^31-1)
                               ;   into general register rd (sel selects shift source, 0 in esp-dl)
```

**QACC mode (per-lane accumulation, 20-bit or 40-bit per lane):**
```
EE.VMULAS.S8.QACC    qx, qy    ; QACC[i] += qx[i]*qy[i], 20-bit signed saturated per lane
EE.VMULAS.S16.QACC   qx, qy    ; QACC[i] += qx[i]*qy[i], 40-bit signed saturated per lane
EE.VMULAS.U8.QACC    qx, qy    ; unsigned
EE.VMULAS.U16.QACC   qx, qy    ; unsigned
; Extract QACC→QR with right-shift+sat:
EE.SRCMB.S8.QACC     qu, as, 0 ; 16×20-bit→16×8-bit, R-shift SAR, saturate to s8→qu
EE.SRCMB.S16.QACC    qu, as, 0 ; 8×40-bit→8×16-bit, R-shift SAR, saturate to s16→qu
```

### 7. COMPARISON (result = 0xFF on true, 0 on false)
```
EE.VMAX.S8 / S16 / S32   qa, qx, qy    ; qa[i] = max(qx[i], qy[i])
EE.VMIN.S8 / S16 / S32   qa, qx, qy    ; qa[i] = min(qx[i], qy[i])
EE.VCMP.EQ.S8 / S16 / S32  qa, qx, qy  ; qa[i] = (qx[i]==qy[i]) ? 0xFF : 0
EE.VCMP.LT.S8 / S16 / S32  qa, qx, qy  ; qa[i] = (qx[i] < qy[i]) ? 0xFF : 0
EE.VCMP.GT.S8 / S16 / S32  qa, qx, qy  ; qa[i] = (qx[i] > qy[i]) ? 0xFF : 0
; Use ANDQ with mask to implement conditional selection (e.g. ReLU)
```

### 8. BITWISE
```
EE.ANDQ             qa, qx, qy    ; qa = qx & qy  (128-bit)
EE.ORQ              qa, qx, qy    ; qa = qx | qy
EE.XORQ             qa, qx, qy    ; qa = qx ^ qy
EE.NOTQ             qa, qx       ; qa = ~qx
```

### 9. SHIFT
```
EE.VSL.32           qa, qs        ; qa[i] = qs[i] << SAR[5:0]  (4×32-bit logical left)
EE.VSR.32           qa, qs        ; qa[i] = qs[i] >> SAR[5:0]  (4×32-bit arithmetic right)
EE.SRC.Q            qa, qs0, qs1  ; qa = ({qs1,qs0} >> SAR_BYTE*8)[127:0]
; SRC.Q used with LD.128.USAR for unaligned access
```

### 10. DATA MOVEMENT
```
MV.QR               qu, qs        ; qu = qs
EE.ZERO.Q           qa            ; qa = 0
EE.MOVI.32.A        qs, au, sel4  ; au = qs[sel4*32+31:sel4*32]  (extract 32-bit lane)
EE.MOVI.32.Q        qu, as, sel4  ; qu[sel4*32+31:sel4*32] = as  (insert 32-bit lane)
```

### 11. ACTIVATION
```
EE.VRELU.S8         qs, ax, ay    ; qs[i] = (qs[i]<=0) ? (qs[i]*ax[7:0])>>ay[4:0] : qs[i]
EE.VRELU.S16        qs, ax, ay    ; qs[i] = (qs[i]<=0) ? (qs[i]*ax[15:0])>>ay[5:0] : qs[i]
EE.VPRELU.S8        qz, qx, qy, ay  ; qz[i]=(qx[i]<=0) ? (qx[i]*qy[i])>>ay[4:0] : qx[i]
EE.VPRELU.S16       qz, qx, qy, ay  ; qz[i]=(qx[i]<=0) ? (qx[i]*qy[i])>>ay[5:0] : qx[i]
```

### 12. ZIP/UNZIP (interleave/deinterleave)
```
EE.VZIP.8 / 16 / 32   qs0, qs1    ; interleave qs0 and qs1 elements in-place
EE.VUNZIP.8 / 16 / 32  qs0, qs1   ; deinterleave qs0 and qs1 elements in-place
```

### 13. FFT-SPECIFIC
```
EE.FFT.R2BF.S16                qa0, qa1, qx, qy, sel2  ; radix-2 butterfly
EE.FFT.R2BF.S16.ST.INCP        qa0, qx, qy, as, sar4   ; butterfly + store sums
EE.FFT.CMUL.S16.LD.XP          qu, as, ad, qz, qx, qy, sel8  ; complex mul + load
EE.FFT.CMUL.S16.ST.XP          qx, qy, qv, as, ad, sel8, upd4, sar4  ; complex mul + store
EE.FFT.AMS.S16.LD.INCP         qu, as, qz, qz1, qx, qy, qm, sel2  ; FFT AMS + load
EE.FFT.AMS.S16.LD.INCP.UAUP    qu, as, qz, qz1, qx, qy, qm, sel2  ; AMS+unaligned load
EE.FFT.AMS.S16.LD.R32.DECP     qu, as, qz, qz1, qx, qy, qm, sel2  ; AMS+reversed32 load
EE.FFT.AMS.S16.ST.INCP         qv, qz1, at, as, qx, qy, qm, sel2  ; AMS + store
EE.FFT.VST.R32.DECP            qv, as, sar2           ; reversed32 store, as-=16
EE.BITREV                      qa, as                 ; bit-reverse (needs FFT_BIT_WIDTH)
```

---

## C → PIE SIMD CONVERSION PATTERNS

### Pattern A: Element-wise Arithmetic (Add/Sub/Mul)
```c
// C scalar: int16_t out[i] = in0[i] + in1[i] for i in 0..N-1
```
→
```asm
; a2=out, a3=in0, a4=in1, a5=N
    srai    a5, a5, 3            ; N/8 (8 elements per 128-bit load)
    loopgtz a5, 0f
    ee.vld.128.ip   q0, a3, 16   ; load 8×int16_t
    ee.vld.128.ip   q1, a4, 16
    ee.vadds.s16    q2, q0, q1   ; 8 lane adds with saturation
    ee.vst.128.ip   q2, a2, 16
0:
```

### Pattern B: Dot Product (Reduce Sum)
```c
// C scalar: sum += a[i] * b[i]  (int8_t)
```
→
```asm
; Use ACCX for scalar accumulation
    ee.zero.accx
    srai    a4, a4, 4            ; N/16 (16 elements per load)
    loopgtz a4, 0f
    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a3, 16
    ee.vmulas.s8.accx  q0, q1    ; ACCX += Σall_lanes(q0[i]*q1[i])
0:
    rur.accx_0  a2               ; result low 32
    rur.accx_1  a3               ; result high 8
```

### Pattern C: Multiply with Scale (Weighted Sum)
```c
// C: out[i] = (in[i] * weight[i]) >> shift
```
→
```asm
    movi    a6, shift_amount
    wsr.sar a6                   ; SAR controls right-shift
    srai    a5, a5, 3
    loopgtz a5, 0f
    ee.vld.128.ip   q0, a3, 16
    ee.vld.128.ip   q1, a4, 16
    ee.vmul.s16     q2, q0, q1   ; multiply, >>SAR
    ee.vst.128.ip   q2, a2, 16
0:
```

### Pattern D: Multiply-Accumulate per Lane (e.g. Quantized Conv)
```c
// C: accum[i] += input[i] * weight[i]  (per-lane accumulation)
```
→
```asm
; Use QACC for per-lane accumulation
    ee.zero.qacc
    movi    a6, shift_amount
    wsr.sar a6
    srai    a5, a5, 4            ; 16×int8 per load
    loopgtz a5, 0f
    ee.vld.128.ip     q0, a3, 16
    ee.vld.128.ip     q1, a4, 16
    ee.vmulas.s8.qacc q0, q1     ; QACC[i] += q0[i]*q1[i] for i=0..15
0:
    ee.srcmb.s8.qacc  q2, a2, 0  ; extract: 20-bit→8-bit with >>SAR, saturate
```

### Pattern E: ReLU Activation
```c
// C: out[i] = in[i] > 0 ? in[i] : 0  (int16_t)
```
→
```asm
    ee.zero.q   q3               ; zeros vector
    srai    a5, a5, 3
    loopgtz a5, 0f
    ee.vld.128.ip    q0, a3, 16
    ee.vcmp.gt.s16   q1, q0, q3  ; mask = (q0 > 0) → 0xFFFF or 0
    ee.andq          q2, q0, q1  ; apply mask
    ee.vst.128.ip    q2, a2, 16
0:
```
**Alternative using VRELU (faster, fewer ops):**
```asm
; EE.VRELU.S16 qs, ax, ay: if qs[i]<=0 then qs[i]=(qs[i]*ax)>>ay else qs[i]=qs[i]
; For standard ReLU: ax=0 (multiply by 0), ay=0 (shift 0)
    movi    a6, 0
    srai    a5, a5, 3
    loopgtz a5, 0f
    ee.vld.128.ip    q0, a3, 16
    ee.vrelu.s16     q0, a6, a6  ; in-place ReLU: negative→0
    ee.vst.128.ip    q0, a2, 16
0:
```

### Pattern F: PReLU / LeakyReLU
```c
// C: out[i] = in[i] > 0 ? in[i] : in[i] * slope  (int16_t, slope quantized)
```
→
```asm
; Load slope once, broadcast to all lanes
    ee.vldbc.16      q3, a5      ; broadcast slope to 8 lanes
    srai    a6, a6, 3
    loopgtz a6, 0f
    ee.vld.128.ip    q0, a3, 16
    ee.vprelu.s16    q2, q0, q3, a7  ; q2[i]=(q0[i]<=0)?(q0[i]*q3[i])>>a7:q0[i]
    ee.vst.128.ip    q2, a2, 16
0:
```

### Pattern G: Loop with Unaligned Load
```c
// C: process int16_t array at arbitrary (non-16-byte-aligned) address
```
→
```asm
    ee.ld.128.usar.ip  q0, a3, 16    ; first unaligned load, offset→SAR_BYTE
    ee.vld.128.ip      q1, a3, 16    ; second aligned load
    srai    a5, a5, 3
    addi    a5, a5, -1               ; one iteration already consumed
    loopgtz a5, 0f
    ee.src.q           q2, q0, q1    ; align: q2 = ({q1,q0}>>SAR_BYTE*8)[127:0]
    ee.vld.128.ip      q0, a3, 16    ; next aligned chunk
    ; ... process q2 ...
    ; swap q0↔q1 for next iteration:
    ee.src.q           q2, q1, q0    ; use the newly loaded data
    ee.vld.128.ip      q1, a3, 16
    ; ... process q2 ...
0:
```

### Pattern H: Bias Addition (broadcast scalar to vector)
```c
// C: out[i] = in[i] + bias  (same bias for all elements)
```
→
```asm
    ee.vldbc.16      q3, a4      ; broadcast bias to all 8 lanes
    srai    a5, a5, 3
    loopgtz a5, 0f
    ee.vld.128.ip    q0, a3, 16
    ee.vadds.s16     q2, q0, q3
    ee.vst.128.ip    q2, a2, 16
0:
```

### Pattern I: Max Pooling (2x1)
```c
// C: out[i] = max(in[i*2], in[i*2+1])
```
→
```asm
    srai    a5, a5, 4            ; output has N/8 elements, input has N/4
    loopgtz a5, 0f
    ee.vld.128.ip   q0, a3, 16   ; load 8 pairs = 16 elements
    ee.vld.128.ip   q1, a3, 16
    ; Use VUNZIP to separate evens and odds, then VMAX
    ee.vunzip.16    q0, q1       ; q0=evens, q1=odds
    ee.vmax.s16     q2, q0, q1
    ee.vst.128.ip   q2, a2, 16
0:
```

---

## LOOP STRUCTURES

### Label Naming Convention (REQUIRED)
For branch/loop targets, use **local labels**: either a plain number (`0:`, `1:`, referenced as `0f`/`0b`) or a `.L`-prefixed name (`.Lloop`, `.Lremainder`). Do **not** use full descriptive labels like `loop_start:` / `end_label:`.

- Numeric local labels keep tight inner loops compact and avoid name clashes.
- `.L` labels stay local to the file (not emitted into the symbol table) and clearly mark internal jump targets.

```asm
; GOOD — numeric local label
    loopgtz a6, 0f
        ee.vmin.s16.ld.incp q0, a3, q2, q0, q1
        ee.vld.128.ip       q1, a4, 16
        ee.vst.128.ip       q2, a2, 16
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

### Zero-Overhead Loop (preferred)
```asm
; LOOPGTZ: enter loop body if aN > 0; decrements LCOUNT each iteration
    srai    a5, a5, 3            ; convert element count to iteration count
    loopgtz a5, 0f
    ; ... loop body (max ~56 instructions) ...
0:
```

### Manual Loop (when zero-overhead loop constraints violated)
```asm
    srai    a5, a5, 3
    beqz    a5, 1f
.Lloop:
    ; ... loop body ...
    addi    a5, a5, -1
    bnez    a5, .Lloop
1:
```

---

## PIPELINE & SCHEDULING

### Data Dependencies (insert NOPs or reorder to avoid stalls)
- QR load → use: 2 cycle latency (result ready in W stage, used in R stage)
- QR multiply → use: result ready in E stage (same as arithmetic), 0-1 cycle
- SAR write → use: requires ISYNC or at least 1 instruction gap

### Prefer LD+OP fused instructions when available
Instead of separate load + compute:
```asm
; Sub-optimal: 3 instructions
ee.vld.128.ip  q0, a3, 16
ee.vld.128.ip  q1, a4, 16
ee.vadds.s16   q2, q0, q1

; Optimal when loading new data for next iteration:
ee.vadds.s16.ld.incp  q0, a3, q2, q4, q5  ; computes q2 while loading q0 for next iter
```

---

## KEY CONSTRAINTS & GOTCHAS

1. **SAR must be set before VMUL/VMULAS/CMUL** — forgotten SAR causes wrong scaling.
2. **Only 8 QR registers (q0-q7)** — register pressure is real. Plan allocation carefully.
3. **VLD/VST force address alignment** — unaligned addresses silently read/write wrong data.
4. **No 32-bit multiply** — only 8-bit and 16-bit vector multiply. Use multiple 16-bit ops for 32-bit.
5. **Zero-overhead loop limit** — max ~56 instructions in loop body, no nested zero-overhead loops.
6. **ACCX is 40-bit** — saturated to [-2^39, 2^39-1]. Read with RUR ACCX_0 + ACCX_1.
7. **QACC is per-lane** — 16 lanes of 20-bit for S8/U8, 8 lanes of 40-bit for S16/U16.
8. **VADDS/VSUBS saturate** — use when overflow is a concern; VMUL shifts and truncates (no sat).
9. **EE.VRELU modifies input register in-place** — qs is both source and destination.
10. **Don't mix EE.* and Xtensa native load/store on same address** without MEMW/DSYNC if caching matters.

---

## QUICK DECISION TABLE: Which Instruction to Use

| C Operation | Data Type | Best PIE Instruction(s) |
|-------------|-----------|------------------------|
| `c = a + b` (element-wise) | int8/int16/int32 | EE.VADDS.S8/S16/S32 |
| `c = a - b` (element-wise) | int8/int16/int32 | EE.VSUBS.S8/S16/S32 |
| `c = a * b >> shift` | int8/int16 | EE.VMUL.S8/S16 (set SAR first) |
| `sum += a[i] * b[i]` (scalar result) | int8/int16 | EE.VMULAS.S8/S16.ACCX |
| `accum[i] += a[i] * b[i]` (per-lane) | int8/int16 | EE.VMULAS.S8/S16.QACC → EE.SRCMB |
| `c = max(a, b)` | int8/int16/int32 | EE.VMAX.S8/S16/S32 |
| `c = min(a, b)` | int8/int16/int32 | EE.VMIN.S8/S16/S32 |
| `c = a > 0 ? a : 0` (ReLU) | int16 | EE.VRELU.S16 qs, zero, zero |
| `c = a > 0 ? a : a*slope` (PReLU) | int8/int16 | EE.VPRELU.S8/S16 |
| `c = a > b ? 0xFF : 0` (mask) | int8/int16 | EE.VCMP.GT.S8/S16 |
| `c = a & b` | any | EE.ANDQ |
| `c = a | b` | any | EE.ORQ |
| `c = broadcast(scalar)` | int8/int16/int32 | EE.VLDBC.8/16/32 |
| unaligned load | any | EE.LD.128.USAR.IP + EE.SRC.Q |
| `c = a << n` (per 32-bit lane) | int32 | EE.VSL.32 (set SAR first) |
| `c = a >> n` (per 32-bit lane) | int32 | EE.VSR.32 (set SAR first) |
| clear accumulator | - | EE.ZERO.ACCX / EE.ZERO.QACC |
| complex multiply | int16 | EE.CMUL.S16 |
| Radix-2 FFT butterfly | int16 | EE.FFT.R2BF.S16 |

---

## FUNCTION TEMPLATE

```asm
    .align  4
    .text
    .global my_kernel_name
    .type   my_kernel_name, @function
my_kernel_name:
    # a2: output pointer (int16_t*, 16-byte aligned)
    # a3: input0 pointer  (int16_t*, 16-byte aligned)
    # a4: input1 pointer  (int16_t*, 16-byte aligned)
    # a5: element count

    entry   sp, 64              # allocate stack frame

    # Pre-loop setup
    ; set SAR if multiplying
    ; zero accumulators if accumulating
    ; broadcast scalars if needed

    srai    a5, a5, 3           # count → iterations (8 × int16 per 128-bit)
    loopgtz a5, 0f

    ee.vld.128.ip   q0, a3, 16  # load input0
    ee.vld.128.ip   q1, a4, 16  # load input1
    ee.vadds.s16    q2, q0, q1  # compute
    ee.vst.128.ip   q2, a2, 16  # store output
0:
    retw
```

---

## SUMMARY: Conversion Workflow

1. **Identify** the C loop operation and data type (int8/int16/int32).
2. **Check alignment** — ensure buffers are 16-byte aligned; use unaligned load pattern if not.
3. **Select** the right PIE instruction from the decision table above.
4. **Set SAR** if the instruction uses it (multiply, shift).
5. **Zero accumulators** if using ACCX or QACC.
6. **Compute iteration count**: N / (128/bits_per_element). E.g., int16 → N/8.
7. **Write** the zero-overhead loop body using `loopgtz`.
8. **Extract** accumulator results (RUR.*) after the loop if needed.
9. **Watch register pressure** — you only have q0-q7, plan which hold constants, which are temporaries.

---

## REFERENCE DOCUMENTS

When you need to query complete instruction encodings, all instruction variants, detailed bit fields of special registers, exceptions/interrupts, or Xtensa native instructions, refer to the following reference files:

| Document | Path | Content |
|----------|------|---------|
| PIE Instruction Set Complete Reference | [references/pie_instruction_set.md](references/pie_instruction_set.md) | All EE.* instructions with encoding, pseudocode, SAR/QACC/ACCX special register access, unaligned data handling, pipeline hazards |
| Xtensa ISA Reference | [references/xtensa_isa.md](references/xtensa_isa.md) | AR/FR/BR registers, all Xtensa native instructions (load/store/branch/call/loop/float/cache/TLB), exception vectors, special register map |

**Reading Strategy**: This skill file provides quick lookup tables and conversion templates. Please read the corresponding reference files when you need the following information:
- Exact bit encoding of instructions (instruction word encoding)
- Complete operation pseudocode (e.g., detailed formulas for all 16 lanes)
- Full segment access list for special registers (e.g., QACC_H_0~4, QACC_L_0~4)
- Xtensa native instructions (instructions without EE.* prefix)
- Exception handling, TLB, cache management

