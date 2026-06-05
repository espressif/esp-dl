# Xtensa CPU 指令集参考手册

> 本文档汇总了Xtensa CPU的所有指令、寄存器及使用说明，方便code agent进行指令查询和使用。

---

## 目录

1. [寄存器概述](#1-寄存器概述)
2. [特殊寄存器(Special Registers)](#2-特殊寄存器special-registers)
3. [用户寄存器(User Registers)](#3-用户寄存器user-registers)
4. [指令格式](#4-指令格式)
5. [指令列表](#5-指令列表)
6. [指令详细说明](#6-指令详细说明)
7. [异常和中断](#7-异常和中断)
8. [内存管理](#8-内存管理)

---

## 1. 寄存器概述

### 1.1 地址寄存器 (AR - Address Registers)

| 寄存器 | 名称 | 描述 |
|--------|------|------|
| a0-a15 | 地址寄存器 | 通用32位寄存器，用于地址计算和数据操作 |
| a0 | 返回地址 | 函数调用返回地址 |
| a1 (sp) | 栈指针 | 指向当前栈顶 |
| a2-a7 | 参数/返回值 | 函数参数和返回值传递 |

### 1.2 布尔寄存器 (BR - Boolean Registers)

| 寄存器 | 名称 | 描述 |
|--------|------|------|
| b0-b15 | 布尔寄存器 | 1位布尔值，用于条件判断 |

### 1.3 浮点寄存器 (FR - Floating-Point Registers)

| 寄存器 | 名称 | 描述 |
|--------|------|------|
| f0-f15 | 浮点寄存器 | 32位IEEE754单精度浮点数 |

---

## 2. 特殊寄存器(Special Registers)

### 2.1 程序状态寄存器

| SR# | 名称 | 描述 | 宽度 | 特权 | 复位值 |
|-----|------|------|------|------|--------|
| 0 | LBEG | 循环起始地址 | 32 | 否 | 0 |
| 1 | LEND | 循环结束地址 | 32 | 否 | 0 |
| 2 | LCOUNT | 循环计数器 | 32 | 否 | 0 |
| 3 | SAR | 移位量寄存器 | 6 | 否 | 0 |
| 5 | LITBASE | 字面量基址 | 32 | 否 | 0 |

### 2.2 窗口寄存器

| SR# | 名称 | 描述 | 宽度 | 特权 |
|-----|------|------|------|------|
| 48 | WINDOWBASE | 窗口基址 | 4-8 | 是 |
| 49 | WINDOWSTART | 窗口起始 | 16 | 是 |

### 2.3 程序状态寄存器 (PS)

| SR# | 名称 | 描述 | 宽度 | 特权 |
|-----|------|------|------|------|
| 230 | PS | 程序状态 | 32 | 是 |

**PS寄存器字段:**
- **INTLEVEL** (位0-3): 中断级别
- **EXCM** (位4): 异常模式
- **UM** (位5): 用户模式
- **RING** (位6-7): 特权环
- **OWB** (位8-11): 旧窗口基址
- **CALLINC** (位16-17): 调用增量
- **WOE** (位18): 窗口溢出使能

### 2.4 异常相关寄存器

| SR# | 名称 | 描述 | 宽度 | 特权 |
|-----|------|------|------|------|
| 128 | EXCCAUSE | 异常原因 | 32 | 是 |
| 129 | DEBUGCAUSE | 调试原因 | 32 | 是 |
| 136-142 | EPC1-EPC7 | 异常程序计数器 | 32 | 是 |
| 144-150 | EPS2-EPS7 | 异常程序状态 | 32 | 是 |
| 160 | EXCVADDR | 异常虚拟地址 | 32 | 是 |
| 176-182 | EXCSAVE1-EXCSAVE7 | 异常保存 | 32 | 是 |
| 192 | DEPC | 双异常程序计数器 | 32 | 是 |

### 2.5 中断相关寄存器

| SR# | 名称 | 描述 | 宽度 | 特权 |
|-----|------|------|------|------|
| 226 | INTENABLE | 中断使能 | 32 | 是 |
| 227 | INTENABLE | 中断使能(别名) | 32 | 是 |
| 228 | INTERRUPT | 中断状态 | 32 | 是 |
| 229 | INTCLEAR | 中断清除 | 32 | 是 |

### 2.6 定时器寄存器

| SR# | 名称 | 描述 | 宽度 | 特权 |
|-----|------|------|------|------|
| 234 | CCOUNT | 时钟计数器 | 32 | 是 |
| 240-242 | CCOMPARE0-CCOMPARE2 | 时钟比较器 | 32 | 是 |

### 2.7 MAC16相关寄存器

| SR# | 名称 | 描述 | 宽度 | 特权 |
|-----|------|------|------|------|
| 176 | ACCLO | MAC累加器低32位 | 32 | 否 |
| 177 | ACCHI | MAC累加器高8位 | 8 | 否 |
| 178-181 | M0-M3 | MAC16数据寄存器 | 32 | 否 |
| 182 | MR | MAC16读寄存器 | 32 | 否 |

### 2.8 调试相关寄存器

| SR# | 名称 | 描述 | 宽度 | 特权 |
|-----|------|------|------|------|
| 129 | DEBUGCAUSE | 调试原因 | 32 | 是 |
| 197 | ICOUNT | 指令计数 | 32 | 是 |
| 198 | ICOUNTLEVEL | 指令计数级别 | 4 | 是 |
| 200-201 | IBREAKA0-IBREAKA1 | 指令断点地址 | 32 | 是 |
| 202 | IBREAKENABLE | 指令断点使能 | 2 | 是 |
| 204-205 | DBREAKA0-DBREAKA1 | 数据断点地址 | 32 | 是 |
| 208-209 | DBREAKC0-DBREAKC1 | 数据断点控制 | 32 | 是 |
| 210 | DDR | 调试数据寄存器 | 32 | 是 |

### 2.9 TLB相关寄存器

| SR# | 名称 | 描述 | 宽度 | 特权 |
|-----|------|------|------|------|
| 176 | PTEVADDR | PTE虚拟地址 | 32 | 是 |
| 177 | RASID | 环ASID | 32 | 是 |
| 178 | ITLBCFG | 指令TLB配置 | 32 | 是 |
| 179 | DTLBCFG | 数据TLB配置 | 32 | 是 |

### 2.10 其他寄存器

| SR# | 名称 | 描述 | 宽度 | 特权 |
|-----|------|------|------|------|
| 89 | VECBASE | 向量基址 | 32 | 是 |
| 195 | MISC0-MISC3 | 杂项寄存器 | 32 | 是 |
| 196 | THREADPTR | 线程指针 | 32 | 否 |
| 197 | PRID | 处理器ID | 32 | 是 |
| 204 | MMID | 内存映射ID | 32 | 是 |
| 235 | ATOMCTL | 原子控制 | 32 | 是 |
| 236 | CPENABLE | 协处理器使能 | 32 | 是 |

---

## 3. 用户寄存器(User Registers)

| 编号 | 名称 | 描述 | 宽度 |
|------|------|------|------|
| 0-15 | FCR | 浮点控制寄存器 | 32 |
| 0-15 | FSR | 浮点状态寄存器 | 32 |
| 231 | THREADPTR | 线程指针 | 32 |

---

## 4. 指令格式

### 4.1 RRR格式 (24位)

```
23 20 19 16 15 12 11 8 7 4 3 0
+--+--+--+--+--+--+--+--+--+--+
| op2 | op1 | t | s | r | op0 |
+--+--+--+--+--+--+--+--+--+--+
```

### 4.2 RRI8格式 (24位)

```
23 20 19 16 15 12 11 8 7 0
+--+--+--+--+--+--+--+--+
| op2 | op1 | t | s | imm8 |
+--+--+--+--+--+--+--+--+
```

### 4.3 RRI4格式 (24位)

```
23 20 19 16 15 12 11 8 7 4 3 0
+--+--+--+--+--+--+--+--+--+--+
| op2 | op1 | t | s | r | imm4 |
+--+--+--+--+--+--+--+--+--+--+
```

### 4.4 RSR格式 (24位)

```
23 20 19 16 15 8 7 4 3 0
+--+--+--+--+--+--+--+--+
| op2 | op1 | sr | t | op0 |
+--+--+--+--+--+--+--+--+
```

### 4.5 CALL格式 (24位)

```
23 20 19 18 17 16 15 0
+--+--+--+--+--+--+--+
| op0 | n | offset(18位) |
+--+--+--+--+--+--+--+
```

### 4.6 BRI8格式 (24位)

```
23 20 19 18 17 16 15 12 11 8 7 0
+--+--+--+--+--+--+--+--+--+--+
| op0 | n | m | r | s | imm8 |
+--+--+--+--+--+--+--+--+--+--+
```

### 4.7 BRI12格式 (24位)

```
23 20 19 18 17 16 15 12 11 0
+--+--+--+--+--+--+--+--+--+
| op0 | n | m | s | imm12 |
+--+--+--+--+--+--+--+--+--+
```

### 4.8 RI16格式 (24位)

```
23 20 19 16 15 0
+--+--+--+--+--+
| op0 | t | imm16 |
+--+--+--+--+--+
```

### 4.9 RRRN格式 (16位)

```
15 12 11 8 7 4 3 0
+--+--+--+--+--+--+
| op0 | t | s | r |
+--+--+--+--+--+--+
```

### 4.10 RI7格式 (16位)

```
15 12 11 8 7 4 3 0
+--+--+--+--+--+--+
| op0 | t | s | imm7(4+3) |
+--+--+--+--+--+--+
```

### 4.11 RI6格式 (16位)

```
15 12 11 8 7 6 5 4 3 0
+--+--+--+--+--+--+--+--+
| op0 | t | z | s | imm6(2+4) |
+--+--+--+--+--+--+--+--+
```

---

## 5. 指令列表

### 5.1 算术指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| ADD | 加法 | RRR | Core |
| ADDI | 立即数加法 | RRI8 | Core |
| ADDMI | 立即数加法(16位) | RRI8 | Core |
| ADDX2 | 左移1位加法 | RRR | Core |
| ADDX4 | 左移2位加法 | RRR | Core |
| ADDX8 | 左移3位加法 | RRR | Core |
| SUB | 减法 | RRR | Core |
| SUBX2 | 左移1位减法 | RRR | Core |
| SUBX4 | 左移2位减法 | RRR | Core |
| SUBX8 | 左移3位减法 | RRR | Core |
| NEG | 取负 | RRR | Core |
| ABS | 绝对值 | RRR | Core |

### 5.2 逻辑指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| AND | 按位与 | RRR | Core |
| OR | 按位或 | RRR | Core |
| XOR | 按位异或 | RRR | Core |
| ANDB | 布尔与 | RRR | Boolean |
| ORB | 布尔或 | RRR | Boolean |
| XORB | 布尔异或 | RRR | Boolean |

### 5.3 移位指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| SLL | 逻辑左移 | RRR | Core |
| SRL | 逻辑右移 | RRR | Core |
| SRA | 算术右移 | RRR | Core |
| SLLI | 立即数逻辑左移 | RRR | Core |
| SRLI | 立即数逻辑右移 | RRR | Core |
| SRAI | 立即数算术右移 | RRR | Core |
| SRC | 循环移位 | RRR | Core |
| SSR | 设置右移量 | RRR | Core |
| SSL | 设置左移量 | RRR | Core |
| SSA8L | 设置8位左移对齐 | RRR | Core |
| SSA8B | 设置8位右移对齐 | RRR | Core |
| SSAI | 设置立即数移位对齐 | RRR | Core |

### 5.4 乘法指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| MULL | 32位乘法(低32位) | RRR | 32-bit Multiply |
| MULUH | 无符号32位乘法(高32位) | RRR | 32-bit Multiply |
| MULSH | 有符号32位乘法(高32位) | RRR | 32-bit Multiply |
| MUL16U | 无符号16位乘法 | RRR | 16-bit Multiply |
| MUL16S | 有符号16位乘法 | RRR | 16-bit Multiply |

### 5.5 除法指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| QUOU | 无符号除法 | RRR | 32-bit Divide |
| QUOS | 有符号除法 | RRR | 32-bit Divide |
| REMU | 无符号取余 | RRR | 32-bit Divide |
| REMS | 有符号取余 | RRR | 32-bit Divide |

### 5.6 MAC16指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| MUL.AA.* | 乘法 | RRR | MAC16 |
| MULA.AA.* | 乘加 | RRR | MAC16 |
| MULS.AA.* | 乘减 | RRR | MAC16 |
| MUL.AD.* | 乘法(混合) | RRR | MAC16 |
| MULA.AD.* | 乘加(混合) | RRR | MAC16 |
| MULS.AD.* | 乘减(混合) | RRR | MAC16 |
| MUL.DA.* | 乘法(混合) | RRR | MAC16 |
| MULA.DA.* | 乘加(混合) | RRR | MAC16 |
| MULS.DA.* | 乘减(混合) | RRR | MAC16 |
| MUL.DD.* | 乘法 | RRR | MAC16 |
| MULA.DD.* | 乘加 | RRR | MAC16 |
| MULS.DD.* | 乘减 | RRR | MAC16 |
| UMUL.AA.* | 无符号乘法 | RRR | MAC16 |
| LDINC | 加载并递增 | RRR | MAC16 |
| LDDEC | 加载并递减 | RRR | MAC16 |

**注:** * 可以是 LL, LH, HL, HH (表示高低半字)

### 5.7 加载指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| L8UI | 加载无符号8位 | RRI8 | Core |
| L16UI | 加载无符号16位 | RRI8 | Core |
| L16SI | 加载有符号16位 | RRI8 | Core |
| L32I | 加载32位 | RRI8 | Core |
| L32R | 从字面量池加载 | RI16 | Core |
| L32E | 加载32位(异常处理) | RRI4 | Windowed Register |
| L32AI | 原子加载32位 | RRI8 | Multiprocessor Synchronization |

### 5.8 存储指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| S8I | 存储8位 | RRI8 | Core |
| S16I | 存储16位 | RRI8 | Core |
| S32I | 存储32位 | RRI8 | Core |
| S32E | 存储32位(异常处理) | RRI4 | Windowed Register |
| S32RI | 原子存储32位 | RRI8 | Multiprocessor Synchronization |
| S32C1I | 条件存储32位 | RRI8 | Conditional Store |

### 5.9 浮点指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| ADD.S | 浮点加法 | RRR | Floating-Point |
| SUB.S | 浮点减法 | RRR | Floating-Point |
| MUL.S | 浮点乘法 | RRR | Floating-Point |
| MADD.S | 浮点乘加 | RRR | Floating-Point |
| MSUB.S | 浮点乘减 | RRR | Floating-Point |
| ABS.S | 浮点绝对值 | RRR | Floating-Point |
| NEG.S | 浮点取负 | RRR | Floating-Point |
| MOV.S | 浮点移动 | RRR | Floating-Point |
| RFR | 从AR读取到FR | RRR | Floating-Point |
| WFR | 从FR写入到AR | RRR | Floating-Point |
| FLOAT.S | 整数转浮点 | RRR | Floating-Point |
| UFLOAT.S | 无符号整数转浮点 | RRR | Floating-Point |
| TRUNC.S | 浮点截断转整数 | RRR | Floating-Point |
| UTRUNC.S | 浮点截断转无符号整数 | RRR | Floating-Point |
| ROUND.S | 浮点四舍五入 | RRR | Floating-Point |
| CEIL.S | 浮点向上取整 | RRR | Floating-Point |
| FLOOR.S | 浮点向下取整 | RRR | Floating-Point |

### 5.10 浮点比较指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| OEQ.S | 有序相等比较 | RRR | Floating-Point |
| OLD.S | 有序小于比较 | RRR | Floating-Point |
| OLE.S | 有序小于等于比较 | RRR | Floating-Point |
| UEQ.S | 无序或相等比较 | RRR | Floating-Point |
| ULT.S | 无序或小于比较 | RRR | Floating-Point |
| ULE.S | 无序或小于等于比较 | RRR | Floating-Point |
| UN.S | 无序比较(NaN检查) | RRR | Floating-Point |

### 5.11 分支指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| J | 无条件跳转 | CALL | Core |
| JX | 寄存器跳转 | CALLX | Core |
| BEQ | 相等跳转 | RRI8 | Core |
| BNE | 不等跳转 | RRI8 | Core |
| BLT | 小于跳转(有符号) | RRI8 | Core |
| BGE | 大于等于跳转(有符号) | RRI8 | Core |
| BLTU | 小于跳转(无符号) | RRI8 | Core |
| BGEU | 大于等于跳转(无符号) | RRI8 | Core |
| BEQI | 立即数相等跳转 | BRI8 | Core |
| BNEI | 立即数不等跳转 | BRI8 | Core |
| BLTI | 立即数小于跳转 | BRI8 | Core |
| BGEI | 立即数大于等于跳转 | BRI8 | Core |
| BLTUI | 立即数小于跳转(无符号) | BRI8 | Core |
| BGEUI | 立即数大于等于跳转(无符号) | BRI8 | Core |
| BEQZ | 等于零跳转 | BRI12 | Core |
| BNEZ | 不等于零跳转 | BRI12 | Core |
| BLTZ | 小于零跳转 | BRI12 | Core |
| BGEZ | 大于等于零跳转 | BRI12 | Core |
| BNONE | 无位设置跳转 | RRI8 | Core |
| BALL | 所有位设置跳转 | RRI8 | Core |
| BANY | 任意位设置跳转 | RRI8 | Core |
| BNALL | 非所有位设置跳转 | RRI8 | Core |
| BBC | 位清除跳转 | RRI8 | Core |
| BBS | 位设置跳转 | RRI8 | Core |
| BBCI | 立即数位清除跳转 | RRI8 | Core |
| BBSI | 立即数位设置跳转 | RRI8 | Core |

### 5.12 调用指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| CALL0 | 调用(无窗口) | CALL | Core |
| CALL4 | 调用(4寄存器窗口) | CALL | Windowed Register |
| CALL8 | 调用(8寄存器窗口) | CALL | Windowed Register |
| CALL12 | 调用(12寄存器窗口) | CALL | Windowed Register |
| CALLX0 | 寄存器调用(无窗口) | CALLX | Core |
| CALLX4 | 寄存器调用(4寄存器窗口) | CALLX | Windowed Register |
| CALLX8 | 寄存器调用(8寄存器窗口) | CALLX | Windowed Register |
| CALLX12 | 寄存器调用(12寄存器窗口) | CALLX | Windowed Register |

### 5.13 返回指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| RET | 返回 | CALLX | Core |
| RETW | 窗口返回 | CALLX | Windowed Register |
| RETW.N | 窗口返回(窄指令) | RRRN | Windowed Register + Code Density |

### 5.14 循环指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| LOOP | 零开销循环 | BRI8 | Loop |
| LOOPGTZ | 大于零循环 | BRI8 | Loop |
| LOOPNEZ | 不等于零循环 | BRI8 | Loop |

### 5.15 特殊寄存器访问指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| RSR.* | 读特殊寄存器 | RSR | Core |
| WSR.* | 写特殊寄存器 | RSR | Core |
| XSR.* | 交换特殊寄存器 | RSR | Core |

### 5.16 用户寄存器访问指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| RUR.* | 读用户寄存器 | RSR | Coprocessor |
| WUR.* | 写用户寄存器 | RSR | Coprocessor |

### 5.17 同步指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| ISYNC | 指令同步 | RRR | Core |
| DSYNC | 数据同步 | RRR | Core |
| RSYNC | 寄存器同步 | RRR | Core |
| ESYNC | 执行同步 | RRR | Core |
| MEMW | 内存等待 | RRR | Core |
| EXTW | 扩展等待 | RRR | Core |
| EXCW | 异常等待 | RRR | Core |

### 5.18 缓存指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| IHI | 指令缓存命中无效 | RRI4 | Instruction Cache |
| III | 指令缓存索引无效 | RRI4 | Instruction Cache |
| IPF | 指令缓存预取 | RRI4 | Instruction Cache |
| IPFL | 指令缓存预取并锁定 | RRI4 | Instruction Cache |
| DHI | 数据缓存命中无效 | RRI4 | Data Cache |
| DII | 数据缓存索引无效 | RRI4 | Data Cache |
| DPF | 数据缓存预取 | RRI4 | Data Cache |
| DPFL | 数据缓存预取并锁定 | RRI4 | Data Cache |
| DWB | 数据缓存写回 | RRI4 | Data Cache |
| DWBI | 数据缓存写回并无效 | RRI4 | Data Cache |
| DIWB | 数据缓存索引写回 | RRI4 | Data Cache |
| DIWBI | 数据缓存索引写回并无效 | RRI4 | Data Cache |
| DHU | 数据缓存命中解锁 | RRI4 | Data Cache |
| DIU | 数据缓存索引解锁 | RRI4 | Data Cache |
| IHUL | 指令缓存命中解锁 | RRI4 | Instruction Cache |
| IIUL | 指令缓存索引解锁 | RRI4 | Instruction Cache |
| LICW | 锁定指令缓存路 | RRR | Instruction Cache Index Lock |
| SICW | 解锁指令缓存路 | RRR | Instruction Cache Index Lock |
| LDCW | 锁定数据缓存路 | RRR | Data Cache Index Lock |
| SDCW | 解锁数据缓存路 | RRR | Data Cache Index Lock |

### 5.19 TLB指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| RITLB0 | 读指令TLB条目0 | RRR | Region Translation/MMU |
| RITLB1 | 读指令TLB条目1 | RRR | Region Translation/MMU |
| RDTLB0 | 读数据TLB条目0 | RRR | Region Translation/MMU |
| RDTLB1 | 读数据TLB条目1 | RRR | Region Translation/MMU |
| WITLB | 写指令TLB条目 | RRR | Region Translation/MMU |
| WDTLB | 写数据TLB条目 | RRR | Region Translation/MMU |
| IITLB | 无效指令TLB条目 | RRR | Region Translation/MMU |
| IDTLB | 无效数据TLB条目 | RRR | Region Translation/MMU |
| PITLB | 探测指令TLB | RRR | Region Translation/MMU |
| PDTLB | 探测数据TLB | RRR | Region Translation/MMU |

### 5.20 异常和中断指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| RFE | 从异常返回 | RRR | Exception |
| RFUE | 从用户异常返回 | RRR | Exception (XEA1) |
| RFI | 从中断返回 | RRR | Interrupt |
| RFME | 从内存错误返回 | RRR | Exception |
| RFDE | 从双异常返回 | RRR | Exception |
| RFWO | 从窗口溢出返回 | RRR | Windowed Register |
| RFWU | 从窗口下溢返回 | RRR | Windowed Register |
| SYSCALL | 系统调用 | RRR | Exception |
| WAITI | 等待中断 | RRR | Interrupt |
| RSIL | 读并设置中断级别 | RRR | Interrupt |
| BREAK | 断点 | RRR | Debug |
| BREAK.N | 断点(窄指令) | RRRN | Debug + Code Density |

### 5.21 其他指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| MOV | 移动 | RRR | Core |
| MOVI | 立即数移动 | RRI8 | Core |
| MOV.N | 移动(窄指令) | RRRN | Code Density |
| MOVI.N | 立即数移动(窄指令) | RI7 | Code Density |
| MOVEQZ | 等于零则移动 | RRR | Miscellaneous Operations |
| MOVNEZ | 不等于零则移动 | RRR | Miscellaneous Operations |
| MOVLTZ | 小于零则移动 | RRR | Miscellaneous Operations |
| MOVGEZ | 大于等于零则移动 | RRR | Miscellaneous Operations |
| MOVFP | 浮点假则移动 | RRR | Floating-Point |
| MOVTP | 浮点真则移动 | RRR | Floating-Point |
| ENTRY | 进入函数 | BRI8 | Windowed Register |
| ILL | 非法指令 | CALLX | Core |
| ILL.N | 非法指令(窄指令) | RRRN | Code Density |
| NOP | 空操作 | RRR | Core |
| NOP.N | 空操作(窄指令) | RRRN | Code Density |
| RER | 读外部寄存器 | RRR | Core |
| WER | 写外部寄存器 | RRR | Core |

### 5.22 杂项操作指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| MIN | 最小值 | RRR | Miscellaneous Operations |
| MAX | 最大值 | RRR | Miscellaneous Operations |
| MINU | 无符号最小值 | RRR | Miscellaneous Operations |
| MAXU | 无符号最大值 | RRR | Miscellaneous Operations |
| CLAMPS | 有符号钳制 | RRR | Miscellaneous Operations |
| NSA | 最高有效位前导零个数 | RRR | Miscellaneous Operations |
| NSAU | 无符号最高有效位前导零个数 | RRR | Miscellaneous Operations |
| SEXT | 符号扩展 | RRR | Miscellaneous Operations |

### 5.23 布尔操作指令

| 指令 | 操作 | 格式 | 选项 |
|------|------|------|------|
| ANDB | 布尔与 | RRR | Boolean |
| ORB | 布尔或 | RRR | Boolean |
| XORB | 布尔异或 | RRR | Boolean |
| ANDBC | 布尔与(补码) | RRR | Boolean |
| ORBC | 布尔或(补码) | RRR | Boolean |
| ANY4 | 任意4个布尔为真 | RRR | Boolean |
| ALL4 | 所有4个布尔为真 | RRR | Boolean |
| ANY8 | 任意8个布尔为真 | RRR | Boolean |
| ALL8 | 所有8个布尔为真 | RRR | Boolean |

---

## 6. 指令详细说明

### 6.1 算术指令

#### ADD - 加法

**语法:** `ADD ar, as, at`

**操作:** `AR[r] ← AR[s] + AR[t]`

**描述:** 将地址寄存器as和at的内容相加，结果写入ar。

**异常:** EveryInstR Group

---

#### ADDI - 立即数加法

**语法:** `ADDI ar, as, -128..127`

**操作:** `AR[r] ← AR[s] + sign_extend(imm8)`

**描述:** 将地址寄存器as的内容与8位有符号立即数相加，结果写入ar。

**异常:** EveryInstR Group

---

#### ADDMI - 立即数加法(16位)

**语法:** `ADDMI ar, as, -32768..32512` (256的倍数)

**操作:** `AR[r] ← AR[s] + sign_extend(imm8 << 8)`

**描述:** 将地址寄存器as的内容与16位有符号立即数(256的倍数)相加，结果写入ar。

**异常:** EveryInstR Group

---

#### ADDX2 - 左移1位加法

**语法:** `ADDX2 ar, as, at`

**操作:** `AR[r] ← (AR[s] << 1) + AR[t]`

**描述:** 将地址寄存器as左移1位后与at相加，结果写入ar。

**异常:** EveryInstR Group

---

#### ADDX4 - 左移2位加法

**语法:** `ADDX4 ar, as, at`

**操作:** `AR[r] ← (AR[s] << 2) + AR[t]`

**描述:** 将地址寄存器as左移2位后与at相加，结果写入ar。

**异常:** EveryInstR Group

---

#### ADDX8 - 左移3位加法

**语法:** `ADDX8 ar, as, at`

**操作:** `AR[r] ← (AR[s] << 3) + AR[t]`

**描述:** 将地址寄存器as左移3位后与at相加，结果写入ar。

**异常:** EveryInstR Group

---

#### SUB - 减法

**语法:** `SUB ar, as, at`

**操作:** `AR[r] ← AR[s] - AR[t]`

**描述:** 将地址寄存器as减去at的内容，结果写入ar。

**异常:** EveryInstR Group

---

#### SUBX2 - 左移1位减法

**语法:** `SUBX2 ar, as, at`

**操作:** `AR[r] ← (AR[s] << 1) - AR[t]`

**描述:** 将地址寄存器as左移1位后减去at，结果写入ar。

**异常:** EveryInstR Group

---

#### SUBX4 - 左移2位减法

**语法:** `SUBX4 ar, as, at`

**操作:** `AR[r] ← (AR[s] << 2) - AR[t]`

**描述:** 将地址寄存器as左移2位后减去at，结果写入ar。

**异常:** EveryInstR Group

---

#### SUBX8 - 左移3位减法

**语法:** `SUBX8 ar, as, at`

**操作:** `AR[r] ← (AR[s] << 3) - AR[t]`

**描述:** 将地址寄存器as左移3位后减去at，结果写入ar。

**异常:** EveryInstR Group

---

#### NEG - 取负

**语法:** `NEG ar, at`

**操作:** `AR[r] ← 0 - AR[t]`

**描述:** 将地址寄存器at的内容取负，结果写入ar。

**异常:** EveryInstR Group

---

#### ABS - 绝对值

**语法:** `ABS ar, at`

**操作:** `AR[r] ← |AR[t]|`

**描述:** 将地址寄存器at的内容取绝对值，结果写入ar。

**异常:** EveryInstR Group

---

### 6.2 逻辑指令

#### AND - 按位与

**语法:** `AND ar, as, at`

**操作:** `AR[r] ← AR[s] and AR[t]`

**描述:** 将地址寄存器as和at的内容按位与，结果写入ar。

**异常:** EveryInstR Group

---

#### OR - 按位或

**语法:** `OR ar, as, at`

**操作:** `AR[r] ← AR[s] or AR[t]`

**描述:** 将地址寄存器as和at的内容按位或，结果写入ar。

**异常:** EveryInstR Group

---

#### XOR - 按位异或

**语法:** `XOR ar, as, at`

**操作:** `AR[r] ← AR[s] xor AR[t]`

**描述:** 将地址寄存器as和at的内容按位异或，结果写入ar。

**异常:** EveryInstR Group

---

### 6.3 移位指令

#### SLL - 逻辑左移

**语法:** `SLL ar, as`

**操作:** `AR[r] ← AR[s] << SAR`

**描述:** 将地址寄存器as的内容逻辑左移SAR位，结果写入ar。

**异常:** EveryInstR Group

---

#### SRL - 逻辑右移

**语法:** `SRL ar, at`

**操作:** `AR[r] ← AR[t] >> SAR` (逻辑右移)

**描述:** 将地址寄存器at的内容逻辑右移SAR位，结果写入ar。

**异常:** EveryInstR Group

---

#### SRA - 算术右移

**语法:** `SRA ar, at`

**操作:** `AR[r] ← AR[t] >> SAR` (算术右移)

**描述:** 将地址寄存器at的内容算术右移SAR位，结果写入ar。

**异常:** EveryInstR Group

---

#### SLLI - 立即数逻辑左移

**语法:** `SLLI ar, as, 0..31`

**操作:** `AR[r] ← AR[s] << imm5`

**描述:** 将地址寄存器as的内容逻辑左移imm5位，结果写入ar。

**异常:** EveryInstR Group

---

#### SRLI - 立即数逻辑右移

**语法:** `SRLI ar, as, 0..31`

**操作:** `AR[r] ← AR[s] >> imm5` (逻辑右移)

**描述:** 将地址寄存器as的内容逻辑右移imm5位，结果写入ar。

**异常:** EveryInstR Group

---

#### SRAI - 立即数算术右移

**语法:** `SRAI ar, as, 0..31`

**操作:** `AR[r] ← AR[s] >> imm5` (算术右移)

**描述:** 将地址寄存器as的内容算术右移imm5位，结果写入ar。

**异常:** EveryInstR Group

---

#### SRC - 循环移位

**语法:** `SRC ar, as, at`

**操作:** `AR[r] ← (AR[s] << (32-SAR)) or (AR[t] >> SAR)`

**描述:** 将AR[s]和AR[t]拼接后右移SAR位，结果写入ar。用于多字移位。

**异常:** EveryInstR Group

---

#### SSR - 设置右移量

**语法:** `SSR as`

**操作:** `SAR ← AR[s]4..0`

**描述:** 将地址寄存器as的低5位写入SAR，用于后续的右移操作。

**异常:** EveryInstR Group

---

#### SSL - 设置左移量

**语法:** `SSL as`

**操作:** `SAR ← 32 - AR[s]4..0`

**描述:** 将32减去地址寄存器as的低5位后写入SAR，用于后续的左移操作。

**异常:** EveryInstR Group

---

#### SSA8L - 设置8位左移对齐

**语法:** `SSA8L as`

**操作:** `SAR ← (AR[s] and 3) << 3`

**描述:** 将地址寄存器as的低2位乘以8后写入SAR，用于字节对齐的左移。

**异常:** EveryInstR Group

---

#### SSA8B - 设置8位右移对齐

**语法:** `SSA8B as`

**操作:** `SAR ← 32 - ((AR[s] and 3) << 3)`

**描述:** 将32减去地址寄存器as的低2位乘以8后写入SAR，用于字节对齐的右移。

**异常:** EveryInstR Group

---

#### SSAI - 设置立即数移位对齐

**语法:** `SSAI 0..31`

**操作:** `SAR ← 32 - imm5`

**描述:** 将32减去5位立即数后写入SAR。

**异常:** EveryInstR Group

---

### 6.4 乘法指令

#### MULL - 32位乘法(低32位)

**语法:** `MULL ar, as, at`

**操作:** `AR[r] ← (AR[s] × AR[t])31..0`

**描述:** 将地址寄存器as和at的内容相乘，取低32位结果写入ar。

**异常:** EveryInstR Group

---

#### MULUH - 无符号32位乘法(高32位)

**语法:** `MULUH ar, as, at`

**操作:** `AR[r] ← (AR[s] × AR[t])63..32` (无符号)

**描述:** 将地址寄存器as和at的内容无符号相乘，取高32位结果写入ar。

**异常:** EveryInstR Group

---

#### MULSH - 有符号32位乘法(高32位)

**语法:** `MULSH ar, as, at`

**操作:** `AR[r] ← (AR[s] × AR[t])63..32` (有符号)

**描述:** 将地址寄存器as和at的内容有符号相乘，取高32位结果写入ar。

**异常:** EveryInstR Group

---

#### MUL16U - 无符号16位乘法

**语法:** `MUL16U ar, as, at`

**操作:** `AR[r] ← (AR[s]15..0 × AR[t]15..0)`

**描述:** 将地址寄存器as和at的低16位无符号相乘，结果写入ar。

**异常:** EveryInstR Group

---

#### MUL16S - 有符号16位乘法

**语法:** `MUL16S ar, as, at`

**操作:** `AR[r] ← (AR[s]15..0 × AR[t]15..0)` (有符号)

**描述:** 将地址寄存器as和at的低16位有符号相乘，结果写入ar。

**异常:** EveryInstR Group

---

### 6.5 除法指令

#### QUOU - 无符号除法

**语法:** `QUOU ar, as, at`

**操作:** `AR[r] ← AR[s] ÷ AR[t]` (无符号)

**描述:** 将地址寄存器as的内容除以at的内容(无符号)，商写入ar。

**异常:** 
- EveryInstR Group
- IntegerDivideByZeroCause (除数为零)

---

#### QUOS - 有符号除法

**语法:** `QUOS ar, as, at`

**操作:** `AR[r] ← AR[s] ÷ AR[t]` (有符号)

**描述:** 将地址寄存器as的内容除以at的内容(有符号)，商写入ar。

**异常:** 
- EveryInstR Group
- IntegerDivideByZeroCause (除数为零)

---

#### REMU - 无符号取余

**语法:** `REMU ar, as, at`

**操作:** `AR[r] ← AR[s] mod AR[t]` (无符号)

**描述:** 将地址寄存器as的内容除以at的内容(无符号)，余数写入ar。

**异常:** 
- EveryInstR Group
- IntegerDivideByZeroCause (除数为零)

---

#### REMS - 有符号取余

**语法:** `REMS ar, as, at`

**操作:** `AR[r] ← AR[s] mod AR[t]` (有符号)

**描述:** 将地址寄存器as的内容除以at的内容(有符号)，余数写入ar。

**异常:** 
- EveryInstR Group
- IntegerDivideByZeroCause (除数为零)

---

### 6.6 加载指令

#### L8UI - 加载无符号8位

**语法:** `L8UI ar, as, 0..255`

**操作:** `AR[r] ← zero_extend(Mem[AR[s] + imm8]7..0)`

**描述:** 从内存地址as+imm8处加载8位无符号数据，零扩展到32位后写入ar。

**异常:** 
- EveryInstR Group
- GenExcep(LoadStoreErrorCause)
- GenExcep(LoadStoreAlignmentCause)

---

#### L16UI - 加载无符号16位

**语法:** `L16UI ar, as, 0..510` (2字节对齐)

**操作:** `AR[r] ← zero_extend(Mem[AR[s] + (imm8 << 1)]15..0)`

**描述:** 从内存地址as+(imm8<<1)处加载16位无符号数据，零扩展到32位后写入ar。

**异常:** 
- EveryInstR Group
- GenExcep(LoadStoreErrorCause)
- GenExcep(LoadStoreAlignmentCause)

---

#### L16SI - 加载有符号16位

**语法:** `L16SI ar, as, 0..510` (2字节对齐)

**操作:** `AR[r] ← sign_extend(Mem[AR[s] + (imm8 << 1)]15..0)`

**描述:** 从内存地址as+(imm8<<1)处加载16位有符号数据，符号扩展到32位后写入ar。

**异常:** 
- EveryInstR Group
- GenExcep(LoadStoreErrorCause)
- GenExcep(LoadStoreAlignmentCause)

---

#### L32I - 加载32位

**语法:** `L32I ar, as, 0..1020` (4字节对齐)

**操作:** `AR[r] ← Mem[AR[s] + (imm8 << 2)]`

**描述:** 从内存地址as+(imm8<<2)处加载32位数据，写入ar。

**异常:** 
- EveryInstR Group
- GenExcep(LoadStoreErrorCause)
- GenExcep(LoadStoreAlignmentCause)

---

#### L32R - 从字面量池加载

**语法:** `L32R ar, label`

**操作:** `AR[r] ← Mem[(PC + 3)4..2 || 00 + imm16]`

**描述:** 从PC相对地址的字面量池加载32位常量，写入ar。

**异常:** EveryInstR Group

---

### 6.7 存储指令

#### S8I - 存储8位

**语法:** `S8I ar, as, 0..255`

**操作:** `Mem[AR[s] + imm8]7..0 ← AR[r]7..0`

**描述:** 将地址寄存器ar的低8位存储到内存地址as+imm8处。

**异常:** 
- EveryInstR Group
- GenExcep(LoadStoreErrorCause)

---

#### S16I - 存储16位

**语法:** `S16I ar, as, 0..510` (2字节对齐)

**操作:** `Mem[AR[s] + (imm8 << 1)]15..0 ← AR[r]15..0`

**描述:** 将地址寄存器ar的低16位存储到内存地址as+(imm8<<1)处。

**异常:** 
- EveryInstR Group
- GenExcep(LoadStoreErrorCause)
- GenExcep(LoadStoreAlignmentCause)

---

#### S32I - 存储32位

**语法:** `S32I ar, as, 0..1020` (4字节对齐)

**操作:** `Mem[AR[s] + (imm8 << 2)] ← AR[r]`

**描述:** 将地址寄存器ar的内容存储到内存地址as+(imm8<<2)处。

**异常:** 
- EveryInstR Group
- GenExcep(LoadStoreErrorCause)
- GenExcep(LoadStoreAlignmentCause)

---

### 6.8 分支指令

#### J - 无条件跳转

**语法:** `J target`

**操作:** `PC ← PC + sign_extend(imm18) + 4`

**描述:** 跳转到PC相对地址target处。

**异常:** EveryInst Group

---

#### BEQ - 相等跳转

**语法:** `BEQ as, at, label`

**操作:** `if AR[s] == AR[t] then PC ← PC + sign_extend(imm8) + 4`

**描述:** 如果as等于at，则跳转到label处。

**异常:** EveryInst Group

---

#### BNE - 不等跳转

**语法:** `BNE as, at, label`

**操作:** `if AR[s] != AR[t] then PC ← PC + sign_extend(imm8) + 4`

**描述:** 如果as不等于at，则跳转到label处。

**异常:** EveryInst Group

---

#### BLT - 小于跳转(有符号)

**语法:** `BLT as, at, label`

**操作:** `if AR[s] < AR[t] (有符号) then PC ← PC + sign_extend(imm8) + 4`

**描述:** 如果as小于at(有符号比较)，则跳转到label处。

**异常:** EveryInst Group

---

#### BGE - 大于等于跳转(有符号)

**语法:** `BGE as, at, label`

**操作:** `if AR[s] >= AR[t] (有符号) then PC ← PC + sign_extend(imm8) + 4`

**描述:** 如果as大于等于at(有符号比较)，则跳转到label处。

**异常:** EveryInst Group

---

#### BLTU - 小于跳转(无符号)

**语法:** `BLTU as, at, label`

**操作:** `if AR[s] < AR[t] (无符号) then PC ← PC + sign_extend(imm8) + 4`

**描述:** 如果as小于at(无符号比较)，则跳转到label处。

**异常:** EveryInst Group

---

#### BGEU - 大于等于跳转(无符号)

**语法:** `BGEU as, at, label`

**操作:** `if AR[s] >= AR[t] (无符号) then PC ← PC + sign_extend(imm8) + 4`

**描述:** 如果as大于等于at(无符号比较)，则跳转到label处。

**异常:** EveryInst Group

---

#### BEQZ - 等于零跳转

**语法:** `BEQZ as, label`

**操作:** `if AR[s] == 0 then PC ← PC + sign_extend(imm12) + 4`

**描述:** 如果as等于0，则跳转到label处。

**异常:** EveryInst Group

---

#### BNEZ - 不等于零跳转

**语法:** `BNEZ as, label`

**操作:** `if AR[s] != 0 then PC ← PC + sign_extend(imm12) + 4`

**描述:** 如果as不等于0，则跳转到label处。

**异常:** EveryInst Group

---

### 6.9 调用和返回指令

#### CALL0 - 调用(无窗口)

**语法:** `CALL0 target`

**操作:** 
```
AR[0] ← PC + 3
PC ← PC + sign_extend(imm18) + 4
```

**描述:** 将返回地址保存到a0，跳转到target处。

**异常:** EveryInst Group

---

#### CALL4 - 调用(4寄存器窗口)

**语法:** `CALL4 target`

**操作:** 
```
AR[0] ← (PC + 3) and 0x3FFFFFFF or 0x01
PS.CALLINC ← 1
WindowCheck(1, 1, 1)
PC ← PC + sign_extend(imm18) + 4
```

**描述:** 将返回地址保存到a0(编码窗口大小为4)，检查窗口溢出，跳转到target处。

**异常:** 
- EveryInst Group
- GenExcep(WindowOverflowCause)

---

#### CALL8 - 调用(8寄存器窗口)

**语法:** `CALL8 target`

**操作:** 
```
AR[0] ← (PC + 3) and 0x3FFFFFFF or 0x02
PS.CALLINC ← 2
WindowCheck(2, 2, 2)
PC ← PC + sign_extend(imm18) + 4
```

**描述:** 将返回地址保存到a0(编码窗口大小为8)，检查窗口溢出，跳转到target处。

**异常:** 
- EveryInst Group
- GenExcep(WindowOverflowCause)

---

#### CALL12 - 调用(12寄存器窗口)

**语法:** `CALL12 target`

**操作:** 
```
AR[0] ← (PC + 3) and 0x3FFFFFFF or 0x03
PS.CALLINC ← 3
WindowCheck(3, 3, 3)
PC ← PC + sign_extend(imm18) + 4
```

**描述:** 将返回地址保存到a0(编码窗口大小为12)，检查窗口溢出，跳转到target处。

**异常:** 
- EveryInst Group
- GenExcep(WindowOverflowCause)

---

#### RET - 返回

**语法:** `RET`

**操作:** `PC ← AR[0]`

**描述:** 返回到a0中保存的地址。

**异常:** EveryInst Group

---

#### RETW - 窗口返回

**语法:** `RETW`

**操作:** 
```
if (AR[0]1..0 == 0) then
  PC ← AR[0]
else
  PS.CALLINC ← AR[0]1..0
  WindowUnderflow()
  PC ← AR[0] and 0x3FFFFFFF
endif
```

**描述:** 根据a0中的窗口信息返回，可能需要处理窗口下溢。

**异常:** 
- EveryInst Group
- GenExcep(WindowUnderflowCause)

---

### 6.10 循环指令

#### LOOP - 零开销循环

**语法:** `LOOP as, label`

**操作:** 
```
LBEG ← PC + 4
LEND ← PC + sign_extend(imm8) + 4
LCOUNT ← AR[s]
```

**描述:** 设置零开销循环，循环次数由as指定，循环体从当前指令后开始到label处结束。

**异常:** EveryInst Group

**限制:**
- 循环体最多56条指令
- LEND必须大于LBEG
- 循环体内不能有跳转指令

---

#### LOOPGTZ - 大于零循环

**语法:** `LOOPGTZ as, label`

**操作:** 
```
if AR[s] > 0 then
  LBEG ← PC + 4
  LEND ← PC + sign_extend(imm8) + 4
  LCOUNT ← AR[s]
else
  PC ← PC + sign_extend(imm8) + 4
endif
```

**描述:** 如果as大于0，则设置零开销循环，否则跳过循环体。

**异常:** EveryInst Group

---

#### LOOPNEZ - 不等于零循环

**语法:** `LOOPNEZ as, label`

**操作:** 
```
if AR[s] != 0 then
  LBEG ← PC + 4
  LEND ← PC + sign_extend(imm8) + 4
  LCOUNT ← AR[s]
else
  PC ← PC + sign_extend(imm8) + 4
endif
```

**描述:** 如果as不等于0，则设置零开销循环，否则跳过循环体。

**异常:** EveryInst Group

---

### 6.11 特殊寄存器访问指令

#### RSR.* - 读特殊寄存器

**语法:** `RSR.* at` 或 `RSR at, *`

**操作:** `AR[t] ← SpecialRegister[*]`

**描述:** 读取指定的特殊寄存器内容到地址寄存器at。

**异常:** 
- EveryInstR Group
- GenExcep(IllegalInstructionCause) (如果寄存器未配置)
- GenExcep(PrivilegedCause) (如果sr >= 64且非特权模式)

---

#### WSR.* - 写特殊寄存器

**语法:** `WSR.* at` 或 `WSR at, *`

**操作:** `SpecialRegister[*] ← AR[t]`

**描述:** 将地址寄存器at的内容写入指定的特殊寄存器。

**异常:** 
- EveryInstR Group
- GenExcep(IllegalInstructionCause) (如果寄存器未配置)
- GenExcep(PrivilegedCause) (如果sr >= 64且非特权模式)

---

#### XSR.* - 交换特殊寄存器

**语法:** `XSR.* at` 或 `XSR at, *`

**操作:** 
```
temp ← SpecialRegister[*]
SpecialRegister[*] ← AR[t]
AR[t] ← temp
```

**描述:** 交换地址寄存器at和指定特殊寄存器的内容。

**异常:** 
- EveryInstR Group
- GenExcep(IllegalInstructionCause) (如果寄存器未配置)
- GenExcep(PrivilegedCause) (如果sr >= 64且非特权模式)

---

### 6.12 同步指令

#### ISYNC - 指令同步

**语法:** `ISYNC`

**操作:** 同步指令流

**描述:** 确保之前的指令对后续指令可见，用于自修改代码和特殊寄存器修改后的同步。

**异常:** EveryInst Group

---

#### DSYNC - 数据同步

**语法:** `DSYNC`

**操作:** 同步数据流

**描述:** 确保之前的数据操作对后续数据操作可见，用于内存映射I/O和TLB修改后的同步。

**异常:** EveryInst Group

---

#### RSYNC - 寄存器同步

**语法:** `RSYNC`

**操作:** 同步寄存器访问

**描述:** 确保之前的特殊寄存器写操作对后续特殊寄存器读操作可见。

**异常:** EveryInst Group

---

#### ESYNC - 执行同步

**语法:** `ESYNC`

**操作:** 执行同步

**描述:** 确保之前的特殊寄存器写操作对后续所有操作可见，是最强的同步指令。

**异常:** EveryInst Group

---

#### MEMW - 内存等待

**语法:** `MEMW`

**操作:** 等待内存操作完成

**描述:** 等待所有未完成的内存操作完成，用于多处理器同步。

**异常:** EveryInst Group

---

### 6.13 浮点指令

#### ADD.S - 浮点加法

**语法:** `ADD.S fr, fs, ft`

**操作:** `FR[r] ← FR[s] +s FR[t]`

**描述:** 将浮点寄存器fs和ft的内容相加，结果写入fr。

**异常:** 
- EveryInstR Group
- GenExcep(Coprocessor0Disabled)

---

#### SUB.S - 浮点减法

**语法:** `SUB.S fr, fs, ft`

**操作:** `FR[r] ← FR[s] -s FR[t]`

**描述:** 将浮点寄存器fs减去ft的内容，结果写入fr。

**异常:** 
- EveryInstR Group
- GenExcep(Coprocessor0Disabled)

---

#### MUL.S - 浮点乘法

**语法:** `MUL.S fr, fs, ft`

**操作:** `FR[r] ← FR[s] ×s FR[t]`

**描述:** 将浮点寄存器fs和ft的内容相乘，结果写入fr。

**异常:** 
- EveryInstR Group
- GenExcep(Coprocessor0Disabled)

---

#### RFR - 从AR读取到FR

**语法:** `RFR fr, as`

**操作:** `FR[r] ← AR[s]`

**描述:** 将地址寄存器as的内容移动到浮点寄存器fr，非算术操作。

**异常:** 
- EveryInstR Group
- GenExcep(Coprocessor0Disabled)

---

#### WFR - 从FR写入到AR

**语法:** `WFR ar, fs`

**操作:** `AR[r] ← FR[s]`

**描述:** 将浮点寄存器fs的内容移动到地址寄存器ar，非算术操作。

**异常:** 
- EveryInstR Group
- GenExcep(Coprocessor0Disabled)

---

### 6.14 异常和中断指令

#### RFE - 从异常返回

**语法:** `RFE`

**操作:** 
```
PS ← EPS[1]
PC ← EPC[1]
```

**描述:** 从异常返回，恢复程序状态和程序计数器。

**异常:** 
- EveryInst Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### RFI - 从中断返回

**语法:** `RFI level` (level = 2..7)

**操作:** 
```
PS ← EPS[level]
PC ← EPC[level]
```

**描述:** 从中断返回，恢复程序状态和程序计数器。

**异常:** 
- EveryInst Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### SYSCALL - 系统调用

**语法:** `SYSCALL`

**操作:** 
```
EPC[1] ← PC
EXCCAUSE ← SyscallCause
PC ← UserExceptionVector
```

**描述:** 触发系统调用异常。

**异常:** 
- EveryInst Group
- GenExcep(SyscallCause)

---

#### WAITI - 等待中断

**语法:** `WAITI 0..15`

**操作:** 
```
PS.INTLEVEL ← imm4
// 可能进入低功耗模式等待中断
```

**描述:** 设置中断级别并等待中断，通常用于空闲循环以降低功耗。

**异常:** 
- EveryInst Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### RSIL - 读并设置中断级别

**语法:** `RSIL ar, 0..15`

**操作:** 
```
AR[r] ← PS
PS.INTLEVEL ← imm4
```

**描述:** 读取当前程序状态到ar，并设置新的中断级别。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### BREAK - 断点

**语法:** `BREAK 0..15, 0..15`

**操作:** 触发调试异常

**描述:** 触发调试异常，用于断点调试。

**异常:** 
- EveryInst Group
- GenExcep(DebugCause)

---

### 6.15 缓存指令

#### IHI - 指令缓存命中无效

**语法:** `IHI as, 0..240` (16字节对齐)

**操作:** 使指令缓存中命中的行无效

**描述:** 使指令缓存中与地址as+(imm4<<4)匹配的行无效。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### III - 指令缓存索引无效

**语法:** `III as, 0..240` (16字节对齐)

**操作:** 使指令缓存中指定索引的行无效

**描述:** 使指令缓存中索引为as+(imm4<<4)的行无效。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### DHI - 数据缓存命中无效

**语法:** `DHI as, 0..240` (16字节对齐)

**操作:** 使数据缓存中命中的行无效

**描述:** 使数据缓存中与地址as+(imm4<<4)匹配的行无效。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### DII - 数据缓存索引无效

**语法:** `DII as, 0..240` (16字节对齐)

**操作:** 使数据缓存中指定索引的行无效

**描述:** 使数据缓存中索引为as+(imm4<<4)的行无效。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### DWB - 数据缓存写回

**语法:** `DWB as, 0..240` (16字节对齐)

**操作:** 写回数据缓存中命中的行

**描述:** 将数据缓存中与地址as+(imm4<<4)匹配的脏行写回内存。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### DWBI - 数据缓存写回并无效

**语法:** `DWBI as, 0..240` (16字节对齐)

**操作:** 写回并无效数据缓存中命中的行

**描述:** 将数据缓存中与地址as+(imm4<<4)匹配的脏行写回内存，然后使该行无效。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

### 6.16 TLB指令

#### RITLB0 - 读指令TLB条目0

**语法:** `RITLB0 at, as`

**操作:** `AR[t] ← InstTLB[0][AR[s]]`

**描述:** 读取指令TLB way 0中由as指定的条目到at。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### RITLB1 - 读指令TLB条目1

**语法:** `RITLB1 at, as`

**操作:** `AR[t] ← InstTLB[1][AR[s]]`

**描述:** 读取指令TLB way 1中由as指定的条目到at。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### RDTLB0 - 读数据TLB条目0

**语法:** `RDTLB0 at, as`

**操作:** `AR[t] ← DataTLB[0][AR[s]]`

**描述:** 读取数据TLB way 0中由as指定的条目到at。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### RDTLB1 - 读数据TLB条目1

**语法:** `RDTLB1 at, as`

**操作:** `AR[t] ← DataTLB[1][AR[s]]`

**描述:** 读取数据TLB way 1中由as指定的条目到at。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### WITLB - 写指令TLB条目

**语法:** `WITLB at, as`

**操作:** `InstTLB[way][index] ← AR[t]`

**描述:** 将at的内容写入由as指定的指令TLB条目。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

#### WDTLB - 写数据TLB条目

**语法:** `WDTLB at, as`

**操作:** `DataTLB[way][index] ← AR[t]`

**描述:** 将at的内容写入由as指定的数据TLB条目。

**异常:** 
- EveryInstR Group
- GenExcep(PrivilegedCause) (如果非特权模式)

---

### 6.17 杂项指令

#### MOV - 移动

**语法:** `MOV ar, as`

**操作:** `AR[r] ← AR[s]`

**描述:** 将地址寄存器as的内容复制到ar。

**异常:** EveryInstR Group

---

#### MOVI - 立即数移动

**语法:** `MOVI ar, -2048..2047`

**操作:** `AR[r] ← sign_extend(imm12)`

**描述:** 将12位有符号立即数符号扩展到32位后写入ar。

**异常:** EveryInstR Group

---

#### MOVEQZ - 等于零则移动

**语法:** `MOVEQZ ar, as, at`

**操作:** `if AR[t] == 0 then AR[r] ← AR[s]`

**描述:** 如果at等于0，则将as的内容复制到ar。

**异常:** EveryInstR Group

---

#### MOVNEZ - 不等于零则移动

**语法:** `MOVNEZ ar, as, at`

**操作:** `if AR[t] != 0 then AR[r] ← AR[s]`

**描述:** 如果at不等于0，则将as的内容复制到ar。

**异常:** EveryInstR Group

---

#### MOVLTZ - 小于零则移动

**语法:** `MOVLTZ ar, as, at`

**操作:** `if AR[t] < 0 then AR[r] ← AR[s]`

**描述:** 如果at小于0，则将as的内容复制到ar。

**异常:** EveryInstR Group

---

#### MOVGEZ - 大于等于零则移动

**语法:** `MOVGEZ ar, as, at`

**操作:** `if AR[t] >= 0 then AR[r] ← AR[s]`

**描述:** 如果at大于等于0，则将as的内容复制到ar。

**异常:** EveryInstR Group

---

#### ENTRY - 进入函数

**语法:** `ENTRY as, 0..32767` (8字节对齐)

**操作:** 
```
AR[s - 16] ← AR[0]
AR[s - 12] ← AR[1]
AR[s - 8] ← AR[2]
AR[s - 4] ← AR[3]
AR[1] ← AR[s] - (imm12 << 3)
```

**描述:** 保存寄存器窗口并分配栈空间，用于函数入口。

**异常:** 
- EveryInst Group
- GenExcep(WindowOverflowCause)

---

#### NOP - 空操作

**语法:** `NOP`

**操作:** 无

**描述:** 空操作，不产生任何效果。

**异常:** EveryInst Group

---

### 6.18 杂项操作指令

#### MIN - 最小值

**语法:** `MIN ar, as, at`

**操作:** `AR[r] ← min(AR[s], AR[t])` (有符号)

**描述:** 取as和at中的较小值(有符号比较)，结果写入ar。

**异常:** EveryInstR Group

---

#### MAX - 最大值

**语法:** `MAX ar, as, at`

**操作:** `AR[r] ← max(AR[s], AR[t])` (有符号)

**描述:** 取as和at中的较大值(有符号比较)，结果写入ar。

**异常:** EveryInstR Group

---

#### MINU - 无符号最小值

**语法:** `MINU ar, as, at`

**操作:** `AR[r] ← min(AR[s], AR[t])` (无符号)

**描述:** 取as和at中的较小值(无符号比较)，结果写入ar。

**异常:** EveryInstR Group

---

#### MAXU - 无符号最大值

**语法:** `MAXU ar, as, at`

**操作:** `AR[r] ← max(AR[s], AR[t])` (无符号)

**描述:** 取as和at中的较大值(无符号比较)，结果写入ar。

**异常:** EveryInstR Group

---

#### NSA - 最高有效位前导零个数

**语法:** `NSA ar, as`

**操作:** `AR[r] ← count_leading_zeros(AR[s])`

**描述:** 计算as中最高有效位前导零的个数，结果写入ar。

**异常:** EveryInstR Group

---

#### NSAU - 无符号最高有效位前导零个数

**语法:** `NSAU ar, as`

**操作:** `AR[r] ← count_leading_zeros(AR[s])` (无符号)

**描述:** 计算as中最高有效位前导零的个数(无符号)，结果写入ar。

**异常:** EveryInstR Group

---

#### CLAMPS - 有符号钳制

**语法:** `CLAMPS ar, as, 0..31`

**操作:** `AR[r] ← clamp(AR[s], -(1 << imm4), (1 << imm4) - 1)`

**描述:** 将as钳制在指定范围内，结果写入ar。

**异常:** EveryInstR Group

---

#### SEXT - 符号扩展

**语法:** `SEXT ar, as, 7..22`

**操作:** `AR[r] ← sign_extend(AR[s]imm4..0)`

**描述:** 将as的指定位符号扩展到32位，结果写入ar。

**异常:** EveryInstR Group

---

## 7. 异常和中断

### 7.1 异常向量

| 向量 | 地址 | 描述 |
|------|------|------|
| ResetVector | VECBASE + 0x00 | 复位向量 |
| UserExceptionVector | VECBASE + 0x40 | 用户异常向量 |
| KernelExceptionVector | VECBASE + 0x50 | 内核异常向量 |
| DoubleExceptionVector | VECBASE + 0x70 | 双异常向量 |
| WindowOverflow4 | VECBASE + 0x80 | 4寄存器窗口溢出 |
| WindowUnderflow4 | VECBASE + 0x90 | 4寄存器窗口下溢 |
| WindowOverflow8 | VECBASE + 0xA0 | 8寄存器窗口溢出 |
| WindowUnderflow8 | VECBASE + 0xB0 | 8寄存器窗口下溢 |
| WindowOverflow12 | VECBASE + 0xC0 | 12寄存器窗口溢出 |
| WindowUnderflow12 | VECBASE + 0xD0 | 12寄存器窗口下溢 |
| InterruptVector2..7 | VECBASE + 0x100 + (level-2)*0x10 | 中断向量 |
| MemoryErrorVector | VECBASE + 0x60 | 内存错误向量 |

### 7.2 异常原因 (EXCCAUSE)

| 值 | 名称 | 描述 |
|----|------|------|
| 0 | IllegalInstructionCause | 非法指令 |
| 1 | SyscallCause | 系统调用 |
| 2 | InstructionFetchErrorCause | 指令取指错误 |
| 3 | LoadStoreErrorCause | 加载/存储错误 |
| 4 | Level1InterruptCause | 1级中断 |
| 5 | AllocaCause | 栈分配错误 |
| 6 | IntegerDivideByZeroCause | 整数除以零 |
| 8 | PrivilegedCause | 特权违规 |
| 9 | LoadStoreAlignmentCause | 加载/存储对齐错误 |
| 12 | InstrPIFDataErrorCause | 指令PIF数据错误 |
| 13 | LoadStorePIFDataErrorCause | 加载/存储PIF数据错误 |
| 14 | InstrPIFAddrErrorCause | 指令PIF地址错误 |
| 15 | LoadStorePIFAddrErrorCause | 加载/存储PIF地址错误 |
| 16 | InstTLBMissCause | 指令TLB未命中 |
| 17 | InstTLBMultiHitCause | 指令TLB多重命中 |
| 18 | InstFetchPrivilegeCause | 指令取指特权错误 |
| 19 | InstFetchProhibitedCause | 指令取指禁止 |
| 24 | LoadStoreTLBMissCause | 加载/存储TLB未命中 |
| 25 | LoadStoreTLBMultiHitCause | 加载/存储TLB多重命中 |
| 26 | LoadStorePrivilegeCause | 加载/存储特权错误 |
| 27 | LoadProhibitedCause | 加载禁止 |
| 28 | StoreProhibitedCause | 存储禁止 |
| 29 | Coprocessor0DisabledCause | 协处理器0禁用 |
| 32 | Coprocessor1DisabledCause | 协处理器1禁用 |
| 33 | Coprocessor2DisabledCause | 协处理器2禁用 |
| 34 | Coprocessor3DisabledCause | 协处理器3禁用 |
| 35 | Coprocessor4DisabledCause | 协处理器4禁用 |
| 36 | Coprocessor5DisabledCause | 协处理器5禁用 |
| 37 | Coprocessor6DisabledCause | 协处理器6禁用 |
| 38 | Coprocessor7DisabledCause | 协处理器7禁用 |

---

## 8. 内存管理

### 8.1 TLB条目格式

#### 指令TLB读取格式 (RITLB)

| 位 | 字段 | 描述 |
|----|------|------|
| 31-12 | VPN | 虚拟页号 |
| 11-0 | - | 保留 |

#### 指令TLB写入格式 (WITLB)

| 位 | 字段 | 描述 |
|----|------|------|
| 31-12 | PPN | 物理页号 |
| 11-8 | CA | 缓存属性 |
| 7-4 | SR | 状态/权限 |
| 3-0 | RING | 环号 |

#### 数据TLB读取格式 (RDTLB)

| 位 | 字段 | 描述 |
|----|------|------|
| 31-12 | VPN | 虚拟页号 |
| 11-0 | ASID | 地址空间标识符 |

#### 数据TLB写入格式 (WDTLB)

| 位 | 字段 | 描述 |
|----|------|------|
| 31-12 | PPN | 物理页号 |
| 11-8 | CA | 缓存属性 |
| 7-4 | SR | 状态/权限 |
| 3-0 | RING | 环号 |

### 8.2 缓存属性 (CA)

| 值 | 名称 | 描述 |
|----|------|------|
| 0 | - | 保留 |
| 1 | Cached | 缓存(写回) |
| 2 | Bypass | 旁路(不缓存) |
| 3 | Cached(WriteThrough) | 缓存(写通) |
| 4 | - | 保留 |
| 5-13 | - | 保留 |
| 14 | Isolate | 隔离 |
| 15 | - | 保留 |

### 8.3 内存属性

| 属性 | 描述 |
|------|------|
| Bypass | 不缓存，直接访问内存 |
| Cached | 缓存访问 |
| WriteThrough | 写通缓存 |
| WriteBack | 写回缓存 |
| Isolate | 隔离访问 |
| Guarded | 受保护访问 |

---

## 附录A: 指令操作码汇总

### A.1 主要操作码 (op0)

| op0 | 描述 |
|-----|------|
| 00xx | QRST - 快速操作 |
| 01xx | L32R - 从字面量池加载 |
| 10xx | LSAI - 加载/存储/立即数 |
| 11xx | 保留/密度指令 |

### A.2 QRST子操作码 (op1)

| op1 | 描述 |
|-----|------|
| 00xx | RST0 - 寄存器/移位/测试0 |
| 01xx | RST1 - 寄存器/移位/测试1 |
| 10xx | RST2 - 寄存器/移位/测试2 |
| 11xx | RST3 - 寄存器/移位/测试3 |

### A.3 架构选项缩写

| 缩写 | 选项 |
|------|------|
| C | Instruction Cache or Data Cache |
| D | MAC16 |
| F | Floating-Point Coprocessor |
| I | 32-Bit Integer Multiply/Divide |
| L | Cache Index Lock |
| M | MMU |
| N | Code Density (Narrow instructions) |
| P | Coprocessor |
| S | Speculation |
| U | Miscellaneous Operations |
| W | Windowed Registers |
| X | Exception or Interrupt |
| Y | Multiprocessor Synchronization |

---

**文档版本:** 1.0  
**最后更新:** 2026-04-03  
**来源:** Xtensa Instruction Set Architecture (ISA) Reference Manual
