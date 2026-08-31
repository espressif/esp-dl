# ESP-DL Operator Cross-Target Benchmark Report

- esp-dl version: **3.3.10**
- Generated at: 2026-08-31 15:24:33
- Baseline target: **esp32** (normalized to 1.00x)
- Compared targets: **esp32s3**, **esp32p4**
- Test cases: 955 (cases common to every target)
- Aggregation: per-operator speedup is the **geometric mean** over its test cases; the range in parentheses is the min-max across those cases

## Per-Operator Speedup (esp32 = 1.00x)

65 operators in total:

| Operator | Cases | esp32 | esp32s3 | esp32p4 |
|---|--:|---:|---:|---:|
| Add | 54 | 1.00x | 3.69x (1.51-89.4) | 9.52x (3.29-148) |
| AveragePool | 20 | 1.00x | 2.47x (1.50-7.04) | 3.94x (3.20-9.19) |
| Clip | 4 | 1.00x | 3.74x (2.82-6.54) | 9.74x (4.31-25.8) |
| Concat | 6 | 1.00x | 1.98x (1.21-3.37) | 8.32x (2.07-75.9) |
| Conv | 78 | 1.00x | 22.9x (3.39-171) | 39.0x (3.66-282) |
| ConvTranspose | 14 | 1.00x | 31.7x (9.28-100) | 64.3x (18.6-213) |
| DepthToSpace | 8 | 1.00x | 1.49x (1.44-1.53) | 3.42x (2.75-3.86) |
| Div | 42 | 1.00x | 1.28x (0.95-2.32) | 2.73x (1.42-4.34) |
| Elu | 2 | 1.00x | 2.31x (2.24-2.38) | 5.11x (4.88-5.35) |
| Equal | 12 | 1.00x | 2.22x (1.91-2.96) | 5.22x (4.37-7.22) |
| Exp | 6 | 1.00x | 3.49x (2.57-5.27) | 11.1x (7.39-23.1) |
| Flatten | 6 | 1.00x | 0.87x (0.75-1.00) | 1.23x (1.12-1.25) |
| GRU | 12 | 1.00x | 3.79x (1.37-8.00) | 8.06x (1.68-42.7) |
| Gather | 6 | 1.00x | 1.59x (1.56-1.61) | 4.11x (3.97-4.17) |
| Gemm | 9 | 1.00x | 2.37x (1.80-3.35) | 5.59x (4.34-7.59) |
| GlobalAveragePool | 12 | 1.00x | 3.02x (1.75-5.52) | 4.96x (3.65-10.6) |
| Greater | 12 | 1.00x | 2.42x (1.99-3.34) | 5.78x (4.66-8.05) |
| GreaterOrEqual | 12 | 1.00x | 2.58x (2.20-3.48) | 6.20x (5.37-8.72) |
| HardSigmoid | 6 | 1.00x | 3.44x (2.52-5.08) | 10.9x (7.42-22.4) |
| HardSwish | 2 | 1.00x | 4.89x (4.72-5.05) | 18.7x (15.3-22.9) |
| LSTM | 12 | 1.00x | 3.91x (1.37-8.31) | 8.18x (1.79-41.8) |
| LayerNormalization | 6 | 1.00x | 1.36x (1.17-1.74) | 3.08x (2.70-3.83) |
| LeakyRelu | 6 | 1.00x | 3.51x (2.73-5.14) | 11.3x (7.65-23.4) |
| Less | 12 | 1.00x | 2.26x (1.95-3.16) | 5.36x (4.51-7.64) |
| LessOrEqual | 12 | 1.00x | 2.32x (2.01-3.06) | 5.73x (4.95-7.86) |
| Log | 6 | 1.00x | 1.57x (1.29-2.27) | 4.05x (3.21-6.24) |
| LogSoftmax | 6 | 1.00x | 1.79x (1.25-3.64) | 4.13x (2.99-7.50) |
| LpNormalization | 10 | 1.00x | 1.39x (1.25-1.68) | 2.37x (1.46-3.33) |
| MatMul | 38 | 1.00x | 3.04x (1.18-8.25) | 7.02x (2.34-22.2) |
| MaxPool | 20 | 1.00x | 36.8x (6.63-154) | 76.7x (9.04-270) |
| Mod | 12 | 1.00x | 1.63x (1.38-2.28) | 3.60x (2.29-5.61) |
| Mul | 56 | 1.00x | 3.20x (1.47-18.1) | 10.8x (3.33-145) |
| Neg | 6 | 1.00x | 2.11x (1.67-4.16) | 4.59x (3.35-10.2) |
| PRelu | 4 | 1.00x | 5.18x (3.41-9.74) | 14.1x (11.0-15.4) |
| Pad | 26 | 1.00x | 2.47x (1.74-4.57) | 6.85x (4.48-28.6) |
| Pow | 10 | 1.00x | 1.66x (1.24-2.11) | 3.57x (2.50-4.94) |
| RMSNormalization | 10 | 1.00x | 2.89x (2.26-4.73) | 5.78x (4.36-8.67) |
| ReduceL1 | 20 | 1.00x | 1.46x (0.90-3.12) | 2.85x (1.62-6.39) |
| ReduceL2 | 20 | 1.00x | 1.31x (0.92-1.95) | 2.87x (1.64-4.91) |
| ReduceLogSum | 20 | 1.00x | 1.88x (1.45-2.75) | 4.08x (1.94-7.43) |
| ReduceLogSumExp | 20 | 1.00x | 1.32x (1.19-1.67) | 3.23x (2.82-4.21) |
| ReduceMax | 20 | 1.00x | 1.47x (0.78-2.86) | 2.70x (1.38-6.07) |
| ReduceMean | 20 | 1.00x | 1.42x (0.87-2.63) | 2.72x (1.09-5.63) |
| ReduceMin | 20 | 1.00x | 1.46x (0.78-2.85) | 2.71x (1.41-6.05) |
| ReduceProd | 10 | 1.00x | 1.20x (1.08-1.48) | 2.26x (1.58-3.57) |
| ReduceSum | 20 | 1.00x | 1.49x (0.86-3.34) | 3.23x (2.03-6.17) |
| ReduceSumSquare | 20 | 1.00x | 1.49x (0.87-3.46) | 3.21x (1.92-6.47) |
| Relu | 6 | 1.00x | 3.83x (2.34-6.42) | 9.55x (4.38-15.3) |
| Requantize | 14 | 1.00x | 7.06x (5.00-10.8) | 34.6x (26.4-42.5) |
| Reshape | 6 | 1.00x | 0.87x (0.75-1.00) | 1.18x (1.00-1.25) |
| Resize | 14 | 1.00x | 8.04x (1.42-58.7) | 17.9x (3.65-132) |
| ReverseSequence | 4 | 1.00x | 1.50x (1.47-1.54) | 3.69x (3.20-4.00) |
| ScatterND | 12 | 1.00x | 1.58x (1.51-1.86) | 3.97x (3.57-5.33) |
| Sigmoid | 6 | 1.00x | 3.53x (2.73-5.23) | 11.3x (7.85-23.0) |
| Slice | 20 | 1.00x | 2.17x (1.62-6.28) | 5.40x (3.71-17.2) |
| Softmax | 6 | 1.00x | 1.95x (1.26-3.96) | 4.49x (2.99-8.00) |
| SpaceToDepth | 8 | 1.00x | 1.66x (1.50-1.98) | 4.09x (3.58-4.84) |
| Split | 4 | 1.00x | 2.65x (1.72-7.02) | 7.03x (4.98-14.5) |
| Sqrt | 6 | 1.00x | 1.36x (1.15-1.89) | 11.0x (8.74-15.9) |
| Squeeze | 6 | 1.00x | 0.87x (0.75-1.00) | 1.10x (0.90-1.25) |
| Sub | 56 | 1.00x | 3.10x (1.50-20.8) | 9.75x (3.32-108) |
| Swish | 2 | 1.00x | 3.06x (2.84-3.30) | 9.81x (9.53-10.1) |
| Tanh | 6 | 1.00x | 3.48x (2.64-5.08) | 11.0x (7.55-22.8) |
| Transpose | 6 | 1.00x | 1.81x (1.43-2.23) | 3.50x (2.54-4.20) |
| Unsqueeze | 8 | 1.00x | 0.88x (0.75-1.00) | 1.15x (1.00-1.25) |

## Per-Operator Speedup Bar Chart (esp32 = 1.00x)

![Per-Operator Speedup Bar Chart (esp32 = 1.00x)](benchmark_speedup_bars.png)

## Total Test Time

Sum of the mean execution time (us) of all common test cases per target:

| Target | Total time (us) |
|--------|----------------:|
| esp32 | 11,483,657.0 |
| esp32s3 | 964,143.0 |
| esp32p4 | 320,531.0 |

## Notes

1. Speedup = baseline (esp32) time / target time; larger is faster. A value below 1.00x means the target is slower than the baseline on that operator.
2. Raw measurements below 2.0 us are clamped in the source data; 32 test cases are affected, so their speedups may be overestimated.
3. Test cases unique to `esp32p4` (352 cases) are excluded from the comparison; see the raw JSON for that target.
