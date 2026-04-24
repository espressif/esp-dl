# Operator Kernelwise Performance on ESP32-S3 and ESP32-P4

* Numbers are kernel execution time in microseconds
* **ESP32-S3** config: ESP32-S3 @ 240MHz, SPI: QPI 80MHz, Data cache: 64KB, cache line: 64B, PSRAM: Octal
* **ESP32-P4** config: ESP32-P4 @ 360MHz, PSRAM: Octal 200MHz, L1 cache: 64KB, L1 cache line: 64B, L2 cache: 256KB, L2 cache line: 128B

## ESP32-S3 Performance Summary

| Function | ANSI C (μs) | SIMD Optimized (μs) | Speedup (×) | Data info |
|----------|-------------|---------------------|-------------|-----------|
| conv2d | 11068 | 422 | **26.23** | input(1,16,20,20), filter(32,16,1,1), pad(0,0), stride(1,1), bias |
| conv2d | 25142 | 482 | **52.16** | input(1,32,19,23), filter(16,32,3,3), pad(1,1), stride(2,2), bias |
| conv2d+ReLU | 75451 | 974 | **77.47** | input(1,16,22,22), filter(32,16,5,5), pad(2,2), stride(2,2), bias |
| conv2d | 137933 | 3447 | **40.02** | input(1,22,19,23), filter(33,22,3,3), pad(1,1), stride(1,1), bias |
| dwconv2d+ReLU | 96073 | 5542 | **17.34** | input(1,32,100,50), depthwise filter(32,1,3,3), pad(0,0), stride(1,1), groups=32, bias |
| dwconv2d+ReLU | 12772 | 918 | **13.91** | input(1,25,20,20), depthwise filter(25,1,7,7), pad(2,2), stride(1,2), groups=25, bias |

## ESP32-P4 Performance Summary

| Function | ANSI C (μs) | SIMD Optimized (μs) | Speedup (×) | Data info |
|----------|-------------|---------------------|-------------|-----------|
| conv2d | 6665 | 289 | **23.06** | input(1,16,20,20), filter(32,16,1,1), pad(0,0), stride(1,1), bias |
| conv2d | 10886 | 421 | **25.86** | input(1,32,19,23), filter(16,32,3,3), pad(1,1), stride(2,2), bias |
| conv2d+ReLU | 36570 | 666 | **54.91** | input(1,16,22,22), filter(32,16,5,5), pad(2,2), stride(2,2), bias |
| conv2d | 58941 | 2517 | **23.42** | input(1,22,19,23), filter(33,22,3,3), pad(1,1), stride(1,1), bias |
| dwconv2d+ReLU | 52862 | 2333 | **22.66** | input(1,32,100,50), depthwise filter(32,1,3,3), pad(0,0), stride(1,1), groups=32, bias |
| dwconv2d+ReLU | 5901 | 660 | **8.94** | input(1,25,20,20), depthwise filter(25,1,7,7), pad(2,2), stride(1,2), groups=25, bias |


## Notes

1. **Speedup** = `ANSI C time / SIMD Optimized time`. Higher values indicate greater SIMD acceleration benefit.
2. **Cross-platform comparison**: ESP32-P4 runs at 360MHz vs ESP32-S3 at 240MHz. Raw times are not directly comparable without accounting for clock speed differences.
3. **conv2d+ReLU** — fused convolution and ReLU activation in a single kernel pass.
4. **dwconv2d** — depthwise convolution where `groups == in_channels == out_channels`, each input channel convolves with its own filter.
5. **Data layout** — The filter shape notation in this table follows PyTorch's NCHW format.(Note: This is not the actual memory layout used by ESP-DL during computation.)
6. **Memory** — Although all tests are performed on PSRAM, the performance of different operator shapes can vary significantly due to cache effects.
