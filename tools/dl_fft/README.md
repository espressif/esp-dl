# DL_FFT

DL_FFT is a lightweight FFT library supporting both float32 and int16 data types.

The FFT implementation is come from esp-dsp. And we further optimized the int16 FFT to achieving better precision.
For int16 FFT, we recommend to use `dl_fft_s16_hp_run` or `dl_rfft_s16_hp_run` interface. `hp` means "high precision".

## User Guide:
```

#include "dl_fft.h"
#include "dl_rfft.h"

// float fft
float  x[nfft*2];
dl_fft_f32_t *fft_handle = dl_fft_f32_init(nfft, MALLOC_CAP_8BIT);
dl_fft_f32_run(fft_handle, x);
dl_fft_f32_deinit(fft_handle);

// float rfft
float  x[nfft];
dl_fft_f32_t *fft_handle = dl_rfft_f32_init(nfft, MALLOC_CAP_8BIT);
dl_rfft_f32_run(fft_handle, x);
dl_rfft_f32_deinit(fft_handle);

// int16 fft
int16_t  x[nfft*2];
float  y[nfft*2];
int in_exponent = -15;  //  float y = x * 2^in_exponent;
int out_exponent;
dl_fft_s16_t *fft_handle = dl_fft_s16_init(nfft, MALLOC_CAP_8BIT);
dl_fft_s16_hp_run(fft_handle, x, in_exponent, &out_exponent);
dl_short_to_float(x, nfft, out_exponent, y); // convert output from int16_t to float
dl_fft_s16_deinit(fft_handle);

// int16 rfft
int16_t  x[nfft];
float  y[nfft];
int in_exponent = -15;  //  float y = x * 2^in_exponent;
int out_exponent;
dl_fft_s16_t *fft_handle = dl_rfft_s16_init(nfft, MALLOC_CAP_8BIT);
dl_rfft_s16_hp_run(fft_handle, x, in_exponent, &out_exponent);
dl_short_to_float(x, nfft, out_exponent, y); // convert output from int16_t to float
dl_rfft_s16_deinit(fft_handle);


```
Please refer to [dl_fft.h](./dl_fft.h) and [dl_rfft.h](./dl_rfft.h) for more details. 

## FAQ:

#### 1. Why not just use esp-dsp directly? 

Because esp-dsp uses global variables to share FFT tables and other parameters in order to minimize memory consumption. This introduces significant risks for independent components—your FFT results might be corrupted by other programs, and this is something you have little control over.  

#### 2. What does dl_fft do?

1. Provides a re-repackaged interface. Users no longer need to worry about their FFT results being affected by other programs. All FFT tables are allocated and released within the function scope.  
2. Reimplements an int16 FFT. Performs dynamic quantization during butterfly operations, achieving better precision.  
3. Leverages hardware acceleration. Uses built-in FFT instructions on ESP32-S3 and ESP32-P4 to further speed up int16 FFT computations.


## Benchmark

#### ES32-S3 fft benchmark:
| FFT Method         | 128-point (time/snr/rmse) | 256-point (time/snr/rmse) | 512-point (time/snr/rmse) | 1024-point (time/snr/rmse) | 2048-point (time/snr/rmse) |
|--------------------|--------------------------|--------------------------|--------------------------|---------------------------|---------------------------|
| dl_fft_f32         | 72μs / 104.87dB / 0.00032 | 157μs / 107.64dB / 0.00032 | 342μs / 110.55dB / 0.00032 | 739μs / 113.58dB / 0.00032 | 1587μs / 116.91dB / 0.00032 |
| dl_rfft_f32        | 26μs / 101.36dB / 0.00032 | 76μs / 105.29dB / 0.00032 | 124μs / 107.98dB / 0.00032 | 359μs / 110.49dB / 0.00032 | 564μs / 113.90dB / 0.00032 |
| dl_fft_s16         | 141μs / 65.92dB / 0.00172 | 312μs / 61.95dB / 0.00352 | 683μs / 59.24dB / 0.00661 | 1487μs / 56.81dB / 0.01319 | 3216μs / 53.68dB / 0.02622 |
| dl_fft_hp_s16      | 181μs / 76.13dB / 0.00062 | 405μs / 73.60dB / 0.00098 | 869μs / 72.60dB / 0.00155 | 1923μs / 73.05dB / 0.00208 | 4098μs / 69.90dB / 0.00439 |
| dl_rfft_s16        | 74μs / 63.78dB / 0.00140 | 161μs / 60.65dB / 0.00289 | 352μs / 58.20dB / 0.00543 | 764μs / 54.49dB / 0.01128 | 1650μs / 51.85dB / 0.02230 |
| dl_rfft_hp_s16     | 94μs / 75.73dB / 0.00046 | 201μs / 74.20dB / 0.00073 | 446μs / 72.74dB / 0.00110 | 952μs / 72.96dB / 0.00143 | 2087μs / 70.30dB / 0.00295 |
| kiss_rfft_s16      | 98μs / 61.26dB / 0.00180 | 229μs / 60.24dB / 0.00326 | 464μs / 56.97dB / 0.00651 | 1063μs / 53.54dB / 0.01324 | 2153μs / 51.32dB / 0.02560 |


#### ESP3-2C5 fft benchmark:

| FFT Method         | 128-point (time/snr/rmse) | 256-point (time/snr/rmse) | 512-point (time/snr/rmse) | 1024-point (time/snr/rmse) | 2048-point (time/snr/rmse) |
|--------------------|--------------------------|--------------------------|--------------------------|---------------------------|---------------------------|
| dl_fft_f32     | 1227μs / 104.87dB / 0.00032 | 2815μs / 107.64dB / 0.00032 | 6351μs / 110.55dB / 0.00032 | 14143μs / 113.58dB / 0.00032 | 31186μs / 116.91dB / 0.00032 |
| dl_rfft_f32    | 615μs / 101.36dB / 0.00032 | 1538μs / 105.29dB / 0.00032 | 3089μs / 107.98dB / 0.00032 | 7588μs / 110.49dB / 0.00032 | 14890μs / 113.90dB / 0.00032 |
| dl_fft_s16     | 87μs / 65.92dB / 0.00172 | 194μs / 61.95dB / 0.00352 | 429μs / 59.24dB / 0.00661 | 939μs / 56.81dB / 0.01319 |   2039μs/ 53.68dB / 0.02622 |
| dl_fft_hp_s16  | 114μs / 76.13dB / 0.00062 | 258μs / 73.60dB / 0.00098 | 557μs / 72.60dB / 0.00155 | 1237μs / 73.05dB / 0.00208 | 2643μs / 69.90dB / 0.00439 |
| dl_rfft_s16        | 45μs / 63.78dB / 0.00140 | 99μs / 60.65dB / 0.00289 | 218μs / 58.20dB / 0.00543 | 477μs / 54.49dB / 0.01128 | 1035μs / 51.85dB / 0.02230 |
| dl_rfft_hp_s16     | 59μs / 75.73dB / 0.00046 | 126μs / 74.20dB / 0.00073 | 282μs / 72.74dB / 0.00110 | 605μs / 72.96dB / 0.00143 | 1332μs / 70.30dB / 0.00295 |
| kiss_rfft_s16      | 63μs / 61.26dB / 0.00180 | 154μs / 60.24dB / 0.00326 | 296μs / 56.97dB / 0.00651 | 702μs / 53.54dB / 0.01324 | 1366μs / 51.32dB / 0.02560 |

## Reference

- [esp-dsp](https://github.com/espressif/esp-dsp)
- [kissfft](https://github.com/mborgerding/kissfft)
- [fftw](https://github.com/FFTW/fftw3)
