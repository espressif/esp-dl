#pragma once

#include "dl_fft_platform.h"
#include "dsp_common.h"
#include "esp_attr.h"
#include "esp_dsp.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int dl_power_of_two(uint32_t n);
float *dl_short_to_float(int16_t *x, int len, int exponent, float *y);

// float fftr2
esp_err_t dl_fft2r_fc32_ansi(float *data, int N, float *w);
esp_err_t dl_bit_rev_lookup_fc32_ansi(float *data, int reverse_size, uint16_t *reverse_tab);
esp_err_t dl_bit_rev2r_fc32(float *data, int N);
float *dl_gen_fftr2_table_f32(int fft_point, uint32_t caps);
esp_err_t dl_bit_rev_fc32_ansi(float *data, int N);

// float fftr4
esp_err_t dl_bit_rev4r_fc32(float *data, int N);
esp_err_t dl_cplx2real_fc32_ansi(float *data, int N, float *table, int table_size);
esp_err_t dl_fft4r_fc32_ansi(float *data, int length, float *table, int table_size);
float *dl_gen_rfft_table_f32(int fft_point, uint32_t caps);

// int16 fft and rfft
esp_err_t dl_fft2r_sc16_ansi(int16_t *data, int N, int16_t *table);
int16_t *dl_gen_fftr2_table_sc16(int fft_point, uint32_t caps);
esp_err_t dl_bit_rev_sc16_ansi(int16_t *data, int N);
esp_err_t dl_cplx2real_sc16_ansi(int16_t *data, int N, int16_t *table);
esp_err_t dl_fft2r_sc16_hp_ansi(int16_t *data, int N, int16_t *table, int *shift);
int16_t *dl_gen_rfft_table_s16(int fft_point, uint32_t caps);
esp_err_t dl_cplx2real_sc16_hp_ansi(int16_t *data, int N, int16_t *table, int *shift);
int16_t dl_get_fft_shift_s16(int16_t *x, int size);

#if CONFIG_IDF_TARGET_ESP32
#define dl_fft2r_fc32 dl_fft2r_fc32_ae32_
#define dl_fft4r_fc32 dl_fft4r_fc32_ae32_
#elif CONFIG_IDF_TARGET_ESP32S3
#define dl_fft2r_fc32 dl_fft2r_fc32_aes3_
#define dl_fft4r_fc32 dl_fft4r_fc32_aes3_
#elif CONFIG_IDF_TARGET_ESP32P4
#define dl_fft2r_fc32 dl_fft2r_fc32_arp4_
#define dl_fft4r_fc32 dl_fft4r_fc32_arp4_
#else
#define dl_fft2r_fc32 dl_fft2r_fc32_ansi
#define dl_fft4r_fc32 dl_fft4r_fc32_ansi
#endif

#define dl_fft2r_sc16 dl_fft2r_sc16_ansi
#define dl_fft2r_sc16_hp dl_fft2r_sc16_hp_ansi

#ifdef __cplusplus
}
#endif
