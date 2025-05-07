#pragma once

#include "dl_fft_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int log2N;
    int cpx_point;
    int16_t *win;
    int16_t *win2;
} fft_s16_info_t;

fft_s16_info_t *fft_info_init(int fft_point);
void fft_info_destroy(fft_s16_info_t *fft_info);
void test_real_fft(int16_t *fft_input, void *fft_handle);
int test_real_fft_hp(int16_t *fft_input, void *fft_handle);
void test_real_ifft(int16_t *fft_input, void *fft_handle);
int test_real_ifft_hp(int16_t *fft_input, void *fft_handle);

void test_radix2_fft_bf_s16(int16_t *, int16_t *, int16_t, int16_t, int16_t);
int test_radix2_fft_bf_s16_hp(int16_t *, int16_t *, int16_t, int16_t, int16_t);
void test_radix2_bit_reverse(int16_t *, int16_t, int16_t);
void test_fftr_s16(int16_t *, int16_t *, int16_t);
void test_ffti_s16(int16_t *, int16_t *, int16_t);
void test_radix2_ifft_bf_s16(int16_t *, int16_t *, int16_t, int16_t, int16_t);
int test_radix2_ifft_bf_s16_hp(int16_t *, int16_t *, int16_t, int16_t, int16_t);
int test_real_fft_hp2(int16_t *fft_input, void *fft_handle);
#ifdef __cplusplus
}
#endif
