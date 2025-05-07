
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"
#pragma GCC diagnostic ignored "-Wint-conversion"
#pragma GCC diagnostic ignored "-Wreturn-mismatch"
#include "dl_s16.h"
#include "esp_system.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PI (3.14159265358979323846)

int get_log2N(int n)
{
    int i = 0;
    while (n > 1) {
        i++;
        n >>= 1;
    }
    return i;
}

void direct_radix2_bit_reverse(int16_t *data, int fft_point, int log2N)
{
    int i, j;
    for (i = 0; i < fft_point; i++) {
        int num = log2N;
        int bit, reversed_idx = 0;
        uint32_t temp;
        uint32_t *in_data = (uint32_t *)data;

        j = i;
        while (num > 0) {
            bit = j & 1;
            reversed_idx = (reversed_idx << 1) + bit;
            j = j >> 1;
            num--;
        }

        if (reversed_idx > i) {
            temp = in_data[i];
            in_data[i] = in_data[reversed_idx];
            in_data[reversed_idx] = temp;
        }
    }
}

fft_s16_info_t *fft_info_init(int fft_point)
{
    fft_s16_info_t *fft_info = (fft_s16_info_t *)malloc(sizeof(fft_s16_info_t));
    int cpx_point = fft_point >> 1;
    fft_info->cpx_point = cpx_point;

    int log2N = get_log2N(cpx_point);
    fft_info->log2N = log2N;

    fft_info->win = (int16_t *)heap_caps_aligned_alloc(16, fft_point * sizeof(int16_t), MALLOC_CAP_8BIT);
    fft_info->win2 = (int16_t *)heap_caps_aligned_alloc(16, cpx_point * sizeof(int16_t), MALLOC_CAP_8BIT);

    int count = cpx_point;
    int i, j;
    int16_t *win_ptr = fft_info->win;
    for (i = 1; i <= log2N; i++) {
        count = count >> 1;
        for (j = 0; j < count; j++) {
            double angle = PI * pow(2, i) * j / (double)cpx_point;
            *win_ptr++ = (int16_t)((double)cos(angle) * pow(2, 14));
            *win_ptr++ = (int16_t)((double)sin(angle) * pow(2, 14));
        }
    }

    for (i = 0; i < cpx_point / 2; i++) {
        double phase = -PI * ((double)(i + 1) / cpx_point + 0.5);
        fft_info->win2[i * 2 + 0] = (int16_t)((double)cos(phase) * pow(2, 14));
        fft_info->win2[i * 2 + 1] = (int16_t)((double)sin(phase) * pow(2, 14));
    }
    return (void *)fft_info;
}

// void* fft_info_init_psram(int fft_point){
// 	fft_s16_info_t *fft_info = heap_caps_calloc(sizeof(fft_s16_info_t), 1, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
// 	int cpx_point = fft_point >> 1;
// 	fft_info->cpx_point = cpx_point;

// 	int log2N = get_log2N(cpx_point);
// 	fft_info->log2N = log2N;

// 	fft_info->win = dl_lib_calloc_psram(1, sizeof(int16_t) * fft_point, 16);
// 	fft_info->win2 = dl_lib_calloc_psram(1, sizeof(int16_t) * cpx_point, 16);

// 	int count = cpx_point;
// 	int i, j;
// 	int16_t* win_ptr = fft_info->win;
// 	for(i = 1; i<=log2N; i++){
// 		count = count >> 1;
// 		for(j = 0; j < count; j ++)
// 		{
// 			double angle = PI * pow(2,i) * j / (double)cpx_point;
// 			*win_ptr++ = (int16_t)((double)cos(angle) * pow(2 , 14));
// 			*win_ptr++ = (int16_t)((double)sin(angle) * pow(2 , 14));
// 		}
// 	}

// 	for (i = 0; i < cpx_point / 2; i++) {
// 		double phase = -PI * ((double)(i + 1) / cpx_point + 0.5);
// 		fft_info->win2[i * 2 + 0] = (int16_t) ((double)cos(phase) * pow(2,14));
// 		fft_info->win2[i * 2 + 1] = (int16_t) ((double)sin(phase) * pow(2,14));
// 	}
// 	return (void*) fft_info;
// }

void fft_info_destroy(fft_s16_info_t *fft_info)
{
    heap_caps_free(fft_info->win);
    heap_caps_free(fft_info->win2);
    heap_caps_free(fft_info);
    return;
}

void test_real_fft(int16_t *fft_input, void *fft_handle)
{
    fft_s16_info_t *fft_info = (fft_s16_info_t *)fft_handle;

    // printf("fft_info->cpx_point = %d\n", fft_info->cpx_point);
    // printf("fft_info->log2N = %d\n", fft_info->log2N);

    test_radix2_fft_bf_s16(fft_input, fft_info->win, 15, fft_info->log2N, fft_info->cpx_point); // pase
    direct_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    // if(fft_info->cpx_point > 8 && fft_info->cpx_point < 2048)
    // 	test_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    // else
    // 	direct_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    test_fftr_s16(fft_input, fft_info->win2, fft_info->cpx_point);
}

int test_real_fft_hp(int16_t *fft_input, void *fft_handle)
{
    fft_s16_info_t *fft_info = (fft_s16_info_t *)fft_handle;
    int ls = test_radix2_fft_bf_s16_hp(fft_input, fft_info->win, 15, fft_info->log2N, fft_info->cpx_point);

    // if(fft_info->cpx_point > 8 && fft_info->cpx_point < 2048)
    // 	test_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    // else
    // 	direct_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    direct_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);

    test_fftr_s16(fft_input, fft_info->win2, fft_info->cpx_point);

    return ls;
}

int test_real_fft_hp2(int16_t *fft_input, void *fft_handle)
{
    fft_s16_info_t *fft_info = (fft_s16_info_t *)fft_handle;
    int ls = test_radix2_fft_bf_s16_hp(fft_input, fft_info->win, 15, fft_info->log2N, fft_info->cpx_point);

    // if(fft_info->cpx_point > 8 && fft_info->cpx_point < 2048)
    // 	test_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    // else
    // 	direct_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    direct_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);

    return ls;
}

void test_real_ifft(int16_t *fft_input, void *fft_handle)
{
    fft_s16_info_t *fft_info = (fft_s16_info_t *)fft_handle;
    test_ffti_s16(fft_input, fft_info->win2, fft_info->cpx_point);
    test_radix2_ifft_bf_s16(fft_input, fft_info->win, 15, fft_info->log2N, fft_info->cpx_point);
    // if(fft_info->cpx_point > 8 && fft_info->cpx_point < 2048)
    // 	test_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    // else
    // 	direct_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    direct_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
}

int test_real_ifft_hp(int16_t *fft_input, void *fft_handle)
{
    fft_s16_info_t *fft_info = (fft_s16_info_t *)fft_handle;
    test_ffti_s16(fft_input, fft_info->win2, fft_info->cpx_point);
    int ls = test_radix2_ifft_bf_s16_hp(fft_input, fft_info->win, 15, fft_info->log2N, fft_info->cpx_point);
    ls = fft_info->log2N + 1 - ls;
    direct_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    // if(fft_info->cpx_point > 8 && fft_info->cpx_point < 2048)
    // 	test_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    // else
    // 	direct_radix2_bit_reverse(fft_input, fft_info->cpx_point, fft_info->log2N);
    return ls;
}

#pragma GCC diagnostic pop
