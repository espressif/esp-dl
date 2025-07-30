#pragma once

#include "rfft_test_1024.h"
#include "rfft_test_128.h"
#include "rfft_test_2048.h"
#include "rfft_test_256.h"
#include "rfft_test_512.h"

#include "fft_test_1024.h"
#include "fft_test_128.h"
#include "fft_test_2048.h"
#include "fft_test_256.h"
#include "fft_test_512.h"

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "unity.h"

#ifdef __cplusplus
extern "C" {
#endif

float cal_snr_item(float x, float y);

float get_snr(const float *x, const float *gt, int size);

float get_rmse(const float *x, const float *gt, int size);

bool check_fft_results(const float *x, const float *gt, int size, float snr_threshold, float rmse_threshold);

bool check_is_same(const float *x, const float *gt, int size, float threshold);
#ifdef __cplusplus
}
#endif
