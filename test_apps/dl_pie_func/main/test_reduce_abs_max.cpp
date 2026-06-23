#include "dl_base_reduce.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "unity.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>

static const char *TAG = "TEST DL REDUCE_ABS_MAX";

// --- Scalar reference implementations ---

static int8_t reduce_abs_max_s8_naive(const int8_t *input, int32_t size, int32_t stride)
{
    int8_t result = 0;
    const int8_t *ptr = input;
    for (int32_t i = 0; i < size; i++) {
        int8_t abs_val = *ptr < 0 ? (*ptr == INT8_MIN ? INT8_MAX : (int8_t)(-*ptr)) : *ptr;
        if (abs_val > result)
            result = abs_val;
        ptr += stride;
    }
    return result;
}

static int16_t reduce_abs_max_s16_naive(const int16_t *input, int32_t size, int32_t stride)
{
    int16_t result = 0;
    const int16_t *ptr = input;
    for (int32_t i = 0; i < size; i++) {
        int16_t abs_val = *ptr < 0 ? (*ptr == INT16_MIN ? INT16_MAX : (int16_t)(-*ptr)) : *ptr;
        if (abs_val > result)
            result = abs_val;
        ptr += stride;
    }
    return result;
}

static float reduce_abs_max_f32_naive(const float *input, int32_t size, int32_t stride)
{
    float result = 0.0f;
    const float *ptr = input;
    for (int32_t i = 0; i < size; i++) {
        float abs_val = fabsf(*ptr);
        if (abs_val > result)
            result = abs_val;
        ptr += stride;
    }
    return result;
}

// --- reduce_abs_max int8 precision ---

TEST_CASE("Test dl base reduce_abs_max int8: precision", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_abs_max int8: precision");

    const int sizes[] = {1, 7, 15, 16, 17, 31, 63, 64, 127, 256, 1024, 4093};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int offsets[] = {0, 1, 3, 8};
    const int num_offsets = sizeof(offsets) / sizeof(offsets[0]);

    for (int si = 0; si < num_sizes; si++) {
        int n = sizes[si];
        for (int oi = 0; oi < num_offsets; oi++) {
            int off = offsets[oi];

            int8_t *buf = (int8_t *)heap_caps_aligned_alloc(16, n + off + 16, MALLOC_CAP_DEFAULT);
            TEST_ASSERT_NOT_NULL(buf);
            int8_t *input = buf + off;

            srand(0xabcd + n * 31 + off);
            for (int i = 0; i < n; i++) {
                input[i] = (int8_t)(rand() % 256 - 128);
            }

            int8_t ref = reduce_abs_max_s8_naive(input, n, 1);
            int8_t opt = dl::base::reduce_abs_max(input, n, 1);

            TEST_ASSERT_EQUAL_INT8_MESSAGE(ref, opt, "int8 reduce_abs_max (stride 1) mismatch");

            heap_caps_free(buf);
        }
    }

    // Edge case: INT8_MIN present
    {
        int n = 64;
        int8_t *input = (int8_t *)heap_caps_aligned_alloc(16, n, MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        for (int i = 0; i < n; i++) {
            input[i] = INT8_MIN;
        }
        int8_t ref = reduce_abs_max_s8_naive(input, n, 1);
        int8_t opt = dl::base::reduce_abs_max(input, n, 1);
        TEST_ASSERT_EQUAL_INT8_MESSAGE(ref, opt, "int8 reduce_abs_max INT8_MIN saturation mismatch");
        heap_caps_free(input);
    }

    // Strided path
    {
        int n = 333;
        int stride = 3;
        int8_t *input = (int8_t *)heap_caps_aligned_alloc(16, n * stride, MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0xccdd);
        for (int i = 0; i < n * stride; i++) {
            input[i] = (int8_t)(rand() % 256 - 128);
        }
        int8_t ref = reduce_abs_max_s8_naive(input, n, stride);
        int8_t opt = dl::base::reduce_abs_max(input, n, stride);
        TEST_ASSERT_EQUAL_INT8_MESSAGE(ref, opt, "int8 reduce_abs_max (strided) mismatch");
        heap_caps_free(input);
    }
}

// --- reduce_abs_max int16 precision ---

TEST_CASE("Test dl base reduce_abs_max int16: precision", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_abs_max int16: precision");

    const int sizes[] = {1, 7, 8, 9, 15, 31, 64, 127, 256, 1024, 4093, 8192};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int offsets[] = {0, 1, 2, 4};
    const int num_offsets = sizeof(offsets) / sizeof(offsets[0]);

    for (int si = 0; si < num_sizes; si++) {
        int n = sizes[si];
        for (int oi = 0; oi < num_offsets; oi++) {
            int off = offsets[oi];

            int16_t *buf = (int16_t *)heap_caps_aligned_alloc(16, (n + off + 8) * sizeof(int16_t), MALLOC_CAP_DEFAULT);
            TEST_ASSERT_NOT_NULL(buf);
            int16_t *input = buf + off;

            srand(0xdcba + n * 17 + off);
            for (int i = 0; i < n; i++) {
                input[i] = (int16_t)(rand() % 65536 - 32768);
            }

            int16_t ref = reduce_abs_max_s16_naive(input, n, 1);
            int16_t opt = dl::base::reduce_abs_max(input, n, 1);

            TEST_ASSERT_EQUAL_INT16_MESSAGE(ref, opt, "int16 reduce_abs_max (stride 1) mismatch");

            heap_caps_free(buf);
        }
    }

    // Edge case: INT16_MIN present
    {
        int n = 64;
        int16_t *input = (int16_t *)heap_caps_aligned_alloc(16, n * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        for (int i = 0; i < n; i++) {
            input[i] = INT16_MIN;
        }
        int16_t ref = reduce_abs_max_s16_naive(input, n, 1);
        int16_t opt = dl::base::reduce_abs_max(input, n, 1);
        TEST_ASSERT_EQUAL_INT16_MESSAGE(ref, opt, "int16 reduce_abs_max INT16_MIN saturation mismatch");
        heap_caps_free(input);
    }

    // Strided path
    {
        int n = 257;
        int stride = 4;
        int16_t *input = (int16_t *)heap_caps_aligned_alloc(16, n * stride * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0xeeff);
        for (int i = 0; i < n * stride; i++) {
            input[i] = (int16_t)(rand() % 65536 - 32768);
        }
        int16_t ref = reduce_abs_max_s16_naive(input, n, stride);
        int16_t opt = dl::base::reduce_abs_max(input, n, stride);
        TEST_ASSERT_EQUAL_INT16_MESSAGE(ref, opt, "int16 reduce_abs_max (strided) mismatch");
        heap_caps_free(input);
    }
}

// --- reduce_abs_max float precision ---

TEST_CASE("Test dl base reduce_abs_max float: precision", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_abs_max float: precision");

    const int sizes[] = {1, 7, 15, 16, 17, 31, 63, 64, 127, 256, 1024, 4093};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < num_sizes; si++) {
        int n = sizes[si];

        float *input = (float *)heap_caps_aligned_alloc(16, n * sizeof(float), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);

        srand(0x1122 + n * 31);
        for (int i = 0; i < n; i++) {
            // Generate values in range [-100, 100)
            input[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }

        float ref = reduce_abs_max_f32_naive(input, n, 1);
        float opt = dl::base::reduce_abs_max(input, n, 1);

        TEST_ASSERT_EQUAL_FLOAT_MESSAGE(ref, opt, "float reduce_abs_max (stride 1) mismatch");

        heap_caps_free(input);
    }

    // Strided path
    {
        int n = 257;
        int stride = 4;
        float *input = (float *)heap_caps_aligned_alloc(16, n * stride * sizeof(float), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0x3344);
        for (int i = 0; i < n * stride; i++) {
            input[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }
        float ref = reduce_abs_max_f32_naive(input, n, stride);
        float opt = dl::base::reduce_abs_max(input, n, stride);
        TEST_ASSERT_EQUAL_FLOAT_MESSAGE(ref, opt, "float reduce_abs_max (strided) mismatch");
        heap_caps_free(input);
    }
}

// --- Speed tests ---

TEST_CASE("Test dl base reduce_abs_max int8: speed", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_abs_max int8: speed");

    const int lengths[] = {64, 256, 1024, 4096};
    const int num_lengths = sizeof(lengths) / sizeof(lengths[0]);
    const int iters = 4000;

    for (int li = 0; li < num_lengths; li++) {
        int n = lengths[li];
        int8_t *input = (int8_t *)heap_caps_aligned_alloc(16, n, MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0xabcd + n);
        for (int i = 0; i < n; i++) {
            input[i] = (int8_t)(rand() % 256 - 128);
        }

        TEST_ASSERT_EQUAL_INT8_MESSAGE(reduce_abs_max_s8_naive(input, n, 1),
                                       dl::base::reduce_abs_max(input, n, 1),
                                       "int8 reduce_abs_max speed sanity");

        volatile int8_t sink = 0;

        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += dl::base::reduce_abs_max(input, n, 1);
        }
        int64_t t_opt = esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += reduce_abs_max_s8_naive(input, n, 1);
        }
        int64_t t_naive = esp_timer_get_time() - t0;

        float speedup = t_opt > 0 ? (float)t_naive / (float)t_opt : 0.0f;
        ESP_LOGI(TAG,
                 "abs_max_s8 n=%6d | %d iters | esp-dl=%8lld us | naive=%8lld us | speedup=%.2fx",
                 n,
                 iters,
                 (long long)t_opt,
                 (long long)t_naive,
                 (double)speedup);

        (void)sink;
        heap_caps_free(input);
    }
}

TEST_CASE("Test dl base reduce_abs_max int16: speed", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_abs_max int16: speed");

    const int lengths[] = {64, 256, 1024, 4096};
    const int num_lengths = sizeof(lengths) / sizeof(lengths[0]);
    const int iters = 4000;

    for (int li = 0; li < num_lengths; li++) {
        int n = lengths[li];
        int16_t *input = (int16_t *)heap_caps_aligned_alloc(16, n * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0xdcba + n);
        for (int i = 0; i < n; i++) {
            input[i] = (int16_t)(rand() % 65536 - 32768);
        }

        TEST_ASSERT_EQUAL_INT16_MESSAGE(reduce_abs_max_s16_naive(input, n, 1),
                                        dl::base::reduce_abs_max(input, n, 1),
                                        "int16 reduce_abs_max speed sanity");

        volatile int16_t sink = 0;

        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += dl::base::reduce_abs_max(input, n, 1);
        }
        int64_t t_opt = esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += reduce_abs_max_s16_naive(input, n, 1);
        }
        int64_t t_naive = esp_timer_get_time() - t0;

        float speedup = t_opt > 0 ? (float)t_naive / (float)t_opt : 0.0f;
        ESP_LOGI(TAG,
                 "abs_max_s16 n=%6d | %d iters | esp-dl=%8lld us | naive=%8lld us | speedup=%.2fx",
                 n,
                 iters,
                 (long long)t_opt,
                 (long long)t_naive,
                 (double)speedup);

        (void)sink;
        heap_caps_free(input);
    }
}

TEST_CASE("Test dl base reduce_abs_max float: speed", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_abs_max float: speed");

    const int lengths[] = {64, 256, 1024, 4096};
    const int num_lengths = sizeof(lengths) / sizeof(lengths[0]);
    const int iters = 4000;

    for (int li = 0; li < num_lengths; li++) {
        int n = lengths[li];
        float *input = (float *)heap_caps_aligned_alloc(16, n * sizeof(float), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0x1122 + n);
        for (int i = 0; i < n; i++) {
            input[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }

        TEST_ASSERT_EQUAL_FLOAT_MESSAGE(reduce_abs_max_f32_naive(input, n, 1),
                                        dl::base::reduce_abs_max(input, n, 1),
                                        "float reduce_abs_max speed sanity");

        volatile float sink = 0.0f;

        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += dl::base::reduce_abs_max(input, n, 1);
        }
        int64_t t_opt = esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += reduce_abs_max_f32_naive(input, n, 1);
        }
        int64_t t_naive = esp_timer_get_time() - t0;

        float speedup = t_opt > 0 ? (float)t_naive / (float)t_opt : 0.0f;
        ESP_LOGI(TAG,
                 "abs_max_f32 n=%6d | %d iters | esp-dl=%8lld us | naive=%8lld us | speedup=%.2fx",
                 n,
                 iters,
                 (long long)t_opt,
                 (long long)t_naive,
                 (double)speedup);

        (void)sink;
        heap_caps_free(input);
    }
}
