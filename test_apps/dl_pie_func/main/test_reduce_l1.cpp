#include "dl_base_reduce.hpp"
#include "dl_tool.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "unity.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>

static const char *TAG = "TEST DL REDUCE_L1";

// --- Scalar reference implementations for reduce_l1 ---

static int32_t reduce_l1_s8_naive(const int8_t *input, int32_t size, int32_t stride)
{
    int32_t sum = 0;
    const int8_t *ptr = input;
    for (int32_t i = 0; i < size; i++) {
        int32_t v = *ptr;
        // True absolute value: |INT8_MIN| = 128 is accumulated exactly (no saturation).
        sum += v < 0 ? -v : v;
        ptr += stride;
    }
    return sum;
}

static int64_t reduce_l1_s16_naive(const int16_t *input, int32_t size, int32_t stride)
{
    int64_t sum = 0;
    const int16_t *ptr = input;
    for (int32_t i = 0; i < size; i++) {
        int64_t v = *ptr;
        // True absolute value: |INT16_MIN| = 32768 is accumulated exactly (no saturation).
        sum += v < 0 ? -v : v;
        ptr += stride;
    }
    return sum;
}

static float reduce_l1_f32_naive(const float *input, int32_t size, int32_t stride)
{
    float sum = 0.0f;
    const float *ptr = input;
    for (int32_t i = 0; i < size; i++) {
        sum += fabsf(*ptr);
        ptr += stride;
    }
    return sum;
}

static void assert_equal_int64(int64_t expected, int64_t actual, const char *msg)
{
    if (expected != actual) {
        ESP_LOGE(TAG, "%s: expected=%lld actual=%lld", msg, (long long)expected, (long long)actual);
    }
    TEST_ASSERT_TRUE_MESSAGE(expected == actual, msg);
}

static void assert_float_near(float ref, float opt, const char *msg)
{
    float denom = fabsf(ref) > fabsf(opt) ? fabsf(ref) : fabsf(opt);
    if (denom < 1e-6f)
        denom = 1e-6f;
    float rel_err = fabsf(ref - opt) / denom;
    if (rel_err >= 1e-3f) {
        ESP_LOGE(TAG,
                 "%s: ref=%.6f opt=%.6f | rel_err=%.6e (threshold 1e-3)",
                 msg,
                 (double)ref,
                 (double)opt,
                 (double)rel_err);
    }
    TEST_ASSERT_TRUE_MESSAGE(rel_err < 1e-3f, msg);
}

// ==========================================================================
// reduce_l1 precision tests
// ==========================================================================

TEST_CASE("Test dl base reduce_l1 int8: precision", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_l1 int8: precision");

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

            srand(0x9abc + n * 31 + off);
            for (int i = 0; i < n; i++) {
                input[i] = (int8_t)(rand() % 256 - 128);
            }

            int32_t ref = reduce_l1_s8_naive(input, n, 1);
            int32_t opt = dl::base::reduce_l1(input, n, 1);

            TEST_ASSERT_EQUAL_INT32_MESSAGE(ref, opt, "int8 reduce_l1 (stride 1) mismatch");

            heap_caps_free(buf);
        }
    }

    // Strided path.
    {
        int n = 333;
        int stride = 3;
        int8_t *input = (int8_t *)heap_caps_aligned_alloc(16, n * stride, MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0x1a2b);
        for (int i = 0; i < n * stride; i++) {
            input[i] = (int8_t)(rand() % 256 - 128);
        }
        int32_t ref = reduce_l1_s8_naive(input, n, stride);
        int32_t opt = dl::base::reduce_l1(input, n, stride);
        TEST_ASSERT_EQUAL_INT32_MESSAGE(ref, opt, "int8 reduce_l1 (strided) mismatch");
        heap_caps_free(input);
    }

    // Edge: INT8_MIN (|−128| = 128 must be accumulated exactly, not saturated to 127).
    {
        int n = 64;
        int8_t *input = (int8_t *)heap_caps_aligned_alloc(16, n, MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        for (int i = 0; i < n; i++) {
            input[i] = INT8_MIN;
        }
        int32_t ref = reduce_l1_s8_naive(input, n, 1);
        int32_t opt = dl::base::reduce_l1(input, n, 1);
        ESP_LOGI(TAG, "INT8_MIN stress n=%d | ref=%ld | opt=%ld", n, (long)ref, (long)opt);
        TEST_ASSERT_EQUAL_INT32_MESSAGE(ref, opt, "int8 reduce_l1 INT8_MIN stress mismatch");
        heap_caps_free(input);
    }
}

TEST_CASE("Test dl base reduce_l1 int16: precision", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_l1 int16: precision");

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

            srand(0xdef0 + n * 17 + off);
            for (int i = 0; i < n; i++) {
                int v = rand() % 65536 - 32768;
                input[i] = (int16_t)v;
            }

            int64_t ref = reduce_l1_s16_naive(input, n, 1);
            int64_t opt = dl::base::reduce_l1(input, n, 1);

            assert_equal_int64(ref, opt, "int16 reduce_l1 (stride 1) mismatch");

            heap_caps_free(buf);
        }
    }

    // Strided path.
    {
        int n = 257;
        int stride = 4;
        int16_t *input = (int16_t *)heap_caps_aligned_alloc(16, n * stride * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0x3c4d);
        for (int i = 0; i < n * stride; i++) {
            input[i] = (int16_t)(rand() % 65536 - 32768);
        }
        int64_t ref = reduce_l1_s16_naive(input, n, stride);
        int64_t opt = dl::base::reduce_l1(input, n, stride);
        assert_equal_int64(ref, opt, "int16 reduce_l1 (strided) mismatch");
        heap_caps_free(input);
    }

    // Edge: INT16_MIN (|−32768| = 32768 must be accumulated exactly, not saturated to 32767).
    {
        int n = 64;
        int16_t *input = (int16_t *)heap_caps_aligned_alloc(16, n * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        for (int i = 0; i < n; i++) {
            input[i] = INT16_MIN;
        }
        int64_t ref = reduce_l1_s16_naive(input, n, 1);
        int64_t opt = dl::base::reduce_l1(input, n, 1);
        ESP_LOGI(TAG, "INT16_MIN stress n=%d | ref=%lld | opt=%lld", n, (long long)ref, (long long)opt);
        assert_equal_int64(ref, opt, "int16 reduce_l1 INT16_MIN stress mismatch");
        heap_caps_free(input);
    }
}

TEST_CASE("Test dl base reduce_l1 float: precision", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_l1 float: precision");

    const int sizes[] = {1, 7, 15, 16, 17, 31, 63, 64, 127, 256, 1024, 4093};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < num_sizes; si++) {
        int n = sizes[si];

        float *input = (float *)heap_caps_aligned_alloc(16, n * sizeof(float), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);

        srand(0xfe01 + n * 41);
        for (int i = 0; i < n; i++) {
            input[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }

        float ref = reduce_l1_f32_naive(input, n, 1);
        float opt = dl::base::reduce_l1(input, n, 1);

        assert_float_near(ref, opt, "float reduce_l1 (stride 1) mismatch");

        heap_caps_free(input);
    }

    // Strided path.
    {
        int n = 257;
        int stride = 4;
        float *input = (float *)heap_caps_aligned_alloc(16, n * stride * sizeof(float), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0xcafe);
        for (int i = 0; i < n * stride; i++) {
            input[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }
        float ref = reduce_l1_f32_naive(input, n, stride);
        float opt = dl::base::reduce_l1(input, n, stride);
        assert_float_near(ref, opt, "float reduce_l1 (strided) mismatch");
        heap_caps_free(input);
    }
}

// ==========================================================================
// reduce_l1 speed tests
// ==========================================================================

TEST_CASE("Test dl base reduce_l1 int8: speed", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_l1 int8: speed");

    const int lengths[] = {64, 256, 1024, 4096};
    const int num_lengths = sizeof(lengths) / sizeof(lengths[0]);
    const int iters = 4000;

    for (int li = 0; li < num_lengths; li++) {
        int n = lengths[li];
        int8_t *input = (int8_t *)heap_caps_aligned_alloc(16, n, MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0x9abc + n);
        for (int i = 0; i < n; i++) {
            input[i] = (int8_t)(rand() % 256 - 128);
        }

        TEST_ASSERT_EQUAL_INT32_MESSAGE(
            reduce_l1_s8_naive(input, n, 1), dl::base::reduce_l1(input, n, 1), "int8 reduce_l1 speed sanity");

        volatile int32_t sink = 0;

        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += dl::base::reduce_l1(input, n, 1);
        }
        int64_t t_opt = esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += reduce_l1_s8_naive(input, n, 1);
        }
        int64_t t_naive = esp_timer_get_time() - t0;

        float speedup = t_opt > 0 ? (float)t_naive / (float)t_opt : 0.0f;
        ESP_LOGI(TAG,
                 "n=%6d | %d iters | esp-dl=%8lld us | naive=%8lld us | speedup=%.2fx",
                 n,
                 iters,
                 (long long)t_opt,
                 (long long)t_naive,
                 (double)speedup);

        (void)sink;
        heap_caps_free(input);
    }
}

TEST_CASE("Test dl base reduce_l1 int16: speed", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_l1 int16: speed");

    const int lengths[] = {64, 256, 1024, 4096};
    const int num_lengths = sizeof(lengths) / sizeof(lengths[0]);
    const int iters = 4000;

    for (int li = 0; li < num_lengths; li++) {
        int n = lengths[li];
        int16_t *input = (int16_t *)heap_caps_aligned_alloc(16, n * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0xdef0 + n);
        for (int i = 0; i < n; i++) {
            input[i] = (int16_t)(rand() % 65536 - 32768);
        }

        assert_equal_int64(
            reduce_l1_s16_naive(input, n, 1), dl::base::reduce_l1(input, n, 1), "int16 reduce_l1 speed sanity");

        volatile int64_t sink = 0;

        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += dl::base::reduce_l1(input, n, 1);
        }
        int64_t t_opt = esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += reduce_l1_s16_naive(input, n, 1);
        }
        int64_t t_naive = esp_timer_get_time() - t0;

        float speedup = t_opt > 0 ? (float)t_naive / (float)t_opt : 0.0f;
        ESP_LOGI(TAG,
                 "n=%6d | %d iters | esp-dl=%8lld us | naive=%8lld us | speedup=%.2fx",
                 n,
                 iters,
                 (long long)t_opt,
                 (long long)t_naive,
                 (double)speedup);

        (void)sink;
        heap_caps_free(input);
    }
}

TEST_CASE("Test dl base reduce_l1 float: speed", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_l1 float: speed");

    const int lengths[] = {64, 256, 1024, 4096};
    const int num_lengths = sizeof(lengths) / sizeof(lengths[0]);
    const int iters = 4000;

    for (int li = 0; li < num_lengths; li++) {
        int n = lengths[li];
        float *input = (float *)heap_caps_aligned_alloc(16, n * sizeof(float), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0xfe01 + n);
        for (int i = 0; i < n; i++) {
            input[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }

        assert_float_near(
            reduce_l1_f32_naive(input, n, 1), dl::base::reduce_l1(input, n, 1), "float reduce_l1 speed sanity");

        volatile float sink = 0.0f;

        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += dl::base::reduce_l1(input, n, 1);
        }
        int64_t t_opt = esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += reduce_l1_f32_naive(input, n, 1);
        }
        int64_t t_naive = esp_timer_get_time() - t0;

        float speedup = t_opt > 0 ? (float)t_naive / (float)t_opt : 0.0f;
        ESP_LOGI(TAG,
                 "n=%6d | %d iters | esp-dl=%8lld us | naive=%8lld us | speedup=%.2fx",
                 n,
                 iters,
                 (long long)t_opt,
                 (long long)t_naive,
                 (double)speedup);

        (void)sink;
        heap_caps_free(input);
    }
}
