#include "dl_base_reduce.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "unity.h"
#include <cstdint>
#include <cstdlib>

static const char *TAG = "TEST DL REDUCE_L2";

// Plain scalar reference: sum of squares over `size` elements with the given stride.
static int32_t reduce_l2_s8_naive(const int8_t *input, int32_t size, int32_t stride)
{
    int32_t sum = 0;
    const int8_t *ptr = input;
    for (int32_t i = 0; i < size; i++) {
        int32_t v = *ptr;
        sum += v * v;
        ptr += stride;
    }
    return sum;
}

static int64_t reduce_l2_s16_naive(const int16_t *input, int32_t size, int32_t stride)
{
    int64_t sum = 0;
    const int16_t *ptr = input;
    for (int32_t i = 0; i < size; i++) {
        int64_t v = *ptr;
        sum += v * v;
        ptr += stride;
    }
    return sum;
}

// Unity's TEST_ASSERT_EQUAL_INT64 needs UNITY_SUPPORT_64, which is disabled in this
// build, so compare the 64-bit values directly and log both operands on mismatch so
// the divergence is visible in the report.
static void assert_equal_int64(int64_t expected, int64_t actual, const char *msg)
{
    if (expected != actual) {
        ESP_LOGE(TAG, "%s: expected=%lld actual=%lld", msg, (long long)expected, (long long)actual);
    }
    TEST_ASSERT_TRUE_MESSAGE(expected == actual, msg);
}

// Exercises the int8 base::reduce_l2 (ISA-accelerated where available) against a
// scalar reference across a range of sizes and alignment offsets so the SIMD
// prologue (unaligned head) and scalar tail (remainder) are all covered.
TEST_CASE("Test dl base reduce_l2 int8: precision", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_l2 int8: precision");

    const int sizes[] = {1, 7, 15, 16, 17, 31, 63, 64, 127, 256, 1024, 4093};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    // Byte offsets into a 16-byte aligned buffer to force unaligned SIMD heads.
    const int offsets[] = {0, 1, 3, 8};
    const int num_offsets = sizeof(offsets) / sizeof(offsets[0]);

    for (int si = 0; si < num_sizes; si++) {
        int n = sizes[si];
        for (int oi = 0; oi < num_offsets; oi++) {
            int off = offsets[oi];

            int8_t *buf = (int8_t *)heap_caps_aligned_alloc(16, n + off + 16, MALLOC_CAP_DEFAULT);
            TEST_ASSERT_NOT_NULL(buf);
            int8_t *input = buf + off;

            srand(0x1234 + n * 31 + off);
            for (int i = 0; i < n; i++) {
                input[i] = (int8_t)(rand() % 256 - 128);
            }

            int32_t ref = reduce_l2_s8_naive(input, n, 1);
            int32_t opt = dl::base::reduce_l2(input, n, 1);

            TEST_ASSERT_EQUAL_INT32_MESSAGE(ref, opt, "int8 reduce_l2 (stride 1) mismatch");

            heap_caps_free(buf);
        }
    }

    // Strided path (stride != 1 always uses the scalar implementation).
    {
        int n = 333;
        int stride = 3;
        int8_t *input = (int8_t *)heap_caps_aligned_alloc(16, n * stride, MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0x55aa);
        for (int i = 0; i < n * stride; i++) {
            input[i] = (int8_t)(rand() % 256 - 128);
        }
        int32_t ref = reduce_l2_s8_naive(input, n, stride);
        int32_t opt = dl::base::reduce_l2(input, n, stride);
        TEST_ASSERT_EQUAL_INT32_MESSAGE(ref, opt, "int8 reduce_l2 (strided) mismatch");
        heap_caps_free(input);
    }
}

// Exercises the int16 base::reduce_l2. The int16 sum-of-squares can exceed 32 bits and
// the 40-bit saturating ACCX/XACC (PIE V1 and V2 alike), so large element counts with
// near-full-scale magnitudes are included to cover the 256-element chunking path.
TEST_CASE("Test dl base reduce_l2 int16: precision", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_l2 int16: precision");

    const int sizes[] = {1, 7, 8, 9, 15, 31, 64, 127, 256, 1024, 4093, 8192};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    // Element offsets into a 16-byte aligned buffer to force unaligned SIMD heads.
    const int offsets[] = {0, 1, 2, 4};
    const int num_offsets = sizeof(offsets) / sizeof(offsets[0]);

    for (int si = 0; si < num_sizes; si++) {
        int n = sizes[si];
        for (int oi = 0; oi < num_offsets; oi++) {
            int off = offsets[oi];

            int16_t *buf = (int16_t *)heap_caps_aligned_alloc(16, (n + off + 8) * sizeof(int16_t), MALLOC_CAP_DEFAULT);
            TEST_ASSERT_NOT_NULL(buf);
            int16_t *input = buf + off;

            srand(0x4321 + n * 17 + off);
            for (int i = 0; i < n; i++) {
                // Bias towards large magnitudes so the accumulator grows quickly.
                int v = rand() % 65536 - 32768;
                input[i] = (int16_t)v;
            }

            int64_t ref = reduce_l2_s16_naive(input, n, 1);
            int64_t opt = dl::base::reduce_l2(input, n, 1);

            assert_equal_int64(ref, opt, "int16 reduce_l2 (stride 1) mismatch");

            heap_caps_free(buf);
        }
    }

    // Strided path (stride != 1 always uses the scalar implementation).
    {
        int n = 257;
        int stride = 4;
        int16_t *input = (int16_t *)heap_caps_aligned_alloc(16, n * stride * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0x99cc);
        for (int i = 0; i < n * stride; i++) {
            input[i] = (int16_t)(rand() % 65536 - 32768);
        }
        int64_t ref = reduce_l2_s16_naive(input, n, stride);
        int64_t opt = dl::base::reduce_l2(input, n, stride);
        assert_equal_int64(ref, opt, "int16 reduce_l2 (strided) mismatch");
        heap_caps_free(input);
    }
}

// Measures throughput of the optimized (ISA-accelerated) int8 reduce_l2 against the
// scalar reference and reports the speedup (naive_time / opt_time) for each length.
TEST_CASE("Test dl base reduce_l2 int8: speed", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_l2 int8: speed");

    const int lengths[] = {64, 256, 1024, 4096};
    const int num_lengths = sizeof(lengths) / sizeof(lengths[0]);
    const int iters = 4000;

    for (int li = 0; li < num_lengths; li++) {
        int n = lengths[li];
        int8_t *input = (int8_t *)heap_caps_aligned_alloc(16, n, MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0x1234 + n);
        for (int i = 0; i < n; i++) {
            input[i] = (int8_t)(rand() % 256 - 128);
        }

        // Correctness sanity check before timing.
        TEST_ASSERT_EQUAL_INT32_MESSAGE(
            reduce_l2_s8_naive(input, n, 1), dl::base::reduce_l2(input, n, 1), "int8 reduce_l2 speed sanity");

        volatile int32_t sink = 0;

        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += dl::base::reduce_l2(input, n, 1);
        }
        int64_t t_opt = esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += reduce_l2_s8_naive(input, n, 1);
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

// Measures throughput of the optimized (ISA-accelerated) int16 reduce_l2 against the
// scalar reference and reports the speedup (naive_time / opt_time) for each length.
TEST_CASE("Test dl base reduce_l2 int16: speed", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base reduce_l2 int16: speed");

    const int lengths[] = {64, 256, 1024, 4096};
    const int num_lengths = sizeof(lengths) / sizeof(lengths[0]);
    const int iters = 4000;

    for (int li = 0; li < num_lengths; li++) {
        int n = lengths[li];
        int16_t *input = (int16_t *)heap_caps_aligned_alloc(16, n * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        srand(0x4321 + n);
        for (int i = 0; i < n; i++) {
            input[i] = (int16_t)(rand() % 65536 - 32768);
        }

        // Correctness sanity check before timing.
        assert_equal_int64(
            reduce_l2_s16_naive(input, n, 1), dl::base::reduce_l2(input, n, 1), "int16 reduce_l2 speed sanity");

        volatile int64_t sink = 0;

        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += dl::base::reduce_l2(input, n, 1);
        }
        int64_t t_opt = esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += reduce_l2_s16_naive(input, n, 1);
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
