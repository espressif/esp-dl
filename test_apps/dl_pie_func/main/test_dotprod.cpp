#include "dl_base_dotprod.hpp"
#include "esp_dsp.h"
#include "esp_heap_caps.h"
#include "unity.h"
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static const char *TAG = "TEST DL FUNCTION";

// Plain scalar reference dot product (single-precision accumulation).
static float dotprod_f32_naive(const float *a, const float *b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Compares the optimized base::dotprod (ISA-accelerated where available) against
// a scalar reference and esp-dsp, reporting numerical accuracy and throughput.
TEST_CASE("Test dl base dotprod f32: precision and speed", "[pie]")
{
    ESP_LOGI(TAG, "Test dl base dotprod f32: precision and speed");

    const int lengths[] = {16, 64, 127, 256, 1024, 4093};
    const int num_lengths = sizeof(lengths) / sizeof(lengths[0]);
    const int iters = 2000;

    for (int li = 0; li < num_lengths; li++) {
        int n = lengths[li];

        // 16-byte aligned buffers to satisfy PIE load/store requirements.
        float *a = (float *)heap_caps_aligned_alloc(16, n * sizeof(float), MALLOC_CAP_DEFAULT);
        float *b = (float *)heap_caps_aligned_alloc(16, n * sizeof(float), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(a);
        TEST_ASSERT_NOT_NULL(b);

        // Fill with deterministic pseudo-random values in [-1, 1].
        srand(0x1234 + n);
        double ref = 0.0; // double accumulation = high-precision ground truth
        for (int i = 0; i < n; i++) {
            a[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            b[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            ref += (double)a[i] * (double)b[i];
        }

        float out_opt = 0.0f;
        float out_dsp = 0.0f;
        float out_naive = 0.0f;

        // base::dotprod dispatches to the ISA-accelerated kernel when available.
        dl::base::dotprod(a, b, &out_opt, n);
        dsps_dotprod_f32(a, b, &out_dsp, n);
        out_naive = dotprod_f32_naive(a, b, n);

        // Precision: error relative to the double-precision ground truth.
        float abs_ref = fabsf((float)ref);
        float tol = abs_ref * 1e-5f + 1e-5f;
        float err_opt = fabsf(out_opt - (float)ref);
        float err_dsp = fabsf(out_dsp - (float)ref);

        ESP_LOGI(TAG,
                 "n=%5d | ref=%+.6f | esp-dl=%+.6f (err=%.2e) | esp-dsp=%+.6f (err=%.2e) | naive=%+.6f",
                 n,
                 (double)ref,
                 (double)out_opt,
                 (double)err_opt,
                 (double)out_dsp,
                 (double)err_dsp,
                 (double)out_naive);

        TEST_ASSERT_TRUE_MESSAGE(err_opt <= tol, "optimized dotprod_f32 result deviates from reference");

        // Speed: average time over many iterations (volatile sink prevents elision).
        volatile float sink = 0.0f;

        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            float r;
            dl::base::dotprod(a, b, &r, n);
            sink += r;
        }
        int64_t t_opt = esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            float r;
            dsps_dotprod_f32(a, b, &r, n);
            sink += r;
        }
        int64_t t_dsp = esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int it = 0; it < iters; it++) {
            sink += dotprod_f32_naive(a, b, n);
        }
        int64_t t_naive = esp_timer_get_time() - t0;

        ESP_LOGI(TAG,
                 "n=%5d | %d iters | esp-dl=%lld us | esp-dsp=%lld us | naive=%lld us",
                 n,
                 iters,
                 (long long)t_opt,
                 (long long)t_dsp,
                 (long long)t_naive);

        (void)sink;
        heap_caps_free(a);
        heap_caps_free(b);
    }
}

static uint32_t mat_vecs_lcg(uint32_t &state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

static void fill_mat_vecs_inputs(int8_t *matrix, int matrix_size, int8_t *vectors, int vectors_size, int pattern)
{
    if (pattern == 1 || pattern == 2) {
        for (int i = 0; i < matrix_size; ++i) {
            matrix[i] = 127;
        }
        const int8_t value = pattern == 1 ? 127 : -128;
        for (int i = 0; i < vectors_size; ++i) {
            vectors[i] = value;
        }
        return;
    }

    uint32_t state = 0x6d2b79f5u + (uint32_t)matrix_size * 17u + (uint32_t)vectors_size;
    for (int i = 0; i < matrix_size; ++i) {
        matrix[i] = (int8_t)(mat_vecs_lcg(state) >> 24);
    }
    for (int i = 0; i < vectors_size; ++i) {
        vectors[i] = (int8_t)(mat_vecs_lcg(state) >> 24);
    }
}

static int16_t dotprod_s8_scalar_ref(const int8_t *a, const int8_t *b, int cols, int shift)
{
    int64_t sum = 0;
    for (int i = 0; i < cols; ++i) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }

    // Apply round-to-nearest, ties-to-even to match the expected kernel behavior.
    if (shift > 0) {
        const bool negative = sum < 0;
        uint64_t magnitude = negative ? (uint64_t)(-sum) : (uint64_t)sum;
        uint64_t rounded = magnitude >> shift;
        const uint64_t mask = ((uint64_t)1 << shift) - 1;
        const uint64_t remainder = magnitude & mask;
        const uint64_t half = (uint64_t)1 << (shift - 1);
        if (remainder > half || (remainder == half && (rounded & 1))) {
            ++rounded;
        }
        sum = negative ? -(int64_t)rounded : (int64_t)rounded;
    }

    if (sum > INT16_MAX) {
        return INT16_MAX;
    }
    if (sum < INT16_MIN) {
        return INT16_MIN;
    }
    return (int16_t)sum;
}

TEST_CASE("Test dl base mat_vecs s8 PIE: half-even rounding", "[pie][dotprod]")
{
#if !CONFIG_PIE_V2_BOOST
    TEST_IGNORE_MESSAGE("Requires PIE v2 boost support");
#else
    constexpr int rows = 4;
    constexpr int cols = 16;
    constexpr int vectors_num = 1;
    const int shifts[] = {1, 7};
    const int8_t vector_values[] = {1, 64};
    const int16_t expected[] = {0, 2, 0, -2};
    const int8_t coefficients[] = {1, 3, -1, -3};

    alignas(16) int8_t matrix[rows * cols] = {};
    alignas(16) int8_t vector[cols] = {};
    alignas(16) int16_t out_pie[rows] = {};
    alignas(16) int16_t out_dotprod[rows] = {};

    for (int row = 0; row < rows; ++row) {
        matrix[row * cols] = coefficients[row];
    }

    for (int test = 0; test < 2; ++test) {
        vector[0] = vector_values[test];
        dl::base::mat_vecs_dotprod(matrix, vector, out_pie, rows, cols, vectors_num, shifts[test]);

        for (int row = 0; row < rows; ++row) {
            dl::base::dotprod(matrix + row * cols, vector, out_dotprod + row, cols, shifts[test]);
            TEST_ASSERT_EQUAL_INT16_MESSAGE(expected[row], out_pie[row], "mat_vecs half-even mismatch");
            TEST_ASSERT_EQUAL_INT16_MESSAGE(expected[row], out_dotprod[row], "dotprod half-even mismatch");
        }
    }
#endif
}

TEST_CASE("Test dl base mat_vecs s8 PIE: precision and sizes", "[pie][dotprod]")
{
#if !CONFIG_PIE_V2_BOOST
    TEST_IGNORE_MESSAGE("Requires PIE v2 boost support");
#else
    const int cols_cases[] = {16, 32, 48, 64, 80, 512};
    const int rows_cases[] = {1, 3};
    const int vectors_cases[] = {1, 4};
    const int shifts[] = {0, 1, 7, 15};

    for (int cols : cols_cases) {
        for (int rows : rows_cases) {
            for (int vectors_num : vectors_cases) {
                const size_t matrix_count = (size_t)rows * cols;
                const size_t vectors_count = (size_t)vectors_num * cols;
                const size_t output_count = (size_t)rows * vectors_num;
                int8_t *matrix =
                    (int8_t *)heap_caps_aligned_alloc(16, matrix_count * sizeof(int8_t), MALLOC_CAP_DEFAULT);
                int8_t *vectors =
                    (int8_t *)heap_caps_aligned_alloc(16, vectors_count * sizeof(int8_t), MALLOC_CAP_DEFAULT);
                int16_t *out_pie =
                    (int16_t *)heap_caps_aligned_alloc(16, output_count * sizeof(int16_t), MALLOC_CAP_DEFAULT);
                int16_t *out_dotprod =
                    (int16_t *)heap_caps_aligned_alloc(16, output_count * sizeof(int16_t), MALLOC_CAP_DEFAULT);
                int16_t *out_ref =
                    (int16_t *)heap_caps_aligned_alloc(16, output_count * sizeof(int16_t), MALLOC_CAP_DEFAULT);
                TEST_ASSERT_NOT_NULL(matrix);
                TEST_ASSERT_NOT_NULL(vectors);
                TEST_ASSERT_NOT_NULL(out_pie);
                TEST_ASSERT_NOT_NULL(out_dotprod);
                TEST_ASSERT_NOT_NULL(out_ref);

                for (int pattern = 0; pattern < 3; ++pattern) {
                    fill_mat_vecs_inputs(matrix, matrix_count, vectors, vectors_count, pattern);
                    for (int shift : shifts) {
                        dl::base::mat_vecs_dotprod(matrix, vectors, out_pie, rows, cols, vectors_num, shift);
                        for (int v = 0; v < vectors_num; ++v) {
                            for (int row = 0; row < rows; ++row) {
                                dl::base::dotprod(
                                    matrix + row * cols, vectors + v * cols, out_dotprod + v * rows + row, cols, shift);
                                out_ref[v * rows + row] =
                                    dotprod_s8_scalar_ref(matrix + row * cols, vectors + v * cols, cols, shift);
                            }
                        }

                        for (size_t i = 0; i < output_count; ++i) {
                            char message[128];
                            snprintf(message,
                                     sizeof(message),
                                     "cols=%d rows=%d vectors=%d shift=%d pattern=%d index=%u",
                                     cols,
                                     rows,
                                     vectors_num,
                                     shift,
                                     pattern,
                                     (unsigned)i);
                            TEST_ASSERT_EQUAL_INT16_MESSAGE(out_ref[i], out_pie[i], message);
                            TEST_ASSERT_EQUAL_INT16_MESSAGE(out_ref[i], out_dotprod[i], message);
                        }
                    }
                }

                heap_caps_free(matrix);
                heap_caps_free(vectors);
                heap_caps_free(out_pie);
                heap_caps_free(out_dotprod);
                heap_caps_free(out_ref);
            }
        }
    }
#endif
}

TEST_CASE("Test dl base mat_vecs s8 PIE: PSRAM speed", "[pie][dotprod]")
{
#if !CONFIG_PIE_V2_BOOST
    TEST_IGNORE_MESSAGE("Requires PIE v2 boost support");
#else
    constexpr int rows = 384;
    constexpr int cols = 512;
    constexpr int vectors_num = 4;
    constexpr int shift = 7;
    constexpr int iterations = 20;
    const size_t matrix_count = (size_t)rows * cols;
    const size_t vectors_count = (size_t)vectors_num * cols;
    const size_t output_count = (size_t)rows * vectors_num;

    int8_t *matrix =
        (int8_t *)heap_caps_aligned_alloc(16, matrix_count * sizeof(int8_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t *vectors = (int8_t *)heap_caps_aligned_alloc(16, vectors_count * sizeof(int8_t), MALLOC_CAP_DEFAULT);
    int16_t *out_pie = (int16_t *)heap_caps_aligned_alloc(16, output_count * sizeof(int16_t), MALLOC_CAP_DEFAULT);
    int16_t *out_template = (int16_t *)heap_caps_aligned_alloc(16, output_count * sizeof(int16_t), MALLOC_CAP_DEFAULT);
    TEST_ASSERT_NOT_NULL(matrix);
    TEST_ASSERT_NOT_NULL(vectors);
    TEST_ASSERT_NOT_NULL(out_pie);
    TEST_ASSERT_NOT_NULL(out_template);
    fill_mat_vecs_inputs(matrix, matrix_count, vectors, vectors_count, 0);

    dl::base::mat_vecs_dotprod(matrix, vectors, out_pie, rows, cols, vectors_num, shift);
    dl::base::mat_vecs_dotprod<int8_t, int8_t, int16_t>(
        matrix, vectors, out_template, rows, cols, vectors_num, cols, rows, shift);
    for (size_t i = 0; i < output_count; ++i) {
        TEST_ASSERT_EQUAL_INT16(out_template[i], out_pie[i]);
    }

    int64_t start = esp_timer_get_time();
    for (int i = 0; i < iterations; ++i) {
        dl::base::mat_vecs_dotprod(matrix, vectors, out_pie, rows, cols, vectors_num, shift);
    }
    const int64_t pie_us = esp_timer_get_time() - start;

    start = esp_timer_get_time();
    for (int i = 0; i < iterations; ++i) {
        dl::base::mat_vecs_dotprod<int8_t, int8_t, int16_t>(
            matrix, vectors, out_template, rows, cols, vectors_num, cols, rows, shift);
    }
    const int64_t template_us = esp_timer_get_time() - start;
    ESP_LOGI(TAG,
             "mat_vecs rows=%d cols=%d vectors=%d PSRAM | %d iterations | PIE=%lld us | template=%lld us | "
             "speedup=%.2fx",
             rows,
             cols,
             vectors_num,
             iterations,
             (long long)pie_us,
             (long long)template_us,
             pie_us > 0 ? (double)template_us / (double)pie_us : 0.0);

    heap_caps_free(matrix);
    heap_caps_free(vectors);
    heap_caps_free(out_pie);
    heap_caps_free(out_template);
#endif
}
