#include "dl_base_dotprod.hpp"
#include "esp_dsp.h"
#include "esp_heap_caps.h"
#include "unity.h"
#include <cmath>
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
