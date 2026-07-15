#include "sdkconfig.h"

#include "dl_base_lut.hpp"
#include "dl_tool.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "unity.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>

static const char *TAG = "TEST DL LUT";

static inline uint16_t rounded_index(int16_t input, int shift)
{
    uint32_t value = (uint16_t)input ^ 0x8000u;
    uint32_t index = value >> shift;
    uint32_t remainder = value & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);

#if CONFIG_PIE_V2_BOOST
    if (remainder > half || (remainder == half && (index & 1u))) {
#else
    if (remainder >= half) {
#endif
        index++;
    }
    return (uint16_t)index;
}

static inline void run_lut_kernel(int16_t *output, int16_t *input, int size, int16_t *table, int shift)
{
    dl::base::lut_s16_nearest_neighbor(output, input, size, table, 1 << shift);
}

__attribute__((noinline)) static void lut_s16_scalar(
    int16_t *output, const int16_t *input, int size, const int16_t *table, int shift)
{
    for (int i = 0; i < size; i++) {
        output[i] = table[rounded_index(input[i], shift)];
    }
}

static int16_t *allocate_table(int shift)
{
    // Logical table has indices [0, 65536 / step]. One extra int16 pads
    // the final ESP.LDXQ.32 access.
    size_t count = (65536u >> shift) + 2u;
    int16_t *table = (int16_t *)heap_caps_aligned_alloc(16, count * sizeof(int16_t), MALLOC_CAP_DEFAULT);
    TEST_ASSERT_NOT_NULL(table);

    for (size_t i = 0; i + 1 < count; i++) {
        table[i] = (int16_t)((i * 109u + 37u) & 0xffffu);
    }
    table[count - 1] = 0;
    return table;
}

static void fill_input(int16_t *input, int size, unsigned seed)
{
    static const int16_t edge_values[] = {
        INT16_MIN,
        INT16_MIN + 1,
        -32767,
        -16385,
        -16384,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        16383,
        16384,
        32766,
        INT16_MAX,
    };
    const int edge_count = sizeof(edge_values) / sizeof(edge_values[0]);

    srand(seed);
    for (int i = 0; i < size; i++) {
        input[i] = i < edge_count ? edge_values[i] : (int16_t)(rand() & 0xffff);
    }
}

static void assert_equal_buffers(const int16_t *expected, const int16_t *actual, int size, int shift)
{
    for (int i = 0; i < size; i++) {
        if (expected[i] != actual[i]) {
            ESP_LOGE(TAG, "shift=%d index=%d expected=%d actual=%d", shift, i, (int)expected[i], (int)actual[i]);
            TEST_FAIL_MESSAGE("lut_s16_nearest_neighbor output mismatch");
        }
    }
}

#if CONFIG_PIE_V2_BOOST || CONFIG_PIE_V1_BOOST
static inline void run_lut_s8_kernel(int8_t *output, int8_t *input, int size, int8_t *table)
{
    dl::base::lut_s8(output, input, size, table);
}

__attribute__((noinline)) static void lut_s8_scalar(int8_t *output, const int8_t *input, int size, const int8_t *table)
{
    for (int i = 0; i < size; i++) {
        output[i] = table[(uint8_t)input[i] ^ 0x80u];
    }
}

static void fill_s8_input(int8_t *input, int size, unsigned seed)
{
    static const int8_t edge_values[] = {INT8_MIN, INT8_MIN + 1, -1, 0, 1, INT8_MAX - 1, INT8_MAX};
    const int edge_count = sizeof(edge_values) / sizeof(edge_values[0]);

    srand(seed);
    for (int i = 0; i < size; i++) {
        input[i] = i < edge_count ? edge_values[i] : (int8_t)(rand() & 0xff);
    }
}

static void assert_equal_s8_buffers(const int8_t *expected, const int8_t *actual, int size)
{
    for (int i = 0; i < size; i++) {
        if (expected[i] != actual[i]) {
            ESP_LOGE(TAG, "s8 index=%d expected=%d actual=%d", i, (int)expected[i], (int)actual[i]);
            TEST_FAIL_MESSAGE("lut_s8 output mismatch");
        }
    }
}

TEST_CASE("Test base s8 LUT: precision", "[pie][lut]")
{
    const int sizes[] = {1, 7, 8, 15, 16, 63, 64, 1011, 1024};
    int8_t *table = (int8_t *)heap_caps_aligned_alloc(16, 256, MALLOC_CAP_DEFAULT);
    TEST_ASSERT_NOT_NULL(table);
    for (int i = 0; i < 256; i++) {
        table[i] = (int8_t)((i * 109 + 37) & 0xff);
    }

    for (int size : sizes) {
        int8_t *input = (int8_t *)heap_caps_aligned_alloc(16, size, MALLOC_CAP_DEFAULT);
        int8_t *expected = (int8_t *)heap_caps_aligned_alloc(16, size, MALLOC_CAP_DEFAULT);
        int8_t *actual = (int8_t *)heap_caps_aligned_alloc(16, size, MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        TEST_ASSERT_NOT_NULL(expected);
        TEST_ASSERT_NOT_NULL(actual);

        fill_s8_input(input, size, 0x9abcu + (unsigned)size);
        lut_s8_scalar(expected, input, size, table);
        memset(actual, 0xa5, size);
#if CONFIG_PIE_V2_BOOST
        // LUT must not depend on the rounding mode left by a previous PIE kernel.
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
#endif
        run_lut_s8_kernel(actual, input, size, table);
#if CONFIG_PIE_V2_BOOST
        TEST_ASSERT_EQUAL_UINT32(ROUND_MODE_HALF_EVEN, (dl_esp32p4_get_cfg() >> 4) & 0xf);
#endif
        assert_equal_s8_buffers(expected, actual, size);

        heap_caps_free(input);
        heap_caps_free(expected);
        heap_caps_free(actual);
    }
    heap_caps_free(table);
}

TEST_CASE("Test base s8 LUT: speed", "[pie][lut]")
{
    const int sizes[] = {64, 256, 1011, 1024, 4096};
    int8_t *table = (int8_t *)heap_caps_aligned_alloc(16, 256, MALLOC_CAP_DEFAULT);
    TEST_ASSERT_NOT_NULL(table);
    for (int i = 0; i < 256; i++) {
        table[i] = (int8_t)((i * 109 + 37) & 0xff);
    }
    volatile int32_t sink = 0;

    for (int size : sizes) {
        int iterations = 2000000 / size;
        if (iterations < 200) {
            iterations = 200;
        }

        int8_t *input = (int8_t *)heap_caps_aligned_alloc(16, size, MALLOC_CAP_DEFAULT);
        int8_t *scalar_output = (int8_t *)heap_caps_aligned_alloc(16, size, MALLOC_CAP_DEFAULT);
        int8_t *pie_output = (int8_t *)heap_caps_aligned_alloc(16, size, MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        TEST_ASSERT_NOT_NULL(scalar_output);
        TEST_ASSERT_NOT_NULL(pie_output);

        fill_s8_input(input, size, 0xdef0u + (unsigned)size);
        lut_s8_scalar(scalar_output, input, size, table);
        run_lut_s8_kernel(pie_output, input, size, table);
        assert_equal_s8_buffers(scalar_output, pie_output, size);

        int64_t start = esp_timer_get_time();
        for (int i = 0; i < iterations; i++) {
            run_lut_s8_kernel(pie_output, input, size, table);
        }
        int64_t pie_time = esp_timer_get_time() - start;

        start = esp_timer_get_time();
        for (int i = 0; i < iterations; i++) {
            lut_s8_scalar(scalar_output, input, size, table);
        }
        int64_t scalar_time = esp_timer_get_time() - start;

        for (int i = 0; i < size; i++) {
            sink += pie_output[i] + scalar_output[i];
        }

        float speedup = pie_time > 0 ? (float)scalar_time / (float)pie_time : 0.0f;
        ESP_LOGI(TAG,
                 "s8 n=%4d | %5d iterations | PIE=%8lld us | scalar=%8lld us | speedup=%.2fx",
                 size,
                 iterations,
                 (long long)pie_time,
                 (long long)scalar_time,
                 (double)speedup);

        heap_caps_free(input);
        heap_caps_free(scalar_output);
        heap_caps_free(pie_output);
    }

    (void)sink;
    heap_caps_free(table);
}
#endif

TEST_CASE("Test base s16 LUT nearest-neighbor: precision", "[pie][lut]")
{
    const int shifts[] = {1, 2, 3, 4, 8, 15};
    const int sizes[] = {8, 16, 64, 1024};

    for (int shift : shifts) {
        int16_t *table = allocate_table(shift);

        for (int size : sizes) {
            int16_t *input = (int16_t *)heap_caps_aligned_alloc(16, size * sizeof(int16_t), MALLOC_CAP_DEFAULT);
            int16_t *expected = (int16_t *)heap_caps_aligned_alloc(16, size * sizeof(int16_t), MALLOC_CAP_DEFAULT);
            int16_t *actual = (int16_t *)heap_caps_aligned_alloc(16, size * sizeof(int16_t), MALLOC_CAP_DEFAULT);
            TEST_ASSERT_NOT_NULL(input);
            TEST_ASSERT_NOT_NULL(expected);
            TEST_ASSERT_NOT_NULL(actual);

            fill_input(input, size, 0x1234u + (unsigned)shift * 131u + (unsigned)size);
            lut_s16_scalar(expected, input, size, table, shift);
            memset(actual, 0xa5, size * sizeof(int16_t));
            run_lut_kernel(actual, input, size, table, shift);
            assert_equal_buffers(expected, actual, size, shift);

            heap_caps_free(input);
            heap_caps_free(expected);
            heap_caps_free(actual);
        }
        heap_caps_free(table);
    }
}

TEST_CASE("Test base s16 LUT nearest-neighbor: speed", "[pie][lut]")
{
    const int shift = 4;
    const int sizes[] = {64, 256, 1011, 1024, 4096};
    int16_t *table = allocate_table(shift);
    volatile int32_t sink = 0;

    for (int size : sizes) {
        int iterations = 2000000 / size;
        if (iterations < 200) {
            iterations = 200;
        }

        int16_t *input = (int16_t *)heap_caps_aligned_alloc(16, size * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        int16_t *scalar_output = (int16_t *)heap_caps_aligned_alloc(16, size * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        int16_t *pie_output = (int16_t *)heap_caps_aligned_alloc(16, size * sizeof(int16_t), MALLOC_CAP_DEFAULT);
        TEST_ASSERT_NOT_NULL(input);
        TEST_ASSERT_NOT_NULL(scalar_output);
        TEST_ASSERT_NOT_NULL(pie_output);

        fill_input(input, size, 0x5678u + (unsigned)size);
        lut_s16_scalar(scalar_output, input, size, table, shift);
        run_lut_kernel(pie_output, input, size, table, shift);
        assert_equal_buffers(scalar_output, pie_output, size, shift);

        int64_t start = esp_timer_get_time();
        for (int i = 0; i < iterations; i++) {
            run_lut_kernel(pie_output, input, size, table, shift);
        }
        int64_t pie_time = esp_timer_get_time() - start;

        start = esp_timer_get_time();
        for (int i = 0; i < iterations; i++) {
            lut_s16_scalar(scalar_output, input, size, table, shift);
        }
        int64_t scalar_time = esp_timer_get_time() - start;

        for (int i = 0; i < size; i++) {
            sink += pie_output[i] + scalar_output[i];
        }

        float speedup = pie_time > 0 ? (float)scalar_time / (float)pie_time : 0.0f;
        ESP_LOGI(TAG,
                 "n=%4d | %5d iterations | PIE=%8lld us | scalar=%8lld us | speedup=%.2fx",
                 size,
                 iterations,
                 (long long)pie_time,
                 (long long)scalar_time,
                 (double)speedup);

        heap_caps_free(input);
        heap_caps_free(scalar_output);
        heap_caps_free(pie_output);
    }

    (void)sink;
    heap_caps_free(table);
}
