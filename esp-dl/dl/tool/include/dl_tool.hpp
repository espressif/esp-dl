#pragma once

#include <limits>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "dl_define.hpp"
#include "esp_cpu.h"
#include "esp_log.h"
#include "esp_mmu_map.h"
#include "esp_system.h"
#include "esp_timer.h"
#include <functional>
#include "freertos/FreeRTOS.h"
#if CONFIG_ESP32P4_BOOST
#include "dl_esp32p4_cache_reg.hpp"
#include "esp_memory_utils.h"
#endif

extern "C" {
#if CONFIG_TIE728_BOOST
void dl_tie728_memset_8b(void *ptr, const int value, const int n);
void dl_tie728_memset_16b(void *ptr, const int value, const int n);
void dl_tie728_memset_32b(void *ptr, const int value, const int n);
#endif

#if CONFIG_ESP32P4_BOOST
typedef enum {
    ROUND_MODE_FLOOR = 0,
    ROUND_MODE_CEILING = 1,
    ROUND_MODE_UP = 2,
    ROUND_MODE_DOWN = 3,
    ROUND_MODE_HALF_UP = 4,
    ROUND_MODE_HALF_DOWN = 5,
    ROUND_MODE_HALF_EVEN = 6,
    ROUND_MODE_UNNECESSARY = 7,
    MODEL_LOCATION_MAX = ROUND_MODE_UNNECESSARY,
} round_mode_t;

void dl_esp32p4_cfg_round(round_mode_t round_mode);
int dl_esp32p4_round_half_even(float value);
#endif
}

#define DL_LOG_LATENCY_INIT_WITH_SIZE(size) dl::tool::Latency latency(size)
#define DL_LOG_LATENCY_INIT() DL_LOG_LATENCY_INIT_WITH_SIZE(1)
#define DL_LOG_LATENCY_START() latency.start()
#define DL_LOG_LATENCY_END() latency.end()
#define DL_LOG_LATENCY_GET() latency.get_average_period()
#define DL_LOG_LATENCY_PRINT(prefix, key) latency.print(prefix, key)
#define DL_LOG_LATENCY_END_PRINT(prefix, key) \
    DL_LOG_LATENCY_END();                     \
    DL_LOG_LATENCY_PRINT(prefix, key);

#define DL_LOG_LATENCY_ARRAY_INIT_WITH_SIZE(n, size) \
    std::vector<dl::tool::Latency> latencies;        \
    latencies.reserve(n);                            \
    for (int i = 0; i < n; i++) {                    \
        latencies.emplace_back(size);                \
    }
#define DL_LOG_LATENCY_ARRAY_INIT(n) DL_LOG_LATENCY_ARRAY_INIT_WITH_SIZE(n, 1)
#define DL_LOG_LATENCY_ARRAY_START(i) latencies[i].start()
#define DL_LOG_LATENCY_ARRAY_END(i) latencies[i].end()
#define DL_LOG_LATENCY_ARRAY_PRINT(i, prefix, key) latencies[i].print(prefix, key)
#define DL_LOG_LATENCY_ARRAY_END_PRINT(i, prefix, key) \
    DL_LOG_LATENCY_ARRAY_END(i);                       \
    DL_LOG_LATENCY_ARRAY_PRINT(i, prefix, key);

namespace dl {
namespace tool {

/**
 * @brief Encapsulate the round strategies for different platforms.
 * esp32p4:rounding half to even
 * esp32s3:rounding half up
 *
 * @param value The float or double value.
 * @return int
 */
template <typename T>
int round(T value);

/**
 * @brief round(shift(x)). Round strategies is same as round().
 *
 * @param value The Integer value.
 * @return int
 */
int shift_and_round(int value, int shift);

/**
 * @brief Set memory zero.
 *
 * @param ptr pointer of memory
 * @param n   byte number
 */
void set_zero(void *ptr, const int n);

/**
 * @brief Set array value.
 *
 * @tparam T supports all data type, sizeof(T) equals to 1, 2 and 4 will boost by instruction
 * @param ptr   pointer of array
 * @param value value to set
 * @param len   length of array
 */
template <typename T>
void set_value(T *ptr, const T value, const int len)
{
#if CONFIG_TIE728_BOOST
    int *temp = (int *)&value;
    if (sizeof(T) == 1)
        dl_tie728_memset_8b(ptr, *temp, len);
    else if (sizeof(T) == 2)
        dl_tie728_memset_16b(ptr, *temp, len);
    else if (sizeof(T) == 4)
        dl_tie728_memset_32b(ptr, *temp, len);
    else
#endif
        for (size_t i = 0; i < len; i++) ptr[i] = value;
}

/**
 * @brief Copy memory.
 *
 * @param dst pointer of destination
 * @param src pointer of source
 * @param n   byte number
 */
void copy_memory(void *dst, void *src, const size_t n);

/**
 * @brief Get memory addr type.
 *
 * @param address
 * @return memory_addr_type_t
 */
memory_addr_type_t memory_addr_type(void *address);

/**
 * @brief Same as heap_caps_aligned_alloc, only skip TCM in esp32p4.
 *
 * @param alignment
 * @param size
 * @param caps
 * @return void*
 */
void *malloc_aligned(size_t alignment, size_t size, uint32_t caps);

/**
 * @brief Same as heap_caps_aligned_calloc, only skip TCM in esp32p4.
 *
 * @param alignment
 * @param n
 * @param size
 * @param caps
 * @return void*
 */
void *calloc_aligned(size_t alignment, size_t n, size_t size, uint32_t caps);

template <typename T>
struct PSRAMAllocator {
    typedef T value_type;

    PSRAMAllocator() = default;

    template <class U>
    constexpr PSRAMAllocator(const PSRAMAllocator<U> &) noexcept
    {
    }

    template <typename U>
    struct rebind {
        using other = PSRAMAllocator<U>;
    };

    T *allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            return nullptr;
        }
        if (auto p = static_cast<T *>(heap_caps_malloc(n * sizeof(T), MALLOC_CAP_SPIRAM))) {
            return p;
        }
        return nullptr;
    }

    void deallocate(T *p, std::size_t) noexcept { heap_caps_free(p); }
};

template <class T, class U>
bool operator==(const PSRAMAllocator<T> &, const PSRAMAllocator<U> &)
{
    return true;
}

template <class T, class U>
bool operator!=(const PSRAMAllocator<T> &, const PSRAMAllocator<U> &)
{
    return false;
}

/**
 * @brief Truncate the input into int8_t range.
 *
 * @tparam T supports all integer types
 * @param output as an output
 * @param input  as an input
 */
template <typename T>
void truncate(int8_t &output, T input)
{
    output = DL_CLIP(input, INT8_MIN, INT8_MAX);
}

/**
 * @brief Truncate the input into int16_t range.
 *
 * @tparam T supports all integer types
 * @param output as an output
 * @param input  as an input
 */
template <typename T>
void truncate(int16_t &output, T input)
{
    output = DL_CLIP(input, INT16_MIN, INT16_MAX);
}

template <typename T>
void truncate(int32_t &output, T input)
{
    output = DL_CLIP(input, INT32_MIN, INT32_MAX);
}

template <typename T>
void truncate(int64_t &output, T input)
{
    output = DL_CLIP(input, INT64_MIN, INT64_MAX);
}

// void truncate(int32_t &output, int64_t input)
// {
//     output = DL_CLIP(input, INT32_MIN, INT32_MAX);
// }

/**
 * @brief Generate 8bit lut table
 *
 * @param table:    lut table
 * @param exponent: exponent
 * @param func:     function
 *
 * @return return 8-bit lut table
 */
float *gen_lut_8bit(float *table, int exponent, std::function<float(float)> func);

#if CONFIG_ESP32P4_BOOST
inline int calculate_exponent(int n, int max_value)
{
    int exp;
    if (127 == max_value) {
        exp = 6;
    }
    if (32767 == max_value) {
        exp = 14;
    }
    float max_value_float = (float)max_value / (1 << exp);
    while (max_value_float > 1.f / n) {
        exp += 1;
        max_value_float /= 2;
    }

    exp -= 1;
    return exp;
}
#else
/**
 * @brief Calculate the exponent of quantizing 1/n into max_value range.
 *
 * @param n          1/n: value to be quantized
 * @param max_value  the max_range
 */
inline int calculate_exponent(int n, int max_value)
{
    int exp = 0;
    int tmp = 1 / n;
    while (tmp < max_value) {
        exp += 1;
        tmp = (1 << exp) / n;
    }
    exp -= 1;

    return exp;
}
#endif

/**
 * @brief Print vector in format "[x1, x2, ...]\n".
 *
 * @param array to print
 */
inline void print_vector(std::vector<int> &array, const char *message = NULL)
{
    if (message)
        printf("%s: ", message);

    printf("[");
    for (int i = 0; i < array.size(); i++) {
        printf(", %d" + (i ? 0 : 2), array[i]);
    }
    printf("]\n");
}

/**
 * @brief Get the cycle object
 *
 * @return cycle count
 */
inline uint32_t get_cycle()
{
    uint32_t ccount;
    // __asm__ __volatile__("rsr %0, ccount"
    //                      : "=a"(ccount)
    //                      :
    //                      : "memory");
    ccount = esp_cpu_get_cycle_count();
    return ccount;
}

class Latency {
private:
    const uint32_t size; /*!< size of queue */
    uint32_t *queue;     /*!< queue for storing history period */
    uint32_t period;     /*!< current period */
    uint32_t sum;        /*!< sum of period */
    uint32_t count;      /*!< the number of added period */
    uint32_t next;       /*!< point to next element in queue */
    uint32_t timestamp;  /*!< record the start */
#if CONFIG_ESP32P4_BOOST && DL_LOG_CACHE_COUNT
    uint32_t l2dbus_hit_cnt_s;
    uint32_t l2dbus_hit_cnt_e;
    uint32_t l2dbus_miss_cnt_s;
    uint32_t l2dbus_miss_cnt_e;
    uint32_t l2dbus_conflict_cnt_s;
    uint32_t l2dbus_conflict_cnt_e;
    uint32_t l2dbus_nxtlvl_cnt_s; // 访问下一级存储(flash/psram)计数
    uint32_t l2dbus_nxtlvl_cnt_e;
    uint32_t l1dbus_hit_cnt_s;
    uint32_t l1dbus_hit_cnt_e;
    uint32_t l1dbus_miss_cnt_s;
    uint32_t l1dbus_miss_cnt_e;
    uint32_t l1dbus_conflict_cnt_s;
    uint32_t l1dbus_conflict_cnt_e;
    uint32_t l1dbus_nxtlvl_cnt_s;
    uint32_t l1dbus_nxtlvl_cnt_e;

    uint32_t l2ibus_hit_cnt_s;
    uint32_t l2ibus_hit_cnt_e;
    uint32_t l2ibus_miss_cnt_s;
    uint32_t l2ibus_miss_cnt_e;
    uint32_t l2ibus_nxtlvl_cnt_s;
    uint32_t l2ibus_nxtlvl_cnt_e;
    uint32_t l1ibus_hit_cnt_s;
    uint32_t l1ibus_hit_cnt_e;
    uint32_t l1ibus_miss_cnt_s;
    uint32_t l1ibus_miss_cnt_e;
    uint32_t l1ibus_nxtlvl_cnt_s;
    uint32_t l1ibus_nxtlvl_cnt_e;
#endif

public:
    /**
     * @brief Construct a new Latency object.
     *
     * @param size
     */
    Latency(const uint32_t size = 1) : size(size), period(0), sum(0), count(0), next(0)
    {
        this->queue = (this->size > 1) ? (uint32_t *)calloc(this->size, sizeof(uint32_t)) : NULL;
#if CONFIG_ESP32P4_BOOST && DL_LOG_CACHE_COUNT
        REG_WRITE(CACHE_L1_CACHE_ACS_CNT_CTRL_REG, ~0);
        REG_WRITE(CACHE_L2_CACHE_ACS_CNT_CTRL_REG, ~0);
#endif
    }

    /**
     * @brief Destroy the Latency object.
     *
     */
    ~Latency()
    {
        if (this->queue)
            free(this->queue);
    }

    /**
     * @brief Record the start timestamp.
     *
     */
    void start()
    {
#if DL_LOG_LATENCY_UNIT
        this->timestamp = get_cycle();
#if CONFIG_ESP32P4_BOOST && DL_LOG_CACHE_COUNT
        this->l2dbus_hit_cnt_s = REG_READ(L2_DCACHE_ACS_HIT_CNT_REG_n(xPortGetCoreID()));
        this->l2dbus_miss_cnt_s = REG_READ(L2_DCACHE_ACS_MISS_CNT_REG_n(xPortGetCoreID()));
        this->l2dbus_conflict_cnt_s = REG_READ(L2_DCACHE_ACS_CONFLICT_CNT_REG_n(xPortGetCoreID()));
        this->l2dbus_nxtlvl_cnt_s = REG_READ(L2_DCACHE_ACS_NXTLVL_CNT_REG_n(xPortGetCoreID()));

        this->l1dbus_hit_cnt_s = REG_READ(L1_DCACHE_ACS_HIT_CNT_REG_n(xPortGetCoreID()));
        this->l1dbus_miss_cnt_s = REG_READ(L1_DCACHE_ACS_MISS_CNT_REG_n(xPortGetCoreID()));
        this->l1dbus_conflict_cnt_s = REG_READ(L1_DCACHE_ACS_CONFLICT_CNT_REG_n(xPortGetCoreID()));
        this->l1dbus_nxtlvl_cnt_s = REG_READ(L1_DCACHE_ACS_NXTLVL_CNT_REG_n(xPortGetCoreID()));

        this->l2ibus_hit_cnt_s = REG_READ(L2_ICACHE_ACS_HIT_CNT_REG_n(xPortGetCoreID()));
        this->l2ibus_miss_cnt_s = REG_READ(L2_ICACHE_ACS_MISS_CNT_REG_n(xPortGetCoreID()));
        this->l2ibus_nxtlvl_cnt_s = REG_READ(L2_ICACHE_ACS_NXTLVL_CNT_REG_n(xPortGetCoreID()));

        this->l1ibus_hit_cnt_s = REG_READ(L1_ICACHE_ACS_HIT_CNT_REG_n(xPortGetCoreID()));
        this->l1ibus_miss_cnt_s = REG_READ(L1_ICACHE_ACS_MISS_CNT_REG_n(xPortGetCoreID()));
        this->l1ibus_nxtlvl_cnt_s = REG_READ(L1_ICACHE_ACS_NXTLVL_CNT_REG_n(xPortGetCoreID()));
#endif
#else
        this->timestamp = esp_timer_get_time();
#endif
    }

    /**
     * @brief Record the period.
     *
     */
    void end()
    {
#if DL_LOG_LATENCY_UNIT
        this->period = get_cycle() - this->timestamp;
#if CONFIG_ESP32P4_BOOST && DL_LOG_CACHE_COUNT
        this->l2dbus_hit_cnt_e = REG_READ(L2_DCACHE_ACS_HIT_CNT_REG_n(xPortGetCoreID()));
        this->l2dbus_miss_cnt_e = REG_READ(L2_DCACHE_ACS_MISS_CNT_REG_n(xPortGetCoreID()));
        this->l2dbus_conflict_cnt_e = REG_READ(L2_DCACHE_ACS_CONFLICT_CNT_REG_n(xPortGetCoreID()));
        this->l2dbus_nxtlvl_cnt_e = REG_READ(L2_DCACHE_ACS_NXTLVL_CNT_REG_n(xPortGetCoreID()));

        this->l1dbus_hit_cnt_e = REG_READ(L1_DCACHE_ACS_HIT_CNT_REG_n(xPortGetCoreID()));
        this->l1dbus_miss_cnt_e = REG_READ(L1_DCACHE_ACS_MISS_CNT_REG_n(xPortGetCoreID()));
        this->l1dbus_conflict_cnt_e = REG_READ(L1_DCACHE_ACS_CONFLICT_CNT_REG_n(xPortGetCoreID()));
        this->l1dbus_nxtlvl_cnt_e = REG_READ(L1_DCACHE_ACS_NXTLVL_CNT_REG_n(xPortGetCoreID()));

        this->l2ibus_hit_cnt_e = REG_READ(L2_ICACHE_ACS_HIT_CNT_REG_n(xPortGetCoreID()));
        this->l2ibus_miss_cnt_e = REG_READ(L2_ICACHE_ACS_MISS_CNT_REG_n(xPortGetCoreID()));
        this->l2ibus_nxtlvl_cnt_e = REG_READ(L2_ICACHE_ACS_NXTLVL_CNT_REG_n(xPortGetCoreID()));

        this->l1ibus_hit_cnt_e = REG_READ(L1_ICACHE_ACS_HIT_CNT_REG_n(xPortGetCoreID()));
        this->l1ibus_miss_cnt_e = REG_READ(L1_ICACHE_ACS_MISS_CNT_REG_n(xPortGetCoreID()));
        this->l1ibus_nxtlvl_cnt_e = REG_READ(L1_ICACHE_ACS_NXTLVL_CNT_REG_n(xPortGetCoreID()));
#endif
#else
        this->period = esp_timer_get_time() - this->timestamp;
#endif
        if (this->queue) {
            this->sum -= this->queue[this->next];
            this->queue[this->next] = this->period;
            this->sum += this->queue[this->next];
            this->next++;
            this->next = this->next % this->size;
            if (this->count < this->size) {
                this->count++;
            }
        }
    }

    /**
     * @brief Return the period.
     *
     * @return this->timestamp_end - this->timestamp
     */
    uint32_t get_period() { return this->period; }

    /**
     * @brief Get the average period.
     *
     * @return average latency
     */
    uint32_t get_average_period() { return this->queue ? (this->sum / this->count) : this->period; }

    /**
     * @brief Clear the period
     *
     */
    void clear_period() { this->period = 0; }

    /**
     * @brief Print in format "latency: {this->period} {unit}\n".
     */
    void print()
    {
#if DL_LOG_LATENCY_UNIT
        printf("latency: %15lu cycle\n", this->get_average_period());
#else
        printf("latency: %15lu us\n", this->get_average_period());
#endif
    }

    /**
     * @brief Print in format "{message}: {this->period} {unit}\n".
     *
     * @param message message of print
     */
    void print(const char *message)
    {
#if DL_LOG_LATENCY_UNIT
        printf("%s: %15lu cycle\n", message, this->get_average_period());
#else
        printf("%s: %15lu us\n", message, this->get_average_period());
#endif
    }

    /**
     * @brief Log latency info.
     *
     * @param prefix TAG of ESP_LOGI
     * @param key    fomat str of ESP_LOGI
     */
    void print(const char *prefix, const char *key)
    {
        if (!prefix)
            prefix = "";

#if DL_LOG_LATENCY_UNIT
        ESP_LOGI(prefix, "%s: %lu cycle", key, this->get_average_period());
#if CONFIG_ESP32P4_BOOST && DL_LOG_CACHE_COUNT
        ESP_LOGI(prefix, "%s: l2 dcache, hit cnt: %lu", key, this->l2dbus_hit_cnt_e - this->l2dbus_hit_cnt_s);
        ESP_LOGI(prefix, "%s: l2 dcache, miss cnt: %lu", key, this->l2dbus_miss_cnt_e - this->l2dbus_miss_cnt_s);
        ESP_LOGI(
            prefix, "%s: l2 dcache, conflict cnt: %lu", key, this->l2dbus_conflict_cnt_e - this->l2dbus_conflict_cnt_s);
        ESP_LOGI(prefix, "%s: l2 dcache, nxtlvl cnt: %lu", key, this->l2dbus_nxtlvl_cnt_e - this->l2dbus_nxtlvl_cnt_s);
        ESP_LOGI(prefix, "%s: l1 dcache, hit cnt: %lu", key, this->l1dbus_hit_cnt_e - this->l1dbus_hit_cnt_s);
        ESP_LOGI(prefix, "%s: l1 dcache, miss cnt: %lu", key, this->l1dbus_miss_cnt_e - this->l1dbus_miss_cnt_s);
        ESP_LOGI(
            prefix, "%s: l1 dcache, conflict cnt: %lu", key, this->l1dbus_conflict_cnt_e - this->l1dbus_conflict_cnt_s);
        ESP_LOGI(prefix, "%s: l1 dcache, nxtlvl cnt: %lu", key, this->l1dbus_nxtlvl_cnt_e - this->l1dbus_nxtlvl_cnt_s);
        ESP_LOGI(prefix, "%s: l2 icache, hit cnt: %lu", key, this->l2ibus_hit_cnt_e - this->l2ibus_hit_cnt_s);
        ESP_LOGI(prefix, "%s: l2 icache, miss cnt: %lu", key, this->l2ibus_miss_cnt_e - this->l2ibus_miss_cnt_s);
        ESP_LOGI(prefix, "%s: l2 icache, nxtlvl cnt: %lu", key, this->l2ibus_nxtlvl_cnt_e - this->l2ibus_nxtlvl_cnt_s);
        ESP_LOGI(prefix, "%s: l1 icache, hit cnt: %lu", key, this->l1ibus_hit_cnt_e - this->l1ibus_hit_cnt_s);
        ESP_LOGI(prefix, "%s: l1 icache, miss cnt: %lu", key, this->l1ibus_miss_cnt_e - this->l1ibus_miss_cnt_s);
        ESP_LOGI(prefix, "%s: l1 icache, nxtlvl cnt: %lu", key, this->l1ibus_nxtlvl_cnt_e - this->l1ibus_nxtlvl_cnt_s);
#endif
#else
        ESP_LOGI(prefix, "%s: %lu us", key, this->get_average_period());
#endif
    }
};
} // namespace tool
} // namespace dl
