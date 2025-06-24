#include "dl_tool.hpp"
#include <string.h>

extern "C" {
#if CONFIG_XTENSA_BOOST
void dl_xtensa_bzero_32b(void *ptr, const int n);
#endif

#if CONFIG_TIE728_BOOST
void dl_tie728_bzero_128b(void *ptr, const int n);
void dl_tie728_bzero(void *ptr, const int n);
void dl_tie728_memcpy(void *dst, const void *src, const size_t n);
#endif

#if CONFIG_ESP32P4_BOOST
void dl_esp32p4_memcpy(void *dst, const void *src, const size_t n);
#endif
}

namespace dl {
namespace tool {

int round_half_even(float value)
{
#if CONFIG_ESP32P4_BOOST
    return dl_esp32p4_round_half_even(value);
#else
    float rounded;
    if (value < 0) {
        rounded = value - 0.5f;
    } else {
        rounded = value + 0.5f;
    }

    int int_part = (int)rounded;
    if (fabsf(rounded - int_part) < 1e-6) {
        if ((int_part & 1) != 0) {
            if (value < 0)
                int_part++;
            else
                int_part--;
        }
    }
    return int_part;
#endif
}

int round_half_even(double value)
{
    double rounded;
    if (value < 0) {
        rounded = value - 0.5;
    } else {
        rounded = value + 0.5;
    }

    int int_part = (int)rounded;
    if (fabs(rounded - int_part) < 1e-6) {
        if ((int_part & 1) != 0) {
            if (value < 0)
                int_part++;
            else
                int_part--;
        }
    }
    return int_part;
}

int round_half_up(float value)
{
    return (int)floorf(value + 0.5);
}

int round_half_up(double value)
{
    return (int)floor(value + 0.5);
}

template <typename T>
int round(T value)
{
#if CONFIG_IDF_TARGET_ESP32P4
    return round_half_even(value);
#else
    return round_half_up(value);
#endif
}

template int round(float value);
template int round(double value);

template <typename T>
T shift_and_round_half_even(T value, int shift)
{
    T ret = 0;

    if (shift <= 0) {
        int64_t shifted = 0;
        shifted = static_cast<int64_t>(value) << -shift;
        tool::truncate(ret, shifted);
    } else {
        T shifted = value >> shift;
        T remainder = value & ((static_cast<T>(1) << shift) - 1);
        T half = (static_cast<T>(1)) << (shift - 1);

        if (remainder > half) {
            shifted += 1;
        } else if (remainder == half) {
            if ((shifted & 1) != 0) {
                shifted += 1;
            }
        }
        ret = shifted;
    }
    return ret;
}

template <typename T>
T shift_and_round_half_up(T value, int shift)
{
    int64_t shifted = 0;
    T ret = 0;

    if (shift <= 0) {
        shifted = (static_cast<int64_t>(value)) << -shift;
        tool::truncate(ret, shifted);
    } else {
        int64_t half = (static_cast<int64_t>(1)) << (shift - 1);
        shifted = (value + half) >> shift;
        ret = static_cast<T>(shifted);
    }
    return ret;
}

template <typename T>
T shift_and_round(T value, int shift)
{
#if CONFIG_IDF_TARGET_ESP32P4
    return shift_and_round_half_even(value, shift);
#else
    return shift_and_round_half_up(value, shift);
#endif
}

template int32_t shift_and_round(int32_t value, int shift);
template int64_t shift_and_round(int64_t value, int shift);

void set_zero(void *ptr, const int n)
{
#if CONFIG_TIE728_BOOST
    dl_tie728_bzero(ptr, n);
#else
    bzero(ptr, n);
#endif
}

void copy_memory(void *dst, void *src, const size_t n)
{
#if CONFIG_ESP32P4_BOOST
    dl_esp32p4_memcpy(dst, src, n);
#elif CONFIG_TIE728_BOOST
    dl_tie728_memcpy(dst, src, n);
#else
    memcpy(dst, src, n);
#endif
}

memory_addr_type_t memory_addr_type(void *address)
{
#if CONFIG_IDF_TARGET_ESP32P4
    if (esp_ptr_in_tcm(address)) {
        return MEMORY_ADDR_TCM;
    }
#endif

#if CONFIG_SPIRAM_RODATA || CONFIG_SPIRAM_XIP_FROM_PSRAM
    esp_paddr_t paddr;
    mmu_target_t target;
    ESP_ERROR_CHECK(esp_mmu_vaddr_to_paddr(address, &paddr, &target));
    if (target == MMU_TARGET_PSRAM0) {
        return MEMORY_ADDR_PSRAM;
    } else if (target == MMU_TARGET_FLASH0) {
        return MEMORY_ADDR_FLASH;
    }
#else
    if (esp_ptr_external_ram(address)) {
        return MEMORY_ADDR_PSRAM;
    } else if (esp_ptr_in_drom(address)) {
        return MEMORY_ADDR_FLASH;
    }
#endif
    else if (esp_ptr_internal(address)) {
        return MEMORY_ADDR_INTERNAL;
    } else {
        return MEMORY_ADDR_UKN;
    }
}

HEAP_IRAM_ATTR void *malloc_aligned(size_t alignment, size_t size, uint32_t caps)
{
    void *ret = heap_caps_aligned_alloc(alignment, size, caps);
#if CONFIG_IDF_TARGET_ESP32P4
    if (ret && !(caps & MALLOC_CAP_TCM) && esp_ptr_in_tcm(ret)) {
        // skip TCM
        heap_caps_free(ret);
        ret = heap_caps_aligned_alloc(alignment, size, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    }
#endif
    return ret;
}

HEAP_IRAM_ATTR void *calloc_aligned(size_t alignment, size_t n, size_t size, uint32_t caps)
{
    void *ret = heap_caps_aligned_calloc(alignment, n, size, caps);
#if CONFIG_IDF_TARGET_ESP32P4
    if (ret && !(caps & MALLOC_CAP_TCM) && esp_ptr_in_tcm(ret)) {
        // skip TCM
        heap_caps_free(ret);
        ret = heap_caps_aligned_calloc(alignment, n, size, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    }
#endif
    return ret;
}

float *gen_lut_8bit(float *table, int exponent, std::function<float(float)> func)
{
    if (table == nullptr) {
        return table;
    }
    float scale = DL_SCALE(exponent);
    for (int i = 0; i < 256; i++) {
        table[i] = func(scale * (i - 128));
    }
    return table;
}

} // namespace tool
} // namespace dl
