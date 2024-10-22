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
    if (rounded == (float)int_part) {
        if (int_part % 2 != 0) {
            if (value < 0)
                int_part++;
            else
                int_part--;
        }
    }
    return int_part;
#endif
}

int round_half_up(float value)
{
    return (int)floorf(value + 0.5);
}

int round_down(float value)
{
    return (int)floorf(value);
}

int round(float value)
{
#if CONFIG_IDF_TARGET_ESP32P4
    return round_half_even(value);
#else
    return round_half_up(value);
#endif
}

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
