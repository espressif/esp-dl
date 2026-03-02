#pragma once
#include "esp_heap_caps.h"
#include <algorithm>
#include <cstdint>

namespace dl {
namespace image {
inline uint8_t extract_channel1_from_rgb565le(uint16_t x)
{
    return static_cast<uint8_t>((x & 0xF800) >> 8);
}

inline uint8_t extract_channel2_from_rgb565le(uint16_t x)
{
    return static_cast<uint8_t>((x & 0x7E0) >> 3);
}

inline uint8_t extract_channel3_from_rgb565le(uint16_t x)
{
    return static_cast<uint8_t>((x & 0x1F) << 3);
}

inline uint8_t extract_channel1_from_rgb565be(uint16_t x)
{
    return static_cast<uint8_t>(x & 0xF8);
}

inline uint8_t extract_channel2_from_rgb565be(uint16_t x)
{
    return static_cast<uint8_t>(((x & 0x7) << 5) | ((x & 0xE000) >> 11));
}

inline uint8_t extract_channel3_from_rgb565be(uint16_t x)
{
    return static_cast<uint8_t>((x & 0x1F00) >> 5);
}

inline uint8_t rgb8882gray(uint8_t r, uint8_t g, uint8_t b)
{
    constexpr int coeff_r = 9798;
    constexpr int coeff_g = 19235;
    constexpr int coeff_b = 3735;
    constexpr int shift = 15;
    constexpr int round_delta = 1 << (shift - 1);
    return static_cast<uint8_t>((coeff_r * r + coeff_g * g + coeff_b * b + round_delta) >> shift);
}

inline uint16_t rgb8882rgb565(uint8_t r, uint8_t g, uint8_t b)
{
    return static_cast<uint16_t>(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}

struct HSVTablesSingleton {
    static constexpr int hsv_shift = 12;
    int *m_sdiv_table;
    int *m_hdiv_table;

protected:
    HSVTablesSingleton()
    {
        m_sdiv_table = (int *)heap_caps_aligned_alloc(4, 256 * sizeof(int), MALLOC_CAP_DEFAULT);
        m_hdiv_table = (int *)heap_caps_aligned_alloc(4, 256 * sizeof(int), MALLOC_CAP_DEFAULT);
        m_sdiv_table[0] = m_hdiv_table[0] = 0;
        for (int i = 1; i < 256; i++) {
            m_sdiv_table[i] = static_cast<int>((255 << hsv_shift) / (1. * i));
            m_hdiv_table[i] = static_cast<int>((180 << hsv_shift) / (6. * i));
        }
    }

public:
    static HSVTablesSingleton &get_instance()
    {
        static HSVTablesSingleton g_tables;
        return g_tables;
    }
};

inline void rgb8882hsv(const HSVTablesSingleton &hsv_tables,
                       uint8_t r_u8,
                       uint8_t g_u8,
                       uint8_t b_u8,
                       uint8_t *h_u8,
                       uint8_t *s_u8,
                       uint8_t *v_u8)
{
    constexpr int hsv_shift = HSVTablesSingleton::hsv_shift;
    constexpr int round_delta = 1 << (hsv_shift - 1);
    int r = r_u8, g = g_u8, b = b_u8;
    int h, s, v = b;
    int vmin = b;
    int vr, vg;

    v = std::max(std::max(v, g), r);
    vmin = std::min(std::min(vmin, g), r);

    uint8_t diff = (uint8_t)(v - vmin);
    vr = v == r ? -1 : 0;
    vg = v == g ? -1 : 0;

    s = (diff * hsv_tables.m_sdiv_table[v] + round_delta) >> hsv_shift;
    h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    h = (h * hsv_tables.m_hdiv_table[diff] + round_delta) >> hsv_shift;
    h += h < 0 ? 180 : 0;

    *h_u8 = (uint8_t)h;
    *s_u8 = (uint8_t)s;
    *v_u8 = (uint8_t)v;
}

inline bool is_valid_hsv_thr(const std::array<uint8_t, 3> &hsv_min, const std::array<uint8_t, 3> &hsv_max)
{
    bool h_across_zero = hsv_min[0] > hsv_max[0];
    if (h_across_zero) {
        return (hsv_min[0] > hsv_max[0] && hsv_min[0] <= 180) && (hsv_min[1] < hsv_max[1]) && (hsv_min[2] < hsv_max[2]);
    } else {
        return (hsv_min[0] < hsv_max[0] && hsv_max[0] <= 180) && (hsv_min[1] < hsv_max[1]) && (hsv_min[2] < hsv_max[2]);
    }
}
} // namespace image
} // namespace dl
