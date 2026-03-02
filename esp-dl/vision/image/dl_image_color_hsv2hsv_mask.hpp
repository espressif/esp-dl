#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_isa.hpp"
#include <array>

namespace dl {
namespace image {
template <bool HAcrossZero>
struct HSV2HSVMask {
    static constexpr bool h_across_zero = HAcrossZero;
    std::array<uint8_t, 3> m_hsv_min;
    std::array<uint8_t, 3> m_hsv_max;

    HSV2HSVMask(const std::array<uint8_t, 3> &hsv_min, const std::array<uint8_t, 3> &hsv_max) :
        m_hsv_min(hsv_min), m_hsv_max(hsv_max)
    {
        assert(is_valid_hsv_thr(hsv_min, hsv_max));
    }

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t h = src[0], s = src[1], v = src[2];
        if constexpr (HAcrossZero) {
            if ((h >= m_hsv_min[0] || h <= m_hsv_max[0]) && s >= m_hsv_min[1] && s <= m_hsv_max[1] &&
                v >= m_hsv_min[2] && v <= m_hsv_max[2]) {
                *dst = 255;
            } else {
                *dst = 0;
            }
        } else {
            if (h >= m_hsv_min[0] && h <= m_hsv_max[0] && s >= m_hsv_min[1] && s <= m_hsv_max[1] && v >= m_hsv_min[2] &&
                v <= m_hsv_max[2]) {
                *dst = 255;
            } else {
                *dst = 0;
            }
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (HAcrossZero) {
            cvt_color_simd_helper_hsv2hsv_mask1(src, dst, n, m_hsv_min.data(), m_hsv_max.data());
        } else {
            cvt_color_simd_helper_hsv2hsv_mask0(src, dst, n, m_hsv_min.data(), m_hsv_max.data());
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (HAcrossZero) {
            resize_nn_simd_helper_hsv2hsv_mask1(&src, offsets, dst, n, m_hsv_min.data(), m_hsv_max.data());
        } else {
            resize_nn_simd_helper_hsv2hsv_mask0(&src, offsets, dst, n, m_hsv_min.data(), m_hsv_max.data());
        }
    }
};
} // namespace image
} // namespace dl
