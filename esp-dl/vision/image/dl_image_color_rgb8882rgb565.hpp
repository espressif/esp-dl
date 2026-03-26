#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_isa.hpp"

namespace dl {
namespace image {
template <bool RGB565BE, bool RGBSwap>
struct RGB8882RGB565 {
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint16_t *dst_data = reinterpret_cast<uint16_t *>(dst);
        if constexpr (RGB565BE && RGBSwap) {
            *dst_data = __builtin_bswap16(rgb8882rgb565(src[2], src[1], src[0]));
        } else if constexpr (RGB565BE && !RGBSwap) {
            *dst_data = __builtin_bswap16(rgb8882rgb565(src[0], src[1], src[2]));
        } else if constexpr (!RGB565BE && RGBSwap) {
            *dst_data = rgb8882rgb565(src[2], src[1], src[0]);
        } else {
            *dst_data = rgb8882rgb565(src[0], src[1], src[2]);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            cvt_color_simd_helper_bgr8882rgb565be(src, dst, n);
        } else if constexpr (RGB565BE && !RGBSwap) {
            cvt_color_simd_helper_rgb8882rgb565be(src, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap) {
            cvt_color_simd_helper_bgr8882rgb565le(src, dst, n);
        } else {
            cvt_color_simd_helper_rgb8882rgb565le(src, dst, n);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            resize_nn_simd_helper_bgr8882rgb565be(&src, offsets, dst, n);
        } else if constexpr (RGB565BE && !RGBSwap) {
            resize_nn_simd_helper_rgb8882rgb565be(&src, offsets, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap) {
            resize_nn_simd_helper_bgr8882rgb565le(&src, offsets, dst, n);
        } else {
            resize_nn_simd_helper_rgb8882rgb565le(&src, offsets, dst, n);
        }
    }
};
} // namespace image
} // namespace dl
