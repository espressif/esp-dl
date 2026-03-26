#pragma once
#include "dl_image_color_isa.hpp"

namespace dl {
namespace image {
template <bool RGB565BE, bool RGBSwap, bool ByteSwap>
struct RGB5652RGB565 {
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        const uint16_t *src_data = reinterpret_cast<const uint16_t *>(src);
        uint16_t *dst_data = reinterpret_cast<uint16_t *>(dst);
        if constexpr (!RGBSwap && ByteSwap) {
            *dst_data = __builtin_bswap16(*src_data);
        } else if constexpr (!RGBSwap && !ByteSwap) {
            *dst_data = *src_data;
        } else if constexpr (RGB565BE && RGBSwap && ByteSwap) {
            *dst_data = static_cast<uint16_t>(((*src_data & 0xf8) >> 3) | ((*src_data & 0xe000) >> 8) |
                                              ((*src_data & 0x7) << 8) | ((*src_data & 0x1f00) << 3));
        } else if constexpr (!RGB565BE && RGBSwap && ByteSwap) {
            *dst_data = static_cast<uint16_t>(((*src_data & 0xf800) >> 3) | ((*src_data & 0x700) >> 8) |
                                              ((*src_data & 0xe0) << 8) | ((*src_data & 0x1f) << 3));
        } else if constexpr (RGB565BE && RGBSwap && !ByteSwap) {
            *dst_data =
                static_cast<uint16_t>(((*src_data & 0xf8) << 5) | (*src_data & 0xe007) | ((*src_data & 0x1f00) >> 5));
        } else if constexpr (!RGB565BE && RGBSwap && !ByteSwap) {
            *dst_data =
                static_cast<uint16_t>(((*src_data & 0xf800) >> 11) | (*src_data & 0x7e0) | ((*src_data & 0x1f) << 11));
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (!RGBSwap && ByteSwap) {
            cvt_color_simd_helper_rgb565le2rgb565be(src, dst, n);
        } else if constexpr (!RGBSwap && !ByteSwap) {
            cvt_color_simd_helper_rgb5652rgb565(src, dst, n);
        } else if constexpr (RGB565BE && RGBSwap && ByteSwap) {
            cvt_color_simd_helper_rgb565be2bgr565le(src, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap && ByteSwap) {
            cvt_color_simd_helper_rgb565le2bgr565be(src, dst, n);
        } else if constexpr (RGB565BE && RGBSwap && !ByteSwap) {
            cvt_color_simd_helper_rgb565be2bgr565be(src, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap && !ByteSwap) {
            cvt_color_simd_helper_rgb565le2bgr565le(src, dst, n);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (!RGBSwap && ByteSwap) {
            resize_nn_simd_helper_rgb565le2rgb565be(&src, offsets, dst, n);
        } else if constexpr (!RGBSwap && !ByteSwap) {
            resize_nn_simd_helper_rgb5652rgb565(&src, offsets, dst, n);
        } else if constexpr (RGB565BE && RGBSwap && ByteSwap) {
            resize_nn_simd_helper_rgb565be2bgr565le(&src, offsets, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap && ByteSwap) {
            resize_nn_simd_helper_rgb565le2bgr565be(&src, offsets, dst, n);
        } else if constexpr (RGB565BE && RGBSwap && !ByteSwap) {
            resize_nn_simd_helper_rgb565be2bgr565be(&src, offsets, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap && !ByteSwap) {
            resize_nn_simd_helper_rgb565le2bgr565le(&src, offsets, dst, n);
        }
    }
};
} // namespace image
} // namespace dl
