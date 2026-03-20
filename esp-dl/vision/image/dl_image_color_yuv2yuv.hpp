#pragma once
#include "dl_image_color_isa.hpp"

namespace dl {
namespace image {

template <bool ByteSwap>
struct YUV2YUV {
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        if constexpr (ByteSwap) {
            const uint16_t *src_data = reinterpret_cast<const uint16_t *>(src);
            uint16_t *dst_data = reinterpret_cast<uint16_t *>(dst);
            *dst_data = __builtin_bswap16(*src_data);
        } else {
            memcpy(dst, src, 2);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const {}

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const {}
};

template <typename T>
inline constexpr bool is_yuv2yuv_v = false;

template <bool ByteSwap>
inline constexpr bool is_yuv2yuv_v<YUV2YUV<ByteSwap>> = true;
} // namespace image
} // namespace dl
