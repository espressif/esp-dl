#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_isa.hpp"

namespace dl {
namespace image {
template <bool YUYVOrUYVY, bool RGB565BE, bool RGBSwap>
struct YUYV2RGB565 {
    static constexpr int src_bpp = 2;
    static constexpr int dst_bpp = 2;

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t y1, u, y2, v;
        get_yuv<YUYVOrUYVY>(src, &y1, &u, &y2, &v);
        uint8_t r, g, b;
        uint16_t *dst_data = reinterpret_cast<uint16_t *>(dst);
        if constexpr (RGB565BE && RGBSwap) {
            yuv2rgb888(y1, u, v, &b, &g, &r);
            dst_data[0] = __builtin_bswap16(rgb8882rgb565(r, g, b));
            yuv2rgb888(y2, u, v, &b, &g, &r);
            dst_data[1] = __builtin_bswap16(rgb8882rgb565(r, g, b));
        } else if constexpr (RGB565BE && !RGBSwap) {
            yuv2rgb888(y1, u, v, &r, &g, &b);
            dst_data[0] = __builtin_bswap16(rgb8882rgb565(r, g, b));
            yuv2rgb888(y2, u, v, &r, &g, &b);
            dst_data[1] = __builtin_bswap16(rgb8882rgb565(r, g, b));
        } else if constexpr (!RGB565BE && RGBSwap) {
            yuv2rgb888(y1, u, v, &b, &g, &r);
            dst_data[0] = rgb8882rgb565(r, g, b);
            yuv2rgb888(y2, u, v, &b, &g, &r);
            dst_data[1] = rgb8882rgb565(r, g, b);
        } else {
            yuv2rgb888(y1, u, v, &r, &g, &b);
            dst_data[0] = rgb8882rgb565(r, g, b);
            yuv2rgb888(y2, u, v, &r, &g, &b);
            dst_data[1] = rgb8882rgb565(r, g, b);
        }
    }

    void operator()(const uint8_t *src, uint8_t *dst, bool odd) const
    {
        uint8_t y, u, v;
        get_yuv<YUYVOrUYVY>(src, odd, &y, &u, &v);
        uint8_t r, g, b;
        uint16_t *dst_data = reinterpret_cast<uint16_t *>(dst);
        if constexpr (RGB565BE && RGBSwap) {
            yuv2rgb888(y, u, v, &b, &g, &r);
            *dst_data = __builtin_bswap16(rgb8882rgb565(r, g, b));
        } else if constexpr (RGB565BE && !RGBSwap) {
            yuv2rgb888(y, u, v, &r, &g, &b);
            *dst_data = __builtin_bswap16(rgb8882rgb565(r, g, b));
        } else if constexpr (!RGB565BE && RGBSwap) {
            yuv2rgb888(y, u, v, &b, &g, &r);
            *dst_data = rgb8882rgb565(r, g, b);
        } else {
            yuv2rgb888(y, u, v, &r, &g, &b);
            *dst_data = rgb8882rgb565(r, g, b);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (YUYVOrUYVY && RGB565BE && RGBSwap) {
            cvt_color_simd_helper_yuyv2bgr565be(src, dst, n);
        } else if constexpr (YUYVOrUYVY && RGB565BE && !RGBSwap) {
            cvt_color_simd_helper_yuyv2rgb565be(src, dst, n);
        } else if constexpr (YUYVOrUYVY && !RGB565BE && RGBSwap) {
            cvt_color_simd_helper_yuyv2bgr565le(src, dst, n);
        } else if constexpr (YUYVOrUYVY && !RGB565BE && !RGBSwap) {
            cvt_color_simd_helper_yuyv2rgb565le(src, dst, n);
        } else if constexpr (!YUYVOrUYVY && RGB565BE && RGBSwap) {
            cvt_color_simd_helper_uyvy2bgr565be(src, dst, n);
        } else if constexpr (!YUYVOrUYVY && RGB565BE && !RGBSwap) {
            cvt_color_simd_helper_uyvy2rgb565be(src, dst, n);
        } else if constexpr (!YUYVOrUYVY && !RGB565BE && RGBSwap) {
            cvt_color_simd_helper_uyvy2bgr565le(src, dst, n);
        } else {
            cvt_color_simd_helper_uyvy2rgb565le(src, dst, n);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (YUYVOrUYVY && RGB565BE && RGBSwap) {
            resize_nn_simd_helper_yuyv2bgr565be(&src, offsets, dst, n);
        } else if constexpr (YUYVOrUYVY && RGB565BE && !RGBSwap) {
            resize_nn_simd_helper_yuyv2rgb565be(&src, offsets, dst, n);
        } else if constexpr (YUYVOrUYVY && !RGB565BE && RGBSwap) {
            resize_nn_simd_helper_yuyv2bgr565le(&src, offsets, dst, n);
        } else if constexpr (YUYVOrUYVY && !RGB565BE && !RGBSwap) {
            resize_nn_simd_helper_yuyv2rgb565le(&src, offsets, dst, n);
        } else if constexpr (!YUYVOrUYVY && RGB565BE && RGBSwap) {
            resize_nn_simd_helper_uyvy2bgr565be(&src, offsets, dst, n);
        } else if constexpr (!YUYVOrUYVY && RGB565BE && !RGBSwap) {
            resize_nn_simd_helper_uyvy2rgb565be(&src, offsets, dst, n);
        } else if constexpr (!YUYVOrUYVY && !RGB565BE && RGBSwap) {
            resize_nn_simd_helper_uyvy2bgr565le(&src, offsets, dst, n);
        } else {
            resize_nn_simd_helper_uyvy2rgb565le(&src, offsets, dst, n);
        }
    }
};

template <typename T>
inline constexpr bool is_yuyv2rgb565_v = false;

template <bool YUYVOrUYVY, bool RGB565BE, bool RGBSwap>
inline constexpr bool is_yuyv2rgb565_v<YUYV2RGB565<YUYVOrUYVY, RGB565BE, RGBSwap>> = true;
} // namespace image
} // namespace dl
