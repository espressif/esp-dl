#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_hsv2hsv_mask.hpp"
#include "dl_image_color_isa.hpp"
#include <concepts>

namespace dl {
namespace image {
template <bool YUYVOrUYVY, typename ExtraProcess = void>
struct YUV2HSV;

template <bool YUYVOrUYVY>
struct YUV2HSV<YUYVOrUYVY, void> {
    const HSVTablesSingleton &m_hsv_tables;

    YUV2HSV() : m_hsv_tables(HSVTablesSingleton::get_instance()) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t y1, u, y2, v;
        get_yuv<YUYVOrUYVY>(src, &y1, &u, &y2, &v);
        uint8_t r, g, b;
        yuv2rgb888(y1, u, v, &r, &g, &b);
        rgb8882hsv(m_hsv_tables, r, g, b, dst, dst + 1, dst + 2);
        yuv2rgb888(y2, u, v, &r, &g, &b);
        rgb8882hsv(m_hsv_tables, r, g, b, dst + 3, dst + 4, dst + 5);
    }

    void operator()(const uint8_t *src, uint8_t *dst, bool odd) const
    {
        uint8_t y, u, v;
        get_yuv<YUYVOrUYVY>(src, odd, &y, &u, &v);
        uint8_t r, g, b;
        yuv2rgb888(y, u, v, &r, &g, &b);
        rgb8882hsv(m_hsv_tables, r, g, b, dst, dst + 1, dst + 2);
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const {}

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const {}
};

template <bool YUYVOrUYVY, typename ExtraProcess>
    requires std::same_as<ExtraProcess, HSV2HSVMask<false>> || std::same_as<ExtraProcess, HSV2HSVMask<true>>
struct YUV2HSV<YUYVOrUYVY, ExtraProcess> {
    const HSVTablesSingleton &m_hsv_tables;
    const ExtraProcess &m_extra_process;

    YUV2HSV(const ExtraProcess &extra_process) :
        m_hsv_tables(HSVTablesSingleton::get_instance()), m_extra_process(extra_process)
    {
    }

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t y1, u, y2, v;
        get_yuv<YUYVOrUYVY>(src, &y1, &u, &y2, &v);
        uint8_t r, g, b;
        uint8_t hsv[3];
        yuv2rgb888(y1, u, v, &r, &g, &b);
        rgb8882hsv(m_hsv_tables, r, g, b, hsv, hsv + 1, hsv + 2);
        m_extra_process(hsv, dst);
        yuv2rgb888(y2, u, v, &r, &g, &b);
        rgb8882hsv(m_hsv_tables, r, g, b, hsv, hsv + 1, hsv + 2);
        m_extra_process(hsv, dst + 1);
    }

    void operator()(const uint8_t *src, uint8_t *dst, bool odd) const
    {
        uint8_t y, u, v;
        get_yuv<YUYVOrUYVY>(src, odd, &y, &u, &v);
        uint8_t r, g, b;
        uint8_t hsv[3];
        yuv2rgb888(y, u, v, &r, &g, &b);
        rgb8882hsv(m_hsv_tables, r, g, b, hsv, hsv + 1, hsv + 2);
        m_extra_process(hsv, dst);
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const {}

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const {}
};

template <typename T>
inline constexpr bool is_yuv2hsv_v = false;

template <bool YUYVOrUYVY, typename ExtraProcess>
inline constexpr bool is_yuv2hsv_v<YUV2HSV<YUYVOrUYVY, ExtraProcess>> = true;
} // namespace image
} // namespace dl
