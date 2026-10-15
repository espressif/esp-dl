#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_isa.hpp"
#include "dl_image_norm_quant.hpp"
#include <type_traits>

namespace dl {
namespace image {
template <bool YUYVOrUYVY, bool RGBSwap, typename ExtraProcess = void>
struct YUYV2RGB888;

template <bool YUYVOrUYVY, bool RGBSwap>
struct YUYV2RGB888<YUYVOrUYVY, RGBSwap, void> {
    static constexpr int src_bpp = 2;
    static constexpr int dst_bpp = 3;

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t y1, u, y2, v;
        get_yuv<YUYVOrUYVY>(src, &y1, &u, &y2, &v);
        if constexpr (RGBSwap) {
            yuv2rgb888(y1, u, v, dst + 2, dst + 1, dst);
            yuv2rgb888(y2, u, v, dst + 5, dst + 4, dst + 3);
        } else {
            yuv2rgb888(y1, u, v, dst, dst + 1, dst + 2);
            yuv2rgb888(y2, u, v, dst + 3, dst + 4, dst + 5);
        }
    }

    void operator()(const uint8_t *src, uint8_t *dst, bool odd) const
    {
        uint8_t y, u, v;
        get_yuv<YUYVOrUYVY>(src, odd, &y, &u, &v);
        if constexpr (RGBSwap) {
            yuv2rgb888(y, u, v, dst + 2, dst + 1, dst);
        } else {
            yuv2rgb888(y, u, v, dst, dst + 1, dst + 2);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (YUYVOrUYVY) {
            if constexpr (RGBSwap) {
                cvt_color_simd_helper_yuyv2bgr888(src, dst, n);
            } else {
                cvt_color_simd_helper_yuyv2rgb888(src, dst, n);
            }
        } else if constexpr (RGBSwap) {
            cvt_color_simd_helper_uyvy2bgr888(src, dst, n);
        } else {
            cvt_color_simd_helper_uyvy2rgb888(src, dst, n);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (YUYVOrUYVY) {
            if constexpr (RGBSwap) {
                resize_nn_simd_helper_yuyv2bgr888(&src, offsets, dst, n);
            } else {
                resize_nn_simd_helper_yuyv2rgb888(&src, offsets, dst, n);
            }
        } else if constexpr (RGBSwap) {
            resize_nn_simd_helper_uyvy2bgr888(&src, offsets, dst, n);
        } else {
            resize_nn_simd_helper_uyvy2rgb888(&src, offsets, dst, n);
        }
    }
};

template <bool YUYVOrUYVY, bool RGBSwap, typename ExtraProcess>
    requires std::same_as<ExtraProcess, NormQuant<int8_t, 3>> || std::same_as<ExtraProcess, NormQuant<int16_t, 3>>
struct YUYV2RGB888<YUYVOrUYVY, RGBSwap, ExtraProcess> {
    using QuantType = ExtraProcess::QT;
    static constexpr int src_bpp = 2;
    static constexpr int dst_bpp = ExtraProcess::channels * static_cast<int>(sizeof(QuantType));
    const ExtraProcess &m_extra_process;

    YUYV2RGB888(const ExtraProcess &extra_process) : m_extra_process(extra_process) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t y1, u, y2, v;
        get_yuv<YUYVOrUYVY>(src, &y1, &u, &y2, &v);
        uint8_t r, g, b;
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        if constexpr (RGBSwap) {
            yuv2rgb888(y1, u, v, &b, &g, &r);
            dst_data[0] = m_extra_process.norm_quant_chn1(r);
            dst_data[1] = m_extra_process.norm_quant_chn2(g);
            dst_data[2] = m_extra_process.norm_quant_chn3(b);
            yuv2rgb888(y2, u, v, &b, &g, &r);
            dst_data[3] = m_extra_process.norm_quant_chn1(r);
            dst_data[4] = m_extra_process.norm_quant_chn2(g);
            dst_data[5] = m_extra_process.norm_quant_chn3(b);
        } else {
            yuv2rgb888(y1, u, v, &r, &g, &b);
            dst_data[0] = m_extra_process.norm_quant_chn1(r);
            dst_data[1] = m_extra_process.norm_quant_chn2(g);
            dst_data[2] = m_extra_process.norm_quant_chn3(b);
            yuv2rgb888(y2, u, v, &r, &g, &b);
            dst_data[3] = m_extra_process.norm_quant_chn1(r);
            dst_data[4] = m_extra_process.norm_quant_chn2(g);
            dst_data[5] = m_extra_process.norm_quant_chn3(b);
        }
    }

    void operator()(const uint8_t *src, uint8_t *dst, bool odd) const
    {
        uint8_t y, u, v;
        get_yuv<YUYVOrUYVY>(src, odd, &y, &u, &v);
        uint8_t r, g, b;
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        if constexpr (RGBSwap) {
            yuv2rgb888(y, u, v, &b, &g, &r);
        } else {
            yuv2rgb888(y, u, v, &r, &g, &b);
        }
        dst_data[0] = m_extra_process.norm_quant_chn1(r);
        dst_data[1] = m_extra_process.norm_quant_chn2(g);
        dst_data[2] = m_extra_process.norm_quant_chn3(b);
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (YUYVOrUYVY) {
                if constexpr (RGBSwap) {
                    cvt_color_simd_helper_yuyv2bgr888_qint8(src, dst, n, m_extra_process.m_lut_32);
                } else {
                    cvt_color_simd_helper_yuyv2rgb888_qint8(src, dst, n, m_extra_process.m_lut_32);
                }
            } else if constexpr (RGBSwap) {
                cvt_color_simd_helper_uyvy2bgr888_qint8(src, dst, n, m_extra_process.m_lut_32);
            } else {
                cvt_color_simd_helper_uyvy2rgb888_qint8(src, dst, n, m_extra_process.m_lut_32);
            }
        } else if constexpr (YUYVOrUYVY) {
            if constexpr (RGBSwap) {
                cvt_color_simd_helper_yuyv2bgr888_qint16(src, dst, n, m_extra_process.m_lut_32);
            } else {
                cvt_color_simd_helper_yuyv2rgb888_qint16(src, dst, n, m_extra_process.m_lut_32);
            }
        } else if constexpr (RGBSwap) {
            cvt_color_simd_helper_uyvy2bgr888_qint16(src, dst, n, m_extra_process.m_lut_32);
        } else {
            cvt_color_simd_helper_uyvy2rgb888_qint16(src, dst, n, m_extra_process.m_lut_32);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (YUYVOrUYVY) {
                if constexpr (RGBSwap) {
                    resize_nn_simd_helper_yuyv2bgr888_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
                } else {
                    resize_nn_simd_helper_yuyv2rgb888_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
                }
            } else if constexpr (RGBSwap) {
                resize_nn_simd_helper_uyvy2bgr888_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else {
                resize_nn_simd_helper_uyvy2rgb888_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
            }
        } else if constexpr (YUYVOrUYVY) {
            if constexpr (RGBSwap) {
                resize_nn_simd_helper_yuyv2bgr888_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else {
                resize_nn_simd_helper_yuyv2rgb888_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
            }
        } else if constexpr (RGBSwap) {
            resize_nn_simd_helper_uyvy2bgr888_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
        } else {
            resize_nn_simd_helper_uyvy2rgb888_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
        }
    }
};

template <typename T>
inline constexpr bool is_yuyv2rgb888_v = false;

template <bool YUYVOrUYVY, bool RGBSwap, typename ExtraProcess>
inline constexpr bool is_yuyv2rgb888_v<YUYV2RGB888<YUYVOrUYVY, RGBSwap, ExtraProcess>> = true;
} // namespace image
} // namespace dl
