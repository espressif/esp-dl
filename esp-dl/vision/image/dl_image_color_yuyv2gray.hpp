#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_isa.hpp"
#include "dl_image_norm_quant.hpp"
#include <type_traits>

namespace dl {
namespace image {
template <bool YUYVOrUYVY, typename ExtraProcess = void>
struct YUYV2Gray;

template <bool YUYVOrUYVY>
struct YUYV2Gray<YUYVOrUYVY, void> {
    static constexpr int src_bpp = 2;
    static constexpr int dst_bpp = 1;

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        if constexpr (YUYVOrUYVY) {
            *dst = src[0];
        } else {
            *dst = src[1];
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (YUYVOrUYVY) {
            cvt_color_simd_helper_yuyv2gray(src, dst, n);
        } else {
            cvt_color_simd_helper_uyvy2gray(src, dst, n);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (YUYVOrUYVY) {
            resize_nn_simd_helper_yuyv2gray(&src, offsets, dst, n);
        } else {
            resize_nn_simd_helper_uyvy2gray(&src, offsets, dst, n);
        }
    }
};

template <bool YUYVOrUYVY, typename ExtraProcess>
    requires std::same_as<ExtraProcess, NormQuant<int8_t, 1>> || std::same_as<ExtraProcess, NormQuant<int16_t, 1>>
struct YUYV2Gray<YUYVOrUYVY, ExtraProcess> {
    using QuantType = ExtraProcess::QT;
    static constexpr int src_bpp = 2;
    static constexpr int dst_bpp = ExtraProcess::channels * static_cast<int>(sizeof(QuantType));
    const ExtraProcess &m_extra_process;

    YUYV2Gray(const ExtraProcess &extra_process) : m_extra_process(extra_process) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        if constexpr (YUYVOrUYVY) {
            *dst_data = m_extra_process.norm_quant_chn1(src[0]);
        } else {
            *dst_data = m_extra_process.norm_quant_chn1(src[1]);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (YUYVOrUYVY) {
                cvt_color_simd_helper_yuyv2gray_qint8(src, dst, n, m_extra_process.m_lut_32);
            } else {
                cvt_color_simd_helper_uyvy2gray_qint8(src, dst, n, m_extra_process.m_lut_32);
            }
        } else {
            if constexpr (YUYVOrUYVY) {
                cvt_color_simd_helper_yuyv2gray_qint16(src, dst, n, m_extra_process.m_lut_32);
            } else {
                cvt_color_simd_helper_uyvy2gray_qint16(src, dst, n, m_extra_process.m_lut_32);
            }
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (YUYVOrUYVY) {
                resize_nn_simd_helper_yuyv2gray_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else {
                resize_nn_simd_helper_uyvy2gray_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
            }
        } else {
            if constexpr (YUYVOrUYVY) {
                resize_nn_simd_helper_yuyv2gray_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else {
                resize_nn_simd_helper_uyvy2gray_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
            }
        }
    }
};

template <typename T>
inline constexpr bool is_yuyv2gray_v = false;

template <bool YUYVOrUYVY, typename ExtraProcess>
inline constexpr bool is_yuyv2gray_v<YUYV2Gray<YUYVOrUYVY, ExtraProcess>> = true;
} // namespace image
} // namespace dl
