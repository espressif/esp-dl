#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_isa.hpp"
#include "dl_image_norm_quant.hpp"

namespace dl {
namespace image {
template <bool RGBSwap, typename ExtraProcess = void>
struct RGB8882Gray;

template <bool RGBSwap>
struct RGB8882Gray<RGBSwap, void> {
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        if constexpr (RGBSwap) {
            *dst = rgb8882gray(src[2], src[1], src[0]);
        } else {
            *dst = rgb8882gray(src[0], src[1], src[2]);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (RGBSwap) {
            cvt_color_simd_helper_bgr8882gray(src, dst, n);
        } else {
            cvt_color_simd_helper_rgb8882gray(src, dst, n);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (RGBSwap) {
            resize_nn_simd_helper_bgr8882gray(&src, offsets, dst, n);
        } else {
            resize_nn_simd_helper_rgb8882gray(&src, offsets, dst, n);
        }
    }
};
template <bool RGBSwap, typename ExtraProcess>
    requires std::same_as<ExtraProcess, NormQuant<int8_t, 1>> || std::same_as<ExtraProcess, NormQuant<int16_t, 1>>
struct RGB8882Gray<RGBSwap, ExtraProcess> {
    using QuantType = ExtraProcess::QT;
    const ExtraProcess &m_extra_process;

    RGB8882Gray(const ExtraProcess &extra_process) : m_extra_process(extra_process) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        if constexpr (RGBSwap) {
            *dst_data = m_extra_process.norm_quant_chn1(rgb8882gray(src[2], src[1], src[0]));
        } else {
            *dst_data = m_extra_process.norm_quant_chn1(rgb8882gray(src[0], src[1], src[2]));
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGBSwap) {
                cvt_color_simd_helper_bgr8882gray_qint8(src, dst, n, m_extra_process.m_lut_32);
            } else {
                cvt_color_simd_helper_rgb8882gray_qint8(src, dst, n, m_extra_process.m_lut_32);
            }
        } else {
            if constexpr (RGBSwap) {
                cvt_color_simd_helper_bgr8882gray_qint16(src, dst, n, m_extra_process.m_lut_32);
            } else {
                cvt_color_simd_helper_rgb8882gray_qint16(src, dst, n, m_extra_process.m_lut_32);
            }
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGBSwap) {
                resize_nn_simd_helper_bgr8882gray_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else {
                resize_nn_simd_helper_rgb8882gray_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
            }
        } else {
            if constexpr (RGBSwap) {
                resize_nn_simd_helper_bgr8882gray_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else {
                resize_nn_simd_helper_rgb8882gray_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
            }
        }
    }
};
} // namespace image
} // namespace dl
