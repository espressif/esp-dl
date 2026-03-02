#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_isa.hpp"
#include "dl_image_norm_quant.hpp"

namespace dl {
namespace image {
template <bool RGB565BE, bool RGBSwap, typename ExtraProcess = void>
struct RGB5652Gray;

template <bool RGB565BE, bool RGBSwap>
struct RGB5652Gray<RGB565BE, RGBSwap, void> {
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint16_t src_data = *(reinterpret_cast<const uint16_t *>(src));
        if constexpr (RGB565BE && RGBSwap) {
            *dst = rgb8882gray(extract_channel3_from_rgb565be(src_data),
                               extract_channel2_from_rgb565be(src_data),
                               extract_channel1_from_rgb565be(src_data));
        } else if constexpr (RGB565BE && !RGBSwap) {
            *dst = rgb8882gray(extract_channel1_from_rgb565be(src_data),
                               extract_channel2_from_rgb565be(src_data),
                               extract_channel3_from_rgb565be(src_data));
        } else if constexpr (!RGB565BE && RGBSwap) {
            *dst = rgb8882gray(extract_channel3_from_rgb565le(src_data),
                               extract_channel2_from_rgb565le(src_data),
                               extract_channel1_from_rgb565le(src_data));
        } else {
            *dst = rgb8882gray(extract_channel1_from_rgb565le(src_data),
                               extract_channel2_from_rgb565le(src_data),
                               extract_channel3_from_rgb565le(src_data));
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            cvt_color_simd_helper_bgr565be2gray(src, dst, n);
        } else if constexpr (RGB565BE && !RGBSwap) {
            cvt_color_simd_helper_rgb565be2gray(src, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap) {
            cvt_color_simd_helper_bgr565le2gray(src, dst, n);
        } else {
            cvt_color_simd_helper_rgb565le2gray(src, dst, n);
        }
    }
    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            resize_nn_simd_helper_bgr565be2gray(&src, offsets, dst, n);
        } else if constexpr (RGB565BE && !RGBSwap) {
            resize_nn_simd_helper_rgb565be2gray(&src, offsets, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap) {
            resize_nn_simd_helper_bgr565le2gray(&src, offsets, dst, n);
        } else {
            resize_nn_simd_helper_rgb565le2gray(&src, offsets, dst, n);
        }
    }
};

template <bool RGB565BE, bool RGBSwap, typename ExtraProcess>
    requires std::same_as<ExtraProcess, NormQuant<int8_t, 1>> || std::same_as<ExtraProcess, NormQuant<int16_t, 1>>
struct RGB5652Gray<RGB565BE, RGBSwap, ExtraProcess> {
    using QuantType = ExtraProcess::QT;
    const ExtraProcess &m_extra_process;

    RGB5652Gray(const ExtraProcess &extra_process) : m_extra_process(extra_process) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint16_t src_data = *(reinterpret_cast<const uint16_t *>(src));
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        if constexpr (RGB565BE && RGBSwap) {
            *dst_data = m_extra_process.norm_quant_chn1(rgb8882gray(extract_channel3_from_rgb565be(src_data),
                                                                    extract_channel2_from_rgb565be(src_data),
                                                                    extract_channel1_from_rgb565be(src_data)));
        } else if constexpr (RGB565BE && !RGBSwap) {
            *dst_data = m_extra_process.norm_quant_chn1(rgb8882gray(extract_channel1_from_rgb565be(src_data),
                                                                    extract_channel2_from_rgb565be(src_data),
                                                                    extract_channel3_from_rgb565be(src_data)));
        } else if constexpr (!RGB565BE && RGBSwap) {
            *dst_data = m_extra_process.norm_quant_chn1(rgb8882gray(extract_channel3_from_rgb565le(src_data),
                                                                    extract_channel2_from_rgb565le(src_data),
                                                                    extract_channel1_from_rgb565le(src_data)));
        } else {
            *dst_data = m_extra_process.norm_quant_chn1(rgb8882gray(extract_channel1_from_rgb565le(src_data),
                                                                    extract_channel2_from_rgb565le(src_data),
                                                                    extract_channel3_from_rgb565le(src_data)));
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565be2gray_qint8(src, dst, n, m_extra_process.m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                cvt_color_simd_helper_rgb565be2gray_qint8(src, dst, n, m_extra_process.m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565le2gray_qint8(src, dst, n, m_extra_process.m_lut_32);
            } else {
                cvt_color_simd_helper_rgb565le2gray_qint8(src, dst, n, m_extra_process.m_lut_32);
            }
        } else {
            if constexpr (RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565be2gray_qint16(src, dst, n, m_extra_process.m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                cvt_color_simd_helper_rgb565be2gray_qint16(src, dst, n, m_extra_process.m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565le2gray_qint16(src, dst, n, m_extra_process.m_lut_32);
            } else {
                cvt_color_simd_helper_rgb565le2gray_qint16(src, dst, n, m_extra_process.m_lut_32);
            }
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565be2gray_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                resize_nn_simd_helper_rgb565be2gray_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565le2gray_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else {
                resize_nn_simd_helper_rgb565le2gray_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
            }
        } else {
            if constexpr (RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565be2gray_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                resize_nn_simd_helper_rgb565be2gray_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565le2gray_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
            } else {
                resize_nn_simd_helper_rgb565le2gray_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
            }
        }
    }
};
} // namespace image
} // namespace dl
