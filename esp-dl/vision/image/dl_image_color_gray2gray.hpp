#pragma once
#include "dl_image_color_isa.hpp"
#include "dl_image_norm_quant.hpp"

namespace dl {
namespace image {
template <typename ExtraProcess = void>
struct Gray2Gray;

template <>
struct Gray2Gray<void> {
    void operator()(const uint8_t *src, uint8_t *dst) const { *dst = *src; }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        cvt_color_simd_helper_gray2gray(src, dst, n);
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        resize_nn_simd_helper_gray2gray(&src, offsets, dst, n);
    }
};

template <typename ExtraProcess>
    requires std::same_as<ExtraProcess, NormQuant<int8_t, 1>> || std::same_as<ExtraProcess, NormQuant<int16_t, 1>>
struct Gray2Gray<ExtraProcess> {
    using QuantType = ExtraProcess::QT;
    const ExtraProcess &m_extra_process;

    Gray2Gray(const ExtraProcess &extra_process) : m_extra_process(extra_process) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        *dst_data = m_extra_process.norm_quant_chn1(*src);
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            cvt_color_simd_helper_gray2gray_qint8(src, dst, n, m_extra_process.m_lut_32);
        } else {
            cvt_color_simd_helper_gray2gray_qint16(src, dst, n, m_extra_process.m_lut_32);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            resize_nn_simd_helper_gray2gray_qint8(&src, offsets, dst, n, m_extra_process.m_lut_32);
        } else {
            resize_nn_simd_helper_gray2gray_qint16(&src, offsets, dst, n, m_extra_process.m_lut_32);
        }
    }
};
} // namespace image
} // namespace dl
