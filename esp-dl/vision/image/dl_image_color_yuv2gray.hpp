#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_isa.hpp"
#include "dl_image_norm_quant.hpp"

namespace dl {
namespace image {
template <bool YUYVOrUYVY, typename ExtraProcess = void>
struct YUV2Gray;

template <bool YUYVOrUYVY>
struct YUV2Gray<YUYVOrUYVY, void> {
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        if constexpr (YUYVOrUYVY) {
            *dst = src[0];
        } else {
            *dst = src[1];
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const {}

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const {}
};

template <bool YUYVOrUYVY, typename ExtraProcess>
    requires std::same_as<ExtraProcess, NormQuant<int8_t, 1>> || std::same_as<ExtraProcess, NormQuant<int16_t, 1>>
struct YUV2Gray<YUYVOrUYVY, ExtraProcess> {
    using QuantType = ExtraProcess::QT;
    const ExtraProcess &m_extra_process;

    YUV2Gray(const ExtraProcess &extra_process) : m_extra_process(extra_process) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        if constexpr (YUYVOrUYVY) {
            *dst_data = m_extra_process.norm_quant_chn1(src[0]);
        } else {
            *dst_data = m_extra_process.norm_quant_chn1(src[1]);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const {}

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const {}
};

template <typename T>
inline constexpr bool is_yuv2gray_v = false;

template <bool YUYVOrUYVY, typename ExtraProcess>
inline constexpr bool is_yuv2gray_v<YUV2Gray<YUYVOrUYVY, ExtraProcess>> = true;
} // namespace image
} // namespace dl
