#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_hsv2hsv_mask.hpp"
#include "dl_image_color_isa.hpp"
#include <concepts>

namespace dl {
namespace image {
template <bool RGBSwap, typename ExtraProcess = void>
struct RGB8882HSV;

template <bool RGBSwap>
struct RGB8882HSV<RGBSwap, void> {
    const HSVTablesSingleton &m_hsv_tables;

    RGB8882HSV() : m_hsv_tables(HSVTablesSingleton::get_instance()) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        if constexpr (RGBSwap) {
            rgb8882hsv(m_hsv_tables, src[2], src[1], src[0], dst, dst + 1, dst + 2);
        } else {
            rgb8882hsv(m_hsv_tables, src[0], src[1], src[2], dst, dst + 1, dst + 2);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (RGBSwap) {
            cvt_color_simd_helper_bgr8882hsv(src, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        } else {
            cvt_color_simd_helper_rgb8882hsv(src, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (RGBSwap) {
            resize_nn_simd_helper_bgr8882hsv(
                &src, offsets, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        } else {
            resize_nn_simd_helper_rgb8882hsv(
                &src, offsets, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        }
    }
};

template <bool RGBSwap, typename ExtraProcess>
    requires std::same_as<ExtraProcess, HSV2HSVMask<false>> || std::same_as<ExtraProcess, HSV2HSVMask<true>>
struct RGB8882HSV<RGBSwap, ExtraProcess> {
    const HSVTablesSingleton &m_hsv_tables;
    const ExtraProcess &m_extra_process;

    RGB8882HSV(const ExtraProcess &extra_process) :
        m_hsv_tables(HSVTablesSingleton::get_instance()), m_extra_process(extra_process)
    {
    }

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t hsv[3];
        if constexpr (RGBSwap) {
            rgb8882hsv(m_hsv_tables, src[2], src[1], src[0], hsv, hsv + 1, hsv + 2);
        } else {
            rgb8882hsv(m_hsv_tables, src[0], src[1], src[2], hsv, hsv + 1, hsv + 2);
        }
        m_extra_process(hsv, dst);
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (ExtraProcess::h_across_zero) {
            if constexpr (RGBSwap) {
                cvt_color_simd_helper_bgr8882hsv_mask1(src,
                                                       dst,
                                                       n,
                                                       m_hsv_tables.m_sdiv_table,
                                                       m_hsv_tables.m_hdiv_table,
                                                       m_extra_process.m_hsv_min.data(),
                                                       m_extra_process.m_hsv_max.data());
            } else {
                cvt_color_simd_helper_rgb8882hsv_mask1(src,
                                                       dst,
                                                       n,
                                                       m_hsv_tables.m_sdiv_table,
                                                       m_hsv_tables.m_hdiv_table,
                                                       m_extra_process.m_hsv_min.data(),
                                                       m_extra_process.m_hsv_max.data());
            }
        } else {
            if constexpr (RGBSwap) {
                cvt_color_simd_helper_bgr8882hsv_mask0(src,
                                                       dst,
                                                       n,
                                                       m_hsv_tables.m_sdiv_table,
                                                       m_hsv_tables.m_hdiv_table,
                                                       m_extra_process.m_hsv_min.data(),
                                                       m_extra_process.m_hsv_max.data());
            } else {
                cvt_color_simd_helper_rgb8882hsv_mask0(src,
                                                       dst,
                                                       n,
                                                       m_hsv_tables.m_sdiv_table,
                                                       m_hsv_tables.m_hdiv_table,
                                                       m_extra_process.m_hsv_min.data(),
                                                       m_extra_process.m_hsv_max.data());
            }
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (ExtraProcess::h_across_zero) {
            if constexpr (RGBSwap) {
                resize_nn_simd_helper_bgr8882hsv_mask1(&src,
                                                       offsets,
                                                       dst,
                                                       n,
                                                       m_hsv_tables.m_sdiv_table,
                                                       m_hsv_tables.m_hdiv_table,
                                                       m_extra_process.m_hsv_min.data(),
                                                       m_extra_process.m_hsv_max.data());
            } else {
                resize_nn_simd_helper_rgb8882hsv_mask1(&src,
                                                       offsets,
                                                       dst,
                                                       n,
                                                       m_hsv_tables.m_sdiv_table,
                                                       m_hsv_tables.m_hdiv_table,
                                                       m_extra_process.m_hsv_min.data(),
                                                       m_extra_process.m_hsv_max.data());
            }
        } else {
            if constexpr (RGBSwap) {
                resize_nn_simd_helper_bgr8882hsv_mask0(&src,
                                                       offsets,
                                                       dst,
                                                       n,
                                                       m_hsv_tables.m_sdiv_table,
                                                       m_hsv_tables.m_hdiv_table,
                                                       m_extra_process.m_hsv_min.data(),
                                                       m_extra_process.m_hsv_max.data());
            } else {
                resize_nn_simd_helper_rgb8882hsv_mask0(&src,
                                                       offsets,
                                                       dst,
                                                       n,
                                                       m_hsv_tables.m_sdiv_table,
                                                       m_hsv_tables.m_hdiv_table,
                                                       m_extra_process.m_hsv_min.data(),
                                                       m_extra_process.m_hsv_max.data());
            }
        }
    }
};
} // namespace image
} // namespace dl
