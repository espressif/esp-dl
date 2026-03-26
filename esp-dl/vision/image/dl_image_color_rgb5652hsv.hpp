#pragma once
#include "dl_image_color_common.hpp"
#include "dl_image_color_hsv2hsv_mask.hpp"
#include "dl_image_color_isa.hpp"
#include <concepts>

namespace dl {
namespace image {
template <bool RGB565BE, bool RGBSwap, typename ExtraProcess = void>
struct RGB5652HSV;

template <bool RGB565BE, bool RGBSwap>
struct RGB5652HSV<RGB565BE, RGBSwap, void> {
    const HSVTablesSingleton &m_hsv_tables;

    RGB5652HSV() : m_hsv_tables(HSVTablesSingleton::get_instance()) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint16_t src_data = *(reinterpret_cast<const uint16_t *>(src));
        if constexpr (RGB565BE && RGBSwap) {
            rgb8882hsv(m_hsv_tables,
                       extract_channel3_from_rgb565be(src_data),
                       extract_channel2_from_rgb565be(src_data),
                       extract_channel1_from_rgb565be(src_data),
                       dst,
                       dst + 1,
                       dst + 2);
        } else if constexpr (RGB565BE && !RGBSwap) {
            rgb8882hsv(m_hsv_tables,
                       extract_channel1_from_rgb565be(src_data),
                       extract_channel2_from_rgb565be(src_data),
                       extract_channel3_from_rgb565be(src_data),
                       dst,
                       dst + 1,
                       dst + 2);
        } else if constexpr (!RGB565BE && RGBSwap) {
            rgb8882hsv(m_hsv_tables,
                       extract_channel3_from_rgb565le(src_data),
                       extract_channel2_from_rgb565le(src_data),
                       extract_channel1_from_rgb565le(src_data),
                       dst,
                       dst + 1,
                       dst + 2);
        } else {
            rgb8882hsv(m_hsv_tables,
                       extract_channel1_from_rgb565le(src_data),
                       extract_channel2_from_rgb565le(src_data),
                       extract_channel3_from_rgb565le(src_data),
                       dst,
                       dst + 1,
                       dst + 2);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            cvt_color_simd_helper_bgr565be2hsv(src, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        } else if constexpr (RGB565BE && !RGBSwap) {
            cvt_color_simd_helper_rgb565be2hsv(src, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        } else if constexpr (!RGB565BE && RGBSwap) {
            cvt_color_simd_helper_bgr565le2hsv(src, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        } else {
            cvt_color_simd_helper_rgb565le2hsv(src, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            resize_nn_simd_helper_bgr565be2hsv(
                &src, offsets, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        } else if constexpr (RGB565BE && !RGBSwap) {
            resize_nn_simd_helper_rgb565be2hsv(
                &src, offsets, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        } else if constexpr (!RGB565BE && RGBSwap) {
            resize_nn_simd_helper_bgr565le2hsv(
                &src, offsets, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        } else {
            resize_nn_simd_helper_rgb565le2hsv(
                &src, offsets, dst, n, m_hsv_tables.m_sdiv_table, m_hsv_tables.m_hdiv_table);
        }
    }
};

template <bool RGB565BE, bool RGBSwap, typename ExtraProcess>
    requires std::same_as<ExtraProcess, HSV2HSVMask<false>> || std::same_as<ExtraProcess, HSV2HSVMask<true>>
struct RGB5652HSV<RGB565BE, RGBSwap, ExtraProcess> {
    const HSVTablesSingleton &m_hsv_tables;
    const ExtraProcess &m_extra_process;

    RGB5652HSV(const ExtraProcess &extra_process) :
        m_hsv_tables(HSVTablesSingleton::get_instance()), m_extra_process(extra_process)
    {
    }

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t hsv[3];
        uint16_t src_data = *(reinterpret_cast<const uint16_t *>(src));
        if constexpr (RGB565BE && RGBSwap) {
            rgb8882hsv(m_hsv_tables,
                       extract_channel3_from_rgb565be(src_data),
                       extract_channel2_from_rgb565be(src_data),
                       extract_channel1_from_rgb565be(src_data),
                       hsv,
                       hsv + 1,
                       hsv + 2);
        } else if constexpr (RGB565BE && !RGBSwap) {
            rgb8882hsv(m_hsv_tables,
                       extract_channel1_from_rgb565be(src_data),
                       extract_channel2_from_rgb565be(src_data),
                       extract_channel3_from_rgb565be(src_data),
                       hsv,
                       hsv + 1,
                       hsv + 2);
        } else if constexpr (!RGB565BE && RGBSwap) {
            rgb8882hsv(m_hsv_tables,
                       extract_channel3_from_rgb565le(src_data),
                       extract_channel2_from_rgb565le(src_data),
                       extract_channel1_from_rgb565le(src_data),
                       hsv,
                       hsv + 1,
                       hsv + 2);
        } else {
            rgb8882hsv(m_hsv_tables,
                       extract_channel1_from_rgb565le(src_data),
                       extract_channel2_from_rgb565le(src_data),
                       extract_channel3_from_rgb565le(src_data),
                       hsv,
                       hsv + 1,
                       hsv + 2);
        }
        m_extra_process(hsv, dst);
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (ExtraProcess::h_across_zero) {
            if constexpr (RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565be2hsv_mask1(src,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else if constexpr (RGB565BE && !RGBSwap) {
                cvt_color_simd_helper_rgb565be2hsv_mask1(src,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else if constexpr (!RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565le2hsv_mask1(src,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else {
                cvt_color_simd_helper_rgb565le2hsv_mask1(src,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            }
        } else {
            if constexpr (RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565be2hsv_mask0(src,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else if constexpr (RGB565BE && !RGBSwap) {
                cvt_color_simd_helper_rgb565be2hsv_mask0(src,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else if constexpr (!RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565le2hsv_mask0(src,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else {
                cvt_color_simd_helper_rgb565le2hsv_mask0(src,
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
            if constexpr (RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565be2hsv_mask1(&src,
                                                         offsets,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else if constexpr (RGB565BE && !RGBSwap) {
                resize_nn_simd_helper_rgb565be2hsv_mask1(&src,
                                                         offsets,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else if constexpr (!RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565le2hsv_mask1(&src,
                                                         offsets,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else {
                resize_nn_simd_helper_rgb565le2hsv_mask1(&src,
                                                         offsets,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            }
        } else {
            if constexpr (RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565be2hsv_mask0(&src,
                                                         offsets,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else if constexpr (RGB565BE && !RGBSwap) {
                resize_nn_simd_helper_rgb565be2hsv_mask0(&src,
                                                         offsets,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else if constexpr (!RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565le2hsv_mask0(&src,
                                                         offsets,
                                                         dst,
                                                         n,
                                                         m_hsv_tables.m_sdiv_table,
                                                         m_hsv_tables.m_hdiv_table,
                                                         m_extra_process.m_hsv_min.data(),
                                                         m_extra_process.m_hsv_max.data());
            } else {
                resize_nn_simd_helper_rgb565le2hsv_mask0(&src,
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
