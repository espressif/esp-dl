#pragma once
#include "dl_image_color_isa.hpp"
#include "dl_image_define.hpp"
#include "dl_tensor_base.hpp"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include <algorithm> // std::min/std::max
#include <concepts>
#include <cstring> // for memset/memcpy
#include <vector>

namespace dl {
namespace image {
inline uint8_t extract_channel1_from_rgb565le(uint16_t x)
{
    return static_cast<uint8_t>((x & 0xF800) >> 8);
}

inline uint8_t extract_channel2_from_rgb565le(uint16_t x)
{
    return static_cast<uint8_t>((x & 0x7E0) >> 3);
}

inline uint8_t extract_channel3_from_rgb565le(uint16_t x)
{
    return static_cast<uint8_t>((x & 0x1F) << 3);
}

inline uint8_t extract_channel1_from_rgb565be(uint16_t x)
{
    return static_cast<uint8_t>(x & 0xF8);
}

inline uint8_t extract_channel2_from_rgb565be(uint16_t x)
{
    return static_cast<uint8_t>(((x & 0x7) << 5) | ((x & 0xE000) >> 11));
}

inline uint8_t extract_channel3_from_rgb565be(uint16_t x)
{
    return static_cast<uint8_t>((x & 0x1F00) >> 5);
}

inline uint8_t rgb8882gray(uint8_t r, uint8_t g, uint8_t b)
{
    constexpr int coeff_r = 9798;
    constexpr int coeff_g = 19235;
    constexpr int coeff_b = 3735;
    constexpr int shift = 15;
    constexpr int round_delta = 1 << (shift - 1);
    return static_cast<uint8_t>((coeff_r * r + coeff_g * g + coeff_b * b + round_delta) >> shift);
}

inline uint16_t rgb8882rgb565(uint8_t r, uint8_t g, uint8_t b)
{
    return static_cast<uint16_t>(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}

template <typename QuantType, int>
struct NormQuant;

template <typename QuantType>
    requires std::same_as<QuantType, int8_t> || std::same_as<QuantType, std::int16_t>
struct NormQuant<QuantType, 3> {
    using QT = QuantType;
    NormQuant(const std::vector<float> &mean, const std::vector<float> &std, int exp)
    {
        assert(mean.size() == 3 && std.size() == 3);
        m_lut1 = (QuantType *)heap_caps_malloc(mean.size() * 256 * sizeof(QuantType), MALLOC_CAP_DEFAULT);
        m_lut2 = m_lut1 + 256;
        m_lut3 = m_lut2 + 256;
        m_lut_32 = (int *)heap_caps_aligned_alloc(4, mean.size() * 256 * sizeof(int), MALLOC_CAP_DEFAULT);
        float inv_scale = exp > 0 ? 1.f / (1 << exp) : (1 << -exp);
        int idx = 0;
        for (int i = 0; i < mean.size(); i++) {
            float inv_std = 1.f / std[i];
            for (int j = 0; j < 256; j++) {
                QuantType v = quantize<QuantType>((j - mean[i]) * inv_std, inv_scale);
                m_lut1[idx] = v;
                m_lut_32[idx++] = (int)v;
            }
        }
    }

    ~NormQuant()
    {
        heap_caps_free(m_lut1);
        heap_caps_free(m_lut_32);
    }

    QuantType norm_quant_chn1(uint8_t value) const { return m_lut1[value]; }

    QuantType norm_quant_chn2(uint8_t value) const { return m_lut2[value]; }

    QuantType norm_quant_chn3(uint8_t value) const { return m_lut3[value]; }

    QuantType *m_lut1;
    QuantType *m_lut2;
    QuantType *m_lut3;
    int *m_lut_32;
};

template <typename QuantType>
    requires std::same_as<QuantType, int8_t> || std::same_as<QuantType, std::int16_t>
struct NormQuant<QuantType, 1> {
    using QT = QuantType;
    NormQuant(const std::vector<float> &mean, const std::vector<float> &std, int exp)
    {
        assert(mean.size() == 1 && std.size() == 1);
        m_lut1 = (QuantType *)heap_caps_malloc(256 * sizeof(QuantType), MALLOC_CAP_DEFAULT);
        m_lut_32 = (int *)heap_caps_aligned_alloc(4, 256 * sizeof(int), MALLOC_CAP_DEFAULT);
        float inv_scale = exp > 0 ? 1.f / (1 << exp) : (1 << -exp);
        float inv_std = 1.f / std[0];
        for (int i = 0; i < 256; i++) {
            QuantType v = quantize<QuantType>((i - mean[0]) * inv_std, inv_scale);
            m_lut1[i] = v;
            m_lut_32[i] = (int)v;
        }
    }

    ~NormQuant()
    {
        heap_caps_free(m_lut1);
        heap_caps_free(m_lut_32);
    }

    QuantType norm_quant_chn1(uint8_t value) const { return m_lut1[value]; }

    QuantType *m_lut1;
    int *m_lut_32;
};

template <bool RGB565BE, bool RGBSwap, typename ExtraProcess = void>
struct RGB5652RGB888;

template <bool RGB565BE, bool RGBSwap>
struct RGB5652RGB888<RGB565BE, RGBSwap, void> {
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint16_t src_data = *(reinterpret_cast<const uint16_t *>(src));
        if constexpr (RGB565BE && RGBSwap) {
            dst[0] = extract_channel3_from_rgb565be(src_data);
            dst[1] = extract_channel2_from_rgb565be(src_data);
            dst[2] = extract_channel1_from_rgb565be(src_data);
        } else if constexpr (RGB565BE && !RGBSwap) {
            dst[0] = extract_channel1_from_rgb565be(src_data);
            dst[1] = extract_channel2_from_rgb565be(src_data);
            dst[2] = extract_channel3_from_rgb565be(src_data);
        } else if constexpr (!RGB565BE && RGBSwap) {
            dst[0] = extract_channel3_from_rgb565le(src_data);
            dst[1] = extract_channel2_from_rgb565le(src_data);
            dst[2] = extract_channel1_from_rgb565le(src_data);
        } else {
            dst[0] = extract_channel1_from_rgb565le(src_data);
            dst[1] = extract_channel2_from_rgb565le(src_data);
            dst[2] = extract_channel3_from_rgb565le(src_data);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            cvt_color_simd_helper_rgb565be2bgr888(src, dst, n);
        } else if constexpr (RGB565BE && !RGBSwap) {
            cvt_color_simd_helper_rgb565be2rgb888(src, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap) {
            cvt_color_simd_helper_rgb565le2bgr888(src, dst, n);
        } else {
            cvt_color_simd_helper_rgb565le2rgb888(src, dst, n);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            resize_nn_simd_helper_rgb565be2bgr888(&src, offsets, dst, n);
        } else if constexpr (RGB565BE && !RGBSwap) {
            resize_nn_simd_helper_rgb565be2rgb888(&src, offsets, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap) {
            resize_nn_simd_helper_rgb565le2bgr888(&src, offsets, dst, n);
        } else {
            resize_nn_simd_helper_rgb565le2rgb888(&src, offsets, dst, n);
        }
    }
};

template <bool RGB565BE, bool RGBSwap, typename ExtraProcess>
    requires std::same_as<ExtraProcess, NormQuant<int8_t, 3>> || std::same_as<ExtraProcess, NormQuant<int16_t, 3>>
struct RGB5652RGB888<RGB565BE, RGBSwap, ExtraProcess> {
    using QuantType = ExtraProcess::QT;
    ExtraProcess *m_extra_process;

    RGB5652RGB888(void *extra_process) : m_extra_process((ExtraProcess *)extra_process) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint16_t src_data = *(reinterpret_cast<const uint16_t *>(src));
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        if constexpr (RGB565BE && RGBSwap) {
            dst_data[0] = m_extra_process->norm_quant_chn1(extract_channel3_from_rgb565be(src_data));
            dst_data[1] = m_extra_process->norm_quant_chn2(extract_channel2_from_rgb565be(src_data));
            dst_data[2] = m_extra_process->norm_quant_chn3(extract_channel1_from_rgb565be(src_data));
        } else if constexpr (RGB565BE && !RGBSwap) {
            dst_data[0] = m_extra_process->norm_quant_chn1(extract_channel1_from_rgb565be(src_data));
            dst_data[1] = m_extra_process->norm_quant_chn2(extract_channel2_from_rgb565be(src_data));
            dst_data[2] = m_extra_process->norm_quant_chn3(extract_channel3_from_rgb565be(src_data));
        } else if constexpr (!RGB565BE && RGBSwap) {
            dst_data[0] = m_extra_process->norm_quant_chn1(extract_channel3_from_rgb565le(src_data));
            dst_data[1] = m_extra_process->norm_quant_chn2(extract_channel2_from_rgb565le(src_data));
            dst_data[2] = m_extra_process->norm_quant_chn3(extract_channel1_from_rgb565le(src_data));
        } else {
            dst_data[0] = m_extra_process->norm_quant_chn1(extract_channel1_from_rgb565le(src_data));
            dst_data[1] = m_extra_process->norm_quant_chn2(extract_channel2_from_rgb565le(src_data));
            dst_data[2] = m_extra_process->norm_quant_chn3(extract_channel3_from_rgb565le(src_data));
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGB565BE && RGBSwap) {
                cvt_color_simd_helper_rgb565be2bgr888_qint8(src, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                cvt_color_simd_helper_rgb565be2rgb888_qint8(src, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                cvt_color_simd_helper_rgb565le2bgr888_qint8(src, dst, n, m_extra_process->m_lut_32);
            } else {
                cvt_color_simd_helper_rgb565le2rgb888_qint8(src, dst, n, m_extra_process->m_lut_32);
            }
        } else {
            if constexpr (RGB565BE && RGBSwap) {
                cvt_color_simd_helper_rgb565be2bgr888_qint16(src, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                cvt_color_simd_helper_rgb565be2rgb888_qint16(src, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                cvt_color_simd_helper_rgb565le2bgr888_qint16(src, dst, n, m_extra_process->m_lut_32);
            } else {
                cvt_color_simd_helper_rgb565le2rgb888_qint16(src, dst, n, m_extra_process->m_lut_32);
            }
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGB565BE && RGBSwap) {
                resize_nn_simd_helper_rgb565be2bgr888_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                resize_nn_simd_helper_rgb565be2rgb888_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                resize_nn_simd_helper_rgb565le2bgr888_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else {
                resize_nn_simd_helper_rgb565le2rgb888_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            }
        } else {
            if constexpr (RGB565BE && RGBSwap) {
                resize_nn_simd_helper_rgb565be2bgr888_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                resize_nn_simd_helper_rgb565be2rgb888_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                resize_nn_simd_helper_rgb565le2bgr888_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else {
                resize_nn_simd_helper_rgb565le2rgb888_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            }
        }
    }
};

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
    ExtraProcess *m_extra_process;

    RGB5652Gray(void *extra_process) : m_extra_process((ExtraProcess *)(extra_process)) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint16_t src_data = *(reinterpret_cast<const uint16_t *>(src));
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        if constexpr (RGB565BE && RGBSwap) {
            *dst_data = m_extra_process->norm_quant_chn1(rgb8882gray(extract_channel3_from_rgb565be(src_data),
                                                                     extract_channel2_from_rgb565be(src_data),
                                                                     extract_channel1_from_rgb565be(src_data)));
        } else if constexpr (RGB565BE && !RGBSwap) {
            *dst_data = m_extra_process->norm_quant_chn1(rgb8882gray(extract_channel1_from_rgb565be(src_data),
                                                                     extract_channel2_from_rgb565be(src_data),
                                                                     extract_channel3_from_rgb565be(src_data)));
        } else if constexpr (!RGB565BE && RGBSwap) {
            *dst_data = m_extra_process->norm_quant_chn1(rgb8882gray(extract_channel3_from_rgb565le(src_data),
                                                                     extract_channel2_from_rgb565le(src_data),
                                                                     extract_channel1_from_rgb565le(src_data)));
        } else {
            *dst_data = m_extra_process->norm_quant_chn1(rgb8882gray(extract_channel1_from_rgb565le(src_data),
                                                                     extract_channel2_from_rgb565le(src_data),
                                                                     extract_channel3_from_rgb565le(src_data)));
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565be2gray_qint8(src, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                cvt_color_simd_helper_rgb565be2gray_qint8(src, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565le2gray_qint8(src, dst, n, m_extra_process->m_lut_32);
            } else {
                cvt_color_simd_helper_rgb565le2gray_qint8(src, dst, n, m_extra_process->m_lut_32);
            }
        } else {
            if constexpr (RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565be2gray_qint16(src, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                cvt_color_simd_helper_rgb565be2gray_qint16(src, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                cvt_color_simd_helper_bgr565le2gray_qint16(src, dst, n, m_extra_process->m_lut_32);
            } else {
                cvt_color_simd_helper_rgb565le2gray_qint16(src, dst, n, m_extra_process->m_lut_32);
            }
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565be2gray_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                resize_nn_simd_helper_rgb565be2gray_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565le2gray_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else {
                resize_nn_simd_helper_rgb565le2gray_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            }
        } else {
            if constexpr (RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565be2gray_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (RGB565BE && !RGBSwap) {
                resize_nn_simd_helper_rgb565be2gray_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else if constexpr (!RGB565BE && RGBSwap) {
                resize_nn_simd_helper_bgr565le2gray_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else {
                resize_nn_simd_helper_rgb565le2gray_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            }
        }
    }
};

template <bool RGB565BE, bool RGBSwap>
struct RGB8882RGB565 {
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint16_t *dst_data = reinterpret_cast<uint16_t *>(dst);
        if constexpr (RGB565BE && RGBSwap) {
            *dst_data = __builtin_bswap16(rgb8882rgb565(src[2], src[1], src[0]));
        } else if constexpr (RGB565BE && !RGBSwap) {
            *dst_data = __builtin_bswap16(rgb8882rgb565(src[0], src[1], src[2]));
        } else if constexpr (!RGB565BE && RGBSwap) {
            *dst_data = rgb8882rgb565(src[2], src[1], src[0]);
        } else {
            *dst_data = rgb8882rgb565(src[0], src[1], src[2]);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            cvt_color_simd_helper_bgr8882rgb565be(src, dst, n);
        } else if constexpr (RGB565BE && !RGBSwap) {
            cvt_color_simd_helper_rgb8882rgb565be(src, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap) {
            cvt_color_simd_helper_bgr8882rgb565le(src, dst, n);
        } else {
            cvt_color_simd_helper_rgb8882rgb565le(src, dst, n);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            resize_nn_simd_helper_bgr8882rgb565be(&src, offsets, dst, n);
        } else if constexpr (RGB565BE && !RGBSwap) {
            resize_nn_simd_helper_rgb8882rgb565be(&src, offsets, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap) {
            resize_nn_simd_helper_bgr8882rgb565le(&src, offsets, dst, n);
        } else {
            resize_nn_simd_helper_rgb8882rgb565le(&src, offsets, dst, n);
        }
    }
};

template <bool RGBSwap, typename ExtraProcess = void>
struct RGB8882RGB888;

template <bool RGBSwap>
struct RGB8882RGB888<RGBSwap, void> {
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        if constexpr (RGBSwap) {
            dst[2] = src[0];
            dst[1] = src[1];
            dst[0] = src[2];
        } else {
            memcpy(dst, src, 3);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (RGBSwap) {
            cvt_color_simd_helper_rgb8882bgr888(src, dst, n);
        } else {
            cvt_color_simd_helper_rgb8882rgb888(src, dst, n);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (RGBSwap) {
            resize_nn_simd_helper_rgb8882bgr888(&src, offsets, dst, n);
        } else {
            resize_nn_simd_helper_rgb8882rgb888(&src, offsets, dst, n);
        }
    }
};

template <bool RGBSwap, typename ExtraProcess>
    requires std::same_as<ExtraProcess, NormQuant<int8_t, 3>> || std::same_as<ExtraProcess, NormQuant<int16_t, 3>>
struct RGB8882RGB888<RGBSwap, ExtraProcess> {
    using QuantType = ExtraProcess::QT;
    ExtraProcess *m_extra_process;

    RGB8882RGB888(void *extra_process) : m_extra_process((ExtraProcess *)(extra_process)) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        if constexpr (RGBSwap) {
            dst_data[0] = m_extra_process->norm_quant_chn1(src[2]);
            dst_data[1] = m_extra_process->norm_quant_chn2(src[1]);
            dst_data[2] = m_extra_process->norm_quant_chn3(src[0]);
        } else {
            dst_data[0] = m_extra_process->norm_quant_chn1(src[0]);
            dst_data[1] = m_extra_process->norm_quant_chn2(src[1]);
            dst_data[2] = m_extra_process->norm_quant_chn3(src[2]);
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGBSwap) {
                cvt_color_simd_helper_rgb8882bgr888_qint8(src, dst, n, m_extra_process->m_lut_32);
            } else {
                cvt_color_simd_helper_rgb8882rgb888_qint8(src, dst, n, m_extra_process->m_lut_32);
            }
        } else {
            if constexpr (RGBSwap) {
                cvt_color_simd_helper_rgb8882bgr888_qint16(src, dst, n, m_extra_process->m_lut_32);
            } else {
                cvt_color_simd_helper_rgb8882rgb888_qint16(src, dst, n, m_extra_process->m_lut_32);
            }
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGBSwap) {
                resize_nn_simd_helper_rgb8882bgr888_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else {
                resize_nn_simd_helper_rgb8882rgb888_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            }
        } else {
            if constexpr (RGBSwap) {
                resize_nn_simd_helper_rgb8882bgr888_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else {
                resize_nn_simd_helper_rgb8882rgb888_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            }
        }
    }
};

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
    ExtraProcess *m_extra_process;

    RGB8882Gray(void *extra_process) : m_extra_process((ExtraProcess *)(extra_process)) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        if constexpr (RGBSwap) {
            *dst_data = m_extra_process->norm_quant_chn1(rgb8882gray(src[2], src[1], src[0]));
        } else {
            *dst_data = m_extra_process->norm_quant_chn1(rgb8882gray(src[0], src[1], src[2]));
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGBSwap) {
                cvt_color_simd_helper_bgr8882gray_qint8(src, dst, n, m_extra_process->m_lut_32);
            } else {
                cvt_color_simd_helper_rgb8882gray_qint8(src, dst, n, m_extra_process->m_lut_32);
            }
        } else {
            if constexpr (RGBSwap) {
                cvt_color_simd_helper_bgr8882gray_qint16(src, dst, n, m_extra_process->m_lut_32);
            } else {
                cvt_color_simd_helper_rgb8882gray_qint16(src, dst, n, m_extra_process->m_lut_32);
            }
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            if constexpr (RGBSwap) {
                resize_nn_simd_helper_bgr8882gray_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else {
                resize_nn_simd_helper_rgb8882gray_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
            }
        } else {
            if constexpr (RGBSwap) {
                resize_nn_simd_helper_bgr8882gray_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            } else {
                resize_nn_simd_helper_rgb8882gray_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
            }
        }
    }
};

template <bool RGB565BE, bool RGBSwap, bool ByteSwap>
struct RGB5652RGB565 {
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        const uint16_t *src_data = reinterpret_cast<const uint16_t *>(src);
        uint16_t *dst_data = reinterpret_cast<uint16_t *>(dst);
        if constexpr (!RGBSwap && ByteSwap) {
            *dst_data = __builtin_bswap16(*src_data);
        } else if constexpr (!RGBSwap && !ByteSwap) {
            *dst_data = *src_data;
        } else if constexpr (RGB565BE && RGBSwap && ByteSwap) {
            *dst_data = static_cast<uint16_t>(((*src_data & 0xf8) >> 3) | ((*src_data & 0xe000) >> 8) |
                                              ((*src_data & 0x7) << 8) | ((*src_data & 0x1f00) << 3));
        } else if constexpr (!RGB565BE && RGBSwap && ByteSwap) {
            *dst_data = static_cast<uint16_t>(((*src_data & 0xf800) >> 3) | ((*src_data & 0x700) >> 8) |
                                              ((*src_data & 0xe0) << 8) | ((*src_data & 0x1f) << 3));
        } else if constexpr (RGB565BE && RGBSwap && !ByteSwap) {
            *dst_data =
                static_cast<uint16_t>(((*src_data & 0xf8) << 5) | (*src_data & 0xe007) | ((*src_data & 0x1f00) >> 5));
        } else if constexpr (!RGB565BE && RGBSwap && !ByteSwap) {
            *dst_data =
                static_cast<uint16_t>(((*src_data & 0xf800) >> 11) | (*src_data & 0x7e0) | ((*src_data & 0x1f) << 11));
        }
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (!RGBSwap && ByteSwap) {
            cvt_color_simd_helper_rgb565le2rgb565be(src, dst, n);
        } else if constexpr (!RGBSwap && !ByteSwap) {
            cvt_color_simd_helper_rgb5652rgb565(src, dst, n);
        } else if constexpr (RGB565BE && RGBSwap && ByteSwap) {
            cvt_color_simd_helper_rgb565be2bgr565le(src, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap && ByteSwap) {
            cvt_color_simd_helper_rgb565le2bgr565le(src, dst, n);
        } else if constexpr (RGB565BE && RGBSwap && !ByteSwap) {
            cvt_color_simd_helper_rgb565be2bgr565be(src, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap && !ByteSwap) {
            cvt_color_simd_helper_rgb565le2bgr565le(src, dst, n);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (!RGBSwap && ByteSwap) {
            resize_nn_simd_helper_rgb565le2rgb565be(&src, offsets, dst, n);
        } else if constexpr (!RGBSwap && !ByteSwap) {
            resize_nn_simd_helper_rgb5652rgb565(&src, offsets, dst, n);
        } else if constexpr (RGB565BE && RGBSwap && ByteSwap) {
            resize_nn_simd_helper_rgb565be2bgr565le(&src, offsets, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap && ByteSwap) {
            resize_nn_simd_helper_rgb565le2bgr565le(&src, offsets, dst, n);
        } else if constexpr (RGB565BE && RGBSwap && !ByteSwap) {
            resize_nn_simd_helper_rgb565be2bgr565be(&src, offsets, dst, n);
        } else if constexpr (!RGB565BE && RGBSwap && !ByteSwap) {
            resize_nn_simd_helper_rgb565le2bgr565le(&src, offsets, dst, n);
        }
    }
};

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
    ExtraProcess *m_extra_process;

    Gray2Gray(void *extra_process) : m_extra_process((ExtraProcess *)(extra_process)) {}

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        QuantType *dst_data = reinterpret_cast<QuantType *>(dst);
        *dst_data = m_extra_process->norm_quant_chn1(*src);
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            cvt_color_simd_helper_gray2gray_qint8(src, dst, n, m_extra_process->m_lut_32);
        } else {
            cvt_color_simd_helper_gray2gray_qint16(src, dst, n, m_extra_process->m_lut_32);
        }
    }

    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (std::is_same_v<QuantType, int8_t>) {
            resize_nn_simd_helper_gray2gray_qint8(&src, offsets, dst, n, m_extra_process->m_lut_32);
        } else {
            resize_nn_simd_helper_gray2gray_qint16(&src, offsets, dst, n, m_extra_process->m_lut_32);
        }
    }
};

struct HSVTablesSingleton {
    static inline constexpr int hsv_shift = 12;
    int *m_sdiv_table;
    int *m_hdiv_table;

protected:
    HSVTablesSingleton()
    {
        m_sdiv_table = (int *)heap_caps_aligned_alloc(4, 256 * sizeof(int), MALLOC_CAP_DEFAULT);
        m_hdiv_table = (int *)heap_caps_aligned_alloc(4, 256 * sizeof(int), MALLOC_CAP_DEFAULT);
        m_sdiv_table[0] = m_hdiv_table[0] = 0;
        for (int i = 1; i < 256; i++) {
            m_sdiv_table[i] = static_cast<int>((255 << hsv_shift) / (1. * i));
            m_hdiv_table[i] = static_cast<int>((180 << hsv_shift) / (6. * i));
        }
    }

public:
    static HSVTablesSingleton &getInstance()
    {
        static HSVTablesSingleton g_tables;
        return g_tables;
    }
};

template <bool RGBSwap>
struct RGB8882HSV {
    RGB8882HSV()
    {
        const HSVTablesSingleton &global_tables = HSVTablesSingleton::getInstance();
        m_hdiv_table = global_tables.m_hdiv_table;
        m_sdiv_table = global_tables.m_sdiv_table;
    }

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        constexpr int hsv_shift = HSVTablesSingleton::hsv_shift;
        constexpr int round_delta = 1 << (hsv_shift - 1);
        int r, g, b;
        if constexpr (RGBSwap) {
            r = src[2], g = src[1], b = src[0];
        } else {
            r = src[0], g = src[1], b = src[2];
        }
        int h, s, v = b;
        int vmin = b;
        int vr, vg;

        v = std::max(std::max(v, g), r);
        vmin = std::min(std::min(vmin, g), r);

        uint8_t diff = (uint8_t)(v - vmin);
        vr = v == r ? -1 : 0;
        vg = v == g ? -1 : 0;

        s = (diff * m_sdiv_table[v] + round_delta) >> hsv_shift;
        h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
        h = (h * m_hdiv_table[diff] + round_delta) >> hsv_shift;
        h += h < 0 ? 180 : 0;

        dst[0] = (uint8_t)h;
        dst[1] = (uint8_t)s;
        dst[2] = (uint8_t)v;
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (RGBSwap) {
            cvt_color_simd_helper_bgr8882hsv(src, dst, n, m_sdiv_table, m_hdiv_table);
        } else {
            cvt_color_simd_helper_rgb8882hsv(src, dst, n, m_sdiv_table, m_hdiv_table);
        }
    }
    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (RGBSwap) {
            resize_nn_simd_helper_bgr8882hsv(&src, offsets, dst, n, m_sdiv_table, m_hdiv_table);
        } else {
            resize_nn_simd_helper_rgb8882hsv(&src, offsets, dst, n, m_sdiv_table, m_hdiv_table);
        }
    }

    int *m_hdiv_table;
    int *m_sdiv_table;
};

template <bool RGB565BE, bool RGBSwap>
struct RGB5652HSV {
    RGB5652RGB888<RGB565BE, RGBSwap> m_rgb5652rgb888;
    RGB8882HSV<false> m_rgb8882hsv;

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t rgb888[3];
        m_rgb5652rgb888(src, rgb888);
        m_rgb8882hsv(rgb888, dst);
    }

    void cvt_color_simd_helper(uint8_t *src, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            cvt_color_simd_helper_bgr565be2hsv(src, dst, n, m_rgb8882hsv.m_sdiv_table, m_rgb8882hsv.m_hdiv_table);
        } else if constexpr (RGB565BE && !RGBSwap) {
            cvt_color_simd_helper_rgb565be2hsv(src, dst, n, m_rgb8882hsv.m_sdiv_table, m_rgb8882hsv.m_hdiv_table);
        } else if constexpr (!RGB565BE && RGBSwap) {
            cvt_color_simd_helper_bgr565le2hsv(src, dst, n, m_rgb8882hsv.m_sdiv_table, m_rgb8882hsv.m_hdiv_table);
        } else {
            cvt_color_simd_helper_rgb565le2hsv(src, dst, n, m_rgb8882hsv.m_sdiv_table, m_rgb8882hsv.m_hdiv_table);
        }
    }
    void resize_nn_simd_helper(uint8_t *src, int *offsets, uint8_t *dst, int n) const
    {
        if constexpr (RGB565BE && RGBSwap) {
            resize_nn_simd_helper_bgr565be2hsv(
                &src, offsets, dst, n, m_rgb8882hsv.m_sdiv_table, m_rgb8882hsv.m_hdiv_table);
        } else if constexpr (RGB565BE && !RGBSwap) {
            resize_nn_simd_helper_rgb565be2hsv(
                &src, offsets, dst, n, m_rgb8882hsv.m_sdiv_table, m_rgb8882hsv.m_hdiv_table);
        } else if constexpr (!RGB565BE && RGBSwap) {
            resize_nn_simd_helper_bgr565le2hsv(
                &src, offsets, dst, n, m_rgb8882hsv.m_sdiv_table, m_rgb8882hsv.m_hdiv_table);
        } else {
            resize_nn_simd_helper_rgb565le2hsv(
                &src, offsets, dst, n, m_rgb8882hsv.m_sdiv_table, m_rgb8882hsv.m_hdiv_table);
        }
    }
};

template <bool HAcrossZero>
struct HSV2HSVMask {
    HSV2HSVMask(const std::vector<uint8_t> &hsv_min, const std::vector<uint8_t> &hsv_max)
    {
        m_h_min = hsv_min[0];
        m_h_max = hsv_max[0];
        m_s_min = hsv_min[1];
        m_s_max = hsv_max[1];
        m_v_min = hsv_min[2];
        m_v_max = hsv_max[2];
    }

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t h = src[0], s = src[1], v = src[2];
        if constexpr (HAcrossZero) {
            if ((h >= m_h_min || h <= m_h_max) && s >= m_s_min && s <= m_s_max && v >= m_v_min && v <= m_v_max) {
                *dst = 255;
            } else {
                *dst = 0;
            }
        } else {
            if (h >= m_h_min && h <= m_h_max && s >= m_s_min && s <= m_s_max && v >= m_v_min && v <= m_v_max) {
                *dst = 255;
            } else {
                *dst = 0;
            }
        }
    }

    uint8_t m_h_min;
    uint8_t m_h_max;
    uint8_t m_s_min;
    uint8_t m_s_max;
    uint8_t m_v_min;
    uint8_t m_v_max;
};

template <bool RGBSwap, bool HAcrossZero>
struct RGB8882HSVMask {
    RGB8882HSVMask(const std::vector<uint8_t> &hsv_min, const std::vector<uint8_t> &hsv_max) :
        m_hsv2hsv_mask(hsv_min, hsv_max)
    {
    }
    RGB8882HSV<RGBSwap> m_rgb8882hsv;
    HSV2HSVMask<HAcrossZero> m_hsv2hsv_mask;

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t hsv[3];
        m_rgb8882hsv(src, hsv);
        m_hsv2hsv_mask(hsv, dst);
    }
};

template <bool RGBSwap, bool HAcrossZero>
struct RGB8882HSVAndHSVMask {
    RGB8882HSVAndHSVMask(const std::vector<uint8_t> &hsv_min, const std::vector<uint8_t> &hsv_max) :
        m_hsv2hsv_mask(hsv_min, hsv_max)
    {
    }
    RGB8882HSV<RGBSwap> m_rgb8882hsv;
    HSV2HSVMask<HAcrossZero> m_hsv2hsv_mask;

    void operator()(const uint8_t *src, uint8_t *dst_hsv, uint8_t *dst_hsv_mask) const
    {
        m_rgb8882hsv(src, dst_hsv);
        m_hsv2hsv_mask(dst_hsv, dst_hsv_mask);
    }
};

template <bool RGB565BE, bool RGBSwap, bool HAcrossZero>
struct RGB5652HSVMask {
    RGB5652HSVMask(const std::vector<uint8_t> &hsv_min, const std::vector<uint8_t> &hsv_max) :
        m_hsv2hsv_mask(hsv_min, hsv_max)
    {
    }
    RGB5652HSV<RGB565BE, RGBSwap> m_rgb5652hsv;
    HSV2HSVMask<HAcrossZero> m_hsv2hsv_mask;

    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint8_t hsv[3];
        m_rgb5652hsv(src, hsv);
        m_hsv2hsv_mask(hsv, dst);
    }
};

template <bool RGB565BE, bool RGBSwap, bool HAcrossZero>
struct RGB5652HSVAndHSVMask {
    RGB5652HSVAndHSVMask(const std::vector<uint8_t> &hsv_min, const std::vector<uint8_t> &hsv_max) :
        m_hsv2hsv_mask(hsv_min, hsv_max)
    {
    }
    RGB5652HSV<RGB565BE, RGBSwap> m_rgb5652hsv;
    HSV2HSVMask<HAcrossZero> m_hsv2hsv_mask;

    void operator()(const uint8_t *src, uint8_t *dst_hsv, uint8_t *dst_hsv_mask) const
    {
        m_rgb5652hsv(src, dst_hsv);
        m_hsv2hsv_mask(dst_hsv, dst_hsv_mask);
    }
};
} // namespace image
} // namespace dl
