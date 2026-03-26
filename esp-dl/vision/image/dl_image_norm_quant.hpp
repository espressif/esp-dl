#pragma once
#include "dl_tensor_base.hpp"
#include "esp_heap_caps.h"
#include <concepts>

namespace dl {
namespace image {
template <typename QuantType, int>
struct NormQuant;

template <typename QuantType>
    requires std::same_as<QuantType, int8_t> || std::same_as<QuantType, int16_t>
struct NormQuant<QuantType, 3> {
    using QT = QuantType;
    NormQuant(const std::array<float, 3> &mean, const std::array<float, 3> &std, int exp)
    {
        m_lut1 = (QuantType *)heap_caps_malloc(3 * 256 * sizeof(QuantType), MALLOC_CAP_DEFAULT);
        m_lut2 = m_lut1 + 256;
        m_lut3 = m_lut2 + 256;
        m_lut_32 = (int *)heap_caps_aligned_alloc(4, 3 * 256 * sizeof(int), MALLOC_CAP_DEFAULT);
        float inv_scale = exp > 0 ? 1.f / (1 << exp) : (1 << -exp);
        int idx = 0;
        for (int i = 0; i < 3; i++) {
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
    requires std::same_as<QuantType, int8_t> || std::same_as<QuantType, int16_t>
struct NormQuant<QuantType, 1> {
    using QT = QuantType;
    NormQuant(float mean, float std, int exp)
    {
        m_lut1 = (QuantType *)heap_caps_malloc(256 * sizeof(QuantType), MALLOC_CAP_DEFAULT);
        m_lut_32 = (int *)heap_caps_aligned_alloc(4, 256 * sizeof(int), MALLOC_CAP_DEFAULT);
        float inv_scale = exp > 0 ? 1.f / (1 << exp) : (1 << -exp);
        float inv_std = 1.f / std;
        for (int i = 0; i < 256; i++) {
            QuantType v = quantize<QuantType>((i - mean) * inv_std, inv_scale);
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
} // namespace image
} // namespace dl
