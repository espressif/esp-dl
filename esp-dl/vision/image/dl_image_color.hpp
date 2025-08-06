#pragma once
#include "dl_image_define.hpp"
#include "esp_err.h"
#include <concepts>
#include <cstring> // for memset/memcpy

namespace dl {
namespace image {
inline constexpr uint8_t extract_channel1_from_rgb565le(uint16_t x)
{
    return static_cast<uint8_t>((x & 0xF800) >> 8);
}

inline constexpr uint8_t extract_channel2_from_rgb565le(uint16_t x)
{
    return static_cast<uint8_t>((x & 0x7E0) >> 3);
}

inline constexpr uint8_t extract_channel3_from_rgb565le(uint16_t x)
{
    return static_cast<uint8_t>((x & 0x1F) << 3);
}

inline constexpr uint8_t extract_channel1_from_rgb565be(uint16_t x)
{
    return static_cast<uint8_t>(x & 0xF8);
}

inline constexpr uint8_t extract_channel2_from_rgb565be(uint16_t x)
{
    return static_cast<uint8_t>(((x & 0x7) << 5) | ((x & 0xE000) >> 11));
}

inline constexpr uint8_t extract_channel3_from_rgb565be(uint16_t x)
{
    return static_cast<uint8_t>((x & 0x1F00) >> 5);
}

inline constexpr uint8_t rgb8882gray(uint8_t r, uint8_t g, uint8_t b)
{
    constexpr int coeff_r = 9798;
    constexpr int coeff_g = 19235;
    constexpr int coeff_b = 3735;
    constexpr int shift = 15;
    constexpr int round_delta = 1 << (shift - 1);
    return static_cast<uint8_t>((coeff_r * r + coeff_g * g + coeff_b * b + round_delta) >> shift);
}

inline constexpr uint16_t rgb8882rgb565(uint8_t r, uint8_t g, uint8_t b)
{
    return static_cast<uint16_t>(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}

template <typename T>
concept IsQuantType = std::same_as<T, std::int8_t> || std::same_as<T, std::int16_t>;

template <typename QuantType, int>
struct LUT;

template <typename QuantType>
struct LUT<QuantType, 3> {
    LUT(void *lut)
    {
        QuantType *lut_data = static_cast<QuantType *>(lut);
        m_lut1 = lut_data;
        m_lut2 = lut_data + 256;
        m_lut3 = lut_data + 512;
    }

    QuantType *m_lut1;
    QuantType *m_lut2;
    QuantType *m_lut3;
};

template <typename QuantType>
struct LUT<QuantType, 1> {
    LUT(void *lut) { m_lut = static_cast<QuantType *>(lut); }

    QuantType *m_lut;
};

template <bool RGB565BE, bool RGBSwap, typename QuantType = void>
struct RGB5652RGB888;

template <bool RGB565BE, bool RGBSwap>
struct RGB5652RGB888<RGB565BE, RGBSwap, void> {
    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept
    {
        uint16_t src_data = *(reinterpret_cast<const uint16_t *>(src));
        if constexpr (RGB565BE && RGBSwap) {
            dst[2] = extract_channel1_from_rgb565be(src_data);
            dst[1] = extract_channel2_from_rgb565be(src_data);
            dst[0] = extract_channel3_from_rgb565be(src_data);
        } else if constexpr (RGB565BE && !RGBSwap) {
            dst[0] = extract_channel1_from_rgb565be(src_data);
            dst[1] = extract_channel2_from_rgb565be(src_data);
            dst[2] = extract_channel3_from_rgb565be(src_data);
        } else if constexpr (!RGB565BE && RGBSwap) {
            dst[2] = extract_channel1_from_rgb565le(src_data);
            dst[1] = extract_channel2_from_rgb565le(src_data);
            dst[0] = extract_channel3_from_rgb565le(src_data);
        } else {
            dst[0] = extract_channel1_from_rgb565le(src_data);
            dst[1] = extract_channel2_from_rgb565le(src_data);
            dst[2] = extract_channel3_from_rgb565le(src_data);
        }
    }
};

template <bool RGB565BE, bool RGBSwap, typename QuantType>
    requires IsQuantType<QuantType>
struct RGB5652RGB888<RGB565BE, RGBSwap, QuantType> : public LUT<QuantType, 3> {
    using LUT<QuantType, 3>::m_lut1;
    using LUT<QuantType, 3>::m_lut2;
    using LUT<QuantType, 3>::m_lut3;

    RGB5652RGB888(void *lut) : LUT<QuantType, 3>(lut) {}

    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept
    {
        uint16_t src_data = *(reinterpret_cast<const uint16_t *>(src));
        if constexpr (RGB565BE && RGBSwap) {
            dst[2] = m_lut1[extract_channel1_from_rgb565be(src_data)];
            dst[1] = m_lut2[extract_channel2_from_rgb565be(src_data)];
            dst[0] = m_lut3[extract_channel3_from_rgb565be(src_data)];
        } else if constexpr (RGB565BE && !RGBSwap) {
            dst[0] = m_lut1[extract_channel1_from_rgb565be(src_data)];
            dst[1] = m_lut2[extract_channel2_from_rgb565be(src_data)];
            dst[2] = m_lut3[extract_channel3_from_rgb565be(src_data)];
        } else if constexpr (!RGB565BE && RGBSwap) {
            dst[2] = m_lut1[extract_channel1_from_rgb565le(src_data)];
            dst[1] = m_lut2[extract_channel2_from_rgb565le(src_data)];
            dst[0] = m_lut3[extract_channel3_from_rgb565le(src_data)];
        } else {
            dst[0] = m_lut1[extract_channel1_from_rgb565le(src_data)];
            dst[1] = m_lut2[extract_channel2_from_rgb565le(src_data)];
            dst[2] = m_lut3[extract_channel3_from_rgb565le(src_data)];
        }
    }
};

template <bool RGB565BE, bool RGBSwap, typename QuantType = void>
struct RGB5652Gray;

template <bool RGB565BE, bool RGBSwap>
struct RGB5652Gray<RGB565BE, RGBSwap, void> {
    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept
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
};

template <bool RGB565BE, bool RGBSwap, typename QuantType>
    requires IsQuantType<QuantType>
struct RGB5652Gray<RGB565BE, RGBSwap, QuantType> : public LUT<QuantType, 1> {
    using LUT<QuantType, 1>::m_lut;

    RGB5652Gray(void *lut) : LUT<QuantType, 1>(lut) {}

    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept
    {
        uint16_t src_data = *(reinterpret_cast<const uint16_t *>(src));
        if constexpr (RGB565BE && RGBSwap) {
            *dst = m_lut[rgb8882gray(extract_channel3_from_rgb565be(src_data),
                                     extract_channel2_from_rgb565be(src_data),
                                     extract_channel1_from_rgb565be(src_data))];
        } else if constexpr (RGB565BE && !RGBSwap) {
            *dst = m_lut[rgb8882gray(extract_channel1_from_rgb565be(src_data),
                                     extract_channel2_from_rgb565be(src_data),
                                     extract_channel3_from_rgb565be(src_data))];
        } else if constexpr (!RGB565BE && RGBSwap) {
            *dst = m_lut[rgb8882gray(extract_channel3_from_rgb565le(src_data),
                                     extract_channel2_from_rgb565le(src_data),
                                     extract_channel1_from_rgb565le(src_data))];
        } else {
            *dst = m_lut[rgb8882gray(extract_channel1_from_rgb565le(src_data),
                                     extract_channel2_from_rgb565le(src_data),
                                     extract_channel3_from_rgb565le(src_data))];
        }
    }
};

template <bool RGB565BE, bool RGBSwap>
struct RGB8882RGB565 {
    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept
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
};

template <bool RGBSwap, typename QuantType = void>
struct RGB8882RGB888;

template <bool RGBSwap>
struct RGB8882RGB888<RGBSwap, void> {
    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept
    {
        if constexpr (RGBSwap) {
            dst[2] = src[0];
            dst[1] = src[1];
            dst[0] = src[2];
        } else {
            memcpy(dst, src, 3);
        }
    }
};

template <bool RGBSwap, typename QuantType>
    requires IsQuantType<QuantType>
struct RGB8882RGB888<RGBSwap, QuantType> : public LUT<QuantType, 3> {
    using LUT<QuantType, 3>::m_lut1;
    using LUT<QuantType, 3>::m_lut2;
    using LUT<QuantType, 3>::m_lut3;

    RGB8882RGB888(void *lut) : LUT<QuantType, 3>(lut) {}

    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept
    {
        if constexpr (RGBSwap) {
            dst[2] = m_lut1[src[0]];
            dst[1] = m_lut2[src[1]];
            dst[0] = m_lut3[src[2]];
        } else {
            dst[0] = m_lut1[src[0]];
            dst[1] = m_lut2[src[1]];
            dst[2] = m_lut3[src[2]];
        }
    }
};

template <bool RGBSwap, typename QuantType = void>
struct RGB8882Gray;

template <bool RGBSwap>
struct RGB8882Gray<RGBSwap, void> {
    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept
    {
        if constexpr (RGBSwap) {
            *dst = rgb8882gray(src[0], src[1], src[2]);
        } else {
            *dst = rgb8882gray(src[2], src[1], src[0]);
        }
    }
};
template <bool RGBSwap, typename QuantType>
    requires IsQuantType<QuantType>
struct RGB8882Gray<RGBSwap, QuantType> : public LUT<QuantType, 1> {
    using LUT<QuantType, 1>::m_lut;

    RGB8882Gray(void *lut) : LUT<QuantType, 1>(lut) {}

    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept
    {
        if constexpr (RGBSwap) {
            *dst = m_lut[rgb8882gray(src[0], src[1], src[2])];
        } else {
            *dst = m_lut[rgb8882gray(src[2], src[1], src[0])];
        }
    }
};

template <bool RGB565BE, bool RGBSwap, bool ByteSwap>
struct RGB5652RGB565 {
    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept
    {
        const uint16_t *src_data = reinterpret_cast<const uint16_t *>(src);
        uint16_t *dst_data = reinterpret_cast<uint16_t *>(dst);
        if constexpr (!RGBSwap && ByteSwap) {
            *dst_data = __builtin_bswap16(*src_data);
        } else if constexpr (!RGBSwap && !ByteSwap) {
            *dst_data = *src_data;
        } else if constexpr (RGBSwap && ByteSwap && RGB565BE) {
            *dst_data = static_cast<uint16_t>(((*src_data & 0xf8) >> 3) | ((*src_data & 0xe000) >> 8) |
                                              ((*src_data & 0x7) << 8) | ((*src_data & 0x1f00) << 3));
        } else if constexpr (RGBSwap && ByteSwap && !RGB565BE) {
            *dst_data = static_cast<uint16_t>(((*src_data & 0xf800) >> 3) | ((*src_data & 0x700) >> 8) |
                                              ((*src_data & 0xe0) << 8) | ((*src_data & 0x1f) << 3));
        } else if constexpr (RGBSwap && !ByteSwap && RGB565BE) {
            *dst_data =
                static_cast<uint16_t>(((*src_data & 0xf8) << 5) | (*src_data & 0xe007) | ((*src_data & 0x1f00) >> 5));
        } else if constexpr (RGBSwap && !ByteSwap && !RGB565BE) {
            *dst_data =
                static_cast<uint16_t>(((*src_data & 0xf800) >> 11) | (*src_data & 0x7e0) | ((*src_data & 0x1f) << 11));
        }
    }
};

template <typename QuantType>
struct Gray2Gray : public LUT<QuantType, 1> {
    using LUT<QuantType, 1>::m_lut;

    Gray2Gray(void *lut) : LUT<QuantType, 1>(lut) {}

    constexpr void operator()(const uint8_t *src, uint8_t *dst) const noexcept { *dst = m_lut[*src]; }
};

template <typename Func>
esp_err_t pixel_cvt_dispatch(
    const Func &func, pix_type_t src_pix_type, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant_lut)
{
    bool rgb565be = caps & DL_IMAGE_CAP_RGB565_BIG_ENDIAN;
    bool rgb_swap = caps & DL_IMAGE_CAP_RGB_SWAP;
    bool rgb565_swap = caps & DL_IMAGE_CAP_RGB565_BYTE_SWAP;
    bool has_impl = true;
    if (src_pix_type == DL_IMAGE_PIX_TYPE_RGB565) {
        if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB888) {
            if (rgb565be && rgb_swap) {
                func(RGB5652RGB888<true, true>());
            } else if (rgb565be && !rgb_swap) {
                func(RGB5652RGB888<true, false>());
            } else if (!rgb565be && rgb_swap) {
                func(RGB5652RGB888<false, true>());
            } else {
                func(RGB5652RGB888<false, false>());
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB888_QINT8) {
            if (rgb565be && rgb_swap) {
                func(RGB5652RGB888<true, true, int8_t>(norm_quant_lut));
            } else if (rgb565be && !rgb_swap) {
                func(RGB5652RGB888<true, false, int8_t>(norm_quant_lut));
            } else if (!rgb565be && rgb_swap) {
                func(RGB5652RGB888<false, true, int8_t>(norm_quant_lut));
            } else {
                func(RGB5652RGB888<false, false, int8_t>(norm_quant_lut));
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB888_QINT16) {
            if (rgb565be && rgb_swap) {
                func(RGB5652RGB888<true, true, int16_t>(norm_quant_lut));
            } else if (rgb565be && !rgb_swap) {
                func(RGB5652RGB888<true, false, int16_t>(norm_quant_lut));
            } else if (!rgb565be && rgb_swap) {
                func(RGB5652RGB888<false, true, int16_t>(norm_quant_lut));
            } else {
                func(RGB5652RGB888<false, false, int16_t>(norm_quant_lut));
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY) {
            if (rgb565be && rgb_swap) {
                func(RGB5652Gray<true, true>());
            } else if (rgb565be && !rgb_swap) {
                func(RGB5652Gray<true, false>());
            } else if (!rgb565be && rgb_swap) {
                func(RGB5652Gray<false, true>());
            } else {
                func(RGB5652Gray<false, false>());
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT8) {
            if (rgb565be && rgb_swap) {
                func(RGB5652Gray<true, true, int8_t>(norm_quant_lut));
            } else if (rgb565be && !rgb_swap) {
                func(RGB5652Gray<true, false, int8_t>(norm_quant_lut));
            } else if (!rgb565be && rgb_swap) {
                func(RGB5652Gray<false, true, int8_t>(norm_quant_lut));
            } else {
                func(RGB5652Gray<false, false, int8_t>(norm_quant_lut));
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT16) {
            if (rgb565be && rgb_swap) {
                func(RGB5652Gray<true, true, int16_t>(norm_quant_lut));
            } else if (rgb565be && !rgb_swap) {
                func(RGB5652Gray<true, false, int16_t>(norm_quant_lut));
            } else if (!rgb565be && rgb_swap) {
                func(RGB5652Gray<false, true, int16_t>(norm_quant_lut));
            } else {
                func(RGB5652Gray<false, false, int16_t>(norm_quant_lut));
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB565) {
            if (!rgb_swap && rgb565_swap) {
                func(RGB5652RGB565<false, false, true>());
            } else if (!rgb_swap && !rgb565_swap) {
                func(RGB5652RGB565<false, false, false>());
            } else if (rgb_swap && rgb565_swap && rgb565be) {
                func(RGB5652RGB565<true, true, true>());
            } else if (rgb_swap && rgb565_swap && !rgb565be) {
                func(RGB5652RGB565<true, true, false>());
            } else if (rgb_swap && !rgb565_swap && rgb565be) {
                func(RGB5652RGB565<true, false, true>());
            } else if (rgb_swap && !rgb565_swap && !rgb565be) {
                func(RGB5652RGB565<true, false, false>());
            }
        } else {
            has_impl = false;
        }
    } else if (src_pix_type == DL_IMAGE_PIX_TYPE_RGB888) {
        if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB888) {
            if (rgb_swap) {
                func(RGB8882RGB888<true>());
            } else {
                func(RGB8882RGB888<false>());
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB888_QINT8) {
            if (rgb_swap) {
                func(RGB8882RGB888<true, int8_t>(norm_quant_lut));
            } else {
                func(RGB8882RGB888<false, int8_t>(norm_quant_lut));
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB888_QINT16) {
            if (rgb_swap) {
                func(RGB8882RGB888<true, int16_t>(norm_quant_lut));
            } else {
                func(RGB8882RGB888<false, int16_t>(norm_quant_lut));
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY) {
            if (rgb_swap) {
                func(RGB8882Gray<true>());
            } else {
                func(RGB8882Gray<false>());
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT8) {
            if (rgb_swap) {
                func(RGB8882Gray<true, int8_t>(norm_quant_lut));
            } else {
                func(RGB8882Gray<false, int8_t>(norm_quant_lut));
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT16) {
            if (rgb_swap) {
                func(RGB8882Gray<true, int16_t>(norm_quant_lut));
            } else {
                func(RGB8882Gray<false, int16_t>(norm_quant_lut));
            }
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB565) {
            if (rgb565be && rgb_swap) {
                func(RGB8882RGB565<true, true>());
            } else if (rgb565be && !rgb_swap) {
                func(RGB8882RGB565<true, false>());
            } else if (!rgb565be && rgb_swap) {
                func(RGB8882RGB565<false, true>());
            } else {
                func(RGB8882RGB565<false, false>());
            }
        } else {
            has_impl = false;
        }
    } else if (src_pix_type == DL_IMAGE_PIX_TYPE_GRAY) {
        if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT8) {
            func(Gray2Gray<int8_t>(norm_quant_lut));
        } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT16) {
            func(Gray2Gray<int16_t>(norm_quant_lut));
        } else {
            has_impl = false;
        }
    } else {
        has_impl = false;
    }

    if (!has_impl) {
        return ESP_FAIL;
    }
    return ESP_OK;
}

inline esp_err_t cvt_pix(const uint8_t *src,
                         uint8_t *dst,
                         pix_type_t src_pix_type,
                         pix_type_t dst_pix_type,
                         uint32_t caps = 0,
                         void *norm_quant_lut = nullptr)
{
    return pixel_cvt_dispatch(
        [&src, &dst](const auto &pixel_cvt) { pixel_cvt(src, dst); }, src_pix_type, dst_pix_type, caps, norm_quant_lut);
}

} // namespace image
} // namespace dl
