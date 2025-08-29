#pragma once
#include "sdkconfig.h"
#include <cstddef> // for size_t
#include <cstdint>

namespace dl {
namespace image {
inline constexpr int DL_IMAGE_CAP_RGB_SWAP = 1 << 0;
inline constexpr int DL_IMAGE_CAP_RGB565_BYTE_SWAP = 1 << 1;
inline constexpr int DL_IMAGE_CAP_RGB565_BIG_ENDIAN = 1 << 2;

typedef enum {
    DL_IMAGE_PIX_TYPE_RGB888 = 0,
    DL_IMAGE_PIX_TYPE_RGB888_QINT8,
    DL_IMAGE_PIX_TYPE_RGB888_QINT16,
    DL_IMAGE_PIX_TYPE_GRAY,
    DL_IMAGE_PIX_TYPE_GRAY_QINT8,
    DL_IMAGE_PIX_TYPE_GRAY_QINT16,
    DL_IMAGE_PIX_TYPE_RGB565,
    DL_IMAGE_PIX_TYPE_HSV,
} pix_type_t;

typedef struct {
    void *data;
    uint16_t width;
    uint16_t height;
    pix_type_t pix_type;
} img_t;

typedef struct {
    void *data;
    size_t data_len;
} jpeg_img_t;

inline constexpr int align_up(int num, int align) noexcept
{
    return (num + align - 1) & ~(align - 1);
}

inline constexpr bool is_pix_type_quant(pix_type_t pix_type) noexcept
{
    return pix_type == DL_IMAGE_PIX_TYPE_RGB888_QINT8 || pix_type == DL_IMAGE_PIX_TYPE_RGB888_QINT16 ||
        pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT8 || pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT16;
}

inline constexpr size_t get_pix_byte_size(pix_type_t pix_type) noexcept
{
    switch (pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
    case DL_IMAGE_PIX_TYPE_RGB888_QINT8:
    case DL_IMAGE_PIX_TYPE_HSV:
        return 3;
    case DL_IMAGE_PIX_TYPE_RGB565:
    case DL_IMAGE_PIX_TYPE_GRAY_QINT16:
        return 2;
    case DL_IMAGE_PIX_TYPE_GRAY:
    case DL_IMAGE_PIX_TYPE_GRAY_QINT8:
        return 1;
    case DL_IMAGE_PIX_TYPE_RGB888_QINT16:
        return 6;
    default:
        return 0;
    }
}

inline constexpr int get_pix_channel_num(pix_type_t pix_type) noexcept
{
    switch (pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
    case DL_IMAGE_PIX_TYPE_RGB888_QINT8:
    case DL_IMAGE_PIX_TYPE_RGB888_QINT16:
    case DL_IMAGE_PIX_TYPE_RGB565:
    case DL_IMAGE_PIX_TYPE_HSV:
        return 3;
    case DL_IMAGE_PIX_TYPE_GRAY:
    case DL_IMAGE_PIX_TYPE_GRAY_QINT8:
    case DL_IMAGE_PIX_TYPE_GRAY_QINT16:
        return 1;
    default:
        return -1;
    }
}

inline constexpr size_t get_img_byte_size(const img_t &img) noexcept
{
    return get_pix_byte_size(img.pix_type) * img.height * img.width;
}
} // namespace image
} // namespace dl
