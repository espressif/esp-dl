#pragma once
#include <cstddef> // for size_t
#include <cstdint>
#include <string>

namespace dl {
namespace image {

typedef enum {
    DL_IMAGE_PIX_TYPE_RGB888 = 0,
    DL_IMAGE_PIX_TYPE_RGB888_QINT8,
    DL_IMAGE_PIX_TYPE_RGB888_QINT16,
    DL_IMAGE_PIX_TYPE_BGR888,
    DL_IMAGE_PIX_TYPE_BGR888_QINT8,
    DL_IMAGE_PIX_TYPE_BGR888_QINT16,
    DL_IMAGE_PIX_TYPE_GRAY,
    DL_IMAGE_PIX_TYPE_GRAY_QINT8,
    DL_IMAGE_PIX_TYPE_GRAY_QINT16,
    DL_IMAGE_PIX_TYPE_RGB565LE,
    DL_IMAGE_PIX_TYPE_RGB565BE,
    DL_IMAGE_PIX_TYPE_BGR565LE,
    DL_IMAGE_PIX_TYPE_BGR565BE,
    DL_IMAGE_PIX_TYPE_HSV,
    DL_IMAGE_PIX_TYPE_HSV_MASK,
    DL_IMAGE_PIX_TYPE_YUYV,
    DL_IMAGE_PIX_TYPE_UYVY,
} pix_type_t;

inline std::string pix_type2str(pix_type_t pix_type)
{
    switch (pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
        return "DL_IMAGE_PIX_TYPE_RGB888";
    case DL_IMAGE_PIX_TYPE_RGB888_QINT8:
        return "DL_IMAGE_PIX_TYPE_RGB888_QINT8";
    case DL_IMAGE_PIX_TYPE_RGB888_QINT16:
        return "DL_IMAGE_PIX_TYPE_RGB888_QINT16";
    case DL_IMAGE_PIX_TYPE_BGR888:
        return "DL_IMAGE_PIX_TYPE_BGR888";
    case DL_IMAGE_PIX_TYPE_BGR888_QINT8:
        return "DL_IMAGE_PIX_TYPE_BGR888_QINT8";
    case DL_IMAGE_PIX_TYPE_BGR888_QINT16:
        return "DL_IMAGE_PIX_TYPE_BGR888_QINT16";
    case DL_IMAGE_PIX_TYPE_GRAY:
        return "DL_IMAGE_PIX_TYPE_GRAY";
    case DL_IMAGE_PIX_TYPE_GRAY_QINT8:
        return "DL_IMAGE_PIX_TYPE_GRAY_QINT8";
    case DL_IMAGE_PIX_TYPE_GRAY_QINT16:
        return "DL_IMAGE_PIX_TYPE_GRAY_QINT16";
    case DL_IMAGE_PIX_TYPE_RGB565LE:
        return "DL_IMAGE_PIX_TYPE_RGB565LE";
    case DL_IMAGE_PIX_TYPE_RGB565BE:
        return "DL_IMAGE_PIX_TYPE_RGB565BE";
    case DL_IMAGE_PIX_TYPE_BGR565LE:
        return "DL_IMAGE_PIX_TYPE_BGR565LE";
    case DL_IMAGE_PIX_TYPE_BGR565BE:
        return "DL_IMAGE_PIX_TYPE_BGR565BE";
    case DL_IMAGE_PIX_TYPE_HSV:
        return "DL_IMAGE_PIX_TYPE_HSV";
    case DL_IMAGE_PIX_TYPE_HSV_MASK:
        return "DL_IMAGE_PIX_TYPE_HSV_MASK";
    case DL_IMAGE_PIX_TYPE_YUYV:
        return "DL_IMAGE_PIX_TYPE_YUYV";
    case DL_IMAGE_PIX_TYPE_UYVY:
        return "DL_IMAGE_PIX_TYPE_UYVY";
    default:
        return "UKN_PIX_TYPE";
    }
}

constexpr int align_up(int num, int align) noexcept
{
    return (num + align - 1) & ~(align - 1);
}

inline bool is_pix_type_quant(pix_type_t pix_type)
{
    switch (pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888_QINT8:
    case DL_IMAGE_PIX_TYPE_RGB888_QINT16:
    case DL_IMAGE_PIX_TYPE_BGR888_QINT8:
    case DL_IMAGE_PIX_TYPE_BGR888_QINT16:
    case DL_IMAGE_PIX_TYPE_GRAY_QINT8:
    case DL_IMAGE_PIX_TYPE_GRAY_QINT16:
        return true;
    default:
        return false;
    }
}

inline size_t get_pix_byte_size(pix_type_t pix_type)
{
    switch (pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
    case DL_IMAGE_PIX_TYPE_RGB888_QINT8:
    case DL_IMAGE_PIX_TYPE_BGR888:
    case DL_IMAGE_PIX_TYPE_BGR888_QINT8:
    case DL_IMAGE_PIX_TYPE_HSV:
        return 3;
    case DL_IMAGE_PIX_TYPE_RGB565LE:
    case DL_IMAGE_PIX_TYPE_RGB565BE:
    case DL_IMAGE_PIX_TYPE_BGR565LE:
    case DL_IMAGE_PIX_TYPE_BGR565BE:
    case DL_IMAGE_PIX_TYPE_GRAY_QINT16:
    case DL_IMAGE_PIX_TYPE_YUYV:
    case DL_IMAGE_PIX_TYPE_UYVY:
        return 2;
    case DL_IMAGE_PIX_TYPE_GRAY:
    case DL_IMAGE_PIX_TYPE_GRAY_QINT8:
    case DL_IMAGE_PIX_TYPE_HSV_MASK:
        return 1;
    case DL_IMAGE_PIX_TYPE_RGB888_QINT16:
    case DL_IMAGE_PIX_TYPE_BGR888_QINT16:
        return 6;
    default:
        return 0;
    }
}

inline int get_pix_channel_num(pix_type_t pix_type)
{
    switch (pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
    case DL_IMAGE_PIX_TYPE_RGB888_QINT8:
    case DL_IMAGE_PIX_TYPE_RGB888_QINT16:
    case DL_IMAGE_PIX_TYPE_BGR888:
    case DL_IMAGE_PIX_TYPE_BGR888_QINT8:
    case DL_IMAGE_PIX_TYPE_BGR888_QINT16:
    case DL_IMAGE_PIX_TYPE_RGB565LE:
    case DL_IMAGE_PIX_TYPE_RGB565BE:
    case DL_IMAGE_PIX_TYPE_BGR565LE:
    case DL_IMAGE_PIX_TYPE_BGR565BE:
    case DL_IMAGE_PIX_TYPE_HSV:
    case DL_IMAGE_PIX_TYPE_YUYV:
    case DL_IMAGE_PIX_TYPE_UYVY:
        return 3;
    case DL_IMAGE_PIX_TYPE_GRAY:
    case DL_IMAGE_PIX_TYPE_GRAY_QINT8:
    case DL_IMAGE_PIX_TYPE_GRAY_QINT16:
    case DL_IMAGE_PIX_TYPE_HSV_MASK:
        return 1;
    default:
        return -1;
    }
}

typedef struct img_s {
    void *data;
    uint16_t width;
    uint16_t height;
    pix_type_t pix_type;

    int channel() const { return get_pix_channel_num(pix_type); }
    size_t bytes() const { return width * height * get_pix_byte_size(pix_type); }
    bool pix_quant() const { return is_pix_type_quant(pix_type); }
    int col_step() const { return get_pix_byte_size(pix_type); }
    int row_step() const { return width * get_pix_byte_size(pix_type); }
} img_t;

typedef struct {
    void *data;
    size_t data_len;
} jpeg_img_t;
} // namespace image
} // namespace dl
