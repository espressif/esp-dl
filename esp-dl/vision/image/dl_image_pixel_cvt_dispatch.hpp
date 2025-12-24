#pragma once
#include "dl_image_define.hpp"
#include "esp_err.h"

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_rgb5652rgb565(const Func &func, pix_cvt_t pix_cvt_type, void *norm_quant);

template <typename Func>
void pixel_cvt_dispatch_rgb5652rgb888(const Func &func, pix_cvt_t pix_cvt_type, void *norm_quant);

template <typename Func>
void pixel_cvt_dispatch_rgb5652gray(const Func &func, pix_cvt_t pix_cvt_type, void *norm_quant);

template <typename Func>
void pixel_cvt_dispatch_rgb5652hsv(const Func &func, pix_cvt_t pix_cvt_type, void *norm_quant);

template <typename Func>
void pixel_cvt_dispatch_rgb8882rgb888(const Func &func, pix_cvt_t pix_cvt_type, void *norm_quant);

template <typename Func>
void pixel_cvt_dispatch_rgb8882rgb565(const Func &func, pix_cvt_t pix_cvt_type, void *norm_quant);

template <typename Func>
void pixel_cvt_dispatch_rgb8882gray(const Func &func, pix_cvt_t pix_cvt_type, void *norm_quant);

template <typename Func>
void pixel_cvt_dispatch_rgb8882hsv(const Func &func, pix_cvt_t pix_cvt_type, void *norm_quant);

template <typename Func>
void pixel_cvt_dispatch_gray2gray(const Func &func, pix_cvt_t pix_cvt_type, void *norm_quant);

template <typename Func>
esp_err_t pixel_cvt_dispatch(const Func &func, pix_type_t src_pix_type, pix_type_t dst_pix_type, void *norm_quant)
{
    pix_cvt_t pix_cvt_type = (pix_cvt_t)DL_IMAGE_PIX_CVT_ID(src_pix_type, dst_pix_type);
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_RGB8882RGB888:
    case DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_RGB8882BGR888:
    case DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR8882RGB888:
    case DL_IMAGE_PIX_CVT_BGR8882RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR8882RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR8882BGR888:
    case DL_IMAGE_PIX_CVT_BGR8882BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR8882BGR888_QINT16:
        pixel_cvt_dispatch_rgb8882rgb888(func, pix_cvt_type, norm_quant);
        break;
    case DL_IMAGE_PIX_CVT_RGB8882GRAY:
    case DL_IMAGE_PIX_CVT_RGB8882GRAY_QINT8:
    case DL_IMAGE_PIX_CVT_RGB8882GRAY_QINT16:
    case DL_IMAGE_PIX_CVT_BGR8882GRAY:
    case DL_IMAGE_PIX_CVT_BGR8882GRAY_QINT8:
    case DL_IMAGE_PIX_CVT_BGR8882GRAY_QINT16:
        pixel_cvt_dispatch_rgb8882gray(func, pix_cvt_type, norm_quant);
        break;
    case DL_IMAGE_PIX_CVT_RGB8882RGB565LE:
    case DL_IMAGE_PIX_CVT_RGB8882RGB565BE:
    case DL_IMAGE_PIX_CVT_RGB8882BGR565LE:
    case DL_IMAGE_PIX_CVT_RGB8882BGR565BE:
    case DL_IMAGE_PIX_CVT_BGR8882RGB565LE:
    case DL_IMAGE_PIX_CVT_BGR8882RGB565BE:
    case DL_IMAGE_PIX_CVT_BGR8882BGR565LE:
    case DL_IMAGE_PIX_CVT_BGR8882BGR565BE:
        pixel_cvt_dispatch_rgb8882rgb565(func, pix_cvt_type, norm_quant);
        break;
    case DL_IMAGE_PIX_CVT_RGB8882HSV:
    case DL_IMAGE_PIX_CVT_BGR8882HSV:
        pixel_cvt_dispatch_rgb8882hsv(func, pix_cvt_type, norm_quant);
        break;
    case DL_IMAGE_PIX_CVT_GRAY2GRAY:
    case DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT8:
    case DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT16:
        pixel_cvt_dispatch_gray2gray(func, pix_cvt_type, norm_quant);
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB888:
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR888:
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT16:
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB888:
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR888:
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB888:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR888:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB888:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR888:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR888_QINT16:
        pixel_cvt_dispatch_rgb5652rgb888(func, pix_cvt_type, norm_quant);
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2GRAY:
    case DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT8:
    case DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT16:
    case DL_IMAGE_PIX_CVT_RGB565BE2GRAY:
    case DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT8:
    case DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565LE2GRAY:
    case DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565BE2GRAY:
    case DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT16:
        pixel_cvt_dispatch_rgb5652gray(func, pix_cvt_type, norm_quant);
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB565LE:
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB565BE:
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR565LE:
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR565BE:
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB565LE:
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB565BE:
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR565LE:
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR565BE:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB565LE:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB565BE:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR565LE:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR565BE:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB565LE:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB565BE:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR565LE:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR565BE:
        pixel_cvt_dispatch_rgb5652rgb565(func, pix_cvt_type, norm_quant);
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2HSV:
    case DL_IMAGE_PIX_CVT_RGB565BE2HSV:
    case DL_IMAGE_PIX_CVT_BGR565LE2HSV:
    case DL_IMAGE_PIX_CVT_BGR565BE2HSV:
        pixel_cvt_dispatch_rgb5652hsv(func, pix_cvt_type, norm_quant);
        break;
    default:
        ESP_LOGE("pix_cvt_dispatch",
                 "Invalid pix cvt from %s to %s.",
                 pix_type2str(src_pix_type).c_str(),
                 pix_type2str(dst_pix_type).c_str());
        return ESP_FAIL;
    }
    return ESP_OK;
}

struct CvtPixelFunctor {
    const uint8_t *src;
    uint8_t *dst;
    template <typename PixelCvt>
    void operator()(const PixelCvt &pixel_cvt) const
    {
        pixel_cvt(src, dst);
    }
};

inline esp_err_t cvt_pix(
    const uint8_t *src, uint8_t *dst, pix_type_t src_pix_type, pix_type_t dst_pix_type, void *norm_quant = nullptr)
{
    CvtPixelFunctor fn{src, dst};
    return pixel_cvt_dispatch(fn, src_pix_type, dst_pix_type, norm_quant);
}
} // namespace image
} // namespace dl
