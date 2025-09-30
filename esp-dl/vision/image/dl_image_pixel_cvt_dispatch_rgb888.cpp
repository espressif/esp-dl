#include "dl_image_color.hpp"
#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"
#include <functional>

namespace dl {
namespace image {
template <typename Func>
esp_err_t pixel_cvt_dispatch_rgb888(const Func &func, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant)
{
    bool rgb565be = caps & DL_IMAGE_CAP_RGB565_BIG_ENDIAN;
    bool rgb_swap = caps & DL_IMAGE_CAP_RGB_SWAP;
    if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB888) {
        if (rgb_swap) {
            func(RGB8882RGB888<true>());
        } else {
            func(RGB8882RGB888<false>());
        }
    } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB888_QINT8) {
        if (rgb_swap) {
            func(RGB8882RGB888<true, NormQuant<int8_t, 3>>(norm_quant));
        } else {
            func(RGB8882RGB888<false, NormQuant<int8_t, 3>>(norm_quant));
        }
    } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_RGB888_QINT16) {
        if (rgb_swap) {
            func(RGB8882RGB888<true, NormQuant<int16_t, 3>>(norm_quant));
        } else {
            func(RGB8882RGB888<false, NormQuant<int16_t, 3>>(norm_quant));
        }
    } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY) {
        if (rgb_swap) {
            func(RGB8882Gray<true>());
        } else {
            func(RGB8882Gray<false>());
        }
    } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT8) {
        if (rgb_swap) {
            func(RGB8882Gray<true, NormQuant<int8_t, 1>>(norm_quant));
        } else {
            func(RGB8882Gray<false, NormQuant<int8_t, 1>>(norm_quant));
        }
    } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT16) {
        if (rgb_swap) {
            func(RGB8882Gray<true, NormQuant<int16_t, 1>>(norm_quant));
        } else {
            func(RGB8882Gray<false, NormQuant<int16_t, 1>>(norm_quant));
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
    } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_HSV) {
        if (rgb_swap) {
            func(RGB8882HSV<true>());
        } else {
            func(RGB8882HSV<false>());
        }
    } else {
        return ESP_FAIL;
    }
    return ESP_OK;
}

template esp_err_t pixel_cvt_dispatch_rgb888<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                              pix_type_t dst_pix_type,
                                                              uint32_t caps,
                                                              void *norm_quant);
#if CONFIG_IDF_TARGET_ESP32P4
template esp_err_t pixel_cvt_dispatch_rgb888<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant);
#endif
template esp_err_t pixel_cvt_dispatch_rgb888<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant);
} // namespace image
} // namespace dl
