#pragma once
#include "dl_image_define.hpp"
#include "esp_err.h"

namespace dl {
namespace image {
template <typename Func>
esp_err_t pixel_cvt_dispatch_rgb888(const Func &func, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant);

template <typename Func>
esp_err_t pixel_cvt_dispatch_rgb565(const Func &func, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant);

template <typename Func>
esp_err_t pixel_cvt_dispatch_gray(const Func &func, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant);

template <typename Func>
esp_err_t pixel_cvt_dispatch(
    const Func &func, pix_type_t src_pix_type, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant)
{
    if (src_pix_type == DL_IMAGE_PIX_TYPE_RGB565) {
        return pixel_cvt_dispatch_rgb565(func, dst_pix_type, caps, norm_quant);
    } else if (src_pix_type == DL_IMAGE_PIX_TYPE_RGB888) {
        return pixel_cvt_dispatch_rgb888(func, dst_pix_type, caps, norm_quant);
    } else if (src_pix_type == DL_IMAGE_PIX_TYPE_GRAY) {
        return pixel_cvt_dispatch_gray(func, dst_pix_type, caps, norm_quant);
    } else {
        return ESP_FAIL;
    }
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

inline esp_err_t cvt_pix(const uint8_t *src,
                         uint8_t *dst,
                         pix_type_t src_pix_type,
                         pix_type_t dst_pix_type,
                         uint32_t caps = 0,
                         void *norm_quant = nullptr)
{
    CvtPixelFunctor fn{src, dst};
    return pixel_cvt_dispatch(fn, src_pix_type, dst_pix_type, caps, norm_quant);
}
} // namespace image
} // namespace dl
