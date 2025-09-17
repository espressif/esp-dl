#include "dl_image_color.hpp"
#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"
#include <functional>

namespace dl {
namespace image {
template <typename Func>
esp_err_t pixel_cvt_dispatch_gray(const Func &func, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant)
{
    if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY) {
        func(Gray2Gray<>());
    } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT8) {
        func(Gray2Gray<NormQuant<int8_t, 1>>(norm_quant));
    } else if (dst_pix_type == DL_IMAGE_PIX_TYPE_GRAY_QINT16) {
        func(Gray2Gray<NormQuant<int16_t, 1>>(norm_quant));
    } else {
        return ESP_FAIL;
    }
    return ESP_OK;
}

template esp_err_t pixel_cvt_dispatch_gray<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                            pix_type_t dst_pix_type,
                                                            uint32_t caps,
                                                            void *norm_quant);
#if CONFIG_IDF_TARGET_ESP32P4
template esp_err_t pixel_cvt_dispatch_gray<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant);
#endif
template esp_err_t pixel_cvt_dispatch_gray<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_type_t dst_pix_type, uint32_t caps, void *norm_quant);
} // namespace image
} // namespace dl
