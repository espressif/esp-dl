#include "dl_image_color.hpp"
#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"
#include <functional>

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_rgb8882hsv(const Func &func, pix_cvt_t pix_cvt_type, void *norm_quant)
{
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_RGB8882HSV:
        func(RGB8882HSV<false>());
        break;
    case DL_IMAGE_PIX_CVT_BGR8882HSV:
        func(RGB8882HSV<true>());
        break;
    default:
        return;
    }
}

template void pixel_cvt_dispatch_rgb8882hsv<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                             pix_cvt_t pix_cvt_type,
                                                             void *norm_quant);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_rgb8882hsv<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, void *norm_quant);
#endif
template void pixel_cvt_dispatch_rgb8882hsv<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, void *norm_quant);
} // namespace image
} // namespace dl
