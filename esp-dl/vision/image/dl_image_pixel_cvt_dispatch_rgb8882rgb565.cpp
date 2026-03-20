#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_rgb8882rgb565(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_RGB8882RGB565LE:
    case DL_IMAGE_PIX_CVT_BGR8882BGR565LE:
        func(RGB8882RGB565<false, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB8882RGB565BE:
    case DL_IMAGE_PIX_CVT_BGR8882BGR565BE:
        func(RGB8882RGB565<true, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB8882BGR565LE:
    case DL_IMAGE_PIX_CVT_BGR8882RGB565LE:
        func(RGB8882RGB565<false, true>());
        break;
    case DL_IMAGE_PIX_CVT_RGB8882BGR565BE:
    case DL_IMAGE_PIX_CVT_BGR8882RGB565BE:
        func(RGB8882RGB565<true, true>());
        break;
    default:
        return;
    }
}

template void pixel_cvt_dispatch_rgb8882rgb565<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                                pix_cvt_t pix_cvt_type,
                                                                const pix_cvt_param_t &param);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_rgb8882rgb565<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
#endif
template void pixel_cvt_dispatch_rgb8882rgb565<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
} // namespace image
} // namespace dl
