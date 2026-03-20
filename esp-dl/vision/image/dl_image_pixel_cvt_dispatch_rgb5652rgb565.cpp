#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_rgb5652rgb565(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB565LE:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR565LE:
        func(RGB5652RGB565<false, false, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB565BE:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR565BE:
        func(RGB5652RGB565<true, false, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR565LE:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB565LE:
        func(RGB5652RGB565<false, true, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR565BE:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB565BE:
        func(RGB5652RGB565<true, true, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB565BE:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR565BE:
        func(RGB5652RGB565<false, false, true>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB565LE:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR565LE:
        func(RGB5652RGB565<true, false, true>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR565BE:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB565BE:
        func(RGB5652RGB565<false, true, true>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR565LE:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB565LE:
        func(RGB5652RGB565<true, true, true>());
        break;
    default:
        return;
    }
}

template void pixel_cvt_dispatch_rgb5652rgb565<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                                pix_cvt_t pix_cvt_type,
                                                                const pix_cvt_param_t &param);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_rgb5652rgb565<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
#endif
template void pixel_cvt_dispatch_rgb5652rgb565<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
} // namespace image
} // namespace dl
