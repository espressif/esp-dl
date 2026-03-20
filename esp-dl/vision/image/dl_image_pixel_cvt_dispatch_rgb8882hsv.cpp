#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_rgb8882hsv(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_RGB8882HSV:
        func(RGB8882HSV<false>());
        break;
    case DL_IMAGE_PIX_CVT_BGR8882HSV:
        func(RGB8882HSV<true>());
        break;
    case DL_IMAGE_PIX_CVT_RGB8882HSV_MASK:
        if (std::holds_alternative<HSV2HSVMask<false>>(param)) {
            func(RGB8882HSV<false, HSV2HSVMask<false>>(std::get<HSV2HSVMask<false>>(param)));
        } else {
            func(RGB8882HSV<false, HSV2HSVMask<true>>(std::get<HSV2HSVMask<true>>(param)));
        }
        break;
    case DL_IMAGE_PIX_CVT_BGR8882HSV_MASK:
        if (std::holds_alternative<HSV2HSVMask<false>>(param)) {
            func(RGB8882HSV<true, HSV2HSVMask<false>>(std::get<HSV2HSVMask<false>>(param)));
        } else {
            func(RGB8882HSV<true, HSV2HSVMask<true>>(std::get<HSV2HSVMask<true>>(param)));
        }
        break;
    default:
        return;
    }
}

template void pixel_cvt_dispatch_rgb8882hsv<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                             pix_cvt_t pix_cvt_type,
                                                             const pix_cvt_param_t &param);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_rgb8882hsv<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
#endif
template void pixel_cvt_dispatch_rgb8882hsv<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
} // namespace image
} // namespace dl
