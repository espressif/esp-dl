#include "dl_image_color.hpp"
#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"
#include <functional>

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_rgb5652hsv(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_RGB565LE2HSV:
        func(RGB5652HSV<false, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2HSV:
        func(RGB5652HSV<true, false>());
        break;
    case DL_IMAGE_PIX_CVT_BGR565LE2HSV:
        func(RGB5652HSV<false, true>());
        break;
    case DL_IMAGE_PIX_CVT_BGR565BE2HSV:
        func(RGB5652HSV<true, true>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2HSV_MASK: {
        auto hsv_param = std::get<hsv_param_t>(param);
        if (hsv_param.h_across_zero) {
            func(RGB5652HSVMask<false, false, true>(hsv_param.hsv_min, hsv_param.hsv_max));
        } else {
            func(RGB5652HSVMask<false, false, false>(hsv_param.hsv_min, hsv_param.hsv_max));
        }
        break;
    }
    case DL_IMAGE_PIX_CVT_RGB565BE2HSV_MASK: {
        auto hsv_param = std::get<hsv_param_t>(param);
        if (hsv_param.h_across_zero) {
            func(RGB5652HSVMask<true, false, true>(hsv_param.hsv_min, hsv_param.hsv_max));
        } else {
            func(RGB5652HSVMask<true, false, false>(hsv_param.hsv_min, hsv_param.hsv_max));
        }
        break;
    }
    case DL_IMAGE_PIX_CVT_BGR565LE2HSV_MASK: {
        auto hsv_param = std::get<hsv_param_t>(param);
        if (hsv_param.h_across_zero) {
            func(RGB5652HSVMask<false, true, true>(hsv_param.hsv_min, hsv_param.hsv_max));
        } else {
            func(RGB5652HSVMask<false, true, false>(hsv_param.hsv_min, hsv_param.hsv_max));
        }
        break;
    }
    case DL_IMAGE_PIX_CVT_BGR565BE2HSV_MASK: {
        auto hsv_param = std::get<hsv_param_t>(param);
        if (hsv_param.h_across_zero) {
            func(RGB5652HSVMask<true, true, true>(hsv_param.hsv_min, hsv_param.hsv_max));
        } else {
            func(RGB5652HSVMask<true, true, false>(hsv_param.hsv_min, hsv_param.hsv_max));
        }
        break;
    }
    default:
        return;
    }
}

template void pixel_cvt_dispatch_rgb5652hsv<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                             pix_cvt_t pix_cvt_type,
                                                             const pix_cvt_param_t &param);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_rgb5652hsv<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
#endif
template void pixel_cvt_dispatch_rgb5652hsv<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
} // namespace image
} // namespace dl
