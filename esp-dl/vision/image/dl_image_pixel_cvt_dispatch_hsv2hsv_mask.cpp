#include "dl_image_color.hpp"
#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"
#include <functional>

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_hsv2hsv_mask(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    auto hsv_param = std::get<hsv_param_t>(param);
    if (hsv_param.h_across_zero) {
        func(HSV2HSVMask<true>(hsv_param.hsv_min, hsv_param.hsv_max));
    } else {
        func(HSV2HSVMask<false>(hsv_param.hsv_min, hsv_param.hsv_max));
    }
}

template void pixel_cvt_dispatch_hsv2hsv_mask<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                               pix_cvt_t pix_cvt_type,
                                                               const pix_cvt_param_t &param);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_hsv2hsv_mask<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
#endif
template void pixel_cvt_dispatch_hsv2hsv_mask<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
} // namespace image
} // namespace dl
