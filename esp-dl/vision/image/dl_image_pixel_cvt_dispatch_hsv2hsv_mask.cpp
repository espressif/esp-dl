#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_hsv2hsv_mask(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    if (std::holds_alternative<HSV2HSVMask<false>>(param)) {
        func(std::get<HSV2HSVMask<false>>(param));
    } else {
        func(std::get<HSV2HSVMask<true>>(param));
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
