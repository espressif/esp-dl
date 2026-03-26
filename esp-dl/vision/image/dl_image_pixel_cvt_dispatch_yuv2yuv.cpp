#include "dl_image_color.hpp"
#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_yuv2yuv(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_YUYV2YUYV:
    case DL_IMAGE_PIX_CVT_UYVY2UYVY:
        func(YUV2YUV<false>());
        break;
    case DL_IMAGE_PIX_CVT_YUYV2UYVY:
    case DL_IMAGE_PIX_CVT_UYVY2YUYV:
        func(YUV2YUV<true>());
        break;
    default:
        return;
    }
}

template void pixel_cvt_dispatch_yuv2yuv<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                          pix_cvt_t pix_cvt_type,
                                                          const pix_cvt_param_t &param);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_yuv2yuv<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
#endif
template void pixel_cvt_dispatch_yuv2yuv<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
} // namespace image
} // namespace dl
