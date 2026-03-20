#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_gray2gray(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_GRAY2GRAY:
        func(Gray2Gray<>());
        break;
    case DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT8:
        func(Gray2Gray<NormQuant<int8_t, 1>>(std::get<NormQuant<int8_t, 1>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT16:
        func(Gray2Gray<NormQuant<int16_t, 1>>(std::get<NormQuant<int16_t, 1>>(param)));
        break;
    default:
        return;
    }
}

template void pixel_cvt_dispatch_gray2gray<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                            pix_cvt_t pix_cvt_type,
                                                            const pix_cvt_param_t &param);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_gray2gray<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
#endif
template void pixel_cvt_dispatch_gray2gray<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
} // namespace image
} // namespace dl
