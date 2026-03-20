#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_rgb5652gray(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_RGB565LE2GRAY:
        func(RGB5652Gray<false, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT8:
        func(RGB5652Gray<false, false, NormQuant<int8_t, 1>>(std::get<NormQuant<int8_t, 1>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT16:
        func(RGB5652Gray<false, false, NormQuant<int16_t, 1>>(std::get<NormQuant<int16_t, 1>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2GRAY:
        func(RGB5652Gray<true, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT8:
        func(RGB5652Gray<true, false, NormQuant<int8_t, 1>>(std::get<NormQuant<int8_t, 1>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT16:
        func(RGB5652Gray<true, false, NormQuant<int16_t, 1>>(std::get<NormQuant<int16_t, 1>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_BGR565LE2GRAY:
        func(RGB5652Gray<false, true>());
        break;
    case DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT8:
        func(RGB5652Gray<false, true, NormQuant<int8_t, 1>>(std::get<NormQuant<int8_t, 1>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT16:
        func(RGB5652Gray<false, true, NormQuant<int16_t, 1>>(std::get<NormQuant<int16_t, 1>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_BGR565BE2GRAY:
        func(RGB5652Gray<true, true>());
        break;
    case DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT8:
        func(RGB5652Gray<true, true, NormQuant<int8_t, 1>>(std::get<NormQuant<int8_t, 1>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT16:
        func(RGB5652Gray<true, true, NormQuant<int16_t, 1>>(std::get<NormQuant<int16_t, 1>>(param)));
        break;
    default:
        return;
    }
}

template void pixel_cvt_dispatch_rgb5652gray<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                              pix_cvt_t pix_cvt_type,
                                                              const pix_cvt_param_t &param);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_rgb5652gray<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
#endif
template void pixel_cvt_dispatch_rgb5652gray<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
} // namespace image
} // namespace dl
