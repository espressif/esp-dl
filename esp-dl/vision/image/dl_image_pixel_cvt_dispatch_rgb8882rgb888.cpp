#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_rgb8882rgb888(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_RGB8882RGB888:
    case DL_IMAGE_PIX_CVT_BGR8882BGR888:
        func(RGB8882RGB888<false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR8882BGR888_QINT8:
        func(RGB8882RGB888<false, NormQuant<int8_t, 3>>(std::get<NormQuant<int8_t, 3>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR8882BGR888_QINT16:
        func(RGB8882RGB888<false, NormQuant<int16_t, 3>>(std::get<NormQuant<int16_t, 3>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB8882BGR888:
    case DL_IMAGE_PIX_CVT_BGR8882RGB888:
        func(RGB8882RGB888<true>());
        break;
    case DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR8882RGB888_QINT8:
        func(RGB8882RGB888<true, NormQuant<int8_t, 3>>(std::get<NormQuant<int8_t, 3>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR8882RGB888_QINT16:
        func(RGB8882RGB888<true, NormQuant<int16_t, 3>>(std::get<NormQuant<int16_t, 3>>(param)));
        break;
    default:
        return;
    }
}

template void pixel_cvt_dispatch_rgb8882rgb888<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                                pix_cvt_t pix_cvt_type,
                                                                const pix_cvt_param_t &param);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_rgb8882rgb888<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
#endif
template void pixel_cvt_dispatch_rgb8882rgb888<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
} // namespace image
} // namespace dl
