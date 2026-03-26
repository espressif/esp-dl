#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_image_process.hpp"

namespace dl {
namespace image {
template <typename Func>
void pixel_cvt_dispatch_rgb5652rgb888(const Func &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param)
{
    switch (pix_cvt_type) {
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB888:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR888:
        func(RGB5652RGB888<false, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR888_QINT8:
        func(RGB5652RGB888<false, false, NormQuant<int8_t, 3>>(std::get<NormQuant<int8_t, 3>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR888_QINT16:
        func(RGB5652RGB888<false, false, NormQuant<int16_t, 3>>(std::get<NormQuant<int16_t, 3>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR888:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB888:
        func(RGB5652RGB888<false, true>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB888_QINT8:
        func(RGB5652RGB888<false, true, NormQuant<int8_t, 3>>(std::get<NormQuant<int8_t, 3>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB888_QINT16:
        func(RGB5652RGB888<false, true, NormQuant<int16_t, 3>>(std::get<NormQuant<int16_t, 3>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB888:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR888:
        func(RGB5652RGB888<true, false>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR888_QINT8:
        func(RGB5652RGB888<true, false, NormQuant<int8_t, 3>>(std::get<NormQuant<int8_t, 3>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR888_QINT16:
        func(RGB5652RGB888<true, false, NormQuant<int16_t, 3>>(std::get<NormQuant<int16_t, 3>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR888:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB888:
        func(RGB5652RGB888<true, true>());
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB888_QINT8:
        func(RGB5652RGB888<true, true, NormQuant<int8_t, 3>>(std::get<NormQuant<int8_t, 3>>(param)));
        break;
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB888_QINT16:
        func(RGB5652RGB888<true, true, NormQuant<int16_t, 3>>(std::get<NormQuant<int16_t, 3>>(param)));
        break;
    default:
        return;
    }
}

template void pixel_cvt_dispatch_rgb5652rgb888<CvtPixelFunctor>(const CvtPixelFunctor &func,
                                                                pix_cvt_t pix_cvt_type,
                                                                const pix_cvt_param_t &param);
#if CONFIG_IDF_TARGET_ESP32P4
template void pixel_cvt_dispatch_rgb5652rgb888<ImageTransformer::TransformNNFunctor<true>>(
    const ImageTransformer::TransformNNFunctor<true> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
#endif
template void pixel_cvt_dispatch_rgb5652rgb888<ImageTransformer::TransformNNFunctor<false>>(
    const ImageTransformer::TransformNNFunctor<false> &func, pix_cvt_t pix_cvt_type, const pix_cvt_param_t &param);
} // namespace image
} // namespace dl
