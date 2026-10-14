#pragma once
#include "dl_image_pixel_kernel.hpp"

namespace dl {
namespace image {

template <pix_cvt_t Id>
struct plain_traits;

template <pix_cvt_t Id>
struct norm_quant_traits;

template <pix_cvt_t Id, bool Across>
struct hsv_mask_extra_traits;

template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB8882RGB888> {
    using type = RGB8882RGB888<false>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT8> {
    using type = RGB8882RGB888<false, NormQuant<int8_t, 3>>;
    using extra = NormQuant<int8_t, 3>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT16> {
    using type = RGB8882RGB888<false, NormQuant<int16_t, 3>>;
    using extra = NormQuant<int16_t, 3>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB8882BGR888> {
    using type = RGB8882RGB888<true>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT8> {
    using type = RGB8882RGB888<true, NormQuant<int8_t, 3>>;
    using extra = NormQuant<int8_t, 3>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT16> {
    using type = RGB8882RGB888<true, NormQuant<int16_t, 3>>;
    using extra = NormQuant<int16_t, 3>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB8882GRAY> {
    using type = RGB8882Gray<false>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB8882GRAY_QINT8> {
    using type = RGB8882Gray<false, NormQuant<int8_t, 1>>;
    using extra = NormQuant<int8_t, 1>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB8882GRAY_QINT16> {
    using type = RGB8882Gray<false, NormQuant<int16_t, 1>>;
    using extra = NormQuant<int16_t, 1>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_BGR8882GRAY> {
    using type = RGB8882Gray<true>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_BGR8882GRAY_QINT8> {
    using type = RGB8882Gray<true, NormQuant<int8_t, 1>>;
    using extra = NormQuant<int8_t, 1>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_BGR8882GRAY_QINT16> {
    using type = RGB8882Gray<true, NormQuant<int16_t, 1>>;
    using extra = NormQuant<int16_t, 1>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB8882RGB565LE> {
    using type = RGB8882RGB565<false, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB8882RGB565BE> {
    using type = RGB8882RGB565<true, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB8882BGR565LE> {
    using type = RGB8882RGB565<false, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB8882BGR565BE> {
    using type = RGB8882RGB565<true, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB8882HSV> {
    using type = RGB8882HSV<false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_BGR8882HSV> {
    using type = RGB8882HSV<true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_GRAY2GRAY> {
    using type = Gray2Gray<>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT8> {
    using type = Gray2Gray<NormQuant<int8_t, 1>>;
    using extra = NormQuant<int8_t, 1>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT16> {
    using type = Gray2Gray<NormQuant<int16_t, 1>>;
    using extra = NormQuant<int16_t, 1>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565LE2RGB888> {
    using type = RGB5652RGB888<false, false>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT8> {
    using type = RGB5652RGB888<false, false, NormQuant<int8_t, 3>>;
    using extra = NormQuant<int8_t, 3>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT16> {
    using type = RGB5652RGB888<false, false, NormQuant<int16_t, 3>>;
    using extra = NormQuant<int16_t, 3>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565LE2BGR888> {
    using type = RGB5652RGB888<false, true>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT8> {
    using type = RGB5652RGB888<false, true, NormQuant<int8_t, 3>>;
    using extra = NormQuant<int8_t, 3>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT16> {
    using type = RGB5652RGB888<false, true, NormQuant<int16_t, 3>>;
    using extra = NormQuant<int16_t, 3>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565BE2RGB888> {
    using type = RGB5652RGB888<true, false>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT8> {
    using type = RGB5652RGB888<true, false, NormQuant<int8_t, 3>>;
    using extra = NormQuant<int8_t, 3>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT16> {
    using type = RGB5652RGB888<true, false, NormQuant<int16_t, 3>>;
    using extra = NormQuant<int16_t, 3>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565BE2BGR888> {
    using type = RGB5652RGB888<true, true>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT8> {
    using type = RGB5652RGB888<true, true, NormQuant<int8_t, 3>>;
    using extra = NormQuant<int8_t, 3>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT16> {
    using type = RGB5652RGB888<true, true, NormQuant<int16_t, 3>>;
    using extra = NormQuant<int16_t, 3>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565LE2GRAY> {
    using type = RGB5652Gray<false, false>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT8> {
    using type = RGB5652Gray<false, false, NormQuant<int8_t, 1>>;
    using extra = NormQuant<int8_t, 1>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT16> {
    using type = RGB5652Gray<false, false, NormQuant<int16_t, 1>>;
    using extra = NormQuant<int16_t, 1>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565BE2GRAY> {
    using type = RGB5652Gray<true, false>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT8> {
    using type = RGB5652Gray<true, false, NormQuant<int8_t, 1>>;
    using extra = NormQuant<int8_t, 1>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT16> {
    using type = RGB5652Gray<true, false, NormQuant<int16_t, 1>>;
    using extra = NormQuant<int16_t, 1>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_BGR565LE2GRAY> {
    using type = RGB5652Gray<false, true>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT8> {
    using type = RGB5652Gray<false, true, NormQuant<int8_t, 1>>;
    using extra = NormQuant<int8_t, 1>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT16> {
    using type = RGB5652Gray<false, true, NormQuant<int16_t, 1>>;
    using extra = NormQuant<int16_t, 1>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_BGR565BE2GRAY> {
    using type = RGB5652Gray<true, true>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT8> {
    using type = RGB5652Gray<true, true, NormQuant<int8_t, 1>>;
    using extra = NormQuant<int8_t, 1>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT16> {
    using type = RGB5652Gray<true, true, NormQuant<int16_t, 1>>;
    using extra = NormQuant<int16_t, 1>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565LE2RGB565LE> {
    using type = RGB5652RGB565<false, false, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565BE2RGB565BE> {
    using type = RGB5652RGB565<true, false, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565LE2BGR565LE> {
    using type = RGB5652RGB565<false, true, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565BE2BGR565BE> {
    using type = RGB5652RGB565<true, true, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565LE2RGB565BE> {
    using type = RGB5652RGB565<false, false, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565BE2RGB565LE> {
    using type = RGB5652RGB565<true, false, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565LE2BGR565BE> {
    using type = RGB5652RGB565<false, true, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565BE2BGR565LE> {
    using type = RGB5652RGB565<true, true, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565LE2HSV> {
    using type = RGB5652HSV<false, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_RGB565BE2HSV> {
    using type = RGB5652HSV<true, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_BGR565LE2HSV> {
    using type = RGB5652HSV<false, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_BGR565BE2HSV> {
    using type = RGB5652HSV<true, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_YUYV2GRAY> {
    using type = YUYV2Gray<true>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_YUYV2GRAY_QINT8> {
    using type = YUYV2Gray<true, NormQuant<int8_t, 1>>;
    using extra = NormQuant<int8_t, 1>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_YUYV2GRAY_QINT16> {
    using type = YUYV2Gray<true, NormQuant<int16_t, 1>>;
    using extra = NormQuant<int16_t, 1>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_UYVY2GRAY> {
    using type = YUYV2Gray<false>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_UYVY2GRAY_QINT8> {
    using type = YUYV2Gray<false, NormQuant<int8_t, 1>>;
    using extra = NormQuant<int8_t, 1>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_UYVY2GRAY_QINT16> {
    using type = YUYV2Gray<false, NormQuant<int16_t, 1>>;
    using extra = NormQuant<int16_t, 1>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_YUYV2RGB888> {
    using type = YUYV2RGB888<true, false>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_YUYV2RGB888_QINT8> {
    using type = YUYV2RGB888<true, false, NormQuant<int8_t, 3>>;
    using extra = NormQuant<int8_t, 3>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_YUYV2RGB888_QINT16> {
    using type = YUYV2RGB888<true, false, NormQuant<int16_t, 3>>;
    using extra = NormQuant<int16_t, 3>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_YUYV2BGR888> {
    using type = YUYV2RGB888<true, true>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_YUYV2BGR888_QINT8> {
    using type = YUYV2RGB888<true, true, NormQuant<int8_t, 3>>;
    using extra = NormQuant<int8_t, 3>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_YUYV2BGR888_QINT16> {
    using type = YUYV2RGB888<true, true, NormQuant<int16_t, 3>>;
    using extra = NormQuant<int16_t, 3>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_UYVY2RGB888> {
    using type = YUYV2RGB888<false, false>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_UYVY2RGB888_QINT8> {
    using type = YUYV2RGB888<false, false, NormQuant<int8_t, 3>>;
    using extra = NormQuant<int8_t, 3>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_UYVY2RGB888_QINT16> {
    using type = YUYV2RGB888<false, false, NormQuant<int16_t, 3>>;
    using extra = NormQuant<int16_t, 3>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_UYVY2BGR888> {
    using type = YUYV2RGB888<false, true>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_UYVY2BGR888_QINT8> {
    using type = YUYV2RGB888<false, true, NormQuant<int8_t, 3>>;
    using extra = NormQuant<int8_t, 3>;
};
template <>
struct norm_quant_traits<DL_IMAGE_PIX_CVT_UYVY2BGR888_QINT16> {
    using type = YUYV2RGB888<false, true, NormQuant<int16_t, 3>>;
    using extra = NormQuant<int16_t, 3>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_YUYV2RGB565LE> {
    using type = YUYV2RGB565<true, false, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_YUYV2RGB565BE> {
    using type = YUYV2RGB565<true, true, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_YUYV2BGR565LE> {
    using type = YUYV2RGB565<true, false, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_YUYV2BGR565BE> {
    using type = YUYV2RGB565<true, true, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_UYVY2RGB565LE> {
    using type = YUYV2RGB565<false, false, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_UYVY2RGB565BE> {
    using type = YUYV2RGB565<false, true, false>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_UYVY2BGR565LE> {
    using type = YUYV2RGB565<false, false, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_UYVY2BGR565BE> {
    using type = YUYV2RGB565<false, true, true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_YUYV2HSV> {
    using type = YUYV2HSV<true>;
};
template <>
struct plain_traits<DL_IMAGE_PIX_CVT_UYVY2HSV> {
    using type = YUYV2HSV<false>;
};
template <bool Across>
struct hsv_mask_extra_traits<DL_IMAGE_PIX_CVT_YUYV2HSV_MASK, Across> {
    using type = YUYV2HSV<true, HSV2HSVMask<Across>>;
    using extra = HSV2HSVMask<Across>;
};
template <bool Across>
struct hsv_mask_extra_traits<DL_IMAGE_PIX_CVT_UYVY2HSV_MASK, Across> {
    using type = YUYV2HSV<false, HSV2HSVMask<Across>>;
    using extra = HSV2HSVMask<Across>;
};
template <bool Across>
struct hsv_mask_extra_traits<DL_IMAGE_PIX_CVT_RGB8882HSV_MASK, Across> {
    using type = RGB8882HSV<false, HSV2HSVMask<Across>>;
    using extra = HSV2HSVMask<Across>;
};
template <bool Across>
struct hsv_mask_extra_traits<DL_IMAGE_PIX_CVT_BGR8882HSV_MASK, Across> {
    using type = RGB8882HSV<true, HSV2HSVMask<Across>>;
    using extra = HSV2HSVMask<Across>;
};
template <bool Across>
struct hsv_mask_extra_traits<DL_IMAGE_PIX_CVT_RGB565LE2HSV_MASK, Across> {
    using type = RGB5652HSV<false, false, HSV2HSVMask<Across>>;
    using extra = HSV2HSVMask<Across>;
};
template <bool Across>
struct hsv_mask_extra_traits<DL_IMAGE_PIX_CVT_RGB565BE2HSV_MASK, Across> {
    using type = RGB5652HSV<true, false, HSV2HSVMask<Across>>;
    using extra = HSV2HSVMask<Across>;
};
template <bool Across>
struct hsv_mask_extra_traits<DL_IMAGE_PIX_CVT_BGR565LE2HSV_MASK, Across> {
    using type = RGB5652HSV<false, true, HSV2HSVMask<Across>>;
    using extra = HSV2HSVMask<Across>;
};
template <bool Across>
struct hsv_mask_extra_traits<DL_IMAGE_PIX_CVT_BGR565BE2HSV_MASK, Across> {
    using type = RGB5652HSV<true, true, HSV2HSVMask<Across>>;
    using extra = HSV2HSVMask<Across>;
};

} // namespace image
} // namespace dl
