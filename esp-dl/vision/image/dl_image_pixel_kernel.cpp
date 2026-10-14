#include "dl_image_pixel_cvt_traits.hpp"
#include "sdkconfig.h"
#include <cstring>
#include <type_traits>

namespace dl {
namespace image {

template <pix_cvt_t Id, typename Cvt>
    requires(pix_cvt_is_2pix(Id))
void bind_thunk(PixelKernel *k)
{
    // An odd start consumes one pixel and leaves src on an even pixel. The leftover pixel is even.
    k->cvt_2pix_scalar_fn = [](const PixelKernel *s, uint8_t *src, uint8_t *dst, int n, bool odd) {
        const Cvt &c = *static_cast<const Cvt *>(s->cvt);
        constexpr int src_bpp = Cvt::src_bpp;
        constexpr int dst_bpp = Cvt::dst_bpp;
        constexpr int src_bpp_2x = src_bpp * 2;
        constexpr int dst_bpp_2x = dst_bpp * 2;
        if (odd) {
            c(src, dst, true);
            src += src_bpp;
            dst += dst_bpp;
            --n;
        }
        const int tail = n & 1;
        const int pairs = n >> 1;
        for (int i = 0; i < pairs; ++i, src += src_bpp_2x, dst += dst_bpp_2x) {
            c(src, dst);
        }
        if (tail) {
            c(src, dst, false);
        }
    };
    k->resize_nn_scalar_fn = [](const PixelKernel *s, uint8_t *src_row, int *x, uint8_t *dst, int n) {
        const Cvt &c = *static_cast<const Cvt *>(s->cvt);
        constexpr int dst_bpp = Cvt::dst_bpp;
        for (int i = 0; i < n; ++i, dst += dst_bpp) {
            const int row_offset = x[i];
            c(src_row + row_offset, dst, row_offset & 0b10);
        }
    };
    k->warp_nn_fn = [](const PixelKernel *s, uint8_t *src, const int *offset, uint8_t *dst, int n, const uint8_t *bg) {
        const Cvt &c = *static_cast<const Cvt *>(s->cvt);
        constexpr int dst_bpp = Cvt::dst_bpp;
        for (int i = 0; i < n; ++i, dst += dst_bpp) {
            if (offset[i] >= 0) {
                const int row_offset = offset[i];
                c(src + row_offset, dst, row_offset & 0b10);
            } else {
                memcpy(dst, bg, dst_bpp);
            }
        }
    };
#if CONFIG_PIE_V2_BOOST
    k->cvt_2pix_simd_fn = [](const PixelKernel *s, uint8_t *src, uint8_t *dst, int n, bool odd) {
        const Cvt &c = *static_cast<const Cvt *>(s->cvt);
        constexpr int src_bpp = Cvt::src_bpp;
        constexpr int dst_bpp = Cvt::dst_bpp;
        constexpr int src_bpp_2x = src_bpp * 2;
        constexpr int dst_bpp_2x = dst_bpp * 2;
        if (odd) {
            c(src, dst, true);
            src += src_bpp;
            dst += dst_bpp;
            --n;
        }
        const int n16 = n >> 4;
        if (n16) {
            c.cvt_color_simd_helper(src, dst, n16);
            const int done = n16 << 4;
            src += done * src_bpp;
            dst += done * dst_bpp;
            n -= done;
        }
        const int tail = n & 1;
        const int pairs = n >> 1;
        for (int i = 0; i < pairs; ++i, src += src_bpp_2x, dst += dst_bpp_2x) {
            c(src, dst);
        }
        if (tail) {
            c(src, dst, false);
        }
    };
    k->resize_nn_simd_fn = [](const PixelKernel *s, uint8_t *src_row, int *x, uint8_t *dst, int n) {
        const Cvt &c = *static_cast<const Cvt *>(s->cvt);
        constexpr int dst_bpp = Cvt::dst_bpp;
        int i = 0;
        const int n16 = n >> 4;
        if (n16) {
            c.resize_nn_simd_helper(src_row, x, dst, n16);
            i = n & ~15;
            dst += i * dst_bpp;
        }
        for (; i < n; ++i, dst += dst_bpp) {
            c(src_row + x[i], dst, x[i] & 0b10);
        }
    };
#endif
    k->enable_simd(true);
}

template <pix_cvt_t Id, typename Cvt>
    requires(!pix_cvt_is_2pix(Id))
void bind_thunk(PixelKernel *k)
{
    k->cvt_scalar_fn = [](const PixelKernel *s, uint8_t *src, uint8_t *dst, int n) {
        const Cvt &c = *static_cast<const Cvt *>(s->cvt);
        constexpr int src_bpp = Cvt::src_bpp;
        constexpr int dst_bpp = Cvt::dst_bpp;
        for (int i = 0; i < n; ++i, src += src_bpp, dst += dst_bpp) {
            c(src, dst);
        }
    };
    k->resize_nn_scalar_fn = [](const PixelKernel *s, uint8_t *src_row, int *x, uint8_t *dst, int n) {
        const Cvt &c = *static_cast<const Cvt *>(s->cvt);
        constexpr int dst_bpp = Cvt::dst_bpp;
        for (int i = 0; i < n; ++i, dst += dst_bpp) {
            c(src_row + x[i], dst);
        }
    };
    k->warp_nn_fn = [](const PixelKernel *s, uint8_t *src, const int *offset, uint8_t *dst, int n, const uint8_t *bg) {
        const Cvt &c = *static_cast<const Cvt *>(s->cvt);
        constexpr int dst_bpp = Cvt::dst_bpp;
        for (int i = 0; i < n; ++i, dst += dst_bpp) {
            if (offset[i] >= 0) {
                c(src + offset[i], dst);
            } else {
                memcpy(dst, bg, dst_bpp);
            }
        }
    };
#if CONFIG_PIE_V2_BOOST
    k->cvt_simd_fn = [](const PixelKernel *s, uint8_t *src, uint8_t *dst, int n) {
        const Cvt &c = *static_cast<const Cvt *>(s->cvt);
        constexpr int src_bpp = Cvt::src_bpp;
        constexpr int dst_bpp = Cvt::dst_bpp;
        int i = 0;
        const int n16 = n >> 4;
        if (n16) {
            c.cvt_color_simd_helper(src, dst, n16);
            i = n & ~15;
            src += i * src_bpp;
            dst += i * dst_bpp;
        }
        for (; i < n; ++i, src += src_bpp, dst += dst_bpp) {
            c(src, dst);
        }
    };
    k->resize_nn_simd_fn = [](const PixelKernel *s, uint8_t *src_row, int *x, uint8_t *dst, int n) {
        const Cvt &c = *static_cast<const Cvt *>(s->cvt);
        constexpr int dst_bpp = Cvt::dst_bpp;
        int i = 0;
        const int n16 = n >> 4;
        if (n16) {
            c.resize_nn_simd_helper(src_row, x, dst, n16);
            i = n & ~15;
            dst += i * dst_bpp;
        }
        for (; i < n; ++i, dst += dst_bpp) {
            c(src_row + x[i], dst);
        }
    };
#endif
    k->enable_simd(true);
}

template <pix_cvt_t Id, typename Cvt>
void own_cvt(PixelKernel &k)
{
    k.cvt_id = Id;
    k.destroy = [](PixelKernel *self) {
        delete static_cast<Cvt *>(self->cvt);
        self->cvt = nullptr;
    };
    bind_thunk<Id, Cvt>(&k);
}

template <pix_cvt_t Id, typename Cvt, typename Extra>
void own_cvt_extra(PixelKernel &k)
{
    k.cvt_id = Id;
    k.destroy = [](PixelKernel *self) {
        delete static_cast<Cvt *>(self->cvt);
        self->cvt = nullptr;
        delete static_cast<Extra *>(self->extra);
        self->extra = nullptr;
    };
    bind_thunk<Id, Cvt>(&k);
}

template <pix_cvt_t Id>
PixelKernel make_plain_kernel()
{
    using Cvt = typename plain_traits<Id>::type;
    PixelKernel k{};
    k.cvt = new Cvt();
    own_cvt<Id, Cvt>(k);
    return k;
}

template <pix_cvt_t Id>
PixelKernel make_norm_quant_kernel(const norm_quant_param_t &norm)
{
    if (!norm.set) {
        return {};
    }
    using Traits = norm_quant_traits<Id>;
    using Cvt = typename Traits::type;
    using Extra = typename Traits::extra;
    static_assert(std::is_constructible_v<Cvt, const Extra &>);

    PixelKernel k{};
    if constexpr (Extra::channels == 1) {
        k.extra = new Extra(norm.mean[0], norm.std[0], norm.exp);
    } else {
        k.extra = new Extra(norm.mean, norm.std, norm.exp);
    }
    k.cvt = new Cvt(*static_cast<const Extra *>(k.extra));
    own_cvt_extra<Id, Cvt, Extra>(k);
    return k;
}

template <bool Across>
PixelKernel make_hsv_mask_kernel_impl(const hsv_param_t &hsv)
{
    using Cvt = HSV2HSVMask<Across>;
    PixelKernel k{};
    k.cvt = new Cvt(hsv.min, hsv.max);
    own_cvt<DL_IMAGE_PIX_CVT_HSV2HSV_MASK, Cvt>(k);
    return k;
}

PixelKernel make_hsv_mask_kernel(const hsv_param_t &hsv)
{
    if (!hsv.set) {
        return {};
    }
    // No Id parameter: this factory only builds HSV2HSV_MASK.
    if (hsv.min[0] > hsv.max[0]) {
        return make_hsv_mask_kernel_impl<true>(hsv);
    }
    return make_hsv_mask_kernel_impl<false>(hsv);
}

template <pix_cvt_t Id, bool Across>
PixelKernel make_hsv_mask_extra_kernel_impl(const hsv_param_t &hsv)
{
    using Traits = hsv_mask_extra_traits<Id, Across>;
    using Cvt = typename Traits::type;
    using Extra = typename Traits::extra;
    static_assert(std::is_same_v<Extra, HSV2HSVMask<Across>>);
    static_assert(std::is_constructible_v<Cvt, const Extra &>);

    PixelKernel k{};
    k.extra = new Extra(hsv.min, hsv.max);
    k.cvt = new Cvt(*static_cast<const Extra *>(k.extra));
    own_cvt_extra<Id, Cvt, Extra>(k);
    return k;
}

template <pix_cvt_t Id>
PixelKernel make_hsv_mask_extra_kernel(const hsv_param_t &hsv)
{
    if (!hsv.set) {
        return {};
    }
    if (hsv.min[0] > hsv.max[0]) {
        return make_hsv_mask_extra_kernel_impl<Id, true>(hsv);
    }
    return make_hsv_mask_extra_kernel_impl<Id, false>(hsv);
}

bool pix_cvt_is_identity(pix_cvt_t id)
{
    // These Cvts copy pixels unchanged. Aliases are omitted; make_*_kernel stores them as the canonical id below.
    switch (id) {
    case DL_IMAGE_PIX_CVT_RGB8882RGB888:
    case DL_IMAGE_PIX_CVT_GRAY2GRAY:
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB565LE:
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB565BE:
        return true;
    default:
        return false;
    }
}

PixelKernel make_pixel_kernel(pix_type_t src, pix_type_t dst, const hsv_param_t &hsv, const norm_quant_param_t &norm)
{
    const pix_cvt_t id = static_cast<pix_cvt_t>(pix_cvt_id(src, dst));
    // Later cases in a group are aliases and fall through to one canonical id, which the kernel stores.
    switch (id) {
    case DL_IMAGE_PIX_CVT_RGB8882RGB888:
    case DL_IMAGE_PIX_CVT_BGR8882BGR888:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB8882RGB888>();
    case DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR8882BGR888_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR8882BGR888_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB8882RGB888_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_RGB8882BGR888:
    case DL_IMAGE_PIX_CVT_BGR8882RGB888:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB8882BGR888>();
    case DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR8882RGB888_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR8882RGB888_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB8882BGR888_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_RGB8882GRAY:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB8882GRAY>();
    case DL_IMAGE_PIX_CVT_RGB8882GRAY_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB8882GRAY_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_RGB8882GRAY_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB8882GRAY_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_BGR8882GRAY:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_BGR8882GRAY>();
    case DL_IMAGE_PIX_CVT_BGR8882GRAY_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_BGR8882GRAY_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_BGR8882GRAY_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_BGR8882GRAY_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_RGB8882RGB565LE:
    case DL_IMAGE_PIX_CVT_BGR8882BGR565LE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB8882RGB565LE>();
    case DL_IMAGE_PIX_CVT_RGB8882RGB565BE:
    case DL_IMAGE_PIX_CVT_BGR8882BGR565BE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB8882RGB565BE>();
    case DL_IMAGE_PIX_CVT_RGB8882BGR565LE:
    case DL_IMAGE_PIX_CVT_BGR8882RGB565LE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB8882BGR565LE>();
    case DL_IMAGE_PIX_CVT_RGB8882BGR565BE:
    case DL_IMAGE_PIX_CVT_BGR8882RGB565BE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB8882BGR565BE>();
    case DL_IMAGE_PIX_CVT_RGB8882HSV:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB8882HSV>();
    case DL_IMAGE_PIX_CVT_BGR8882HSV:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_BGR8882HSV>();
    case DL_IMAGE_PIX_CVT_RGB8882HSV_MASK:
        return make_hsv_mask_extra_kernel<DL_IMAGE_PIX_CVT_RGB8882HSV_MASK>(hsv);
    case DL_IMAGE_PIX_CVT_BGR8882HSV_MASK:
        return make_hsv_mask_extra_kernel<DL_IMAGE_PIX_CVT_BGR8882HSV_MASK>(hsv);
    case DL_IMAGE_PIX_CVT_GRAY2GRAY:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_GRAY2GRAY>();
    case DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_GRAY2GRAY_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB888:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR888:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565LE2RGB888>();
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR888_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR888_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565LE2RGB888_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR888:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB888:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565LE2BGR888>();
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB888_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB888_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565LE2BGR888_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB888:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR888:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565BE2RGB888>();
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR888_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR888_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565BE2RGB888_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR888:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB888:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565BE2BGR888>();
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT8:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB888_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT16:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB888_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565BE2BGR888_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_RGB565LE2GRAY:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565LE2GRAY>();
    case DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565LE2GRAY_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_RGB565BE2GRAY:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565BE2GRAY>();
    case DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_RGB565BE2GRAY_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_BGR565LE2GRAY:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_BGR565LE2GRAY>();
    case DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_BGR565LE2GRAY_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_BGR565BE2GRAY:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_BGR565BE2GRAY>();
    case DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_BGR565BE2GRAY_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB565LE:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR565LE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565LE2RGB565LE>();
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB565BE:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR565BE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565BE2RGB565BE>();
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR565LE:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB565LE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565LE2BGR565LE>();
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR565BE:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB565BE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565BE2BGR565BE>();
    case DL_IMAGE_PIX_CVT_RGB565LE2RGB565BE:
    case DL_IMAGE_PIX_CVT_BGR565LE2BGR565BE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565LE2RGB565BE>();
    case DL_IMAGE_PIX_CVT_RGB565BE2RGB565LE:
    case DL_IMAGE_PIX_CVT_BGR565BE2BGR565LE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565BE2RGB565LE>();
    case DL_IMAGE_PIX_CVT_RGB565LE2BGR565BE:
    case DL_IMAGE_PIX_CVT_BGR565LE2RGB565BE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565LE2BGR565BE>();
    case DL_IMAGE_PIX_CVT_RGB565BE2BGR565LE:
    case DL_IMAGE_PIX_CVT_BGR565BE2RGB565LE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565BE2BGR565LE>();
    case DL_IMAGE_PIX_CVT_RGB565LE2HSV:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565LE2HSV>();
    case DL_IMAGE_PIX_CVT_RGB565BE2HSV:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_RGB565BE2HSV>();
    case DL_IMAGE_PIX_CVT_BGR565LE2HSV:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_BGR565LE2HSV>();
    case DL_IMAGE_PIX_CVT_BGR565BE2HSV:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_BGR565BE2HSV>();
    case DL_IMAGE_PIX_CVT_RGB565LE2HSV_MASK:
        return make_hsv_mask_extra_kernel<DL_IMAGE_PIX_CVT_RGB565LE2HSV_MASK>(hsv);
    case DL_IMAGE_PIX_CVT_RGB565BE2HSV_MASK:
        return make_hsv_mask_extra_kernel<DL_IMAGE_PIX_CVT_RGB565BE2HSV_MASK>(hsv);
    case DL_IMAGE_PIX_CVT_BGR565LE2HSV_MASK:
        return make_hsv_mask_extra_kernel<DL_IMAGE_PIX_CVT_BGR565LE2HSV_MASK>(hsv);
    case DL_IMAGE_PIX_CVT_BGR565BE2HSV_MASK:
        return make_hsv_mask_extra_kernel<DL_IMAGE_PIX_CVT_BGR565BE2HSV_MASK>(hsv);
    case DL_IMAGE_PIX_CVT_HSV2HSV_MASK:
        return make_hsv_mask_kernel(hsv);
    case DL_IMAGE_PIX_CVT_YUYV2GRAY:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_YUYV2GRAY>();
    case DL_IMAGE_PIX_CVT_YUYV2GRAY_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_YUYV2GRAY_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_YUYV2GRAY_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_YUYV2GRAY_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_UYVY2GRAY:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_UYVY2GRAY>();
    case DL_IMAGE_PIX_CVT_UYVY2GRAY_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_UYVY2GRAY_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_UYVY2GRAY_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_UYVY2GRAY_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_YUYV2RGB888:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_YUYV2RGB888>();
    case DL_IMAGE_PIX_CVT_YUYV2RGB888_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_YUYV2RGB888_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_YUYV2RGB888_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_YUYV2RGB888_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_YUYV2BGR888:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_YUYV2BGR888>();
    case DL_IMAGE_PIX_CVT_YUYV2BGR888_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_YUYV2BGR888_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_YUYV2BGR888_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_YUYV2BGR888_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_UYVY2RGB888:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_UYVY2RGB888>();
    case DL_IMAGE_PIX_CVT_UYVY2RGB888_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_UYVY2RGB888_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_UYVY2RGB888_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_UYVY2RGB888_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_UYVY2BGR888:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_UYVY2BGR888>();
    case DL_IMAGE_PIX_CVT_UYVY2BGR888_QINT8:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_UYVY2BGR888_QINT8>(norm);
    case DL_IMAGE_PIX_CVT_UYVY2BGR888_QINT16:
        return make_norm_quant_kernel<DL_IMAGE_PIX_CVT_UYVY2BGR888_QINT16>(norm);
    case DL_IMAGE_PIX_CVT_YUYV2RGB565LE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_YUYV2RGB565LE>();
    case DL_IMAGE_PIX_CVT_YUYV2RGB565BE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_YUYV2RGB565BE>();
    case DL_IMAGE_PIX_CVT_YUYV2BGR565LE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_YUYV2BGR565LE>();
    case DL_IMAGE_PIX_CVT_YUYV2BGR565BE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_YUYV2BGR565BE>();
    case DL_IMAGE_PIX_CVT_UYVY2RGB565LE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_UYVY2RGB565LE>();
    case DL_IMAGE_PIX_CVT_UYVY2RGB565BE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_UYVY2RGB565BE>();
    case DL_IMAGE_PIX_CVT_UYVY2BGR565LE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_UYVY2BGR565LE>();
    case DL_IMAGE_PIX_CVT_UYVY2BGR565BE:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_UYVY2BGR565BE>();
    case DL_IMAGE_PIX_CVT_YUYV2HSV:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_YUYV2HSV>();
    case DL_IMAGE_PIX_CVT_UYVY2HSV:
        return make_plain_kernel<DL_IMAGE_PIX_CVT_UYVY2HSV>();
    case DL_IMAGE_PIX_CVT_YUYV2HSV_MASK:
        return make_hsv_mask_extra_kernel<DL_IMAGE_PIX_CVT_YUYV2HSV_MASK>(hsv);
    case DL_IMAGE_PIX_CVT_UYVY2HSV_MASK:
        return make_hsv_mask_extra_kernel<DL_IMAGE_PIX_CVT_UYVY2HSV_MASK>(hsv);
    default:
        return {};
    }
}

} // namespace image
} // namespace dl
