#pragma once
#include "dl_image_color.hpp"
#include "dl_image_define.hpp"
#include <array>
#include <cstdint>
#include <utility>

namespace dl {
namespace image {

// Matches canonical ids only. Aliases fall through in make_pixel_kernel and are stored as that canonical id.
bool pix_cvt_is_identity(pix_cvt_t id);

struct hsv_param_t {
    std::array<uint8_t, 3> min{};
    std::array<uint8_t, 3> max{};
    bool set = false;
};

struct norm_quant_param_t {
    int channels = 0;
    int exp = 0;
    int quant_bits = 0;
    std::array<float, 3> mean{};
    std::array<float, 3> std{};
    bool set = false;
};

struct PixelKernel {
    using CvtFn = void (*)(const PixelKernel *self, uint8_t *src, uint8_t *dst, int n);
    // n is a pixel count. odd is true when src points at the second pixel of a YUV pair.
    using Cvt2PixFn = void (*)(const PixelKernel *self, uint8_t *src, uint8_t *dst, int n, bool odd);
    using ResizeNnFn = void (*)(const PixelKernel *self, uint8_t *src_row, int *x, uint8_t *dst, int n);
    // offset[i] < 0 copies bg. Otherwise offset[i] is a byte offset from src.
    using WarpNnFn =
        void (*)(const PixelKernel *self, uint8_t *src, const int *offset, uint8_t *dst, int n, const uint8_t *bg);

    CvtFn cvt_fn = nullptr;
    CvtFn cvt_scalar_fn = nullptr;
    CvtFn cvt_simd_fn = nullptr;
    Cvt2PixFn cvt_2pix_fn = nullptr;
    Cvt2PixFn cvt_2pix_scalar_fn = nullptr;
    Cvt2PixFn cvt_2pix_simd_fn = nullptr;
    ResizeNnFn resize_nn_fn = nullptr;
    ResizeNnFn resize_nn_scalar_fn = nullptr;
    ResizeNnFn resize_nn_simd_fn = nullptr;
    WarpNnFn warp_nn_fn = nullptr;
    void (*destroy)(PixelKernel *self) = nullptr;
    void *cvt = nullptr;
    void *extra = nullptr;
    // Canonical id of the instantiated Cvt, not pix_cvt_id(src, dst). BGR8882BGR888 is stored as RGB8882RGB888.
    pix_cvt_t cvt_id{};

    PixelKernel() = default;
    PixelKernel(const PixelKernel &) = delete;
    PixelKernel &operator=(const PixelKernel &) = delete;
    PixelKernel(PixelKernel &&rhs) noexcept { *this = std::move(rhs); }
    PixelKernel &operator=(PixelKernel &&rhs) noexcept
    {
        if (this != &rhs) {
            reset();
            cvt_fn = rhs.cvt_fn;
            cvt_scalar_fn = rhs.cvt_scalar_fn;
            cvt_simd_fn = rhs.cvt_simd_fn;
            cvt_2pix_fn = rhs.cvt_2pix_fn;
            cvt_2pix_scalar_fn = rhs.cvt_2pix_scalar_fn;
            cvt_2pix_simd_fn = rhs.cvt_2pix_simd_fn;
            resize_nn_fn = rhs.resize_nn_fn;
            resize_nn_scalar_fn = rhs.resize_nn_scalar_fn;
            resize_nn_simd_fn = rhs.resize_nn_simd_fn;
            warp_nn_fn = rhs.warp_nn_fn;
            destroy = rhs.destroy;
            cvt = rhs.cvt;
            extra = rhs.extra;
            cvt_id = rhs.cvt_id;
            rhs.cvt_fn = nullptr;
            rhs.cvt_scalar_fn = nullptr;
            rhs.cvt_simd_fn = nullptr;
            rhs.cvt_2pix_fn = nullptr;
            rhs.cvt_2pix_scalar_fn = nullptr;
            rhs.cvt_2pix_simd_fn = nullptr;
            rhs.resize_nn_fn = nullptr;
            rhs.resize_nn_scalar_fn = nullptr;
            rhs.resize_nn_simd_fn = nullptr;
            rhs.warp_nn_fn = nullptr;
            rhs.destroy = nullptr;
            rhs.cvt = nullptr;
            rhs.extra = nullptr;
            rhs.cvt_id = {};
        }
        return *this;
    }
    ~PixelKernel() { reset(); }

    void reset()
    {
        if (destroy) {
            destroy(this);
        }
        cvt_fn = nullptr;
        cvt_scalar_fn = nullptr;
        cvt_simd_fn = nullptr;
        cvt_2pix_fn = nullptr;
        cvt_2pix_scalar_fn = nullptr;
        cvt_2pix_simd_fn = nullptr;
        resize_nn_fn = nullptr;
        resize_nn_scalar_fn = nullptr;
        resize_nn_simd_fn = nullptr;
        warp_nn_fn = nullptr;
        destroy = nullptr;
        cvt = nullptr;
        extra = nullptr;
        cvt_id = {};
    }

    void enable_simd(bool enable)
    {
        cvt_fn = (enable && cvt_simd_fn) ? cvt_simd_fn : cvt_scalar_fn;
        cvt_2pix_fn = (enable && cvt_2pix_simd_fn) ? cvt_2pix_simd_fn : cvt_2pix_scalar_fn;
        resize_nn_fn = (enable && resize_nn_simd_fn) ? resize_nn_simd_fn : resize_nn_scalar_fn;
    }

    void cvt_color(uint8_t *src, uint8_t *dst, int n) const { cvt_fn(this, src, dst, n); }

    void cvt_color(uint8_t *src, uint8_t *dst, int n, bool odd) const { cvt_2pix_fn(this, src, dst, n, odd); }

    void resize_nn(uint8_t *src_row, int *x, uint8_t *dst, int n) const { resize_nn_fn(this, src_row, x, dst, n); }

    void warp_nn(uint8_t *src, const int *offset, uint8_t *dst, int n, const uint8_t *bg) const
    {
        warp_nn_fn(this, src, offset, dst, n, bg);
    }
};

PixelKernel make_pixel_kernel(pix_type_t src, pix_type_t dst, const hsv_param_t &hsv, const norm_quant_param_t &norm);

} // namespace image
} // namespace dl
