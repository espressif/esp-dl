#pragma once
#include "dl_image_color.hpp"
#include "dl_image_define.hpp"
#include "dl_image_pixel_kernel.hpp"
#include "dl_math_matrix.hpp"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace dl {
namespace image {
class ImageTransformer {
public:
    ImageTransformer();
    ImageTransformer(const ImageTransformer &) = delete;
    ImageTransformer &operator=(const ImageTransformer &) = delete;
    ImageTransformer(ImageTransformer &&rhs) noexcept;
    ImageTransformer &operator=(ImageTransformer &&rhs) noexcept;

    ~ImageTransformer();
    ImageTransformer &set_src_img(const img_t &src_img);
    ImageTransformer &set_dst_img(const img_t &dst_img);
    ImageTransformer &set_warp_affine_matrix(const math::Matrix<float> &M, bool inv = false);
    ImageTransformer &set_src_img_crop_area(const std::vector<int> &crop_area);
    ImageTransformer &set_dst_img_border(const std::vector<int> &border);
    /**
     * @brief Set the background value. If the dst image uses the different color space besides RGB. The value will be
     * converted automatically.
     *
     * @param bg_value RGB value
     * @return ImageTransformer&
     */
    ImageTransformer &set_bg_value(const std::array<uint8_t, 3> &bg_value);
    /**
     * @brief Set the background value. If the dst image uses the different color space besides Gray. The value will be
     * converted automatically
     *
     * @param bg_value GRAY value
     * @return ImageTransformer&
     */
    ImageTransformer &set_bg_value(const uint8_t bg_value);
    ImageTransformer &set_norm_quant_param(const std::array<float, 3> &mean,
                                           const std::array<float, 3> &std,
                                           int exp,
                                           int quant_bits);
    ImageTransformer &set_norm_quant_param(float mean, float std, int exp, int quant_bits);
    ImageTransformer &set_hsv_thr(const std::array<uint8_t, 3> &hsv_min, const std::array<uint8_t, 3> &hsv_max);
    ImageTransformer &reset();

    const img_t &get_src_img();
    const img_t &get_dst_img();
    math::Matrix<float> &get_warp_affine_matrix();
    std::vector<int> &get_src_img_crop_area();
    std::vector<int> &get_dst_img_border();
    float get_scale_x(bool inv = false);
    float get_scale_y(bool inv = false);

    esp_err_t transform(bool simd = true);

private:
    esp_err_t gen_xy_map();
    esp_err_t gen_bg();
    esp_err_t cvt_color(const PixelKernel &k);
    esp_err_t cvt_color_border(const PixelKernel &k);
    esp_err_t resize_nn(const PixelKernel &k);
    esp_err_t resize_nn_border(const PixelKernel &k);
    esp_err_t warp_affine_nn(const PixelKernel &k);
    esp_err_t warp_affine_nn_border(const PixelKernel &k);
    void fill_warp_offset(int *offset, int row, int n, int x0, int y0, uint32_t x_max, uint32_t y_max) const;
    esp_err_t params_valid() const;
    esp_err_t ensure_kernel(bool simd);

    struct xy_map_t {
        enum class geom_t : uint8_t { same, resize, warp };

        float scale_x = 0;
        float scale_y = 0;
        float inv_scale_x = 0;
        float inv_scale_y = 0;
        int *x = nullptr;
        int *y = nullptr;
        int *x1 = nullptr;
        int *x2 = nullptr;
        int *y1 = nullptr;
        int *y2 = nullptr;
        geom_t geom = geom_t::same;
        bool dirty = false;

        void release()
        {
            heap_caps_free(x);
            heap_caps_free(y);
            heap_caps_free(x1);
            heap_caps_free(x2);
            heap_caps_free(y1);
            heap_caps_free(y2);
            x = nullptr;
            y = nullptr;
            x1 = nullptr;
            x2 = nullptr;
            y1 = nullptr;
            y2 = nullptr;
        }
    };

    struct bg_t {
        uint8_t ori[3]{};
        uint8_t ori_n = 0; // 0: unset, 1: gray, 3: rgb
        uint8_t pix[6]{};
        uint8_t pix_size = 0;
        bool same = false;
        bool dirty = true;
    };

    static constexpr int warp_affine_shift = 10;
    img_t m_src_img;
    img_t m_dst_img;
    math::Matrix<float> m_M;
    std::vector<int> m_crop_area; /*!< left_top_x, left_top_y, bottom_right_x, bottom_right_y */
    std::vector<int> m_border;    /*!< top, bottom, left, right */
    hsv_param_t m_hsv;
    norm_quant_param_t m_norm;
    PixelKernel m_kernel;
    bool m_kernel_dirty = true;
    xy_map_t m_xy;
    bg_t m_bg;
};
} // namespace image
} // namespace dl
