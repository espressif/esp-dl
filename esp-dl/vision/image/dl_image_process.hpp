#pragma once
#include "dl_image_color.hpp"
#include "dl_image_define.hpp"
#include "dl_image_pixel_cvt_dispatch.hpp"
#include "dl_math_matrix.hpp"
#include "esp_heap_caps.h"
#include <algorithm>
#include <variant>
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
    pix_cvt_param_t get_pix_cvt_param();

#if CONFIG_IDF_TARGET_ESP32P4
    static constexpr bool simd = true;
#else
    static constexpr bool simd = false;
#endif
    template <bool SIMD = simd>
    esp_err_t transform();
    template <bool SIMD>
    struct TransformNNFunctor {
        ImageTransformer *self;
        template <typename PixelCvt>
        void operator()(const PixelCvt &pixel_cvt) const
        {
            constexpr bool simd = SIMD && !is_yuv_cvt_v<PixelCvt>;
            self->template transform_nn<PixelCvt, simd>(pixel_cvt);
        }
    };

private:
    void gen_xy_map();

    template <typename PixelCvt, bool SrcCrop, bool SIMD>
    void cvt_color(const PixelCvt &pixel_cvt)
    {
        int src_step = m_src_img.col_step();
        int dst_step = m_dst_img.col_step();
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        if constexpr (std::is_same_v<PixelCvt, RGB5652RGB565<false, false, false>> ||
                      std::is_same_v<PixelCvt, RGB5652RGB565<true, false, false>> ||
                      std::is_same_v<PixelCvt, RGB8882RGB888<false>> || std::is_same_v<PixelCvt, Gray2Gray<>> ||
                      std::is_same_v<PixelCvt, YUV2YUV<false>>) {
            if (src == dst) {
                return;
            }
        }
        if constexpr (!SrcCrop) {
            int n = m_dst_img.width * m_dst_img.height;
            int i = 0;
            if constexpr (SIMD) {
                pixel_cvt.cvt_color_simd_helper(src, dst, n >> 4);
                i = n & ~0xf;
                src += src_step * i;
                dst += dst_step * i;
            }
            for (; i < n; i++, src += src_step, dst += dst_step) {
                pixel_cvt(src, dst);
            }
        } else {
            int dst_width = m_dst_img.width;
            int dst_height = m_dst_img.height;
            int n_simd_loop = dst_width >> 4;
            int n_simd = dst_width & ~0xf;
            int simd_src_step = n_simd * src_step;
            int simd_dst_step = n_simd * dst_step;
            src += (m_src_img.width * m_crop_area[1] + m_crop_area[0]) * src_step;
            int src_step_new_row = (m_crop_area[0] + m_src_img.width - m_crop_area[2]) * src_step;
            for (int i = 0; i < dst_height; i++) {
                int j = 0;
                if constexpr (SIMD) {
                    pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                    src += simd_src_step;
                    dst += simd_dst_step;
                    j = n_simd;
                }
                for (; j < dst_width; j++, src += src_step, dst += dst_step) {
                    pixel_cvt(src, dst);
                }
                src += src_step_new_row;
            }
        }
    }

    template <typename PixelCvt, bool SrcCrop, bool SIMD>
        requires is_2pix_yuv_cvt_v<PixelCvt>
    void cvt_color(const PixelCvt &pixel_cvt)
    {
        if (m_src_img.width & 1) {
            ESP_LOGE("ImageTransformer", "YUV image width should be an even number.");
            return;
        }
        int src_step = m_src_img.col_step();
        int dst_step = m_dst_img.col_step();
        int src_step_2x = src_step << 1;
        int dst_step_2x = dst_step << 1;
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        if constexpr (!SrcCrop) {
            int n = m_dst_img.width * m_dst_img.height / 2;
            int i = 0;
            if constexpr (SIMD) {
                pixel_cvt.cvt_color_simd_helper(src, dst, n >> 4);
                i = n & ~0xf;
                src += src_step_2x * i;
                dst += dst_step_2x * i;
            }
            for (; i < n; i++, src += src_step_2x, dst += dst_step_2x) {
                pixel_cvt(src, dst);
            }
        } else {
            int dst_width = m_dst_img.width;
            int dst_height = m_dst_img.height;
            bool odd_left = m_crop_area[0] & 1;
            bool odd_right = (dst_width & 1) ^ odd_left;
            int n = (dst_width - odd_left - odd_right) / 2;
            int n_simd_loop = n >> 4;
            int n_simd = n & ~0xf;
            int simd_src_step = n_simd * src_step_2x;
            int simd_dst_step = n_simd * dst_step_2x;
            src += (m_src_img.width * m_crop_area[1] + m_crop_area[0]) * src_step;
            int src_step_new_row = (m_crop_area[0] + m_src_img.width - m_crop_area[2]) * src_step;
            if (!odd_left && !odd_right) {
                for (int i = 0; i < dst_height; i++) {
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    src += src_step_new_row;
                }
            } else if (odd_left & !odd_right) {
                for (int i = 0; i < dst_height; i++) {
                    pixel_cvt(src, dst, true);
                    src += src_step;
                    dst += dst_step;
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    src += src_step_new_row;
                }
            } else if (!odd_left & odd_right) {
                for (int i = 0; i < dst_height; i++) {
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    pixel_cvt(src, dst, false);
                    src += src_step;
                    dst += dst_step;
                    src += src_step_new_row;
                }
            } else {
                for (int i = 0; i < dst_height; i++) {
                    pixel_cvt(src, dst, true);
                    src += src_step;
                    dst += dst_step;
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    pixel_cvt(src, dst, false);
                    src += src_step;
                    dst += dst_step;
                    src += src_step_new_row;
                }
            }
        }
    }

    template <typename PixelCvt, bool SrcCrop, bool SIMD>
    void cvt_color_same_value_border(const PixelCvt &pixel_cvt)
    {
        int src_step = m_src_img.col_step();
        int dst_step = m_dst_img.col_step();
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int border_step_top = m_border[0] * m_dst_img.row_step();
        int border_step_bottom = m_border[1] * m_dst_img.row_step();
        int border_step_left = dst_step * m_border[2];
        int border_step_right = dst_step * m_border[3];
        uint8_t v = m_bg_value[0];
        int n_simd_loop = dst_width >> 4;
        int n_simd = dst_width & ~0xf;
        int simd_src_step = n_simd * src_step;
        int simd_dst_step = n_simd * dst_step;
        memset(dst, v, border_step_top);
        dst += border_step_top;
        if constexpr (!SrcCrop) {
            for (int i = 0; i < dst_height; i++) {
                memset(dst, v, border_step_left);
                dst += border_step_left;
                int j = 0;
                if constexpr (SIMD) {
                    pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                    src += simd_src_step;
                    dst += simd_dst_step;
                    j = n_simd;
                }
                for (; j < dst_width; j++, src += src_step, dst += dst_step) {
                    pixel_cvt(src, dst);
                }
                memset(dst, v, border_step_right);
                dst += border_step_right;
            }
        } else {
            src += (m_src_img.width * m_crop_area[1] + m_crop_area[0]) * src_step;
            int src_step_new_row = (m_crop_area[0] + m_src_img.width - m_crop_area[2]) * src_step;
            for (int i = 0; i < dst_height; i++) {
                memset(dst, v, border_step_left);
                dst += border_step_left;
                int j = 0;
                if constexpr (SIMD) {
                    pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                    src += simd_src_step;
                    dst += simd_dst_step;
                    j = n_simd;
                }
                for (; j < dst_width; j++, src += src_step, dst += dst_step) {
                    pixel_cvt(src, dst);
                }
                memset(dst, v, border_step_right);
                dst += border_step_right;
                src += src_step_new_row;
            }
        }
        memset(dst, v, border_step_bottom);
    }

    template <typename PixelCvt, bool SrcCrop, bool SIMD>
        requires is_2pix_yuv_cvt_v<PixelCvt>
    void cvt_color_same_value_border(const PixelCvt &pixel_cvt)
    {
        int src_step = m_src_img.col_step();
        int dst_step = m_dst_img.col_step();
        int src_step_2x = src_step << 1;
        int dst_step_2x = dst_step << 1;
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int border_step_top = m_border[0] * m_dst_img.row_step();
        int border_step_bottom = m_border[1] * m_dst_img.row_step();
        int border_step_left = dst_step * m_border[2];
        int border_step_right = dst_step * m_border[3];
        uint8_t v = m_bg_value[0];
        memset(dst, v, border_step_top);
        dst += border_step_top;
        if constexpr (!SrcCrop) {
            int n = dst_width / 2;
            int n_simd_loop = n >> 4;
            int n_simd = n & ~0xf;
            int simd_src_step = n_simd * src_step_2x;
            int simd_dst_step = n_simd * dst_step_2x;
            for (int i = 0; i < dst_height; i++) {
                memset(dst, v, border_step_left);
                dst += border_step_left;
                int j = 0;
                if constexpr (SIMD) {
                    pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                    src += simd_src_step;
                    dst += simd_dst_step;
                    j = n_simd;
                }
                for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                    pixel_cvt(src, dst);
                }
                memset(dst, v, border_step_right);
                dst += border_step_right;
            }
        } else {
            bool odd_left = m_crop_area[0] & 1;
            bool odd_right = (dst_width & 1) ^ odd_left;
            int n = (dst_width - odd_left - odd_right) / 2;
            int n_simd_loop = n >> 4;
            int n_simd = n & ~0xf;
            int simd_src_step = n_simd * src_step_2x;
            int simd_dst_step = n_simd * dst_step_2x;
            src += (m_src_img.width * m_crop_area[1] + m_crop_area[0]) * src_step;
            int src_step_new_row = (m_crop_area[0] + m_src_img.width - m_crop_area[2]) * src_step;
            if (!odd_left && !odd_right) {
                for (int i = 0; i < dst_height; i++) {
                    memset(dst, v, border_step_left);
                    dst += border_step_left;
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    memset(dst, v, border_step_right);
                    dst += border_step_right;
                    src += src_step_new_row;
                }
            } else if (odd_left & !odd_right) {
                for (int i = 0; i < dst_height; i++) {
                    memset(dst, v, border_step_left);
                    dst += border_step_left;
                    pixel_cvt(src, dst, true);
                    src += src_step;
                    dst += dst_step;
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    memset(dst, v, border_step_right);
                    dst += border_step_right;
                    src += src_step_new_row;
                }
            } else if (!odd_left & odd_right) {
                for (int i = 0; i < dst_height; i++) {
                    memset(dst, v, border_step_left);
                    dst += border_step_left;
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    pixel_cvt(src, dst, false);
                    src += src_step;
                    dst += dst_step;
                    memset(dst, v, border_step_right);
                    dst += border_step_right;
                    src += src_step_new_row;
                }
            } else {
                for (int i = 0; i < dst_height; i++) {
                    memset(dst, v, border_step_left);
                    dst += border_step_left;
                    pixel_cvt(src, dst, true);
                    src += src_step;
                    dst += dst_step;
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    pixel_cvt(src, dst, false);
                    src += src_step;
                    dst += dst_step;
                    memset(dst, v, border_step_right);
                    dst += border_step_right;
                    src += src_step_new_row;
                }
            }
        }
        memset(dst, v, border_step_bottom);
    }

    template <typename PixelCvt, bool SrcCrop, bool SIMD>
    void cvt_color_diff_value_border(const PixelCvt &pixel_cvt)
    {
        int src_step = m_src_img.col_step();
        int dst_step = m_dst_img.col_step();
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int dst_row_step = m_dst_img.row_step();
        int border_step_left = dst_step * m_border[2];
        int border_step_right = dst_step * m_border[3];
        uint8_t *row_border_value = (uint8_t *)heap_caps_malloc(dst_row_step, MALLOC_CAP_DEFAULT);
        uint8_t *p_row_border_value = row_border_value;
        uint8_t *v = m_bg_value.data();
        int n_simd_loop = dst_width >> 4;
        int n_simd = dst_width & ~0xf;
        int simd_src_step = n_simd * src_step;
        int simd_dst_step = n_simd * dst_step;
        for (int i = 0; i < m_dst_img.width; i++, p_row_border_value += dst_step) {
            memcpy(p_row_border_value, v, dst_step);
        }
        for (int i = 0; i < m_border[0]; i++, dst += dst_row_step) {
            memcpy(dst, row_border_value, dst_row_step);
        }
        if constexpr (!SrcCrop) {
            for (int i = 0; i < dst_height; i++) {
                memcpy(dst, row_border_value, border_step_left);
                dst += border_step_left;
                int j = 0;
                if constexpr (SIMD) {
                    pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                    src += simd_src_step;
                    dst += simd_dst_step;
                    j = n_simd;
                }
                for (; j < dst_width; j++, src += src_step, dst += dst_step) {
                    pixel_cvt(src, dst);
                }
                memcpy(dst, row_border_value, border_step_right);
                dst += border_step_right;
            }
        } else {
            src += (m_src_img.width * m_crop_area[1] + m_crop_area[0]) * src_step;
            int src_step_new_row = (m_crop_area[0] + m_src_img.width - m_crop_area[2]) * src_step;
            for (int i = 0; i < dst_height; i++) {
                memcpy(dst, row_border_value, border_step_left);
                dst += border_step_left;
                int j = 0;
                if constexpr (SIMD) {
                    pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                    src += simd_src_step;
                    dst += simd_dst_step;
                    j = n_simd;
                }
                for (; j < dst_width; j++, src += src_step, dst += dst_step) {
                    pixel_cvt(src, dst);
                }
                memcpy(dst, row_border_value, border_step_right);
                dst += border_step_right;
                src += src_step_new_row;
            }
        }
        for (int i = 0; i < m_border[1]; i++, dst += dst_row_step) {
            memcpy(dst, row_border_value, dst_row_step);
        }
        heap_caps_free(row_border_value);
    }

    template <typename PixelCvt, bool SrcCrop, bool SIMD>
        requires is_2pix_yuv_cvt_v<PixelCvt>
    void cvt_color_diff_value_border(const PixelCvt &pixel_cvt)
    {
        int src_step = m_src_img.col_step();
        int dst_step = m_dst_img.col_step();
        int src_step_2x = src_step << 1;
        int dst_step_2x = dst_step << 1;
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int dst_row_step = m_dst_img.row_step();
        int border_step_left = dst_step * m_border[2];
        int border_step_right = dst_step * m_border[3];
        uint8_t *row_border_value = (uint8_t *)heap_caps_malloc(dst_row_step, MALLOC_CAP_DEFAULT);
        uint8_t *p_row_border_value = row_border_value;
        uint8_t *v = m_bg_value.data();
        for (int i = 0; i < m_dst_img.width; i++, p_row_border_value += dst_step) {
            memcpy(p_row_border_value, v, dst_step);
        }
        for (int i = 0; i < m_border[0]; i++, dst += dst_row_step) {
            memcpy(dst, row_border_value, dst_row_step);
        }
        if constexpr (!SrcCrop) {
            int n = dst_width / 2;
            int n_simd_loop = n >> 4;
            int n_simd = n & ~0xf;
            int simd_src_step = n_simd * src_step;
            int simd_dst_step = n_simd * dst_step;
            for (int i = 0; i < dst_height; i++) {
                memcpy(dst, row_border_value, border_step_left);
                dst += border_step_left;
                int j = 0;
                if constexpr (SIMD) {
                    pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                    src += simd_src_step;
                    dst += simd_dst_step;
                    j = n_simd;
                }
                for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                    pixel_cvt(src, dst);
                }
                memcpy(dst, row_border_value, border_step_right);
                dst += border_step_right;
            }
        } else {
            bool odd_left = m_crop_area[0] & 1;
            bool odd_right = (dst_width & 1) ^ odd_left;
            int n = (dst_width - odd_left - odd_right) / 2;
            int n_simd_loop = n >> 4;
            int n_simd = n & ~0xf;
            int simd_src_step = n_simd * src_step_2x;
            int simd_dst_step = n_simd * dst_step_2x;
            src += (m_src_img.width * m_crop_area[1] + m_crop_area[0]) * src_step;
            int src_step_new_row = (m_crop_area[0] + m_src_img.width - m_crop_area[2]) * src_step;
            if (!odd_left && !odd_right) {
                for (int i = 0; i < dst_height; i++) {
                    memcpy(dst, row_border_value, border_step_left);
                    dst += border_step_left;
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    memcpy(dst, row_border_value, border_step_right);
                    dst += border_step_right;
                    src += src_step_new_row;
                }
            } else if (odd_left & !odd_right) {
                for (int i = 0; i < dst_height; i++) {
                    memcpy(dst, row_border_value, border_step_left);
                    dst += border_step_left;
                    pixel_cvt(src, dst, true);
                    src += src_step;
                    dst += dst_step;
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    memcpy(dst, row_border_value, border_step_right);
                    dst += border_step_right;
                    src += src_step_new_row;
                }
            } else if (!odd_left & odd_right) {
                for (int i = 0; i < dst_height; i++) {
                    memcpy(dst, row_border_value, border_step_left);
                    dst += border_step_left;
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    pixel_cvt(src, dst, false);
                    src += src_step;
                    dst += dst_step;
                    memcpy(dst, row_border_value, border_step_right);
                    dst += border_step_right;
                    src += src_step_new_row;
                }
            } else {
                for (int i = 0; i < dst_height; i++) {
                    memcpy(dst, row_border_value, border_step_left);
                    dst += border_step_left;
                    pixel_cvt(src, dst, true);
                    src += src_step;
                    dst += dst_step;
                    int j = 0;
                    if constexpr (SIMD) {
                        pixel_cvt.cvt_color_simd_helper(src, dst, n_simd_loop);
                        src += simd_src_step;
                        dst += simd_dst_step;
                        j = n_simd;
                    }
                    for (; j < n; j++, src += src_step_2x, dst += dst_step_2x) {
                        pixel_cvt(src, dst);
                    }
                    pixel_cvt(src, dst, false);
                    src += src_step;
                    dst += dst_step;
                    memcpy(dst, row_border_value, border_step_right);
                    dst += border_step_right;
                    src += src_step_new_row;
                }
            }
        }
        for (int i = 0; i < m_border[1]; i++, dst += dst_row_step) {
            memcpy(dst, row_border_value, dst_row_step);
        }
        heap_caps_free(row_border_value);
    }

    template <typename PixelCvt, bool SIMD>
    void resize_nn(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width;
        int dst_height = m_dst_img.height;
        int dst_step = m_dst_img.col_step();
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        int n_simd_loop = dst_width >> 4;
        int n_simd = dst_width & ~0xf;
        int simd_dst_step = n_simd * dst_step;
        for (int i = 0; i < dst_height; i++) {
            uint8_t *p_row = src + m_y[i];
            int j = 0;
            if constexpr (SIMD) {
                pixel_cvt.resize_nn_simd_helper(p_row, m_x, dst, n_simd_loop);
                dst += simd_dst_step;
                j = n_simd;
            }
            for (; j < dst_width; j++, dst += dst_step) {
                if constexpr (is_2pix_yuv_cvt_v<PixelCvt>) {
                    int x = m_x[j];
                    pixel_cvt(p_row + x, dst, x & 0b10);
                } else {
                    pixel_cvt(p_row + m_x[j], dst);
                }
            }
        }
    }

    template <typename PixelCvt, bool SIMD>
    void resize_nn_same_value_border(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int dst_step = m_dst_img.col_step();
        int border_step_top = m_border[0] * m_dst_img.row_step();
        int border_step_bottom = m_border[1] * m_dst_img.row_step();
        int border_step_left = dst_step * m_border[2];
        int border_step_right = dst_step * m_border[3];
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        uint8_t v = m_bg_value[0];
        int n_simd_loop = dst_width >> 4;
        int n_simd = dst_width & ~0xf;
        int simd_dst_step = n_simd * dst_step;
        memset(dst, v, border_step_top);
        dst += border_step_top;
        for (int i = 0; i < dst_height; i++) {
            memset(dst, v, border_step_left);
            dst += border_step_left;
            uint8_t *p_row = src + m_y[i];
            int j = 0;
            if constexpr (SIMD) {
                pixel_cvt.resize_nn_simd_helper(p_row, m_x, dst, n_simd_loop);
                dst += simd_dst_step;
                j = n_simd;
            }
            for (; j < dst_width; j++, dst += dst_step) {
                if constexpr (is_2pix_yuv_cvt_v<PixelCvt>) {
                    int x = m_x[j];
                    pixel_cvt(p_row + x, dst, x & 0b10);
                } else {
                    pixel_cvt(p_row + m_x[j], dst);
                }
            }
            memset(dst, v, border_step_right);
            dst += border_step_right;
        }
        memset(dst, v, border_step_bottom);
    }

    template <typename PixelCvt, bool SIMD>
    void resize_nn_diff_value_border(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int dst_step = m_dst_img.col_step();
        int dst_row_step = m_dst_img.row_step();
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        int border_step_left = dst_step * m_border[2];
        int border_step_right = dst_step * m_border[3];
        uint8_t *row_border_value = (uint8_t *)heap_caps_malloc(dst_row_step, MALLOC_CAP_DEFAULT);
        uint8_t *p_row_border_value = row_border_value;
        uint8_t *v = m_bg_value.data();
        int n_simd_loop = dst_width >> 4;
        int n_simd = dst_width & ~0xf;
        int simd_dst_step = n_simd * dst_step;
        for (int i = 0; i < m_dst_img.width; i++, p_row_border_value += dst_step) {
            memcpy(p_row_border_value, v, dst_step);
        }
        for (int i = 0; i < m_border[0]; i++, dst += dst_row_step) {
            memcpy(dst, row_border_value, dst_row_step);
        }
        for (int i = 0; i < dst_height; i++) {
            memcpy(dst, row_border_value, border_step_left);
            dst += border_step_left;
            uint8_t *p_row = src + m_y[i];
            int j = 0;
            if constexpr (SIMD) {
                pixel_cvt.resize_nn_simd_helper(p_row, m_x, dst, n_simd_loop);
                dst += simd_dst_step;
                j = n_simd;
            }
            for (; j < dst_width; j++, dst += dst_step) {
                if constexpr (is_2pix_yuv_cvt_v<PixelCvt>) {
                    int x = m_x[j];
                    pixel_cvt(p_row + x, dst, x & 0b10);
                } else {
                    pixel_cvt(p_row + m_x[j], dst);
                }
            }
            memcpy(dst, row_border_value, border_step_right);
            dst += border_step_right;
        }
        for (int i = 0; i < m_border[1]; i++, dst += dst_row_step) {
            memcpy(dst, row_border_value, dst_row_step);
        }
        heap_caps_free(row_border_value);
    }

    template <typename PixelCvt, bool SrcCrop, bool SameValueBG>
    void warp_affine_nn(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width;
        int dst_height = m_dst_img.height;
        int dst_step = m_dst_img.col_step();
        int src_col_step = m_src_img.col_step();
        int src_row_step = m_src_img.row_step();
        int x1, y1, src_x_max, src_y_max;
        if constexpr (SrcCrop) {
            x1 = m_crop_area[0];
            y1 = m_crop_area[1];
            src_x_max = m_crop_area[2] - x1;
            src_y_max = m_crop_area[3] - y1;
        } else {
            (void)x1;
            (void)y1;
            src_x_max = m_src_img.width;
            src_y_max = m_src_img.height;
        }
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        constexpr int round_delta = 1 << (warp_affine_shift - 1);
        uint8_t *v = m_bg_value.data();
        for (int i = 0; i < dst_height; i++) {
            for (int j = 0; j < dst_width; j++, dst += dst_step) {
                int x = (m_x1[j] + m_x2[i] + round_delta) >> warp_affine_shift;
                int y = (m_y1[j] + m_y2[i] + round_delta) >> warp_affine_shift;
                if (static_cast<uint32_t>(x) < src_x_max && static_cast<uint32_t>(y) < src_y_max) {
                    if constexpr (SrcCrop) {
                        if constexpr (is_2pix_yuv_cvt_v<PixelCvt>) {
                            pixel_cvt((y + y1) * src_row_step + (x + x1) * src_col_step + src, dst, x & 1);
                        } else {
                            pixel_cvt((y + y1) * src_row_step + (x + x1) * src_col_step + src, dst);
                        }
                    } else {
                        if constexpr (is_2pix_yuv_cvt_v<PixelCvt>) {
                            pixel_cvt(y * src_row_step + x * src_col_step + src, dst, x & 1);
                        } else {
                            pixel_cvt(y * src_row_step + x * src_col_step + src, dst);
                        }
                    }
                } else {
                    if constexpr (SameValueBG) {
                        memset(dst, *v, dst_step);
                    } else {
                        memcpy(dst, v, dst_step);
                    }
                }
            }
        }
    }

    template <typename PixelCvt, bool SrcCrop>
    void warp_affine_nn_same_value_border(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int dst_col_step = m_dst_img.col_step();
        int dst_row_step = m_dst_img.row_step();
        int src_col_step = m_src_img.col_step();
        int src_row_step = m_src_img.row_step();
        int x1, y1, src_x_max, src_y_max;
        if constexpr (SrcCrop) {
            x1 = m_crop_area[0];
            y1 = m_crop_area[1];
            src_x_max = m_crop_area[2] - x1;
            src_y_max = m_crop_area[3] - y1;
        } else {
            (void)x1;
            (void)y1;
            src_x_max = m_src_img.width;
            src_y_max = m_src_img.height;
        }
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        int border_step_top = m_border[0] * dst_row_step;
        int border_step_bottom = m_border[1] * dst_row_step;
        int border_step_left = dst_col_step * m_border[2];
        int border_step_right = dst_col_step * m_border[3];
        constexpr int round_delta = 1 << (warp_affine_shift - 1);
        uint8_t v = m_bg_value[0];
        memset(dst, v, border_step_top);
        dst += border_step_top;
        for (int i = 0; i < dst_height; i++) {
            memset(dst, v, border_step_left);
            dst += border_step_left;
            for (int j = 0; j < dst_width; j++, dst += dst_col_step) {
                int x = (m_x1[j] + m_x2[i] + round_delta) >> warp_affine_shift;
                int y = (m_y1[j] + m_y2[i] + round_delta) >> warp_affine_shift;
                if (static_cast<uint32_t>(x) < src_x_max && static_cast<uint32_t>(y) < src_y_max) {
                    if constexpr (SrcCrop) {
                        if constexpr (is_2pix_yuv_cvt_v<PixelCvt>) {
                            pixel_cvt((y + y1) * src_row_step + (x + x1) * src_col_step + src, dst, x & 1);
                        } else {
                            pixel_cvt((y + y1) * src_row_step + (x + x1) * src_col_step + src, dst);
                        }
                    } else {
                        if constexpr (is_2pix_yuv_cvt_v<PixelCvt>) {
                            pixel_cvt(y * src_row_step + x * src_col_step + src, dst, x & 1);
                        } else {
                            pixel_cvt(y * src_row_step + x * src_col_step + src, dst);
                        }
                    }
                } else {
                    memset(dst, v, dst_col_step);
                }
            }
            memset(dst, v, border_step_right);
            dst += border_step_right;
        }
        memset(dst, v, border_step_bottom);
    }

    template <typename PixelCvt, bool SrcCrop>
    void warp_affine_nn_diff_value_border(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int dst_col_step = m_dst_img.col_step();
        int dst_row_step = m_dst_img.row_step();
        int src_col_step = m_src_img.col_step();
        int src_row_step = m_src_img.row_step();
        int x1, y1, src_x_max, src_y_max;
        if constexpr (SrcCrop) {
            x1 = m_crop_area[0];
            y1 = m_crop_area[1];
            src_x_max = m_crop_area[2] - x1;
            src_y_max = m_crop_area[3] - y1;
        } else {
            (void)x1;
            (void)y1;
            src_x_max = m_src_img.width;
            src_y_max = m_src_img.height;
        }
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        int border_step_left = dst_col_step * m_border[2];
        int border_step_right = dst_col_step * m_border[3];
        uint8_t *row_border_value = (uint8_t *)heap_caps_malloc(dst_row_step, MALLOC_CAP_DEFAULT);
        uint8_t *p_row_border_value = row_border_value;
        constexpr int round_delta = 1 << (warp_affine_shift - 1);
        uint8_t *v = m_bg_value.data();
        for (int i = 0; i < m_dst_img.width; i++, p_row_border_value += dst_col_step) {
            memcpy(p_row_border_value, v, dst_col_step);
        }
        for (int i = 0; i < m_border[0]; i++, dst += dst_row_step) {
            memcpy(dst, row_border_value, dst_row_step);
        }
        for (int i = 0; i < dst_height; i++) {
            memcpy(dst, row_border_value, border_step_left);
            dst += border_step_left;
            for (int j = 0; j < dst_width; j++, dst += dst_col_step) {
                int x = (m_x1[j] + m_x2[i] + round_delta) >> warp_affine_shift;
                int y = (m_y1[j] + m_y2[i] + round_delta) >> warp_affine_shift;
                if (static_cast<uint32_t>(x) < src_x_max && static_cast<uint32_t>(y) < src_y_max) {
                    if constexpr (SrcCrop) {
                        if constexpr (is_2pix_yuv_cvt_v<PixelCvt>) {
                            pixel_cvt((y + y1) * src_row_step + (x + x1) * src_col_step + src, dst, x & 1);
                        } else {
                            pixel_cvt((y + y1) * src_row_step + (x + x1) * src_col_step + src, dst);
                        }
                    } else {
                        if constexpr (is_2pix_yuv_cvt_v<PixelCvt>) {
                            pixel_cvt(y * src_row_step + x * src_col_step + src, dst, x & 1);
                        } else {
                            pixel_cvt(y * src_row_step + x * src_col_step + src, dst);
                        }
                    }
                } else {
                    memcpy(dst, v, dst_col_step);
                }
            }
            memcpy(dst, row_border_value, border_step_right);
            dst += border_step_right;
        }
        for (int i = 0; i < m_border[1]; i++, dst += dst_row_step) {
            memcpy(dst, row_border_value, dst_row_step);
        }
        heap_caps_free(row_border_value);
    }

    template <typename PixelCvt, bool SIMD>
    void transform_nn(const PixelCvt &pixel_cvt)
    {
        if (m_gen_xy_map) {
            gen_xy_map();
            m_gen_xy_map = false;
        }

#if CONFIG_IDF_TARGET_ESP32P4
        uint32_t cfg = 0;
        if constexpr (SIMD) {
            cfg = dl_esp32p4_get_cfg();
            dl_esp32p4_cfg_misalign(HW_MISALIGN, HW_MISALIGN);
        }
#endif
        if (m_scale_x == 1 && m_scale_y == 1) {
            if (m_border.empty()) {
                if (m_crop_area.empty()) {
                    cvt_color<PixelCvt, false, SIMD>(pixel_cvt);
                } else {
                    cvt_color<PixelCvt, true, SIMD>(pixel_cvt);
                }
            } else {
                if (m_bg_value_same) {
                    if (m_crop_area.empty()) {
                        cvt_color_same_value_border<PixelCvt, false, SIMD>(pixel_cvt);
                    } else {
                        cvt_color_same_value_border<PixelCvt, true, SIMD>(pixel_cvt);
                    }
                } else {
                    if (m_crop_area.empty()) {
                        cvt_color_diff_value_border<PixelCvt, false, SIMD>(pixel_cvt);
                    } else {
                        cvt_color_diff_value_border<PixelCvt, true, SIMD>(pixel_cvt);
                    }
                }
            }
        } else {
            if (!m_M.array) {
                if (m_border.empty()) {
                    resize_nn<PixelCvt, SIMD>(pixel_cvt);
                } else {
                    if (m_bg_value_same) {
                        resize_nn_same_value_border<PixelCvt, SIMD>(pixel_cvt);
                    } else {
                        resize_nn_diff_value_border<PixelCvt, SIMD>(pixel_cvt);
                    }
                }
            } else {
                if (m_border.empty()) {
                    if (m_crop_area.empty()) {
                        if (m_bg_value_same) {
                            warp_affine_nn<PixelCvt, false, true>(pixel_cvt);
                        } else {
                            warp_affine_nn<PixelCvt, false, false>(pixel_cvt);
                        }
                    } else {
                        if (m_bg_value_same) {
                            warp_affine_nn<PixelCvt, true, true>(pixel_cvt);
                        } else {
                            warp_affine_nn<PixelCvt, true, false>(pixel_cvt);
                        }
                    }
                } else {
                    if (m_crop_area.empty()) {
                        if (m_bg_value_same) {
                            warp_affine_nn_same_value_border<PixelCvt, false>(pixel_cvt);
                        } else {
                            warp_affine_nn_diff_value_border<PixelCvt, false>(pixel_cvt);
                        }
                    } else {
                        if (m_bg_value_same) {
                            warp_affine_nn_same_value_border<PixelCvt, true>(pixel_cvt);
                        } else {
                            warp_affine_nn_diff_value_border<PixelCvt, true>(pixel_cvt);
                        }
                    }
                }
            }
        }
#if CONFIG_IDF_TARGET_ESP32P4
        if constexpr (SIMD) {
            dl_esp32p4_cfg_misalign((misalign_mode_t)(cfg & 0b1), (misalign_mode_t)(cfg & 0b10));
        }
#endif
    }

    static constexpr int warp_affine_shift = 10;
    img_t m_src_img;
    img_t m_dst_img;
    float m_scale_x;
    float m_scale_y;
    float m_inv_scale_x;
    float m_inv_scale_y;
    math::Matrix<float> m_M;
    std::vector<int> m_crop_area; /*!< left_top_x, left_top_y, bottom_right_x, bottom_right_y */
    std::vector<int> m_border;    /*!< top, bottom, left, right */
    pix_cvt_param_t m_pix_cvt_param;
    int *m_x;
    int *m_y;
    int *m_x1;
    int *m_x2;
    int *m_y1;
    int *m_y2;
    bool m_gen_xy_map;
    std::variant<std::monostate, std::array<uint8_t, 3>, uint8_t> m_ori_bg_value;
    std::vector<uint8_t> m_bg_value;
    bool m_bg_value_same;
};
} // namespace image
} // namespace dl
