#pragma once
#include "dl_image_define.hpp"
#include "dl_math_matrix.hpp"
#include "esp_heap_caps.h"
#include <algorithm>
#include <vector>

namespace dl {
namespace image {
class ImageTransformer {
public:
    ImageTransformer();
    ~ImageTransformer();
    ImageTransformer &set_norm_quant_lut(void *lut);
    ImageTransformer &set_src_img_crop_area(const std::vector<int> &crop_area);
    ImageTransformer &set_dst_img_border(const std::vector<int> &border);
    ImageTransformer &set_bg_value(const std::vector<uint8_t> &bg_value);
    ImageTransformer &set_src_img(const img_t &src_img);
    ImageTransformer &set_dst_img(const img_t &dst_img);
    ImageTransformer &set_caps(uint32_t caps);
    ImageTransformer &set_warp_affine_matrix(const math::Matrix<float> &M, bool inv = false);
    std::vector<int> &get_src_img_crop_area();
    std::vector<int> &get_dst_img_border();
    const img_t &get_src_img();
    const img_t &get_dst_img();
    float get_scale_x(bool inv = false);
    float get_scale_y(bool inv = false);
    void reset();
    esp_err_t transform();

private:
    void gen_xy_map();

    template <typename PixelCvt>
    void resize_nn(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width;
        int dst_height = m_dst_img.height;
        int dst_step = get_pix_byte_size(m_dst_img.pix_type);
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        for (int i = 0; i < dst_height; i++) {
            uint8_t *p_row = src + m_y[i];
            for (int j = 0; j < dst_width; j++, dst += dst_step) {
                pixel_cvt(p_row + m_x[j], dst);
            }
        }
    }

    template <typename PixelCvt>
    void resize_nn_same_value_border(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int dst_step = get_pix_byte_size(m_dst_img.pix_type);
        int dst_row_step = dst_step * m_dst_img.width;
        int border_step_top = m_border[0] * dst_row_step;
        int border_step_bottom = m_border[1] * dst_row_step;
        int border_step_left = dst_step * m_border[2];
        int border_step_right = dst_step * m_border[3];
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        uint8_t v = m_bg_value[0];
        memset(dst, v, border_step_top);
        dst += border_step_top;
        for (int i = 0; i < dst_height; i++) {
            memset(dst, v, border_step_left);
            dst += border_step_left;
            uint8_t *p_row = src + m_y[i];
            for (int j = 0; j < dst_width; j++, dst += dst_step) {
                pixel_cvt(p_row + m_x[j], dst);
            }
            memset(dst, v, border_step_right);
            dst += border_step_right;
        }
        memset(dst, v, border_step_bottom);
    }

    template <typename PixelCvt>
    void resize_nn_diff_value_border(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int dst_step = get_pix_byte_size(m_dst_img.pix_type);
        int dst_row_step = dst_step * m_dst_img.width;
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
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
        for (int i = 0; i < dst_height; i++) {
            memcpy(dst, row_border_value, border_step_left);
            dst += border_step_left;
            uint8_t *p_row = src + m_y[i];
            for (int j = 0; j < dst_width; j++, dst += dst_step) {
                pixel_cvt(p_row + m_x[j], dst);
            }
            memcpy(dst, row_border_value, border_step_right);
            dst += border_step_right;
        }
        for (int i = 0; i < m_border[1]; i++, dst += dst_row_step) {
            memcpy(dst, row_border_value, dst_row_step);
        }
        heap_caps_free(row_border_value);
    }

    template <typename PixelCvt, bool SameValueBG>
    void warp_affine_nn(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width;
        int dst_height = m_dst_img.height;
        int dst_step = get_pix_byte_size(m_dst_img.pix_type);
        int src_col_step = get_pix_byte_size(m_src_img.pix_type);
        int src_row_step = m_src_img.width * get_pix_byte_size(m_src_img.pix_type);
        int src_x_max = m_src_img.width;
        int src_y_max = m_src_img.height;
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        constexpr int shift = 10;
        constexpr int round_delta = 1 << (shift - 1);
        if constexpr (SameValueBG) {
            uint8_t v = m_bg_value[0];
            for (int i = 0; i < dst_height; i++) {
                for (int j = 0; j < dst_width; j++, dst += dst_step) {
                    int x = (m_x1[j] + m_x2[i] + round_delta) >> shift;
                    int y = (m_y1[j] + m_y2[i] + round_delta) >> shift;
                    if (static_cast<uint32_t>(x) < src_x_max && static_cast<uint32_t>(y) < src_y_max) {
                        pixel_cvt(y * src_row_step + x * src_col_step + src, dst);
                    } else {
                        memset(dst, v, dst_step);
                    }
                }
            }
        } else {
            uint8_t *v = m_bg_value.data();
            for (int i = 0; i < dst_height; i++) {
                for (int j = 0; j < dst_width; j++, dst += dst_step) {
                    int x = (m_x1[j] + m_x2[i] + round_delta) >> shift;
                    int y = (m_y1[j] + m_y2[i] + round_delta) >> shift;
                    if (static_cast<uint32_t>(x) < src_x_max && static_cast<uint32_t>(y) < src_y_max) {
                        pixel_cvt(y * src_row_step + x * src_col_step + src, dst);
                    } else {
                        memcpy(dst, v, dst_step);
                    }
                }
            }
        }
    }

    template <typename PixelCvt>
    void warp_affine_nn_same_value_border(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int dst_step = get_pix_byte_size(m_dst_img.pix_type);
        int dst_row_step = dst_step * m_dst_img.width;
        int src_col_step = get_pix_byte_size(m_src_img.pix_type);
        int src_row_step = m_src_img.width * get_pix_byte_size(m_src_img.pix_type);
        int src_x_max = m_src_img.width;
        int src_y_max = m_src_img.height;
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        int border_step_top = m_border[0] * dst_row_step;
        int border_step_bottom = m_border[1] * dst_row_step;
        int border_step_left = dst_step * m_border[2];
        int border_step_right = dst_step * m_border[3];
        constexpr int shift = 10;
        constexpr int round_delta = 1 << (shift - 1);
        uint8_t v = m_bg_value[0];
        memset(dst, v, border_step_top);
        dst += border_step_top;
        for (int i = 0; i < dst_height; i++) {
            memset(dst, v, border_step_left);
            dst += border_step_left;
            for (int j = 0; j < dst_width; j++, dst += dst_step) {
                int x = (m_x1[j] + m_x2[i] + round_delta) >> shift;
                int y = (m_y1[j] + m_y2[i] + round_delta) >> shift;
                if (static_cast<uint32_t>(x) < src_x_max && static_cast<uint32_t>(y) < src_y_max) {
                    pixel_cvt(y * src_row_step + x * src_col_step + src, dst);
                } else {
                    memset(dst, v, dst_step);
                }
            }
            memset(dst, v, border_step_right);
            dst += border_step_right;
        }
        memset(dst, v, border_step_bottom);
    }

    template <typename PixelCvt>
    void warp_affine_nn_diff_value_border(const PixelCvt &pixel_cvt)
    {
        int dst_width = m_dst_img.width - m_border[2] - m_border[3];
        int dst_height = m_dst_img.height - m_border[0] - m_border[1];
        int dst_step = get_pix_byte_size(m_dst_img.pix_type);
        int dst_row_step = dst_step * m_dst_img.width;
        int src_col_step = get_pix_byte_size(m_src_img.pix_type);
        int src_row_step = m_src_img.width * get_pix_byte_size(m_src_img.pix_type);
        int src_x_max = m_src_img.width;
        int src_y_max = m_src_img.height;
        uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
        uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
        int border_step_left = dst_step * m_border[2];
        int border_step_right = dst_step * m_border[3];
        uint8_t *row_border_value = (uint8_t *)heap_caps_malloc(dst_row_step, MALLOC_CAP_DEFAULT);
        uint8_t *p_row_border_value = row_border_value;
        constexpr int round_delta = 1 << (warp_affine_shift - 1);
        uint8_t *v = m_bg_value.data();
        for (int i = 0; i < m_dst_img.width; i++, p_row_border_value += dst_step) {
            memcpy(p_row_border_value, v, dst_step);
        }
        for (int i = 0; i < m_border[0]; i++, dst += dst_row_step) {
            memcpy(dst, row_border_value, dst_row_step);
        }
        for (int i = 0; i < dst_height; i++) {
            memcpy(dst, row_border_value, border_step_left);
            dst += border_step_left;
            for (int j = 0; j < dst_width; j++, dst += dst_step) {
                int x = (m_x1[j] + m_x2[i] + round_delta) >> warp_affine_shift;
                int y = (m_y1[j] + m_y2[i] + round_delta) >> warp_affine_shift;
                if (static_cast<uint32_t>(x) < src_x_max && static_cast<uint32_t>(y) < src_y_max) {
                    pixel_cvt(y * src_row_step + x * src_col_step + src, dst);
                } else {
                    memcpy(dst, v, dst_step);
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

    // TODO optimze when resize degenerates to cvtcolor
    template <typename PixelCvt>
    void transform_nn(const PixelCvt &pixel_cvt)
    {
        if (m_bg_value.empty() && (m_M.array || !m_border.empty())) {
            m_bg_value.assign(get_pix_byte_size(m_src_img.pix_type), 0);
            m_new_bg_value = true;
        }
        if (m_new_bg_value) {
            std::vector<uint8_t> dst_bg_value(get_pix_byte_size(m_dst_img.pix_type));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
            pixel_cvt(m_bg_value.data(), dst_bg_value.data());
#pragma GCC diagnostic pop
            m_bg_value.swap(dst_bg_value);
            m_bg_value_same = std::all_of(
                m_bg_value.begin() + 1, m_bg_value.end(), [this](const auto &v) { return v == m_bg_value[0]; });
            m_new_bg_value = false;
        }
        if (!m_M.array) {
            if (m_border.empty()) {
                resize_nn(pixel_cvt);
            } else {
                if (m_bg_value_same) {
                    resize_nn_same_value_border(pixel_cvt);
                } else {
                    resize_nn_diff_value_border(pixel_cvt);
                }
            }
        } else {
            if (m_border.empty()) {
                if (m_bg_value_same) {
                    warp_affine_nn<PixelCvt, true>(pixel_cvt);
                } else {
                    warp_affine_nn<PixelCvt, false>(pixel_cvt);
                }
            } else {
                if (m_bg_value_same) {
                    warp_affine_nn_same_value_border(pixel_cvt);
                } else {
                    warp_affine_nn_diff_value_border(pixel_cvt);
                }
            }
        }
    }

    static inline constexpr int warp_affine_shift = 10;
    img_t m_src_img;
    img_t m_dst_img;
    float m_scale_x;
    float m_scale_y;
    float m_inv_scale_x;
    float m_inv_scale_y;
    math::Matrix<float> m_M;
    std::vector<int> m_crop_area;    /*!< left_top_x, left_top_y, bottom_right_x, bottom_right_y */
    std::vector<int> m_border;       /*!< top, bottom, left, right */
    std::vector<uint8_t> m_bg_value; /*!< the value to fill background */
    uint32_t m_caps;
    int *m_x;
    int *m_y;
    int *m_x1;
    int *m_x2;
    int *m_y1;
    int *m_y2;
    void *m_norm_quant_lut;
    bool m_gen_xy_map;
    bool m_new_bg_value;
    bool m_bg_value_same;
};
} // namespace image
} // namespace dl
