#include "dl_image_process.hpp"
#include "esp_log.h"
#include <cstring>

static const char *TAG = "ImageTransformer";

namespace dl {
namespace image {
ImageTransformer::ImageTransformer() : m_src_img(), m_dst_img()
{
}

ImageTransformer::ImageTransformer(ImageTransformer &&rhs) noexcept :
    m_src_img(rhs.m_src_img),
    m_dst_img(rhs.m_dst_img),
    m_M(rhs.m_M),
    m_crop_area(std::move(rhs.m_crop_area)),
    m_border(std::move(rhs.m_border)),
    m_hsv(rhs.m_hsv),
    m_norm(rhs.m_norm),
    m_kernel(std::move(rhs.m_kernel)),
    m_kernel_dirty(rhs.m_kernel_dirty),
    m_xy(rhs.m_xy),
    m_bg(rhs.m_bg)
{
    rhs.m_src_img = {};
    rhs.m_dst_img = {};
    rhs.m_M = {};
    rhs.m_hsv = {};
    rhs.m_norm = {};
    rhs.m_kernel_dirty = true;
    rhs.m_xy = {};
    rhs.m_bg = {};
}

ImageTransformer &ImageTransformer::operator=(ImageTransformer &&rhs) noexcept
{
    if (this != &rhs) {
        m_xy.release();
        m_src_img = rhs.m_src_img;
        m_dst_img = rhs.m_dst_img;
        m_M = rhs.m_M;
        m_crop_area = std::move(rhs.m_crop_area);
        m_border = std::move(rhs.m_border);
        m_hsv = rhs.m_hsv;
        m_norm = rhs.m_norm;
        m_kernel = std::move(rhs.m_kernel);
        m_kernel_dirty = rhs.m_kernel_dirty;
        m_xy = rhs.m_xy;
        m_bg = rhs.m_bg;
        rhs.m_src_img = {};
        rhs.m_dst_img = {};
        rhs.m_M = {};
        rhs.m_border = {};
        rhs.m_hsv = {};
        rhs.m_norm = {};
        rhs.m_kernel_dirty = true;
        rhs.m_xy = {};
        rhs.m_bg = {};
    }
    return *this;
}

ImageTransformer::~ImageTransformer()
{
    m_xy.release();
}

ImageTransformer &ImageTransformer::set_norm_quant_param(const std::array<float, 3> &mean,
                                                         const std::array<float, 3> &std,
                                                         int exp,
                                                         int quant_bits)
{
    if (quant_bits == 8 || quant_bits == 16) {
        m_norm.channels = 3;
        m_norm.exp = exp;
        m_norm.quant_bits = quant_bits;
        m_norm.mean = mean;
        m_norm.std = std;
        m_norm.set = true;
        m_kernel_dirty = true;
        m_bg.dirty = true;
    } else {
        ESP_LOGE(TAG, "Quant_bits can only be 8/16.");
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_norm_quant_param(float mean, float std, int exp, int quant_bits)
{
    if (quant_bits == 8 || quant_bits == 16) {
        m_norm.channels = 1;
        m_norm.exp = exp;
        m_norm.quant_bits = quant_bits;
        m_norm.mean = {mean, 0, 0};
        m_norm.std = {std, 0, 0};
        m_norm.set = true;
        m_kernel_dirty = true;
        m_bg.dirty = true;
    } else {
        ESP_LOGE(TAG, "Quant_bits can only be 8/16.");
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_hsv_thr(const std::array<uint8_t, 3> &hsv_min,
                                                const std::array<uint8_t, 3> &hsv_max)
{
    if (is_valid_hsv_thr(hsv_min, hsv_max)) {
        m_hsv.min = hsv_min;
        m_hsv.max = hsv_max;
        m_hsv.set = true;
        m_kernel_dirty = true;
        m_bg.dirty = true;
    } else {
        ESP_LOGE(TAG, "Invalid hsv thr.");
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_src_img_crop_area(const std::vector<int> &crop_area)
{
    assert(((crop_area.size() == 4 && crop_area[0] >= 0 && crop_area[1] >= 0 && crop_area[0] < crop_area[2] &&
             crop_area[1] < crop_area[3])) ||
           crop_area.empty());
    if (m_crop_area != crop_area) {
        m_crop_area = crop_area;
        m_xy.dirty = true;
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_dst_img_border(const std::vector<int> &border)
{
    assert((border.size() == 4 && std::all_of(border.begin(), border.end(), [](const auto &v) { return v >= 0; })) ||
           border.empty());
    if (m_border != border) {
        m_border = border;
        m_xy.dirty = true;
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_bg_value(const std::array<uint8_t, 3> &bg_value)
{
    if (m_bg.ori_n != 3 || m_bg.ori[0] != bg_value[0] || m_bg.ori[1] != bg_value[1] || m_bg.ori[2] != bg_value[2]) {
        m_bg.ori[0] = bg_value[0];
        m_bg.ori[1] = bg_value[1];
        m_bg.ori[2] = bg_value[2];
        m_bg.ori_n = 3;
        m_bg.dirty = true;
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_bg_value(uint8_t bg_value)
{
    if (m_bg.ori_n != 1 || m_bg.ori[0] != bg_value) {
        m_bg.ori[0] = bg_value;
        m_bg.ori_n = 1;
        m_bg.dirty = true;
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_src_img(const img_t &src_img)
{
    if (src_img.pix_type != m_src_img.pix_type) {
        m_kernel_dirty = true;
    }
    if (src_img.channel() != m_src_img.channel()) {
        m_bg.dirty = true;
    }
    if ((src_img.width != m_src_img.width) || (src_img.height != m_src_img.height) ||
        src_img.col_step() != m_src_img.col_step()) {
        m_xy.dirty = true;
    }
    m_src_img = src_img;
    return *this;
}

ImageTransformer &ImageTransformer::set_dst_img(const img_t &dst_img)
{
    if (dst_img.pix_type != m_dst_img.pix_type) {
        m_kernel_dirty = true;
        m_bg.dirty = true;
    }
    if ((dst_img.width != m_dst_img.width) || (dst_img.height != m_dst_img.height) ||
        dst_img.col_step() != m_dst_img.col_step()) {
        m_xy.dirty = true;
    }
    m_dst_img = dst_img;
    return *this;
}

ImageTransformer &ImageTransformer::set_warp_affine_matrix(const math::Matrix<float> &M, bool inv)
{
    auto set_M = [this](const math::Matrix<float> &M) -> ImageTransformer & {
        assert((M.array && M.h == 2 && M.w == 3) || !M.array);
        bool flag = false;
        if ((m_M.array && !M.array) || (!m_M.array && M.array)) {
            flag = true;
        } else if (m_M.array && M.array) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) {
                    if (m_M.array[i][j] != M.array[i][j]) {
                        flag = true;
                        break;
                    }
                }
            }
        }
        if (flag) {
            m_M = M;
            m_xy.dirty = true;
        }
        return *this;
    };

    if (!inv && M.array) {
        math::Matrix<float> inv_M(2, 3);
        float **M_ = M.array;
        float **inv_M_ = inv_M.array;
        float M0 = M_[0][0], M1 = M_[0][1], M2 = M_[0][2], M3 = M_[1][0], M4 = M_[1][1], M5 = M_[1][2];
        float D = M0 * M4 - M1 * M3;
        D = D != 0 ? 1. / D : 0;
        inv_M_[0][0] = M4 * D;
        inv_M_[0][1] = M1 * -D;
        inv_M_[1][0] = M3 * -D;
        inv_M_[1][1] = M0 * D;
        inv_M_[0][2] = -M0 * M2 - M1 * M5;
        inv_M_[1][2] = -M3 * M2 - M4 * M5;
        return set_M(inv_M);
    } else {
        return set_M(M);
    }
}

std::vector<int> &ImageTransformer::get_src_img_crop_area()
{
    return m_crop_area;
}

std::vector<int> &ImageTransformer::get_dst_img_border()
{
    return m_border;
}

const img_t &ImageTransformer::get_src_img()
{
    return m_src_img;
}

const img_t &ImageTransformer::get_dst_img()
{
    return m_dst_img;
}

math::Matrix<float> &ImageTransformer::get_warp_affine_matrix()
{
    return m_M;
}

float ImageTransformer::get_scale_x(bool inv)
{
    return inv ? m_xy.inv_scale_x : m_xy.scale_x;
}

float ImageTransformer::get_scale_y(bool inv)
{
    return inv ? m_xy.inv_scale_y : m_xy.scale_y;
}

ImageTransformer &ImageTransformer::reset()
{
    m_xy.release();
    m_src_img = {};
    m_dst_img = {};
    m_xy = {};
    m_M = {};
    m_crop_area = {};
    m_border = {};
    m_hsv = {};
    m_norm = {};
    m_kernel.reset();
    m_kernel_dirty = true;
    m_bg = {};
    return *this;
}

esp_err_t ImageTransformer::params_valid() const
{
    if (!m_src_img.data || !m_src_img.height || !m_src_img.width) {
        ESP_LOGE(TAG, "Invalid src img, call set_src_img().");
        return ESP_FAIL;
    }
    if (!m_dst_img.data || !m_dst_img.height || !m_dst_img.width) {
        ESP_LOGE(TAG, "Invalid dst img, call set_dst_img().");
        return ESP_FAIL;
    }
    if (m_src_img.pix_type == DL_IMAGE_PIX_TYPE_YUYV || m_src_img.pix_type == DL_IMAGE_PIX_TYPE_UYVY) {
        if (m_src_img.width & 1) {
            ESP_LOGE(TAG, "YUV image width should be an even number.");
            return ESP_FAIL;
        }
    }
    if (!m_crop_area.empty() && (m_crop_area[2] > m_src_img.width || m_crop_area[3] > m_src_img.height)) {
        ESP_LOGE(TAG, "Invalid crop area.");
        return ESP_FAIL;
    }
    if (!m_border.empty() &&
        (m_border[0] + m_border[1] > m_dst_img.height - 1 || m_border[2] + m_border[3] > m_dst_img.width - 1)) {
        ESP_LOGE(TAG, "Invalid border.");
        return ESP_FAIL;
    }

    const pix_type_t dst = m_dst_img.pix_type;
    if (dst == DL_IMAGE_PIX_TYPE_HSV_MASK) {
        if (!m_hsv.set) {
            ESP_LOGE(TAG, "Dst img is hsv mask, but hsv_thr is not set. Call set_hsv_thr().");
            return ESP_FAIL;
        }
        return ESP_OK;
    }
    if (!is_pix_type_quant(dst)) {
        return ESP_OK;
    }
    if (!m_norm.set) {
        ESP_LOGE(TAG, "Dst img is quant type, but norm_quant_param is not set. Call set_norm_quant_param().");
        return ESP_FAIL;
    }
    const bool ch1 = m_norm.channels == 1;
    const bool bits8 = m_norm.quant_bits == 8;
    bool match = false;
    if (ch1 && bits8) {
        match = dst == DL_IMAGE_PIX_TYPE_GRAY_QINT8;
    } else if (ch1 && !bits8) {
        match = dst == DL_IMAGE_PIX_TYPE_GRAY_QINT16;
    } else if (!ch1 && bits8) {
        match = dst == DL_IMAGE_PIX_TYPE_RGB888_QINT8 || dst == DL_IMAGE_PIX_TYPE_BGR888_QINT8;
    } else {
        match = dst == DL_IMAGE_PIX_TYPE_RGB888_QINT16 || dst == DL_IMAGE_PIX_TYPE_BGR888_QINT16;
    }
    if (!match) {
        ESP_LOGE(TAG, "norm_quant_param does not match dst img pix_type %s.", pix_type2str(dst).c_str());
        return ESP_FAIL;
    }
    return ESP_OK;
}

esp_err_t ImageTransformer::ensure_kernel(bool simd)
{
    if (m_kernel_dirty) {
        m_kernel = make_pixel_kernel(m_src_img.pix_type, m_dst_img.pix_type, m_hsv, m_norm);
        m_kernel_dirty = false;
    }
    m_kernel.enable_simd(simd);
    if (!m_kernel.cvt_fn && !m_kernel.cvt_2pix_fn) {
        ESP_LOGE(TAG, "Failed to build pixel kernel.");
        return ESP_FAIL;
    }
    return ESP_OK;
}

esp_err_t ImageTransformer::cvt_color(const PixelKernel &k)
{
    uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
    uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
    if (pix_cvt_is_identity(k.cvt_id) && src == dst) {
        return ESP_OK;
    }
    const int src_step = m_src_img.col_step();
    const int dst_step = m_dst_img.col_step();
    const int w = m_dst_img.width;
    const int h = m_dst_img.height;
    const bool two_pix = pix_cvt_is_2pix(k.cvt_id);

    if (m_crop_area.empty()) {
        if (two_pix) {
            k.cvt_color(src, dst, w * h, false);
        } else {
            k.cvt_color(src, dst, w * h);
        }
        return ESP_OK;
    }

    const bool odd = m_crop_area[0] & 1;
    src += (m_src_img.width * m_crop_area[1] + m_crop_area[0]) * src_step;
    const int src_row_skip = (m_crop_area[0] + m_src_img.width - m_crop_area[2]) * src_step;
    const int src_row_bytes = w * src_step + src_row_skip;
    const int dst_row_bytes = w * dst_step;
    for (int y = 0; y < h; ++y) {
        if (two_pix) {
            k.cvt_color(src, dst, w, odd);
        } else {
            k.cvt_color(src, dst, w);
        }
        src += src_row_bytes;
        dst += dst_row_bytes;
    }
    return ESP_OK;
}

esp_err_t ImageTransformer::cvt_color_border(const PixelKernel &k)
{
    const int src_step = m_src_img.col_step();
    const int dst_step = m_dst_img.col_step();
    uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
    uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
    const int n_top = m_border[0];
    const int n_bottom = m_border[1];
    const int content_w = m_dst_img.width - m_border[2] - m_border[3];
    const int content_h = m_dst_img.height - n_top - n_bottom;
    const int dst_row_step = m_dst_img.row_step();
    const int border_top = n_top * dst_row_step;
    const int border_bottom = n_bottom * dst_row_step;
    const int border_left = dst_step * m_border[2];
    const int border_right = dst_step * m_border[3];

    int src_row_skip = 0;
    if (!m_crop_area.empty()) {
        src += (m_src_img.width * m_crop_area[1] + m_crop_area[0]) * src_step;
        src_row_skip = (m_crop_area[0] + m_src_img.width - m_crop_area[2]) * src_step;
    }
    const int src_content_bytes = content_w * src_step + src_row_skip;
    const int dst_content_bytes = content_w * dst_step;
    const bool two_pix = pix_cvt_is_2pix(k.cvt_id);
    const bool odd = !m_crop_area.empty() && (m_crop_area[0] & 1);

    if (m_bg.same) {
        const uint8_t v = m_bg.pix[0];
        memset(dst, v, border_top);
        dst += border_top;
        for (int y = 0; y < content_h; ++y) {
            memset(dst, v, border_left);
            dst += border_left;
            if (two_pix) {
                k.cvt_color(src, dst, content_w, odd);
            } else {
                k.cvt_color(src, dst, content_w);
            }
            src += src_content_bytes;
            dst += dst_content_bytes;
            memset(dst, v, border_right);
            dst += border_right;
        }
        memset(dst, v, border_bottom);
        return ESP_OK;
    }

    uint8_t *row_border = static_cast<uint8_t *>(heap_caps_malloc(dst_row_step, MALLOC_CAP_DEFAULT));
    if (!row_border) {
        ESP_LOGE(TAG, "Failed to allocate border row.");
        return ESP_ERR_NO_MEM;
    }
    uint8_t *p = row_border;
    uint8_t *bg = m_bg.pix;
    for (int i = 0; i < m_dst_img.width; ++i, p += dst_step) {
        memcpy(p, bg, m_bg.pix_size);
    }
    for (int i = 0; i < n_top; ++i, dst += dst_row_step) {
        memcpy(dst, row_border, dst_row_step);
    }
    for (int y = 0; y < content_h; ++y) {
        memcpy(dst, row_border, border_left);
        dst += border_left;
        if (two_pix) {
            k.cvt_color(src, dst, content_w, odd);
        } else {
            k.cvt_color(src, dst, content_w);
        }
        src += src_content_bytes;
        dst += dst_content_bytes;
        memcpy(dst, row_border, border_right);
        dst += border_right;
    }
    for (int i = 0; i < n_bottom; ++i, dst += dst_row_step) {
        memcpy(dst, row_border, dst_row_step);
    }
    heap_caps_free(row_border);
    return ESP_OK;
}

esp_err_t ImageTransformer::resize_nn(const PixelKernel &k)
{
    uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
    uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
    const int dst_step = m_dst_img.col_step();
    const int w = m_dst_img.width;
    const int h = m_dst_img.height;
    const int dst_row_bytes = w * dst_step;
    for (int i = 0; i < h; ++i) {
        k.resize_nn(src + m_xy.y[i], m_xy.x, dst, w);
        dst += dst_row_bytes;
    }
    return ESP_OK;
}

esp_err_t ImageTransformer::resize_nn_border(const PixelKernel &k)
{
    const int dst_step = m_dst_img.col_step();
    uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
    uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
    const int n_top = m_border[0];
    const int n_bottom = m_border[1];
    const int content_w = m_dst_img.width - m_border[2] - m_border[3];
    const int content_h = m_dst_img.height - n_top - n_bottom;
    const int dst_row_step = m_dst_img.row_step();
    const int border_top = n_top * dst_row_step;
    const int border_bottom = n_bottom * dst_row_step;
    const int border_left = dst_step * m_border[2];
    const int border_right = dst_step * m_border[3];
    const int dst_content_bytes = content_w * dst_step;

    if (m_bg.same) {
        const uint8_t v = m_bg.pix[0];
        memset(dst, v, border_top);
        dst += border_top;
        for (int i = 0; i < content_h; ++i) {
            memset(dst, v, border_left);
            dst += border_left;
            k.resize_nn(src + m_xy.y[i], m_xy.x, dst, content_w);
            dst += dst_content_bytes;
            memset(dst, v, border_right);
            dst += border_right;
        }
        memset(dst, v, border_bottom);
        return ESP_OK;
    }

    uint8_t *row_border = static_cast<uint8_t *>(heap_caps_malloc(dst_row_step, MALLOC_CAP_DEFAULT));
    if (!row_border) {
        ESP_LOGE(TAG, "Failed to allocate border row.");
        return ESP_ERR_NO_MEM;
    }
    uint8_t *p = row_border;
    uint8_t *bg = m_bg.pix;
    for (int i = 0; i < m_dst_img.width; ++i, p += dst_step) {
        memcpy(p, bg, m_bg.pix_size);
    }
    for (int i = 0; i < n_top; ++i, dst += dst_row_step) {
        memcpy(dst, row_border, dst_row_step);
    }
    for (int i = 0; i < content_h; ++i) {
        memcpy(dst, row_border, border_left);
        dst += border_left;
        k.resize_nn(src + m_xy.y[i], m_xy.x, dst, content_w);
        dst += dst_content_bytes;
        memcpy(dst, row_border, border_right);
        dst += border_right;
    }
    for (int i = 0; i < n_bottom; ++i, dst += dst_row_step) {
        memcpy(dst, row_border, dst_row_step);
    }
    heap_caps_free(row_border);
    return ESP_OK;
}

void ImageTransformer::fill_warp_offset(
    int *offset, int row, int n, int x0, int y0, uint32_t x_max, uint32_t y_max) const
{
    constexpr int round_delta = 1 << (warp_affine_shift - 1);
    const int src_col_step = m_src_img.col_step();
    const int src_row_step = m_src_img.row_step();
    const int x2 = m_xy.x2[row];
    const int y2 = m_xy.y2[row];
    for (int j = 0; j < n; ++j) {
        const int x = (m_xy.x1[j] + x2 + round_delta) >> warp_affine_shift;
        const int y = (m_xy.y1[j] + y2 + round_delta) >> warp_affine_shift;
        if (static_cast<uint32_t>(x) < x_max && static_cast<uint32_t>(y) < y_max) {
            offset[j] = (y + y0) * src_row_step + (x + x0) * src_col_step;
        } else {
            offset[j] = -1;
        }
    }
}

esp_err_t ImageTransformer::warp_affine_nn(const PixelKernel &k)
{
    uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
    uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
    const int w = m_dst_img.width;
    const int h = m_dst_img.height;
    const int dst_row_bytes = w * m_dst_img.col_step();
    int *offset = static_cast<int *>(heap_caps_malloc(w * sizeof(int), MALLOC_CAP_DEFAULT));
    if (!offset) {
        ESP_LOGE(TAG, "Failed to allocate warp offset.");
        return ESP_ERR_NO_MEM;
    }
    const uint8_t *bg = m_bg.pix;
    int x0 = 0;
    int y0 = 0;
    int src_x_max = m_src_img.width;
    int src_y_max = m_src_img.height;
    if (!m_crop_area.empty()) {
        x0 = m_crop_area[0];
        y0 = m_crop_area[1];
        src_x_max = m_crop_area[2] - x0;
        src_y_max = m_crop_area[3] - y0;
    }
    const uint32_t x_max = static_cast<uint32_t>(src_x_max);
    const uint32_t y_max = static_cast<uint32_t>(src_y_max);
    for (int i = 0; i < h; ++i) {
        fill_warp_offset(offset, i, w, x0, y0, x_max, y_max);
        k.warp_nn(src, offset, dst, w, bg);
        dst += dst_row_bytes;
    }
    heap_caps_free(offset);
    return ESP_OK;
}

esp_err_t ImageTransformer::warp_affine_nn_border(const PixelKernel &k)
{
    const int dst_step = m_dst_img.col_step();
    uint8_t *src = static_cast<uint8_t *>(m_src_img.data);
    uint8_t *dst = static_cast<uint8_t *>(m_dst_img.data);
    const int n_top = m_border[0];
    const int n_bottom = m_border[1];
    const int content_w = m_dst_img.width - m_border[2] - m_border[3];
    const int content_h = m_dst_img.height - n_top - n_bottom;
    const int dst_row_step = m_dst_img.row_step();
    const int border_top = n_top * dst_row_step;
    const int border_bottom = n_bottom * dst_row_step;
    const int border_left = dst_step * m_border[2];
    const int border_right = dst_step * m_border[3];
    const int dst_content_bytes = content_w * dst_step;
    int *offset = static_cast<int *>(heap_caps_malloc(content_w * sizeof(int), MALLOC_CAP_DEFAULT));
    if (!offset) {
        ESP_LOGE(TAG, "Failed to allocate warp offset.");
        return ESP_ERR_NO_MEM;
    }
    const uint8_t *bg = m_bg.pix;
    int x0 = 0;
    int y0 = 0;
    int src_x_max = m_src_img.width;
    int src_y_max = m_src_img.height;
    if (!m_crop_area.empty()) {
        x0 = m_crop_area[0];
        y0 = m_crop_area[1];
        src_x_max = m_crop_area[2] - x0;
        src_y_max = m_crop_area[3] - y0;
    }
    const uint32_t x_max = static_cast<uint32_t>(src_x_max);
    const uint32_t y_max = static_cast<uint32_t>(src_y_max);

    if (m_bg.same) {
        const uint8_t v = m_bg.pix[0];
        memset(dst, v, border_top);
        dst += border_top;
        for (int i = 0; i < content_h; ++i) {
            memset(dst, v, border_left);
            dst += border_left;
            fill_warp_offset(offset, i, content_w, x0, y0, x_max, y_max);
            k.warp_nn(src, offset, dst, content_w, bg);
            dst += dst_content_bytes;
            memset(dst, v, border_right);
            dst += border_right;
        }
        memset(dst, v, border_bottom);
        heap_caps_free(offset);
        return ESP_OK;
    }

    uint8_t *row_border = static_cast<uint8_t *>(heap_caps_malloc(dst_row_step, MALLOC_CAP_DEFAULT));
    if (!row_border) {
        heap_caps_free(offset);
        ESP_LOGE(TAG, "Failed to allocate border row.");
        return ESP_ERR_NO_MEM;
    }
    uint8_t *p = row_border;
    uint8_t *row_bg = m_bg.pix;
    for (int i = 0; i < m_dst_img.width; ++i, p += dst_step) {
        memcpy(p, row_bg, m_bg.pix_size);
    }
    for (int i = 0; i < n_top; ++i, dst += dst_row_step) {
        memcpy(dst, row_border, dst_row_step);
    }
    for (int i = 0; i < content_h; ++i) {
        memcpy(dst, row_border, border_left);
        dst += border_left;
        fill_warp_offset(offset, i, content_w, x0, y0, x_max, y_max);
        k.warp_nn(src, offset, dst, content_w, bg);
        dst += dst_content_bytes;
        memcpy(dst, row_border, border_right);
        dst += border_right;
    }
    for (int i = 0; i < n_bottom; ++i, dst += dst_row_step) {
        memcpy(dst, row_border, dst_row_step);
    }
    heap_caps_free(row_border);
    heap_caps_free(offset);
    return ESP_OK;
}

esp_err_t ImageTransformer::gen_bg()
{
    if (m_border.empty() && !m_M.array) {
        return ESP_OK;
    }
    if (!m_bg.dirty) {
        return ESP_OK;
    }

    const int channel = m_src_img.channel();
    uint8_t ori[3] = {};
    pix_type_t src_pix_type;
    if (channel == 1) {
        if (m_bg.ori_n == 3) {
            ESP_LOGE(TAG, "Background color and src image type mismatch.");
            return ESP_FAIL;
        }
        if (m_bg.ori_n == 1) {
            ori[0] = m_bg.ori[0];
        }
        src_pix_type = DL_IMAGE_PIX_TYPE_GRAY;
    } else if (channel == 3) {
        if (m_bg.ori_n == 1) {
            ESP_LOGE(TAG, "Background color and src image type mismatch.");
            return ESP_FAIL;
        }
        if (m_bg.ori_n == 3) {
            ori[0] = m_bg.ori[0];
            ori[1] = m_bg.ori[1];
            ori[2] = m_bg.ori[2];
        }
        src_pix_type = DL_IMAGE_PIX_TYPE_RGB888;
    } else {
        ESP_LOGE(TAG, "Invalid channel num.");
        return ESP_FAIL;
    }

    const int dst_step = m_dst_img.col_step();
    if (dst_step <= 0 || dst_step > static_cast<int>(sizeof(m_bg.pix))) {
        ESP_LOGE(TAG, "Invalid dst pixel size.");
        return ESP_FAIL;
    }
    PixelKernel bg_kernel = make_pixel_kernel(src_pix_type, m_dst_img.pix_type, m_hsv, m_norm);
    if (!bg_kernel.cvt_fn) {
        ESP_LOGE(TAG, "Failed to build background pixel kernel.");
        return ESP_FAIL;
    }
    bg_kernel.cvt_color(ori, m_bg.pix, 1);
    m_bg.pix_size = static_cast<uint8_t>(dst_step);
    m_bg.same = true;
    for (int i = 1; i < dst_step; ++i) {
        if (m_bg.pix[i] != m_bg.pix[0]) {
            m_bg.same = false;
            break;
        }
    }
    m_bg.dirty = false;
    return ESP_OK;
}

esp_err_t ImageTransformer::transform(bool simd)
{
    esp_err_t err = params_valid();
    if (err != ESP_OK) {
        return err;
    }
    err = ensure_kernel(simd);
    if (err != ESP_OK) {
        return err;
    }
    err = gen_xy_map();
    if (err != ESP_OK) {
        return err;
    }
    err = gen_bg();
    if (err != ESP_OK) {
        return err;
    }

    const bool border = !m_border.empty();

#if CONFIG_PIE_V2_BOOST
    uint32_t cfg = 0;
    if (simd) {
        cfg = dl_esp32p4_get_cfg();
        dl_esp32p4_cfg_misalign(HW_MISALIGN, HW_MISALIGN);
    }
#endif
    switch (m_xy.geom) {
    case xy_map_t::geom_t::same:
        err = border ? cvt_color_border(m_kernel) : cvt_color(m_kernel);
        break;
    case xy_map_t::geom_t::resize:
        err = border ? resize_nn_border(m_kernel) : resize_nn(m_kernel);
        break;
    case xy_map_t::geom_t::warp:
        err = border ? warp_affine_nn_border(m_kernel) : warp_affine_nn(m_kernel);
        break;
    }
#if CONFIG_PIE_V2_BOOST
    if (simd) {
        dl_esp32p4_cfg_misalign((misalign_mode_t)(cfg & 0b1), (misalign_mode_t)((cfg >> 1) & 0b1));
    }
#endif
    return err;
}

esp_err_t ImageTransformer::gen_xy_map()
{
    if (!m_xy.dirty) {
        return ESP_OK;
    }

    int dst_width = m_border.empty() ? m_dst_img.width : (m_dst_img.width - m_border[2] - m_border[3]);
    int dst_height = m_border.empty() ? m_dst_img.height : (m_dst_img.height - m_border[0] - m_border[1]);

    int col_step = m_src_img.col_step();
    int row_step = m_src_img.row_step();

    m_xy.release();

    if (!m_M.array) {
        int src_width = m_crop_area.empty() ? m_src_img.width : (m_crop_area[2] - m_crop_area[0]);
        int src_height = m_crop_area.empty() ? m_src_img.height : (m_crop_area[3] - m_crop_area[1]);
        if (src_width == dst_width && src_height == dst_height) {
            m_xy.scale_x = m_xy.scale_y = m_xy.inv_scale_x = m_xy.inv_scale_y = 1;
            m_xy.geom = xy_map_t::geom_t::same;
            m_xy.dirty = false;
            return ESP_OK;
        }
        m_xy.x = (int *)heap_caps_malloc(dst_width * sizeof(int), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD);
        m_xy.y = (int *)heap_caps_malloc(dst_height * sizeof(int), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD);
        if (!m_xy.x || !m_xy.y) {
            m_xy.release();
            ESP_LOGE(TAG, "Failed to allocate resize map.");
            return ESP_ERR_NO_MEM;
        }
        m_xy.inv_scale_x = static_cast<float>(src_width) / dst_width;
        m_xy.inv_scale_y = static_cast<float>(src_height) / dst_height;
        m_xy.scale_x = 1.f / m_xy.inv_scale_x;
        m_xy.scale_y = 1.f / m_xy.inv_scale_y;
        m_xy.geom = xy_map_t::geom_t::resize;
        if (m_crop_area.empty()) {
            for (int i = 0; i < dst_width; i++) {
                m_xy.x[i] = std::min(static_cast<int>(i * m_xy.inv_scale_x), src_width - 1) * col_step;
            }
            for (int i = 0; i < dst_height; i++) {
                m_xy.y[i] = std::min(static_cast<int>(i * m_xy.inv_scale_y), src_height - 1) * row_step;
            }
        } else {
            for (int i = 0; i < dst_width; i++) {
                m_xy.x[i] =
                    std::min(static_cast<int>(i * m_xy.inv_scale_x) + m_crop_area[0], m_crop_area[2] - 1) * col_step;
            }
            for (int i = 0; i < dst_height; i++) {
                m_xy.y[i] =
                    std::min(static_cast<int>(i * m_xy.inv_scale_y) + m_crop_area[1], m_crop_area[3] - 1) * row_step;
            }
        }
    } else {
        m_xy.x1 = (int *)heap_caps_malloc(dst_width * sizeof(int), MALLOC_CAP_DEFAULT);
        m_xy.x2 = (int *)heap_caps_malloc(dst_height * sizeof(int), MALLOC_CAP_DEFAULT);
        m_xy.y1 = (int *)heap_caps_malloc(dst_width * sizeof(int), MALLOC_CAP_DEFAULT);
        m_xy.y2 = (int *)heap_caps_malloc(dst_height * sizeof(int), MALLOC_CAP_DEFAULT);
        if (!m_xy.x1 || !m_xy.x2 || !m_xy.y1 || !m_xy.y2) {
            m_xy.release();
            ESP_LOGE(TAG, "Failed to allocate warp map.");
            return ESP_ERR_NO_MEM;
        }
        float **M = m_M.array;
        float M0 = M[0][0], M1 = M[0][1], M2 = M[0][2], M3 = M[1][0], M4 = M[1][1], M5 = M[1][2];
        constexpr int scale = 1 << warp_affine_shift;
        for (int i = 0; i < dst_width; i++) {
            m_xy.x1[i] = static_cast<int>(__builtin_lrintf(M0 * i * scale));
            m_xy.y1[i] = static_cast<int>(__builtin_lrintf(M3 * i * scale));
        }
        for (int i = 0; i < dst_height; i++) {
            m_xy.x2[i] = static_cast<int>(__builtin_lrintf((M1 * i + M2) * scale));
            m_xy.y2[i] = static_cast<int>(__builtin_lrintf((M4 * i + M5) * scale));
        }
        m_xy.scale_x = m_xy.scale_y = m_xy.inv_scale_x = m_xy.inv_scale_y = 0;
        m_xy.geom = xy_map_t::geom_t::warp;
    }
    m_xy.dirty = false;
    return ESP_OK;
}
} // namespace image
} // namespace dl
