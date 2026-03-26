#include "dl_image_process.hpp"
#include "esp_log.h"

static const char *TAG = "ImageTransformer";

namespace dl {
namespace image {
ImageTransformer::ImageTransformer() :
    m_src_img(),
    m_dst_img(),
    m_scale_x(0),
    m_scale_y(0),
    m_inv_scale_x(0),
    m_inv_scale_y(0),
    m_x(nullptr),
    m_y(nullptr),
    m_x1(nullptr),
    m_x2(nullptr),
    m_y1(nullptr),
    m_y2(nullptr),
    m_gen_xy_map(false),
    m_bg_value_same(false)
{
}

ImageTransformer::ImageTransformer(ImageTransformer &&rhs) noexcept :
    m_src_img(rhs.m_src_img),
    m_dst_img(rhs.m_dst_img),
    m_scale_x(rhs.m_scale_x),
    m_scale_y(rhs.m_scale_y),
    m_inv_scale_x(rhs.m_inv_scale_x),
    m_inv_scale_y(rhs.m_inv_scale_y),
    m_M(rhs.m_M),
    m_crop_area(std::move(rhs.m_crop_area)),
    m_border(std::move(rhs.m_border)),
    m_pix_cvt_param(std::move(rhs.m_pix_cvt_param)),
    m_x(rhs.m_x),
    m_y(rhs.m_y),
    m_x1(rhs.m_x1),
    m_x2(rhs.m_x2),
    m_y1(rhs.m_y1),
    m_y2(rhs.m_y2),
    m_gen_xy_map(rhs.m_gen_xy_map),
    m_ori_bg_value(std::move(rhs.m_ori_bg_value)),
    m_bg_value(std::move(rhs.m_bg_value)),
    m_bg_value_same(rhs.m_bg_value_same)
{
    rhs.m_src_img = {};
    rhs.m_dst_img = {};
    rhs.m_scale_x = 0;
    rhs.m_scale_y = 0;
    rhs.m_inv_scale_x = 0;
    rhs.m_inv_scale_y = 0;
    rhs.m_M = {};
    rhs.m_x = nullptr;
    rhs.m_y = nullptr;
    rhs.m_x1 = nullptr;
    rhs.m_x2 = nullptr;
    rhs.m_y1 = nullptr;
    rhs.m_y2 = nullptr;
    rhs.m_gen_xy_map = false;
    rhs.m_bg_value_same = false;
}

ImageTransformer &ImageTransformer::operator=(ImageTransformer &&rhs) noexcept
{
    if (this != &rhs) {
        heap_caps_free(m_x);
        heap_caps_free(m_y);
        heap_caps_free(m_x1);
        heap_caps_free(m_x2);
        heap_caps_free(m_y1);
        heap_caps_free(m_y2);
        m_src_img = rhs.m_src_img;
        m_dst_img = rhs.m_dst_img;
        m_scale_x = rhs.m_scale_x;
        m_scale_y = rhs.m_scale_y;
        m_inv_scale_x = rhs.m_inv_scale_x;
        m_inv_scale_y = rhs.m_inv_scale_y;
        m_M = rhs.m_M;
        m_crop_area = std::move(rhs.m_crop_area);
        m_border = std::move(rhs.m_border);
        m_pix_cvt_param = std::move(rhs.m_pix_cvt_param);
        m_x = rhs.m_x;
        m_y = rhs.m_y;
        m_x1 = rhs.m_x1;
        m_x2 = rhs.m_x2;
        m_y1 = rhs.m_y1;
        m_y2 = rhs.m_y2;
        m_gen_xy_map = rhs.m_gen_xy_map;
        m_ori_bg_value = std::move(rhs.m_ori_bg_value);
        m_bg_value = std::move(rhs.m_bg_value);
        m_bg_value_same = rhs.m_bg_value_same;
        rhs.m_src_img = {};
        rhs.m_dst_img = {};
        rhs.m_scale_x = 0;
        rhs.m_scale_y = 0;
        rhs.m_inv_scale_x = 0;
        rhs.m_inv_scale_y = 0;
        rhs.m_M = {};
        rhs.m_x = nullptr;
        rhs.m_y = nullptr;
        rhs.m_x1 = nullptr;
        rhs.m_x2 = nullptr;
        rhs.m_y1 = nullptr;
        rhs.m_y2 = nullptr;
        rhs.m_gen_xy_map = false;
        rhs.m_bg_value_same = false;
    }
    return *this;
}

ImageTransformer::~ImageTransformer()
{
    heap_caps_free(m_x);
    heap_caps_free(m_y);
    heap_caps_free(m_x1);
    heap_caps_free(m_x2);
    heap_caps_free(m_y1);
    heap_caps_free(m_y2);
}

ImageTransformer &ImageTransformer::set_norm_quant_param(const std::array<float, 3> &mean,
                                                         const std::array<float, 3> &std,
                                                         int exp,
                                                         int quant_bits)
{
    if (quant_bits == 8) {
        m_pix_cvt_param.emplace<NormQuant<int8_t, 3>>(mean, std, exp);
    } else if (quant_bits == 16) {
        m_pix_cvt_param.emplace<NormQuant<int16_t, 3>>(mean, std, exp);
    } else {
        ESP_LOGE(TAG, "Quant_bits can only be 8/16.");
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_norm_quant_param(float mean, float std, int exp, int quant_bits)
{
    if (quant_bits == 8) {
        m_pix_cvt_param.emplace<NormQuant<int8_t, 1>>(mean, std, exp);
    } else if (quant_bits == 16) {
        m_pix_cvt_param.emplace<NormQuant<int16_t, 1>>(mean, std, exp);
    } else {
        ESP_LOGE(TAG, "Quant_bits can only be 8/16.");
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_hsv_thr(const std::array<uint8_t, 3> &hsv_min,
                                                const std::array<uint8_t, 3> &hsv_max)
{
    if (is_valid_hsv_thr(hsv_min, hsv_max)) {
        if (hsv_min[0] > hsv_max[0]) {
            m_pix_cvt_param.emplace<HSV2HSVMask<true>>(hsv_min, hsv_max);
        } else {
            m_pix_cvt_param.emplace<HSV2HSVMask<false>>(hsv_min, hsv_max);
        }
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
        m_gen_xy_map = true;
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_dst_img_border(const std::vector<int> &border)
{
    assert((border.size() == 4 && std::all_of(border.begin(), border.end(), [](const auto &v) { return v >= 0; })) ||
           border.empty());
    if (m_border != border) {
        m_border = border;
        m_gen_xy_map = true;
    }
    return *this;
}

ImageTransformer &ImageTransformer::set_bg_value(const std::array<uint8_t, 3> &bg_value)
{
    m_ori_bg_value = bg_value;
    return *this;
}

ImageTransformer &ImageTransformer::set_bg_value(uint8_t bg_value)
{
    m_ori_bg_value = bg_value;
    return *this;
}

ImageTransformer &ImageTransformer::set_src_img(const img_t &src_img)
{
    if ((src_img.width != m_src_img.width) || (src_img.height != m_src_img.height) ||
        src_img.col_step() != m_src_img.col_step()) {
        m_gen_xy_map = true;
    }
    m_src_img = src_img;
    return *this;
}

ImageTransformer &ImageTransformer::set_dst_img(const img_t &dst_img)
{
    if ((dst_img.width != m_dst_img.width) || (dst_img.height != m_dst_img.height) ||
        dst_img.col_step() != m_dst_img.col_step()) {
        m_gen_xy_map = true;
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
            m_gen_xy_map = true;
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
    return inv ? m_inv_scale_x : m_scale_x;
}

float ImageTransformer::get_scale_y(bool inv)
{
    return inv ? m_inv_scale_y : m_scale_y;
}

pix_cvt_param_t ImageTransformer::get_pix_cvt_param()
{
    return m_pix_cvt_param;
}

ImageTransformer &ImageTransformer::reset()
{
    heap_caps_free(m_x);
    heap_caps_free(m_y);
    heap_caps_free(m_x1);
    heap_caps_free(m_x2);
    heap_caps_free(m_y1);
    heap_caps_free(m_y2);
    m_src_img = {};
    m_dst_img = {};
    m_scale_x = 0;
    m_scale_y = 0;
    m_inv_scale_x = 0;
    m_inv_scale_y = 0;
    m_M = {};
    m_crop_area = {};
    m_border = {};
    m_pix_cvt_param = {};
    m_x = nullptr;
    m_y = nullptr;
    m_x1 = nullptr;
    m_x2 = nullptr;
    m_y1 = nullptr;
    m_y2 = nullptr;
    m_gen_xy_map = false;
    m_ori_bg_value = {};
    m_bg_value = {};
    m_bg_value_same = false;
    return *this;
}

template <bool SIMD>
esp_err_t ImageTransformer::transform()
{
    // check src img & dst img.
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

    // check crop_area & border
    if (!m_crop_area.empty() && (m_crop_area[2] > m_src_img.width || m_crop_area[3] > m_src_img.height)) {
        ESP_LOGE(TAG, "Invalid crop area.");
        return ESP_FAIL;
    }
    if (!m_border.empty() &&
        (m_border[0] + m_border[1] > m_dst_img.height - 1 || m_border[2] + m_border[3] > m_dst_img.width - 1)) {
        ESP_LOGE(TAG, "Invalid border.");
        return ESP_FAIL;
    }

    // check pix_cvt_param
    bool invalid_param = std::visit(
        [this](const auto &param) -> bool {
            using T = std::decay_t<decltype(param)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                switch (m_dst_img.pix_type) {
                case DL_IMAGE_PIX_TYPE_GRAY_QINT8:
                case DL_IMAGE_PIX_TYPE_GRAY_QINT16:
                case DL_IMAGE_PIX_TYPE_RGB888_QINT8:
                case DL_IMAGE_PIX_TYPE_BGR888_QINT8:
                case DL_IMAGE_PIX_TYPE_RGB888_QINT16:
                case DL_IMAGE_PIX_TYPE_BGR888_QINT16:
                    ESP_LOGE(TAG,
                             "Dst img is quant type, but norm_quant_param is not set. Call set_norm_quant_param().");
                    return true;
                case DL_IMAGE_PIX_TYPE_HSV_MASK:
                    ESP_LOGE(TAG, "Dst img is hsv mask, but hsv_thr is not set. Call set_hsv_thr().");
                    return true;
                default:
                    return false;
                }
            } else if constexpr (std::is_same_v<T, NormQuant<int8_t, 1>>) {
                if (m_dst_img.pix_type != DL_IMAGE_PIX_TYPE_GRAY_QINT8) {
                    ESP_LOGE(TAG,
                             "1 channel qint8 norm_quant_param is set, but dst img pix_type is %s.",
                             pix_type2str(m_dst_img.pix_type).c_str());
                    return true;
                } else {
                    return false;
                }
            } else if constexpr (std::is_same_v<T, NormQuant<int8_t, 3>>) {
                if (m_dst_img.pix_type != DL_IMAGE_PIX_TYPE_RGB888_QINT8 &&
                    m_dst_img.pix_type != DL_IMAGE_PIX_TYPE_BGR888_QINT8) {
                    ESP_LOGE(TAG,
                             "3 channel qint8 norm_quant_param is set, but dst img pix_type is %s.",
                             pix_type2str(m_dst_img.pix_type).c_str());
                    return true;
                } else {
                    return false;
                }
            } else if constexpr (std::is_same_v<T, NormQuant<int16_t, 1>>) {
                if (m_dst_img.pix_type != DL_IMAGE_PIX_TYPE_GRAY_QINT16) {
                    ESP_LOGE(TAG,
                             "1 channel qint16 norm_quant_param is set, but dst img pix_type is %s.",
                             pix_type2str(m_dst_img.pix_type).c_str());
                    return true;
                } else {
                    return false;
                }
            } else if constexpr (std::is_same_v<T, NormQuant<int16_t, 3>>) {
                if (m_dst_img.pix_type != DL_IMAGE_PIX_TYPE_RGB888_QINT16 &&
                    m_dst_img.pix_type != DL_IMAGE_PIX_TYPE_BGR888_QINT16) {
                    ESP_LOGE(TAG,
                             "3 channel qint16 norm_quant_param is set, but dst img pix_type is %s.",
                             pix_type2str(m_dst_img.pix_type).c_str());
                    return true;
                } else {
                    return false;
                }
            } else if constexpr (std::is_same_v<T, HSV2HSVMask<false>> || std::is_same_v<T, HSV2HSVMask<true>>) {
                if (m_dst_img.pix_type != DL_IMAGE_PIX_TYPE_HSV_MASK) {
                    ESP_LOGE(TAG, "hsv_thr is set, but dst img type mismatch.");
                    return true;
                } else {
                    return false;
                }
            }
        },
        m_pix_cvt_param);

    if (invalid_param) {
        return ESP_FAIL;
    }

    // check & calculate background value
    if (!m_border.empty() || m_M.array) {
        uint8_t *p_ori_bg_value = nullptr;
        pix_type_t src_pix_type;
        switch (m_src_img.channel()) {
        case 1: {
            if (std::holds_alternative<std::monostate>(m_ori_bg_value)) {
                m_ori_bg_value.emplace<uint8_t>();
            }
            auto *ori_bg_value = std::get_if<uint8_t>(&m_ori_bg_value);
            if (!ori_bg_value) {
                ESP_LOGE(TAG, "Background color and src image type mismatch.");
                return ESP_FAIL;
            }
            p_ori_bg_value = ori_bg_value;
            src_pix_type = DL_IMAGE_PIX_TYPE_GRAY;
            break;
        }
        case 3: {
            if (std::holds_alternative<std::monostate>(m_ori_bg_value)) {
                m_ori_bg_value.emplace<std::array<uint8_t, 3>>();
            }
            auto *ori_bg_value = std::get_if<std::array<uint8_t, 3>>(&m_ori_bg_value);
            if (!ori_bg_value) {
                ESP_LOGE(TAG, "Background color and src image type mismatch.");
                return ESP_FAIL;
            }
            p_ori_bg_value = ori_bg_value->data();
            src_pix_type = DL_IMAGE_PIX_TYPE_RGB888;
            break;
        }
        default:
            ESP_LOGE(TAG, "Invalid channel num.");
            return ESP_FAIL;
        }
        m_bg_value.resize(m_dst_img.col_step());
        cvt_pix(p_ori_bg_value, m_bg_value.data(), src_pix_type, m_dst_img.pix_type, m_pix_cvt_param);
        m_bg_value_same =
            std::all_of(m_bg_value.begin() + 1, m_bg_value.end(), [this](const auto &v) { return v == m_bg_value[0]; });
    }

    TransformNNFunctor<SIMD> fn{this};
    return pixel_cvt_dispatch(fn, m_src_img.pix_type, m_dst_img.pix_type, m_pix_cvt_param);
}

#if CONFIG_IDF_TARGET_ESP32P4
template esp_err_t ImageTransformer::transform<true>();
#endif
template esp_err_t ImageTransformer::transform<false>();

void ImageTransformer::gen_xy_map()
{
    int dst_width = m_border.empty() ? m_dst_img.width : (m_dst_img.width - m_border[2] - m_border[3]);
    int dst_height = m_border.empty() ? m_dst_img.height : (m_dst_img.height - m_border[0] - m_border[1]);

    int col_step = m_src_img.col_step();
    int row_step = m_src_img.row_step();

    heap_caps_free(m_x);
    heap_caps_free(m_y);
    heap_caps_free(m_x1);
    heap_caps_free(m_x2);
    heap_caps_free(m_y1);
    heap_caps_free(m_y2);
    m_x = nullptr;
    m_y = nullptr;
    m_x1 = nullptr;
    m_x2 = nullptr;
    m_y1 = nullptr;
    m_y2 = nullptr;

    if (!m_M.array) {
        int src_width = m_crop_area.empty() ? m_src_img.width : (m_crop_area[2] - m_crop_area[0]);
        int src_height = m_crop_area.empty() ? m_src_img.height : (m_crop_area[3] - m_crop_area[1]);
        if (src_width == dst_width && src_height == dst_height) {
            m_scale_x = m_scale_y = m_inv_scale_x = m_inv_scale_y = 1;
            return;
        }
        m_x = (int *)heap_caps_malloc(dst_width * sizeof(int), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD);
        m_y = (int *)heap_caps_malloc(dst_height * sizeof(int), MALLOC_CAP_DEFAULT | MALLOC_CAP_SIMD);
        m_inv_scale_x = static_cast<float>(src_width) / dst_width;
        m_inv_scale_y = static_cast<float>(src_height) / dst_height;
        m_scale_x = 1.f / m_inv_scale_x;
        m_scale_y = 1.f / m_inv_scale_y;
        if (m_crop_area.empty()) {
            for (int i = 0; i < dst_width; i++) {
                m_x[i] = std::min(static_cast<int>(i * m_inv_scale_x), src_width - 1) * col_step;
            }
            for (int i = 0; i < dst_height; i++) {
                m_y[i] = std::min(static_cast<int>(i * m_inv_scale_y), src_height - 1) * row_step;
            }
        } else {
            for (int i = 0; i < dst_width; i++) {
                m_x[i] = std::min(static_cast<int>(i * m_inv_scale_x) + m_crop_area[0], m_crop_area[2] - 1) * col_step;
            }
            for (int i = 0; i < dst_height; i++) {
                m_y[i] = std::min(static_cast<int>(i * m_inv_scale_y) + m_crop_area[1], m_crop_area[3] - 1) * row_step;
            }
        }
    } else {
        m_x1 = (int *)heap_caps_malloc(dst_width * sizeof(int), MALLOC_CAP_DEFAULT);
        m_x2 = (int *)heap_caps_malloc(dst_width * sizeof(int), MALLOC_CAP_DEFAULT);
        m_y1 = (int *)heap_caps_malloc(dst_height * sizeof(int), MALLOC_CAP_DEFAULT);
        m_y2 = (int *)heap_caps_malloc(dst_height * sizeof(int), MALLOC_CAP_DEFAULT);
        float **M = m_M.array;
        float M0 = M[0][0], M1 = M[0][1], M2 = M[0][2], M3 = M[1][0], M4 = M[1][1], M5 = M[1][2];
        constexpr int scale = 1 << warp_affine_shift;
        for (int i = 0; i < dst_width; i++) {
            m_x1[i] = static_cast<int>(__builtin_lrintf(M0 * i * scale));
            m_y1[i] = static_cast<int>(__builtin_lrintf(M3 * i * scale));
        }
        for (int i = 0; i < dst_height; i++) {
            m_x2[i] = static_cast<int>(__builtin_lrintf((M1 * i + M2) * scale));
            m_y2[i] = static_cast<int>(__builtin_lrintf((M4 * i + M5) * scale));
        }
    }
}
} // namespace image
} // namespace dl
