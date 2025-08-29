#include "dl_image_preprocessor.hpp"
#include "dl_image_bmp.hpp"
#include "dl_image_color.hpp"

namespace dl {
namespace image {

ImagePreprocessor::ImagePreprocessor(Model *model,
                                     const std::vector<float> &mean,
                                     const std::vector<float> &std,
                                     uint32_t caps,
                                     const std::string &input_name) :
    m_letter_box(false)
{
    m_model_input = model->get_input(input_name);
    assert(m_model_input->dtype == DATA_TYPE_INT8 || m_model_input->dtype == DATA_TYPE_INT16);
    assert(m_model_input->shape[3] == mean.size() && mean.size() == std.size());
    if (m_model_input->dtype == DATA_TYPE_INT8) {
        create_norm_lut<int8_t>(mean, std);
    } else {
        create_norm_lut<int16_t>(mean, std);
    }
    img_t dst = {.data = m_model_input->data,
                 .width = (uint16_t)m_model_input->shape[2],
                 .height = (uint16_t)m_model_input->shape[1],
                 .pix_type = (m_model_input->dtype == DATA_TYPE_INT8) ? DL_IMAGE_PIX_TYPE_RGB888_QINT8
                                                                      : DL_IMAGE_PIX_TYPE_RGB888_QINT16};
    m_image_transformer.set_dst_img(dst).set_caps(caps).set_norm_quant_lut(m_norm_lut);
}

ImagePreprocessor::~ImagePreprocessor()
{
    heap_caps_free(m_norm_lut);
}

void ImagePreprocessor::enable_letterbox(const std::vector<uint8_t> &bg_value)
{
    assert(bg_value.size() == 3);
    m_rgb888_bg_value = bg_value;
    m_rgb565_bg_value.resize(2);
#if CONFIG_IDF_TARGET_ESP32P4
    uint32_t caps = 0;
#else
    uint32_t caps = dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN;
#endif
    cvt_pix(
        m_rgb888_bg_value.data(), m_rgb565_bg_value.data(), DL_IMAGE_PIX_TYPE_RGB888, DL_IMAGE_PIX_TYPE_RGB565, caps);
    m_letter_box = true;
}

void ImagePreprocessor::enable_letterbox(uint8_t bg_value)
{
    m_gray_bg_value = {bg_value};
    m_letter_box = true;
}

float ImagePreprocessor::get_resize_scale_x(bool inv)
{
    return m_image_transformer.get_scale_x(inv);
}

float ImagePreprocessor::get_resize_scale_y(bool inv)
{
    return m_image_transformer.get_scale_y(inv);
}

int ImagePreprocessor::get_crop_area_top_left_x()
{
    auto crop_area = m_image_transformer.get_src_img_crop_area();
    return crop_area.empty() ? 0 : crop_area[0];
}

int ImagePreprocessor::get_crop_area_top_left_y()
{
    auto crop_area = m_image_transformer.get_src_img_crop_area();
    return crop_area.empty() ? 0 : crop_area[1];
}

int ImagePreprocessor::get_border_top()
{
    auto border = m_image_transformer.get_dst_img_border();
    return border.empty() ? 0 : border[0];
}

int ImagePreprocessor::get_border_left()
{
    auto border = m_image_transformer.get_dst_img_border();
    return border.empty() ? 0 : border[2];
}

TensorBase *ImagePreprocessor::get_model_input()
{
    return m_model_input;
}

template <typename T>
void ImagePreprocessor::create_norm_lut(const std::vector<float> &mean, const std::vector<float> &std)
{
    m_norm_lut = heap_caps_malloc(mean.size() * 256 * sizeof(T), MALLOC_CAP_DEFAULT);
    float inv_scale = 1.f / DL_SCALE(m_model_input->exponent);
    std::vector<float> inv_std(std.size());
    T *norm_lut_ptr = (T *)m_norm_lut;
    for (int i = 0; i < mean.size(); i++) {
        inv_std[i] = 1.f / std[i];
        for (int j = 0; j < 256; j++) {
            norm_lut_ptr[i * 256 + j] = quantize<T>(((float)j - mean[i]) * inv_std[i], inv_scale);
        }
    }
}

template void ImagePreprocessor::create_norm_lut<int8_t>(const std::vector<float> &mean, const std::vector<float> &std);
template void ImagePreprocessor::create_norm_lut<int16_t>(const std::vector<float> &mean,
                                                          const std::vector<float> &std);

void ImagePreprocessor::preprocess(const img_t &img, const std::vector<int> &crop_area)
{
    if (m_letter_box) {
        auto &last_src_img = m_image_transformer.get_src_img();
        auto &dst_img = m_image_transformer.get_dst_img();
        if (img.height != last_src_img.height || img.width != last_src_img.width ||
            crop_area != m_image_transformer.get_src_img_crop_area()) {
            int src_width = crop_area.empty() ? img.width : (crop_area[2] - crop_area[0]);
            int src_height = crop_area.empty() ? img.height : (crop_area[3] - crop_area[1]);
            float scale_x = (float)dst_img.width / (float)src_width;
            float scale_y = (float)dst_img.height / (float)src_height;
            float scale = std::min(scale_x, scale_y);
            int border_top = 0, border_bottom = 0, border_left = 0, border_right = 0;
            if (scale_x < scale_y) {
                int pad_h = dst_img.height - (int)(scale * src_height);
                border_top = pad_h / 2;
                border_bottom = pad_h - border_top;
            } else {
                int pad_w = dst_img.width - (int)(scale * src_width);
                border_left = pad_w / 2;
                border_right = pad_w - border_left;
            }
            m_image_transformer.set_dst_img_border({border_top, border_bottom, border_left, border_right});
        }
        if (!last_src_img.data || img.pix_type != last_src_img.pix_type) {
            switch (img.pix_type) {
            case DL_IMAGE_PIX_TYPE_RGB888:
                m_image_transformer.set_bg_value(m_rgb888_bg_value);
                break;
            case DL_IMAGE_PIX_TYPE_RGB565:
                m_image_transformer.set_bg_value(m_rgb565_bg_value);
                break;
            case DL_IMAGE_PIX_TYPE_GRAY:
                m_image_transformer.set_bg_value(m_gray_bg_value);
                break;
            default:
                ESP_LOGE("ImagePreprocessor", "Unsupported input image pixel type.");
            }
        }
    }
    ESP_ERROR_CHECK(m_image_transformer.set_src_img(img).set_src_img_crop_area(crop_area).transform());
}

void ImagePreprocessor::preprocess(const img_t &img, const dl::math::Matrix<float> &M, bool inv)
{
    ESP_ERROR_CHECK(m_image_transformer.set_src_img(img).set_warp_affine_matrix(M, inv).transform());
}
} // namespace image
} // namespace dl
