#include "dl_image_preprocessor.hpp"

namespace dl {
namespace image {

ImagePreprocessor::ImagePreprocessor(Model *model,
                                     const std::array<float, 3> &mean,
                                     const std::array<float, 3> &std,
                                     bool rgb_swap,
                                     const std::string &input_name) :
    m_letter_box(false)
{
    m_model_input = model->get_input(input_name);
    assert(m_model_input->dtype == DATA_TYPE_INT8 || m_model_input->dtype == DATA_TYPE_INT16);
    assert(m_model_input->shape[3] == 3);
    int quant_bits = (m_model_input->dtype == DATA_TYPE_INT8) ? 8 : 16;
    pix_type_t pix_type;
    if (!rgb_swap) {
        if (m_model_input->dtype == DATA_TYPE_INT8) {
            pix_type = DL_IMAGE_PIX_TYPE_RGB888_QINT8;
        } else {
            pix_type = DL_IMAGE_PIX_TYPE_RGB888_QINT16;
        }
    } else {
        if (m_model_input->dtype == DATA_TYPE_INT8) {
            pix_type = DL_IMAGE_PIX_TYPE_BGR888_QINT8;
        } else {
            pix_type = DL_IMAGE_PIX_TYPE_BGR888_QINT16;
        }
    }
    img_t dst = {.data = m_model_input->data,
                 .width = (uint16_t)m_model_input->shape[2],
                 .height = (uint16_t)m_model_input->shape[1],
                 .pix_type = pix_type};
    m_image_transformer.set_dst_img(dst).set_norm_quant_param(mean, std, m_model_input->exponent, quant_bits);
}

ImagePreprocessor::ImagePreprocessor(Model *model, float mean, float std, const std::string &input_name) :
    m_letter_box(false)
{
    m_model_input = model->get_input(input_name);
    assert(m_model_input->dtype == DATA_TYPE_INT8 || m_model_input->dtype == DATA_TYPE_INT16);
    assert(m_model_input->shape[3] == 1);
    int quant_bits = (m_model_input->dtype == DATA_TYPE_INT8) ? 8 : 16;
    pix_type_t pix_type =
        (m_model_input->dtype == DATA_TYPE_INT8) ? DL_IMAGE_PIX_TYPE_GRAY_QINT8 : DL_IMAGE_PIX_TYPE_GRAY_QINT16;
    img_t dst = {.data = m_model_input->data,
                 .width = (uint16_t)m_model_input->shape[2],
                 .height = (uint16_t)m_model_input->shape[1],
                 .pix_type = pix_type};
    m_image_transformer.set_dst_img(dst).set_norm_quant_param(mean, std, m_model_input->exponent, quant_bits);
}

void ImagePreprocessor::enable_letterbox(const std::array<uint8_t, 3> &bg_value)
{
    m_image_transformer.set_bg_value(bg_value);
    m_letter_box = true;
}

void ImagePreprocessor::enable_letterbox(uint8_t bg_value)
{
    m_image_transformer.set_bg_value(bg_value);
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

void ImagePreprocessor::preprocess(const img_t &img, const std::vector<int> &crop_area)
{
    if (m_letter_box) {
        auto &last_src_img = m_image_transformer.get_src_img();
        if (img.height != last_src_img.height || img.width != last_src_img.width ||
            crop_area != m_image_transformer.get_src_img_crop_area()) {
            auto &dst_img = m_image_transformer.get_dst_img();
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
    }
    ESP_ERROR_CHECK(m_image_transformer.set_src_img(img).set_src_img_crop_area(crop_area).transform());
}

void ImagePreprocessor::preprocess(const img_t &img, const dl::math::Matrix<float> &M, bool inv)
{
    ESP_ERROR_CHECK(m_image_transformer.set_src_img(img).set_warp_affine_matrix(M, inv).transform());
}
} // namespace image
} // namespace dl
