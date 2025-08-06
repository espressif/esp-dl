#include "dl_image_preprocessor.hpp"

namespace dl {
namespace image {

ImagePreprocessor::ImagePreprocessor(Model *model,
                                     const std::vector<float> &mean,
                                     const std::vector<float> &std,
                                     uint32_t caps,
                                     const std::string &input_name)
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

float ImagePreprocessor::get_resize_scale_x(bool inv)
{
    return m_image_transformer.get_scale_x(inv);
}

float ImagePreprocessor::get_resize_scale_y(bool inv)
{
    return m_image_transformer.get_scale_y(inv);
}

int ImagePreprocessor::get_top_left_x()
{
    return m_image_transformer.get_src_img_crop_area()[0];
}

int ImagePreprocessor::get_top_left_y()
{
    return m_image_transformer.get_src_img_crop_area()[1];
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
    ESP_ERROR_CHECK(m_image_transformer.set_src_img(img).set_src_img_crop_area(crop_area).transform());
}

void ImagePreprocessor::preprocess(const img_t &img, const dl::math::Matrix<float> &M, bool inv)
{
    ESP_ERROR_CHECK(m_image_transformer.set_src_img(img).set_warp_affine_matrix(M, inv).transform());
}
} // namespace image
} // namespace dl
