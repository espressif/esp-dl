#pragma once

#include "dl_image_process.hpp"
#include "dl_model_base.hpp"

namespace dl {
namespace image {
class ImagePreprocessor {
public:
    ImagePreprocessor(Model *model,
                      const std::vector<float> &mean,
                      const std::vector<float> &std,
                      uint32_t caps = 0,
                      const std::string &input_name = "");
    ~ImagePreprocessor();
    void enable_letterbox(const std::vector<uint8_t> &bg_value = {0, 0, 0});
    void enable_letterbox(uint8_t bg_value = 0);
    float get_resize_scale_x(bool inv = false);
    float get_resize_scale_y(bool inv = false);
    int get_crop_area_top_left_x();
    int get_crop_area_top_left_y();
    int get_border_top();
    int get_border_left();
    TensorBase *get_model_input();
    void preprocess(const img_t &img, const std::vector<int> &crop_area = {});
    void preprocess(const img_t &img, const dl::math::Matrix<float> &M, bool inv = false);

private:
    template <typename T>
    void create_norm_lut(const std::vector<float> &mean, const std::vector<float> &std);

    ImageTransformer m_image_transformer;
    TensorBase *m_model_input;
    void *m_norm_lut;

    // for letter box
    bool m_letter_box;
    std::vector<uint8_t> m_rgb888_bg_value;
    std::vector<uint8_t> m_rgb565_bg_value;
    std::vector<uint8_t> m_gray_bg_value;
};

} // namespace image
} // namespace dl
