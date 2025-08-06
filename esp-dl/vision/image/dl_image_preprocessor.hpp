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
    float get_resize_scale_x(bool inv = false);
    float get_resize_scale_y(bool inv = false);
    int get_top_left_x();
    int get_top_left_y();
    void preprocess(const img_t &img, const std::vector<int> &crop_area = {});
    void preprocess(const img_t &img, const dl::math::Matrix<float> &M, bool inv = false);

    TensorBase *m_model_input;

protected:
    ImageTransformer m_image_transformer;

private:
    template <typename T>
    void create_norm_lut(const std::vector<float> &mean, const std::vector<float> &std);

    void *m_norm_lut;
};

} // namespace image
} // namespace dl
