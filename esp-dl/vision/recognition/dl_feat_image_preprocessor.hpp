#pragma once
#include "dl_image_preprocessor.hpp"
#include "dl_math_matrix.hpp"

namespace dl {
namespace image {
class FeatImagePreprocessor {
public:
    FeatImagePreprocessor(Model *model,
                          const std::array<float, 3> &mean,
                          const std::array<float, 3> &std,
                          bool rgb_swap = false,
                          const std::string &input_name = "") :
        m_image_preprocessor(new dl::image::ImagePreprocessor(model, mean, std, rgb_swap, input_name)) {};

    FeatImagePreprocessor(Model *model, float mean, float std, const std::string &input_name = "") :
        m_image_preprocessor(new dl::image::ImagePreprocessor(model, mean, std, input_name)) {};

    ~FeatImagePreprocessor();

    void preprocess(const dl::image::img_t &img, const std::vector<int> &landmarks);

private:
    static std::vector<float> s_std_ldks_112;
    dl::image::ImagePreprocessor *m_image_preprocessor;
};
} // namespace image
} // namespace dl
