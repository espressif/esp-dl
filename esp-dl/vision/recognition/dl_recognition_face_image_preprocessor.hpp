#pragma once
#include "dl_image_preprocessor.hpp"
#include "dl_math_matrix.hpp"

namespace dl {
namespace recognition {
template <typename feature_t>
class FaceImagePreprocessor {
public:
    FaceImagePreprocessor(TensorBase *model_input,
                          const std::vector<float> &mean,
                          const std::vector<float> &std,
                          bool byte_rgb = false,
                          bool byte_swap = false,
                          bool use_ppa = false) :
        image_preprocessor(
            new dl::image::ImagePreprocessor<feature_t>(model_input, mean, std, byte_rgb, byte_swap, use_ppa)),
        byte_swap(byte_swap) {};

    ~FaceImagePreprocessor();

    template <typename T>
    void preprocess(T *input_element, const std::vector<int> &input_shape, const std::vector<int> &landmarks);

private:
    static std::vector<float> std_ldks_112;
    dl::image::ImagePreprocessor<feature_t> *image_preprocessor;
    bool byte_swap;
};
} // namespace recognition
} // namespace dl
