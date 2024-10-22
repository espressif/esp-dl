#include "dl_recognition_face_image_preprocessor.hpp"

namespace dl {
namespace recognition {
template <typename feature_t>
std::vector<float> FaceImagePreprocessor<feature_t>::std_ldks_112 = {
    38.2946, 51.6963, 41.5493, 92.3655, 56.0252, 71.7366, 73.5318, 51.5014, 70.7299, 92.2041};

template <typename feature_t>
FaceImagePreprocessor<feature_t>::~FaceImagePreprocessor()
{
    if (this->image_preprocessor) {
        delete this->image_preprocessor;
        this->image_preprocessor = nullptr;
    }
}

template <typename feature_t>
template <typename T>
void FaceImagePreprocessor<feature_t>::preprocess(T *input_element,
                                                  const std::vector<int> &input_shape,
                                                  const std::vector<int> &landmarks)
{
    assert(landmarks.size() == 10);
    // align face
    float h_scale = (float)this->image_preprocessor->model_input->shape[1] / 112.0;
    float w_scale = (float)this->image_preprocessor->model_input->shape[2] / 112.0;
    dl::math::Matrix<float> source_coord(5, 2);
    dl::math::Matrix<float> dest_coord(5, 2);
    dest_coord.set_value(landmarks);
    for (int i = 0; i < source_coord.h; i++) {
        source_coord.array[i][0] = w_scale * std_ldks_112[2 * i];
        source_coord.array[i][1] = h_scale * std_ldks_112[2 * i + 1];
    }
    dl::math::Matrix<float> M_inv = dl::math::get_similarity_transform(source_coord, dest_coord);
    std::vector<int> model_input_shape = {this->image_preprocessor->model_input->shape[1],
                                          this->image_preprocessor->model_input->shape[2],
                                          this->image_preprocessor->model_input->shape[3]};
    if (std::is_same<feature_t, int8_t>::value)
        dl::image::warp_affine(input_element,
                               input_shape,
                               (uint8_t *)this->image_preprocessor->model_input->data,
                               model_input_shape,
                               &M_inv,
                               this->byte_swap);
    else
        dl::image::warp_affine(input_element,
                               input_shape,
                               (int16_t *)this->image_preprocessor->model_input->data,
                               model_input_shape,
                               &M_inv,
                               this->byte_swap);
    // normalize & quantize
    this->image_preprocessor->preprocess((uint8_t *)this->image_preprocessor->model_input->data, model_input_shape);
}

template void FaceImagePreprocessor<int8_t>::preprocess(uint8_t *input_element,
                                                        const std::vector<int> &input_shape,
                                                        const std::vector<int> &landmarks);
template void FaceImagePreprocessor<int8_t>::preprocess(uint16_t *input_element,
                                                        const std::vector<int> &input_shape,
                                                        const std::vector<int> &landmarks);
template void FaceImagePreprocessor<int16_t>::preprocess(uint8_t *input_element,
                                                         const std::vector<int> &input_shape,
                                                         const std::vector<int> &landmarks);
template void FaceImagePreprocessor<int16_t>::preprocess(uint16_t *input_element,
                                                         const std::vector<int> &input_shape,
                                                         const std::vector<int> &landmarks);

template class FaceImagePreprocessor<int8_t>;
template class FaceImagePreprocessor<int16_t>;
} // namespace recognition
} // namespace dl
