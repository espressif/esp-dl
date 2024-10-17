#include "dl_recognition_postprocessor.hpp"

namespace dl {
namespace recognition {

template <typename feature_t>
TensorBase *RecognitionPostprocessor<feature_t>::postprocess()
{
    this->feat->assign(this->model_output);
    this->l2_norm();
    return this->feat;
}

template <typename feature_t>
void RecognitionPostprocessor<feature_t>::l2_norm()
{
    float norm = 0;
    float *ptr = (float *)this->feat->get_element_ptr();
    for (int i = 0; i < this->feat->get_size(); i++) {
        norm += (ptr[i] * ptr[i]);
    }
    norm = dl::math::sqrt_newton(norm);
    for (int i = 0; i < this->feat->get_size(); i++) {
        ptr[i] /= norm;
    }
}

template class RecognitionPostprocessor<int8_t>;
template class RecognitionPostprocessor<int16_t>;
} // namespace recognition
} // namespace dl
