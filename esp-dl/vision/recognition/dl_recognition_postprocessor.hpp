#pragma once
#include "dl_math.hpp"
#include <dl_tensor_base.hpp>
#include <map>

namespace dl {
namespace recognition {
template <typename feature_t>
class RecognitionPostprocessor {
private:
    TensorBase *model_output;
    TensorBase *feat;
    void l2_norm();

public:
    RecognitionPostprocessor(TensorBase *model_output) :
        model_output(model_output),
        feat(new TensorBase(model_output->shape, nullptr, model_output->exponent, DATA_TYPE_FLOAT))
    {
    }
    TensorBase *postprocess();
    ~RecognitionPostprocessor()
    {
        if (this->feat) {
            delete this->feat;
            this->feat = nullptr;
        }
    }
};
} // namespace recognition
} // namespace dl
