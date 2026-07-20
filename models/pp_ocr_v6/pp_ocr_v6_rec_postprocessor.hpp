#pragma once
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include <string>

namespace pp_ocr_v6 {
class RecCTCPostprocessor {
public:
    explicit RecCTCPostprocessor(dl::Model *model);

    // Optional score: mean softmax prob of emitted chars.
    std::string postprocess(float *score = nullptr);

private:
    dl::TensorBase *m_model_output;
};

} // namespace pp_ocr_v6
