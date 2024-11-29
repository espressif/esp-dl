#pragma once
#include "dl_math.hpp"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include <map>

namespace dl {
namespace recognition {
class RecognitionPostprocessor {
private:
    TensorBase *m_model_output;
    TensorBase *m_feat;
    void l2_norm();

public:
    RecognitionPostprocessor(Model *model, const std::string &output_name = "");
    TensorBase *postprocess();
    ~RecognitionPostprocessor()
    {
        if (m_feat) {
            delete m_feat;
            m_feat = nullptr;
        }
    }
};
} // namespace recognition
} // namespace dl
