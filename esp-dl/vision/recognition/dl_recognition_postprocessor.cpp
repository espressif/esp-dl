#include "dl_recognition_postprocessor.hpp"

namespace dl {
namespace recognition {

RecognitionPostprocessor::RecognitionPostprocessor(Model *model, const std::string &output_name)
{
    if (output_name.empty()) {
        std::map<std::string, dl::TensorBase *> model_outputs_map = model->get_outputs();
        assert(model_outputs_map.size() == 1);
        m_model_output = model_outputs_map.begin()->second;
    } else {
        m_model_output = model->get_intermediate(output_name);
    }
    m_feat = new TensorBase(m_model_output->shape, nullptr, 0, DATA_TYPE_FLOAT);
}

TensorBase *RecognitionPostprocessor::postprocess()
{
    m_feat->assign(m_model_output);
    l2_norm();
    return m_feat;
}

void RecognitionPostprocessor::l2_norm()
{
    float norm = 0;
    float *ptr = (float *)m_feat->data;
    for (int i = 0; i < m_feat->get_size(); i++) {
        norm += (ptr[i] * ptr[i]);
    }
    norm = dl::math::sqrt_newton(norm);
    for (int i = 0; i < m_feat->get_size(); i++) {
        ptr[i] /= norm;
    }
}
} // namespace recognition
} // namespace dl
