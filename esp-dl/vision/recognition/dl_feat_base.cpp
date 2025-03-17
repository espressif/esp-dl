#include "dl_feat_base.hpp"

namespace dl {
namespace feat {

FeatImpl::~FeatImpl()
{
    delete m_model;
    delete m_image_preprocessor;
    delete m_postprocessor;
}

TensorBase *FeatImpl::run(const dl::image::img_t &img, const std::vector<int> &landmarks)
{
    DL_LOG_INFER_LATENCY_INIT();
    DL_LOG_INFER_LATENCY_START();
    m_image_preprocessor->preprocess(img, landmarks);
    DL_LOG_INFER_LATENCY_END_PRINT("feat", "pre");

    DL_LOG_INFER_LATENCY_START();
    m_model->run();
    DL_LOG_INFER_LATENCY_END_PRINT("feat", "model");

    DL_LOG_INFER_LATENCY_START();
    dl::TensorBase *feat = m_postprocessor->postprocess();
    DL_LOG_INFER_LATENCY_END_PRINT("feat", "post");

    return feat;
}

} // namespace feat
} // namespace dl
