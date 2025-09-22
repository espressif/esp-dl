#include "dl_feat_base.hpp"

namespace dl {
namespace feat {
FeatWrapper::~FeatWrapper()
{
    delete m_model;
}

TensorBase *FeatWrapper::run(const dl::image::img_t &img, const std::vector<int> &landmarks)
{
    if (!m_model) {
        load_model();
    }
    return m_model->run(img, landmarks);
}

int FeatWrapper::get_feat_len()
{
    if (!m_model) {
        load_model();
    }
    return m_model->get_feat_len();
}

dl::Model *FeatWrapper::get_raw_model()
{
    if (!m_model) {
        load_model();
    }
    return m_model->get_raw_model();
}

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

int FeatImpl::get_feat_len()
{
    return m_model->get_output()->get_size();
    ;
}

dl::Model *FeatImpl::get_raw_model()
{
    return m_model;
}
} // namespace feat
} // namespace dl
