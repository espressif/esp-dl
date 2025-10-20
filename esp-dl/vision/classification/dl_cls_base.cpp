#include "dl_cls_base.hpp"

namespace dl {
namespace cls {
ClsWrapper::~ClsWrapper()
{
    delete m_model;
}

std::vector<dl::cls::result_t> &ClsWrapper::run(const dl::image::img_t &img)
{
    if (!m_model) {
        load_model();
    }
    return m_model->run(img);
}

Cls &ClsWrapper::set_topk(int topk)
{
    if (m_model) {
        m_model->set_topk(m_topk);
    } else {
        m_topk = topk;
    }
    return *this;
}

Cls &ClsWrapper::set_score_thr(float score_thr)
{
    if (m_model) {
        m_model->set_score_thr(m_score_thr);
    } else {
        m_score_thr = score_thr;
    }
    return *this;
}

dl::Model *ClsWrapper::get_raw_model()
{
    if (!m_model) {
        load_model();
    }
    return m_model->get_raw_model();
}

ClsImpl::~ClsImpl()
{
    delete m_model;
    delete m_image_preprocessor;
    delete m_postprocessor;
}

std::vector<dl::cls::result_t> &ClsImpl::run(const dl::image::img_t &img)
{
    DL_LOG_INFER_LATENCY_INIT();
    DL_LOG_INFER_LATENCY_START();
    m_image_preprocessor->preprocess(img);
    DL_LOG_INFER_LATENCY_END_PRINT("cls", "pre");

    DL_LOG_INFER_LATENCY_START();
    m_model->run();
    DL_LOG_INFER_LATENCY_END_PRINT("cls", "model");

    DL_LOG_INFER_LATENCY_START();
    std::vector<dl::cls::result_t> &result = m_postprocessor->postprocess();
    DL_LOG_INFER_LATENCY_END_PRINT("cls", "post");

    return result;
}

Cls &ClsImpl::set_topk(int topk)
{
    m_postprocessor->set_topk(topk);
    return *this;
}

Cls &ClsImpl::set_score_thr(float score_thr)
{
    m_postprocessor->set_score_thr(score_thr);
    return *this;
}

dl::Model *ClsImpl::get_raw_model()
{
    return m_model;
}
} // namespace cls
} // namespace dl
