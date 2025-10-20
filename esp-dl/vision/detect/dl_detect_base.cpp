#include "dl_detect_base.hpp"

namespace dl {
namespace detect {
DetectWrapper::~DetectWrapper()
{
    delete m_model;
}

std::list<dl::detect::result_t> &DetectWrapper::run(const dl::image::img_t &img)
{
    if (!m_model) {
        load_model();
    }
    return m_model->run(img);
}

Detect &DetectWrapper::set_score_thr(float score_thr, int idx)
{
    assert(idx == 0 || idx == 1);
    if (m_model) {
        m_model->set_score_thr(score_thr, idx);
    } else {
        m_score_thr[idx] = score_thr;
    }
    return *this;
}

Detect &DetectWrapper::set_nms_thr(float nms_thr, int idx)
{
    assert(idx == 0 || idx == 1);
    if (m_model) {
        m_model->set_nms_thr(nms_thr, idx);
    } else {
        m_nms_thr[idx] = nms_thr;
    }
    return *this;
}

dl::Model *DetectWrapper::get_raw_model(int idx)
{
    assert(idx == 0 || idx == 1);
    if (!m_model) {
        load_model();
    }
    return m_model->get_raw_model(idx);
}

DetectImpl::~DetectImpl()
{
    delete m_model;
    delete m_image_preprocessor;
    delete m_postprocessor;
}

std::list<dl::detect::result_t> &DetectImpl::run(const dl::image::img_t &img)
{
    DL_LOG_INFER_LATENCY_INIT();
    DL_LOG_INFER_LATENCY_START();
    m_image_preprocessor->preprocess(img);
    DL_LOG_INFER_LATENCY_END_PRINT("detect", "pre");

    DL_LOG_INFER_LATENCY_START();
    m_model->run();
    DL_LOG_INFER_LATENCY_END_PRINT("detect", "model");

    DL_LOG_INFER_LATENCY_START();
    m_postprocessor->clear_result();
    m_postprocessor->postprocess();
    std::list<dl::detect::result_t> &result = m_postprocessor->get_result(img.width, img.height);
    DL_LOG_INFER_LATENCY_END_PRINT("detect", "post");

    return result;
}

Detect &DetectImpl::set_score_thr(float score_thr, int idx)
{
    m_postprocessor->set_score_thr(score_thr);
    return *this;
}

Detect &DetectImpl::set_nms_thr(float nms_thr, int idx)
{
    m_postprocessor->set_nms_thr(nms_thr);
    return *this;
}

dl::Model *DetectImpl::get_raw_model(int idx)
{
    return m_model;
}
} // namespace detect
} // namespace dl
