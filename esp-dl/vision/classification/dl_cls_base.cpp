#include "dl_cls_base.hpp"

namespace dl {
namespace cls {

ClsImpl::~ClsImpl()
{
    if (m_model) {
        delete m_model;
        m_model = nullptr;
    }
    if (m_image_preprocessor) {
        delete m_image_preprocessor;
        m_image_preprocessor = nullptr;
    }
    if (m_postprocessor) {
        delete m_postprocessor;
        m_postprocessor = nullptr;
    }
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

} // namespace cls
} // namespace dl
