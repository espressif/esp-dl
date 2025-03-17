#include "dl_detect_base.hpp"

namespace dl {
namespace detect {

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
    m_postprocessor->set_resize_scale_x(m_image_preprocessor->get_resize_scale_x());
    m_postprocessor->set_resize_scale_y(m_image_preprocessor->get_resize_scale_y());
    m_postprocessor->postprocess();
    std::list<dl::detect::result_t> &result = m_postprocessor->get_result(img.width, img.height);
    DL_LOG_INFER_LATENCY_END_PRINT("detect", "post");

    return result;
}

} // namespace detect
} // namespace dl
