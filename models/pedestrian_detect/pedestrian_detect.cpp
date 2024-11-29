#include "pedestrian_detect.hpp"

extern const uint8_t pedestrian_espdl[] asm("_binary_pedestrian_detect_espdl_start");

PedestrianDetect::PedestrianDetect()
{
    m_model = (void *)new model_zoo::Pedestrian(
        0.5, 0.5, 10, {{8, 8, 4, 4}, {16, 16, 8, 8}, {32, 32, 16, 16}}, {0, 0, 0}, {1, 1, 1});
}

PedestrianDetect::~PedestrianDetect()
{
    if (m_model) {
        delete (model_zoo::Pedestrian *)m_model;
        m_model = nullptr;
    }
}

std::list<dl::detect::result_t> &PedestrianDetect::run(const dl::image::img_t &img)
{
    return ((model_zoo::Pedestrian *)m_model)->run(img);
}

namespace model_zoo {

Pedestrian::Pedestrian(const float score_thr,
                       const float nms_thr,
                       const int top_k,
                       const std::vector<dl::detect::anchor_point_stage_t> &stages,
                       const std::vector<float> &mean,
                       const std::vector<float> &std) :
    m_model(new dl::Model((const char *)pedestrian_espdl)),
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor(new dl::image::ImagePreprocessor(m_model, mean, std, DL_IMAGE_CAP_RGB565_BIG_ENDIAN)),
#else
    m_image_preprocessor(new dl::image::ImagePreprocessor(m_model, mean, std)),
#endif
    m_postprocessor(new dl::detect::PedestrianPostprocessor(m_model, score_thr, nms_thr, top_k, stages))
{
}

Pedestrian::~Pedestrian()
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

std::list<dl::detect::result_t> &Pedestrian::run(const dl::image::img_t &img)
{
    dl::tool::Latency latency[3] = {dl::tool::Latency(), dl::tool::Latency(), dl::tool::Latency()};
    latency[0].start();
    m_image_preprocessor->preprocess(img);
    latency[0].end();

    latency[1].start();
    m_model->run();
    latency[1].end();

    latency[2].start();
    m_postprocessor->clear_result();
    m_postprocessor->set_resize_scale_x(m_image_preprocessor->get_resize_scale_x());
    m_postprocessor->set_resize_scale_y(m_image_preprocessor->get_resize_scale_y());
    m_postprocessor->postprocess();
    std::list<dl::detect::result_t> &result = m_postprocessor->get_result(img.width, img.height);
    latency[2].end();

    latency[0].print("detect", "preprocess");
    latency[1].print("detect", "forward");
    latency[2].print("detect", "postprocess");

    return result;
}
} // namespace model_zoo
