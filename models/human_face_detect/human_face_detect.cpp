#include "human_face_detect.hpp"

extern const uint8_t human_face_detect_espdl[] asm("_binary_human_face_detect_espdl_start");

HumanFaceDetect::HumanFaceDetect()
{
    m_stage1_model =
        (void *)new model_zoo::MSR01(0.5,
                                     0.5,
                                     10,
                                     {{8, 8, 9, 9, {{16, 16}, {32, 32}}}, {16, 16, 9, 9, {{64, 64}, {128, 128}}}},
                                     {0, 0, 0},
                                     {1, 1, 1});
    m_stage2_model = (void *)new model_zoo::MNP01(0.5, 0.5, 10, {{1, 1, 0, 0, {{48, 48}}}}, {0, 0, 0}, {1, 1, 1});
}

HumanFaceDetect::~HumanFaceDetect()
{
    if (m_stage1_model) {
        delete (model_zoo::MSR01 *)m_stage1_model;
        m_stage1_model = nullptr;
    }
    if (m_stage2_model) {
        delete (model_zoo::MNP01 *)m_stage2_model;
        m_stage2_model = nullptr;
    }
}

std::list<dl::detect::result_t> &HumanFaceDetect::run(const dl::image::img_t &img)
{
    std::list<dl::detect::result_t> &candidates = ((model_zoo::MSR01 *)m_stage1_model)->run(img);
    return ((model_zoo::MNP01 *)m_stage2_model)->run(img, candidates);
}
namespace model_zoo {

MSR01::MSR01(const float score_thr,
             const float nms_thr,
             const int top_k,
             const std::vector<dl::detect::anchor_box_stage_t> &stages,
             const std::vector<float> &mean,
             const std::vector<float> &std) :
    m_model(new dl::Model((const char *)human_face_detect_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 1)),
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor(new dl::image::ImagePreprocessor(m_model, mean, std, DL_IMAGE_CAP_RGB565_BIG_ENDIAN)),
#else
    m_image_preprocessor(new dl::image::ImagePreprocessor(m_model, mean, std)),
#endif
    m_postprocessor(new dl::detect::MSR01Postprocessor(m_model, score_thr, nms_thr, top_k, stages))
{
}

MSR01::~MSR01()
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

std::list<dl::detect::result_t> &MSR01::run(const dl::image::img_t &img)
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

MNP01::MNP01(const float score_thr,
             const float nms_thr,
             const int top_k,
             const std::vector<dl::detect::anchor_box_stage_t> &stages,
             const std::vector<float> &mean,
             const std::vector<float> &std) :
    m_model(new dl::Model((const char *)human_face_detect_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0)),
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor(new dl::image::ImagePreprocessor(m_model, mean, std, DL_IMAGE_CAP_RGB565_BIG_ENDIAN)),
#else
    m_image_preprocessor(new dl::image::ImagePreprocessor(m_model, mean, std)),
#endif
    m_postprocessor(new dl::detect::MNP01Postprocessor(m_model, score_thr, nms_thr, top_k, stages))
{
}

MNP01::~MNP01()
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
};

std::list<dl::detect::result_t> &MNP01::run(const dl::image::img_t &img, std::list<dl::detect::result_t> &candidates)
{
    dl::tool::Latency latency[3] = {dl::tool::Latency(10), dl::tool::Latency(10), dl::tool::Latency(10)};
    m_postprocessor->clear_result();
    for (auto &candidate : candidates) {
        int center_x = (candidate.box[0] + candidate.box[2]) >> 1;
        int center_y = (candidate.box[1] + candidate.box[3]) >> 1;
        int side = DL_MAX(candidate.box[2] - candidate.box[0], candidate.box[3] - candidate.box[1]);
        candidate.box[0] = center_x - (side >> 1);
        candidate.box[1] = center_y - (side >> 1);
        candidate.box[2] = candidate.box[0] + side;
        candidate.box[3] = candidate.box[1] + side;
        candidate.limit_box(img.width, img.height);

        latency[0].start();
        m_image_preprocessor->preprocess(img, candidate.box);
        latency[0].end();

        latency[1].start();
        m_model->run();
        latency[1].end();

        latency[2].start();
        m_postprocessor->set_resize_scale_x(m_image_preprocessor->get_resize_scale_x());
        m_postprocessor->set_resize_scale_y(m_image_preprocessor->get_resize_scale_y());
        m_postprocessor->set_top_left_x(m_image_preprocessor->get_top_left_x());
        m_postprocessor->set_top_left_y(m_image_preprocessor->get_top_left_y());
        m_postprocessor->postprocess();
        latency[2].end();
    }
    m_postprocessor->nms();
    std::list<dl::detect::result_t> &result = m_postprocessor->get_result(img.width, img.height);
    if (candidates.size() > 0) {
        latency[0].print("detect", "preprocess");
        latency[1].print("detect", "forward");
        latency[2].print("detect", "postprocess");
    }
    return result;
}

} // namespace model_zoo
