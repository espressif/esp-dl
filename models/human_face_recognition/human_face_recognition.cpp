#include "human_face_recognition.hpp"

extern const uint8_t human_face_feat_espdl[] asm("_binary_human_face_feat_espdl_start");

HumanFaceFeat::HumanFaceFeat(model_type_t model_type) : m_model_type(model_type)
{
    switch (model_type) {
    case model_type_t::MODEL_MFN:
        m_model = (void *)new model_zoo::MFN({127.5, 127.5, 127.5}, {128, 128, 128});
        break;
    case model_type_t::MODEL_MBF:
        m_model = (void *)new model_zoo::MBF({127.5, 127.5, 127.5}, {127.5, 127.5, 127.5});
        break;
    }
}

HumanFaceFeat::~HumanFaceFeat()
{
    if (m_model) {
        switch (m_model_type) {
        case model_type_t::MODEL_MFN:
            delete (model_zoo::MFN *)m_model;
            m_model = nullptr;
            break;
        case model_type_t::MODEL_MBF:
            delete (model_zoo::MBF *)m_model;
            m_model = nullptr;
            break;
        }
    }
}

dl::TensorBase *HumanFaceFeat::run(const dl::image::img_t &img, const std::vector<int> &landmarks)
{
    return ((model_zoo::MFN *)m_model)->run(img, landmarks);
}

HumanFaceRecognizer::~HumanFaceRecognizer()
{
    if (m_feat_extract) {
        delete m_feat_extract;
        m_feat_extract = nullptr;
    }
}

std::vector<dl::recognition::result_t> HumanFaceRecognizer::recognize(const dl::image::img_t &img,
                                                                      std::list<dl::detect::result_t> &detect_res)
{
    std::vector<std::vector<dl::recognition::result_t>> res;
    if (detect_res.empty()) {
        ESP_LOGW("HumanFaceRecognizer", "Failed to recognize. No face detected.");
        return {};
    } else if (detect_res.size() == 1) {
        auto feat = m_feat_extract->run(img, detect_res.back().keypoint);
        return query_feat(feat, m_thr, m_top_k);
    } else {
        auto max_detect_res =
            std::max_element(detect_res.begin(),
                             detect_res.end(),
                             [](const dl::detect::result_t &a, const dl::detect::result_t &b) -> bool {
                                 return a.box_area() > b.box_area();
                             });
        auto feat = m_feat_extract->run(img, max_detect_res->keypoint);
        return query_feat(feat, m_thr, m_top_k);
    }
}

esp_err_t HumanFaceRecognizer::enroll(const dl::image::img_t &img, std::list<dl::detect::result_t> &detect_res)
{
    if (detect_res.empty()) {
        ESP_LOGW("HumanFaceRecognizer", "Failed to enroll. No face detected.");
        return ESP_FAIL;
    } else if (detect_res.size() == 1) {
        auto feat = m_feat_extract->run(img, detect_res.back().keypoint);
        return enroll_feat(feat);
    } else {
        auto max_detect_res =
            std::max_element(detect_res.begin(),
                             detect_res.end(),
                             [](const dl::detect::result_t &a, const dl::detect::result_t &b) -> bool {
                                 return a.box_area() > b.box_area();
                             });
        auto feat = m_feat_extract->run(img, max_detect_res->keypoint);
        return enroll_feat(feat);
    }
}

namespace model_zoo {

MFN::MFN(const std::vector<float> &mean, const std::vector<float> &std) :
    m_model(new dl::Model((const char *)human_face_feat_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0)),
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor(
        new dl::recognition::HumanFaceImagePreprocessor(m_model, mean, std, DL_IMAGE_CAP_RGB565_BIG_ENDIAN)),
#else
    m_image_preprocessor(new dl::recognition::HumanFaceImagePreprocessor(m_model, mean, std)),
#endif
    m_postprocessor(new dl::recognition::RecognitionPostprocessor(m_model))
{
}

MFN::~MFN()
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

dl::TensorBase *MFN::run(const dl::image::img_t &img, const std::vector<int> &landmarks)
{
    dl::tool::Latency latency[3] = {dl::tool::Latency(), dl::tool::Latency(), dl::tool::Latency()};
    latency[0].start();
    m_image_preprocessor->preprocess(img, landmarks);
    latency[0].end();

    latency[1].start();
    // m_model->run();
    m_model->run(dl::RUNTIME_MODE_SINGLE_CORE);
    latency[1].end();

    latency[2].start();
    dl::TensorBase *feat = m_postprocessor->postprocess();
    latency[2].end();

    latency[0].print("recognition", "preprocess");
    latency[1].print("recognition", "forward");
    latency[2].print("recognition", "postprocess");

    return feat;
}
} // namespace model_zoo
