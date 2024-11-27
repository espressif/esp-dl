#include "human_face_recognition.hpp"

extern const uint8_t human_face_feat_espdl[] asm("_binary_human_face_feat_espdl_start");

HumanFaceFeat::HumanFaceFeat(HumanFaceFeat::model_type_t model_type)
{
    switch (model_type) {
    case HumanFaceFeat::model_type_t::MODEL_MFN:
        this->model = (void *)new model_zoo::MFN<int8_t>({127.5, 127.5, 127.5}, {128, 128, 128});
        break;
    case HumanFaceFeat::model_type_t::MODEL_MBF:
        this->model = (void *)new model_zoo::MFN<int8_t>({127.5, 127.5, 127.5}, {127.5, 127.5, 127.5});
        break;
    }
}

HumanFaceFeat::~HumanFaceFeat()
{
    if (this->model) {
        delete (model_zoo::MFN<int8_t> *)this->model;
        this->model = nullptr;
    }
}

template <typename T>
dl::TensorBase *HumanFaceFeat::run(T *input_element,
                                   const std::vector<int> &input_shape,
                                   const std::vector<int> &landmarks)
{
    return ((model_zoo::MFN<int8_t> *)this->model)->run(input_element, input_shape, landmarks);
}
template dl::TensorBase *HumanFaceFeat::run(uint8_t *input_element,
                                            const std::vector<int> &input_shape,
                                            const std::vector<int> &landmarks);
template dl::TensorBase *HumanFaceFeat::run(uint16_t *input_element,
                                            const std::vector<int> &input_shape,
                                            const std::vector<int> &landmarks);

FaceRecognizer::~FaceRecognizer()
{
    if (this->detect) {
        delete this->detect;
        this->detect = nullptr;
    }
    if (this->feat_extract) {
        delete this->feat_extract;
        this->feat_extract = nullptr;
    }
}

template <typename T>
std::vector<std::list<dl::recognition::query_info>> FaceRecognizer::recognize(T *input_element,
                                                                              const std::vector<int> &input_shape)
{
    auto detect_results = this->detect->run(input_element, input_shape);
    std::vector<std::list<dl::recognition::query_info>> res;
    if (detect_results.empty()) {
        ESP_LOGW("FaceRecognizer", "Failed to recognize. No face detected.");
        return {};
    } else if (detect_results.size() == 1) {
        auto feat = this->feat_extract->run(input_element, input_shape, detect_results.back().keypoint);
        res.emplace_back(this->query_feat(feat, this->thr, this->top_k));
    } else {
        for (const auto &detect_res : detect_results) {
            auto feat = this->feat_extract->run(input_element, input_shape, detect_res.keypoint);
            res.emplace_back(this->query_feat(feat, this->thr, this->top_k));
        }
    }
    return res;
}
template std::vector<std::list<dl::recognition::query_info>> FaceRecognizer::recognize(
    uint8_t *input_element, const std::vector<int> &input_shape);
template std::vector<std::list<dl::recognition::query_info>> FaceRecognizer::recognize(
    uint16_t *input_element, const std::vector<int> &input_shape);

template <typename T>
esp_err_t FaceRecognizer::enroll(T *input_element, const std::vector<int> &input_shape)
{
    auto &detect_results = this->detect->run(input_element, input_shape);
    if (detect_results.empty()) {
        ESP_LOGW("FaceRecognizer", "Failed to enroll. No face detected.");
        return ESP_FAIL;
    } else if (detect_results.size() == 1) {
        auto feat = this->feat_extract->run(input_element, input_shape, detect_results.back().keypoint);
        return this->enroll_feat(feat);
    } else {
        ESP_LOGW("FaceRecognizer", "Failed to enroll. Multiple faces detected, please keep one face in picture.");
        return ESP_FAIL;
    }
}
template esp_err_t FaceRecognizer::enroll(uint8_t *input_element, const std::vector<int> &input_shape);
template esp_err_t FaceRecognizer::enroll(uint16_t *input_element, const std::vector<int> &input_shape);

namespace model_zoo {

template <typename feature_t>
MFN<feature_t>::MFN(const std::vector<float> &mean, const std::vector<float> &std) :
    model(new dl::Model((const char *)human_face_feat_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0))
{
    std::map<std::string, dl::TensorBase *> model_inputs_map = this->model->get_inputs();
    assert(model_inputs_map.size() == 1);
    dl::TensorBase *model_input = model_inputs_map.begin()->second;
    this->image_preprocessor = new dl::recognition::FaceImagePreprocessor<feature_t>(model_input, mean, std);

    std::map<std::string, dl::TensorBase *> model_outputs_map = this->model->get_outputs();
    assert(model_outputs_map.size() == 1);
    dl::TensorBase *model_output = model_outputs_map.begin()->second;
    this->postprocessor = new dl::recognition::RecognitionPostprocessor<feature_t>(model_output);
}

template <typename feature_t>
MFN<feature_t>::~MFN()
{
    if (this->model) {
        delete this->model;
        this->model = nullptr;
    }
    if (this->image_preprocessor) {
        delete this->image_preprocessor;
        this->image_preprocessor = nullptr;
    }
    if (this->postprocessor) {
        delete this->postprocessor;
        this->postprocessor = nullptr;
    }
}

template <typename feature_t>
template <typename T>
dl::TensorBase *MFN<feature_t>::run(T *input_element,
                                    const std::vector<int> &input_shape,
                                    const std::vector<int> &landmarks)
{
    dl::tool::Latency latency[3] = {dl::tool::Latency(), dl::tool::Latency(), dl::tool::Latency()};
    latency[0].start();
    this->image_preprocessor->preprocess(input_element, input_shape, landmarks);
    latency[0].end();

    latency[1].start();
    this->model->run();
    latency[1].end();

    latency[2].start();
    dl::TensorBase *feat = this->postprocessor->postprocess();
    latency[2].end();

    latency[0].print("recognition", "preprocess");
    latency[1].print("recognition", "forward");
    latency[2].print("recognition", "postprocess");

    return feat;
}

template dl::TensorBase *MFN<int8_t>::run(uint8_t *input_element,
                                          const std::vector<int> &input_shape,
                                          const std::vector<int> &landmarks);
template dl::TensorBase *MFN<int8_t>::run(uint16_t *input_element,
                                          const std::vector<int> &input_shape,
                                          const std::vector<int> &landmarks);
template dl::TensorBase *MFN<int16_t>::run(uint8_t *input_element,
                                           const std::vector<int> &input_shape,
                                           const std::vector<int> &landmarks);
template dl::TensorBase *MFN<int16_t>::run(uint16_t *input_element,
                                           const std::vector<int> &input_shape,
                                           const std::vector<int> &landmarks);

} // namespace model_zoo
