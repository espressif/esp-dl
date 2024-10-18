#include "human_face_detect.hpp"

extern const uint8_t human_face_detect_espdl[] asm("_binary_human_face_detect_espdl_start");

HumanFaceDetect::HumanFaceDetect()
{
    this->stage1_model = (void *)new model_zoo::MSR01<int8_t>(
        0.5,
        0.5,
        10,
        {{8, 8, 9, 9, {{16, 16}, {32, 32}}}, {16, 16, 9, 9, {{64, 64}, {128, 128}}}},
        {0, 0, 0},
        {1, 1, 1});
    this->stage2_model =
        (void *)new model_zoo::MNP01<int8_t>(0.5, 0.5, 10, {{1, 1, 0, 0, {{48, 48}}}}, {0, 0, 0}, {1, 1, 1});
}

HumanFaceDetect::~HumanFaceDetect()
{
    if (this->stage1_model) {
        delete (model_zoo::MSR01<int8_t> *)this->stage1_model;
        this->stage1_model = nullptr;
    }
    if (this->stage2_model) {
        delete (model_zoo::MNP01<int8_t> *)this->stage2_model;
        this->stage2_model = nullptr;
    }
}

template <typename T>
std::list<dl::detect::result_t> &HumanFaceDetect::run(T *input_element, std::vector<int> input_shape)
{
    std::list<dl::detect::result_t> &candidates =
        ((model_zoo::MSR01<int8_t> *)this->stage1_model)->run(input_element, input_shape);
    return ((model_zoo::MNP01<int8_t> *)this->stage2_model)->run(input_element, input_shape, candidates);
}
template std::list<dl::detect::result_t> &HumanFaceDetect::run(uint16_t *input_element, std::vector<int> input_shape);
template std::list<dl::detect::result_t> &HumanFaceDetect::run(uint8_t *input_element, std::vector<int> input_shape);
namespace model_zoo {

template <typename feature_t>
MSR01<feature_t>::MSR01(const float score_threshold,
                        const float nms_threshold,
                        const int top_k,
                        const std::vector<dl::detect::anchor_box_stage_t> &stages,
                        const std::vector<float> &mean,
                        const std::vector<float> &std) :
    model(new dl::Model((const char *)human_face_detect_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 1)),
    postprocessor(new dl::detect::MSR01Postprocessor<feature_t>(
        this->model->get_outputs(), score_threshold, nms_threshold, top_k, stages))
{
    std::map<std::string, dl::TensorBase *> model_inputs_map = this->model->get_inputs();
    assert(model_inputs_map.size() == 1);
    dl::TensorBase *model_input = model_inputs_map.begin()->second;
    this->image_preprocessor = new dl::image::ImagePreprocessor<feature_t>(model_input, mean, std);
}

template <typename feature_t>
MSR01<feature_t>::~MSR01()
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
std::list<dl::detect::result_t> &MSR01<feature_t>::run(T *input_element, std::vector<int> input_shape)
{
    dl::tool::Latency latency[3] = {dl::tool::Latency(), dl::tool::Latency(), dl::tool::Latency()};
    latency[0].start();
    this->image_preprocessor->preprocess(input_element, input_shape);
    latency[0].end();

    latency[1].start();
    this->model->run();
    latency[1].end();

    latency[2].start();
    this->postprocessor->clear_result();
    this->postprocessor->set_resize_scale_x(this->image_preprocessor->get_resize_scale_x());
    this->postprocessor->set_resize_scale_y(this->image_preprocessor->get_resize_scale_y());
    this->postprocessor->postprocess();
    std::list<dl::detect::result_t> &result = this->postprocessor->get_result(input_shape);
    latency[2].end();

    latency[0].print("detect", "preprocess");
    latency[1].print("detect", "forward");
    latency[2].print("detect", "postprocess");

    return result;
}

template std::list<dl::detect::result_t> &MSR01<int8_t>::run(uint8_t *input_element, std::vector<int> input_shape);
template std::list<dl::detect::result_t> &MSR01<int8_t>::run(uint16_t *input_element, std::vector<int> input_shape);
template std::list<dl::detect::result_t> &MSR01<int16_t>::run(uint8_t *input_element, std::vector<int> input_shape);
template std::list<dl::detect::result_t> &MSR01<int16_t>::run(uint16_t *input_element, std::vector<int> input_shape);

template <typename feature_t>
MNP01<feature_t>::MNP01(const float score_threshold,
                        const float nms_threshold,
                        const int top_k,
                        const std::vector<dl::detect::anchor_box_stage_t> &stages,
                        const std::vector<float> &mean,
                        const std::vector<float> &std) :
    model(new dl::Model((const char *)human_face_detect_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0)),
    postprocessor(new dl::detect::MNP01Postprocessor<feature_t>(
        this->model->get_outputs(), score_threshold, nms_threshold, top_k, stages))
{
    std::map<std::string, dl::TensorBase *> model_inputs_map = this->model->get_inputs();
    assert(model_inputs_map.size() == 1);
    dl::TensorBase *model_input = model_inputs_map.begin()->second;
    this->image_preprocessor = new dl::image::ImagePreprocessor<feature_t>(model_input, mean, std);
}

template <typename feature_t>
MNP01<feature_t>::~MNP01()
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
};

template <typename feature_t>
template <typename T>
std::list<dl::detect::result_t> &MNP01<feature_t>::run(T *input_element,
                                                       std::vector<int> input_shape,
                                                       std::list<dl::detect::result_t> &candidates)
{
    dl::tool::Latency latency[3] = {dl::tool::Latency(10), dl::tool::Latency(10), dl::tool::Latency(10)};
    this->postprocessor->clear_result();
    for (auto &candidate : candidates) {
        int center_x = (candidate.box[0] + candidate.box[2]) >> 1;
        int center_y = (candidate.box[1] + candidate.box[3]) >> 1;
        int side = DL_MAX(candidate.box[2] - candidate.box[0], candidate.box[3] - candidate.box[1]);
        candidate.box[0] = center_x - (side >> 1);
        candidate.box[1] = center_y - (side >> 1);
        candidate.box[2] = candidate.box[0] + side;
        candidate.box[3] = candidate.box[1] + side;

        latency[0].start();
        this->image_preprocessor->preprocess(input_element, input_shape, candidate.box);
        latency[0].end();

        latency[1].start();
        this->model->run();
        latency[1].end();

        latency[2].start();
        this->postprocessor->set_resize_scale_x(this->image_preprocessor->get_resize_scale_x());
        this->postprocessor->set_resize_scale_y(this->image_preprocessor->get_resize_scale_y());
        this->postprocessor->set_top_left_x(this->image_preprocessor->get_top_left_x());
        this->postprocessor->set_top_left_y(this->image_preprocessor->get_top_left_y());
        this->postprocessor->postprocess();
        latency[2].end();
    }
    this->postprocessor->nms();
    std::list<dl::detect::result_t> &result = this->postprocessor->get_result(input_shape);
    if (candidates.size() > 0) {
        latency[0].print("detect", "preprocess");
        latency[1].print("detect", "forward");
        latency[2].print("detect", "postprocess");
    }
    return result;
}

template std::list<dl::detect::result_t> &MNP01<int8_t>::run(uint8_t *input_element,
                                                             std::vector<int> input_shape,
                                                             std::list<dl::detect::result_t> &candidates);
template std::list<dl::detect::result_t> &MNP01<int8_t>::run(uint16_t *input_element,
                                                             std::vector<int> input_shape,
                                                             std::list<dl::detect::result_t> &candidates);
template std::list<dl::detect::result_t> &MNP01<int16_t>::run(uint8_t *input_element,
                                                              std::vector<int> input_shape,
                                                              std::list<dl::detect::result_t> &candidates);
template std::list<dl::detect::result_t> &MNP01<int16_t>::run(uint16_t *input_element,
                                                              std::vector<int> input_shape,
                                                              std::list<dl::detect::result_t> &candidates);

} // namespace model_zoo
