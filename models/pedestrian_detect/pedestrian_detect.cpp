#include "pedestrian_detect.hpp"

extern const uint8_t pedestrian_espdl[] asm("_binary_pedestrian_detect_espdl_start");

PedestrianDetect::PedestrianDetect()
{
    this->model = (void *)new model_zoo::Pedestrian<int8_t>(
        0.5, 0.5, 10, {{8, 8, 4, 4}, {16, 16, 8, 8}, {32, 32, 16, 16}}, {0, 0, 0}, {1, 1, 1});
}

PedestrianDetect::~PedestrianDetect()
{
    if (this->model) {
        delete (model_zoo::Pedestrian<int8_t> *)this->model;
        this->model = nullptr;
    }
}

template <typename T>
std::list<dl::detect::result_t> &PedestrianDetect::run(T *input_element, std::vector<int> input_shape)
{
    return ((model_zoo::Pedestrian<int8_t> *)this->model)->run(input_element, input_shape);
}
template std::list<dl::detect::result_t> &PedestrianDetect::run(uint16_t *input_element, std::vector<int> input_shape);
template std::list<dl::detect::result_t> &PedestrianDetect::run(uint8_t *input_element, std::vector<int> input_shape);

namespace model_zoo {

template <typename feature_t>
Pedestrian<feature_t>::Pedestrian(const float score_threshold,
                                  const float nms_threshold,
                                  const int top_k,
                                  const std::vector<dl::detect::anchor_point_stage_t> &stages,
                                  const std::vector<float> &mean,
                                  const std::vector<float> &std) :
    model(new dl::Model((const char *)pedestrian_espdl)),
    postprocessor(new dl::detect::PedestrianPostprocessor<feature_t>(
        this->model->get_outputs(), score_threshold, nms_threshold, top_k, stages))
{
    std::map<std::string, dl::TensorBase *> model_inputs_map = this->model->get_inputs();
    assert(model_inputs_map.size() == 1);
    dl::TensorBase *model_input = model_inputs_map.begin()->second;
    this->image_preprocessor = new dl::image::ImagePreprocessor<feature_t>(model_input, mean, std);
}

template <typename feature_t>
Pedestrian<feature_t>::~Pedestrian()
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
std::list<dl::detect::result_t> &Pedestrian<feature_t>::run(T *input_element, const std::vector<int> &input_shape)
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

template std::list<dl::detect::result_t> &Pedestrian<int8_t>::run(uint8_t *input_element,
                                                                  const std::vector<int> &input_shape);
template std::list<dl::detect::result_t> &Pedestrian<int8_t>::run(uint16_t *input_element,
                                                                  const std::vector<int> &input_shape);
template std::list<dl::detect::result_t> &Pedestrian<int16_t>::run(uint8_t *input_element,
                                                                   const std::vector<int> &input_shape);
template std::list<dl::detect::result_t> &Pedestrian<int16_t>::run(uint16_t *input_element,
                                                                   const std::vector<int> &input_shape);
} // namespace model_zoo
