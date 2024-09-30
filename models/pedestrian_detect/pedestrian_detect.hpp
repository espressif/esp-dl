#pragma once

#include "dl_detect_pedestrian_postprocessor.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"

extern const uint8_t pedestrian_espdl[] asm("_binary_pedestrian_detect_espdl_start");

class PedestrianDetect {
private:
    void *model;

public:
    /**
     * @brief Construct a new PedestrianDetect object
     */
    PedestrianDetect();

    /**
     * @brief Destroy the PedestrianDetect object
     */
    ~PedestrianDetect();

    /**
     * @brief Inference.
     *
     * @tparam T supports uint8_t and uint16_t
     *         - uint8_t: input image is RGB888
     *         - uint16_t: input image is RGB565
     * @param input_element pointer of input image
     * @param input_shape   shape of input image
     * @return detection result
     */
    template <typename T>
    std::list<dl::detect::result_t> &run(T *input_element, std::vector<int> input_shape);
};

namespace dl {
namespace detect {

template <typename feature_t>
class Pedestrian {
private:
    Model *model;
    image::ImagePreprocessor<feature_t> *image_preprocessor;
    PedestrianPostprocessor<feature_t> *postprocessor;

public:
    Pedestrian(const float score_threshold,
               const float nms_threshold,
               const int top_k,
               const std::vector<anchor_point_stage_t> &stages,
               const std::vector<float> &mean,
               const std::vector<float> &std) :
        model(new Model((const char *)pedestrian_espdl)),
        postprocessor(new PedestrianPostprocessor<feature_t>(score_threshold, nms_threshold, top_k, stages))
    {
        std::map<std::string, TensorBase *> model_inputs_map = this->model->get_inputs();
        assert(model_inputs_map.size() == 1);
        TensorBase *model_input = model_inputs_map.begin()->second;
        this->image_preprocessor = new image::ImagePreprocessor<feature_t>(model_input, mean, std);
    }

    ~Pedestrian()
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

    template <typename T>
    std::list<result_t> &run(T *input_element, std::vector<int> input_shape)
    {
        tool::Latency latency[3] = {tool::Latency(), tool::Latency(), tool::Latency()};
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
        this->postprocessor->postprocess(model->get_outputs());
        std::list<result_t> &result = this->postprocessor->get_result(input_shape);
        latency[2].end();

        latency[0].print("detect", "preprocess");
        latency[1].print("detect", "forward");
        latency[2].print("detect", "postprocess");

        return result;
    }
};

} // namespace detect
} // namespace dl
