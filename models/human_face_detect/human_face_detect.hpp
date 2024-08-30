#pragma once

#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"
#include "dl_detect_msr01_postprocessor.hpp"
#include "dl_detect_mnp01_postprocessor.hpp"

extern const uint8_t human_face_detect_espdl[] asm("_binary_human_face_detect_espdl_start");

class HumanFaceDetect
{
private:
    void *stage1_model;
    void *stage2_model;

public:
    /**
     * @brief Construct a new HumanFaceDetect object
     */
    HumanFaceDetect();

    /**
     * @brief Destroy the HumanFaceDetect object
     */
    ~HumanFaceDetect();

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

    void set_print_info(bool print_info);
};

namespace dl {
namespace detect {

template <typename feature_t>
class MSR01 {
private:
    Model *model;
    image::ImagePreprocessor<feature_t> *image_preprocessor;
    MSR01Postprocessor<feature_t> *postprocessor;
    bool print_info = false;

public:
    MSR01(const float score_threshold,
          const float nms_threshold,
          const int top_k,
          const std::vector<anchor_box_stage_t> &stages,
          const std::vector<float> &mean,
          const std::vector<float> &std) :
        model(new Model((const char *)human_face_detect_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0)),
        postprocessor(new MSR01Postprocessor<feature_t>(score_threshold, nms_threshold, top_k, stages))
    {
        std::map<std::string, TensorBase *> model_inputs_map = this->model->get_inputs();
        assert(model_inputs_map.size() == 1);
        TensorBase *model_input = model_inputs_map.begin()->second;
#if CONFIG_IDF_TARGET_ESP32P4
        this->image_preprocessor = new image::ImagePreprocessor<feature_t>(model_input, mean, std);
#else
        this->image_preprocessor = new image::ImagePreprocessor<feature_t>(model_input, mean, std, false, true);
#endif
    }

    ~MSR01()
    {
        delete this->model;
        delete this->image_preprocessor;
        delete this->postprocessor;
    }

    void set_print_info(bool print_info)
    {
        this->print_info = print_info;
        this->image_preprocessor->set_print_info(print_info);
        this->postprocessor->set_print_info(print_info);
    };

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
        if (this->print_info) {
            latency[0].print("detect", "preprocess");
            latency[1].print("detect", "forward");
            latency[2].print("detect", "postprocess");
        }
        return result;
    }
};

template <typename feature_t>
class MNP01 {
private:
    Model *model;
    image::ImagePreprocessor<feature_t> *image_preprocessor;
    MNP01Postprocessor<feature_t> *postprocessor;
    bool print_info = false;

public:
    MNP01(const float score_threshold,
          const float nms_threshold,
          const int top_k,
          const std::vector<anchor_box_stage_t> &stages,
          const std::vector<float> &mean,
          const std::vector<float> &std) :
        model(new Model((const char *)human_face_detect_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 1)),
        postprocessor(new MNP01Postprocessor<feature_t>(score_threshold, nms_threshold, top_k, stages))
    {
        std::map<std::string, TensorBase *> model_inputs_map = this->model->get_inputs();
        assert(model_inputs_map.size() == 1);
        TensorBase *model_input = model_inputs_map.begin()->second;
#if CONFIG_IDF_TARGET_ESP32P4
        this->image_preprocessor = new image::ImagePreprocessor<feature_t>(model_input, mean, std);
#else
        this->image_preprocessor = new image::ImagePreprocessor<feature_t>(model_input, mean, std, false, true);
#endif
    }

    ~MNP01()
    {
        delete this->model;
        delete this->image_preprocessor;
        delete this->postprocessor;
    };

    void set_print_info(bool print_info)
    {
        this->print_info = print_info;
        this->image_preprocessor->set_print_info(print_info);
        this->postprocessor->set_print_info(print_info);
    };

    template <typename T>
    std::list<result_t> &run(T *input_element,
                                         std::vector<int> input_shape,
                                         std::list<result_t> &candidates)
    {
        tool::Latency latency[3] = {tool::Latency(10), tool::Latency(10), tool::Latency(10)};
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
            this->postprocessor->postprocess(model->get_outputs());
            latency[2].end();
        }
        this->postprocessor->nms();
        std::list<result_t> &result = this->postprocessor->get_result(input_shape);
        if (this->print_info && candidates.size() > 0) {
            latency[0].print("detect", "preprocess");
            latency[1].print("detect", "forward");
            latency[2].print("detect", "postprocess");
        }
        return result;
    }
};

} // namespace detect
} // namespace dl