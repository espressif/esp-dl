#pragma once
#include "dl_detect_define.hpp"
#include <dl_tensor_base.hpp>
#include <list>
#include <map>

namespace dl {
namespace detect {
class DetectPostprocessor {
protected:
    std::map<std::string, TensorBase *> &model_outputs_map;
    const float score_threshold; /*<! Candidate box with lower score than score_threshold will be filtered */
    const float nms_threshold;   /*<! Candidate box with higher IoU than nms_threshold will be filtered */
    const int top_k;             /*<! Keep top_k number of candidate boxes */
    float resize_scale_x;
    float resize_scale_y;
    float top_left_x;
    float top_left_y;
    std::list<result_t> box_list; /*<! Detected box list */
    TensorBase *get_model_output(const char *output_name);

public:
    DetectPostprocessor(std::map<std::string, TensorBase *> &model_outputs_map,
                        const float score_threshold,
                        const float nms_threshold,
                        const int top_k) :
        model_outputs_map(model_outputs_map),
        score_threshold(score_threshold),
        nms_threshold(nms_threshold),
        top_k(top_k) {};
    virtual ~DetectPostprocessor() {};
    virtual void postprocess() = 0;
    void nms();
    void set_resize_scale_x(float resize_scale_x) { this->resize_scale_x = resize_scale_x; };
    void set_resize_scale_y(float resize_scale_y) { this->resize_scale_y = resize_scale_y; };
    void set_top_left_x(float top_left_x) { this->top_left_x = top_left_x; };
    void set_top_left_y(float top_left_y) { this->top_left_y = top_left_y; };
    void clear_result() { this->box_list.clear(); };
    std::list<result_t> &get_result(const std::vector<int> &input_shape);
};

class AnchorPointDetectPostprocessor : public DetectPostprocessor {
protected:
    std::vector<anchor_point_stage_t> stages;

public:
    AnchorPointDetectPostprocessor(std::map<std::string, TensorBase *> &model_outputs_map,
                                   const float score_threshold,
                                   const float nms_threshold,
                                   const int top_k,
                                   const std::vector<anchor_point_stage_t> &stages) :
        DetectPostprocessor(model_outputs_map, score_threshold, nms_threshold, top_k), stages(stages) {};
};

class AnchorBoxDetectPostprocessor : public DetectPostprocessor {
protected:
    std::vector<anchor_box_stage_t> stages;

public:
    AnchorBoxDetectPostprocessor(std::map<std::string, TensorBase *> &model_outputs_map,
                                 const float score_threshold,
                                 const float nms_threshold,
                                 const int top_k,
                                 const std::vector<anchor_box_stage_t> &stages) :
        DetectPostprocessor(model_outputs_map, score_threshold, nms_threshold, top_k), stages(stages) {};
};
} // namespace detect
} // namespace dl
