#pragma once
#include "dl_detect_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"

#include <list>
#include <map>

namespace dl {
namespace detect {
class DetectPostprocessor {
protected:
    Model *m_model;
    image::ImagePreprocessor *m_image_preprocessor;
    const float m_score_thr;        /*!< Candidate box with lower score than score_thr will be filtered */
    const float m_nms_thr;          /*!< Candidate box with higher IoU than nms_thr will be filtered */
    const int m_top_k;              /*!< Keep top_k number of candidate boxes */
    std::list<result_t> m_box_list; /*!< Detected box list */

public:
    DetectPostprocessor(Model *model,
                        image::ImagePreprocessor *image_preprocessor,
                        const float score_thr,
                        const float nms_thr,
                        const int top_k) :
        m_model(model),
        m_image_preprocessor(image_preprocessor),
        m_score_thr(score_thr),
        m_nms_thr(nms_thr),
        m_top_k(top_k) {};
    virtual ~DetectPostprocessor() {};
    virtual void postprocess() = 0;
    void nms();
    void clear_result() { m_box_list.clear(); };
    std::list<result_t> &get_result(int width, int height);
};

class AnchorPointDetectPostprocessor : public DetectPostprocessor {
protected:
    std::vector<anchor_point_stage_t> m_stages;

public:
    AnchorPointDetectPostprocessor(Model *model,
                                   image::ImagePreprocessor *image_preprocessor,
                                   const float score_thr,
                                   const float nms_thr,
                                   const int top_k,
                                   const std::vector<anchor_point_stage_t> &stages) :
        DetectPostprocessor(model, image_preprocessor, score_thr, nms_thr, top_k), m_stages(stages) {};
};

class AnchorBoxDetectPostprocessor : public DetectPostprocessor {
protected:
    std::vector<anchor_box_stage_t> m_stages;

public:
    AnchorBoxDetectPostprocessor(Model *model,
                                 image::ImagePreprocessor *image_preprocessor,
                                 const float score_thr,
                                 const float nms_thr,
                                 const int top_k,
                                 const std::vector<anchor_box_stage_t> &stages) :
        DetectPostprocessor(model, image_preprocessor, score_thr, nms_thr, top_k), m_stages(stages) {};
};
} // namespace detect
} // namespace dl
