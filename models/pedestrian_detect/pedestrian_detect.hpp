#pragma once

#include "dl_detect_pedestrian_postprocessor.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"

class PedestrianDetect {
private:
    void *m_model;

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
     * @param img input image
     * @return detection result
     */
    std::list<dl::detect::result_t> &run(const dl::image::img_t &img);
};

namespace model_zoo {

class Pedestrian {
private:
    dl::Model *m_model;
    dl::image::ImagePreprocessor *m_image_preprocessor;
    dl::detect::PedestrianPostprocessor *m_postprocessor;

public:
    Pedestrian(const float score_thr,
               const float nms_thr,
               const int top_k,
               const std::vector<dl::detect::anchor_point_stage_t> &stages,
               const std::vector<float> &mean,
               const std::vector<float> &std);

    ~Pedestrian();

    std::list<dl::detect::result_t> &run(const dl::image::img_t &img);
};

} // namespace model_zoo
