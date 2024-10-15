#pragma once

#include "dl_detect_pedestrian_postprocessor.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"

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

namespace model_zoo {

template <typename feature_t>
class Pedestrian {
private:
    dl::Model *model;
    dl::image::ImagePreprocessor<feature_t> *image_preprocessor;
    dl::detect::PedestrianPostprocessor<feature_t> *postprocessor;

public:
    Pedestrian(const float score_threshold,
               const float nms_threshold,
               const int top_k,
               const std::vector<dl::detect::anchor_point_stage_t> &stages,
               const std::vector<float> &mean,
               const std::vector<float> &std);

    ~Pedestrian();

    template <typename T>
    std::list<dl::detect::result_t> &run(T *input_element, const std::vector<int> &input_shape);
};

} // namespace model_zoo
