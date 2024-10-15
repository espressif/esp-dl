#pragma once

#include "dl_detect_mnp01_postprocessor.hpp"
#include "dl_detect_msr01_postprocessor.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"

class HumanFaceDetect {
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
};
namespace model_zoo {

template <typename feature_t>
class MSR01 {
private:
    dl::Model *model;
    dl::image::ImagePreprocessor<feature_t> *image_preprocessor;
    dl::detect::MSR01Postprocessor<feature_t> *postprocessor;

public:
    MSR01(const float score_threshold,
          const float nms_threshold,
          const int top_k,
          const std::vector<dl::detect::anchor_box_stage_t> &stages,
          const std::vector<float> &mean,
          const std::vector<float> &std);
    ~MSR01();

    template <typename T>
    std::list<dl::detect::result_t> &run(T *input_element, std::vector<int> input_shape);
};

template <typename feature_t>
class MNP01 {
private:
    dl::Model *model;
    dl::image::ImagePreprocessor<feature_t> *image_preprocessor;
    dl::detect::MNP01Postprocessor<feature_t> *postprocessor;

public:
    MNP01(const float score_threshold,
          const float nms_threshold,
          const int top_k,
          const std::vector<dl::detect::anchor_box_stage_t> &stages,
          const std::vector<float> &mean,
          const std::vector<float> &std);
    ~MNP01();

    template <typename T>
    std::list<dl::detect::result_t> &run(T *input_element,
                                         std::vector<int> input_shape,
                                         std::list<dl::detect::result_t> &candidates);
};

} // namespace model_zoo
