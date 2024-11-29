#pragma once

#include "dl_detect_mnp01_postprocessor.hpp"
#include "dl_detect_msr01_postprocessor.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"

class HumanFaceDetect {
private:
    void *m_stage1_model;
    void *m_stage2_model;

public:
    /**
     * @brief Construct a new HumanFaceDetect object
     */
    HumanFaceDetect();

    /**
     * @brief Destroy the HumanFaceDetect object
     */
    ~HumanFaceDetect();

    std::list<dl::detect::result_t> &run(const dl::image::img_t &img);
};
namespace model_zoo {

class MSR01 {
private:
    dl::Model *m_model;
    dl::image::ImagePreprocessor *m_image_preprocessor;
    dl::detect::MSR01Postprocessor *m_postprocessor;

public:
    MSR01(const float score_thr,
          const float nms_thr,
          const int top_k,
          const std::vector<dl::detect::anchor_box_stage_t> &stages,
          const std::vector<float> &mean,
          const std::vector<float> &std);
    ~MSR01();

    std::list<dl::detect::result_t> &run(const dl::image::img_t &img);
};

class MNP01 {
private:
    dl::Model *m_model;
    dl::image::ImagePreprocessor *m_image_preprocessor;
    dl::detect::MNP01Postprocessor *m_postprocessor;

public:
    MNP01(const float score_thr,
          const float nms_thr,
          const int top_k,
          const std::vector<dl::detect::anchor_box_stage_t> &stages,
          const std::vector<float> &mean,
          const std::vector<float> &std);
    ~MNP01();

    std::list<dl::detect::result_t> &run(const dl::image::img_t &img, std::list<dl::detect::result_t> &candidates);
};

} // namespace model_zoo
