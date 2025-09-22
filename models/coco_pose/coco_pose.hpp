#pragma once
#include "dl_detect_base.hpp"
#include "dl_pose_yolo11_postprocessor.hpp"

namespace coco_pose {
class Yolo11nPose : public dl::detect::DetectImpl {
public:
    static inline constexpr float default_score_thr = 0.25;
    static inline constexpr float default_nms_thr = 0.7;
    Yolo11nPose(const char *model_name, float score_thr, float nms_thr);
};
} // namespace coco_pose

class COCOPose : public dl::detect::DetectWrapper {
public:
    typedef enum {
        YOLO11N_POSE_S8_V1,
        YOLO11N_POSE_S8_V2,
    } model_type_t;
    COCOPose(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_COCO_POSE_MODEL),
             bool lazy_load = true);

private:
    void load_model() override;

    model_type_t m_model_type;
};
