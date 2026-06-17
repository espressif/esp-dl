#pragma once
#include "dl_detect_base.hpp"
#include "dl_seg_yolo11_postprocessor.hpp"

namespace coco_seg {
class Yolo11nSeg : public dl::detect::DetectImpl {
public:
    static constexpr float default_score_thr = 0.25;
    static constexpr float default_nms_thr = 0.7;
    Yolo11nSeg(const char *model_name, float score_thr, float nms_thr);
};
} // namespace coco_seg

class COCOSeg : public dl::detect::DetectWrapper {
public:
    typedef enum {
        YOLO11N_SEG_S8_V1,
    } model_type_t;
    COCOSeg(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_COCO_SEG_MODEL), bool lazy_load = true);

private:
    void load_model() override;

    model_type_t m_model_type;
};
