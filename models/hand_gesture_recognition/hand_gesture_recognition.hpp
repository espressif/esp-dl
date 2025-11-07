#pragma once
#include "dl_cls_base.hpp"
#include "dl_detect_define.hpp"
#include "hand_gesture_cls_postprocessor.hpp"

namespace hand_gesture_recognition {

class MobileNetV2 : public dl::cls::ClsImpl {
public:
    static inline constexpr int default_topk = 1;
    static inline constexpr float default_score_thr = std::numeric_limits<float>::lowest();
    MobileNetV2(const char *model_name, int topk, float score_thr);
    std::vector<dl::cls::result_t> run_crop(const dl::image::img_t &img, const std::vector<int> &crop_area);
};

} // namespace hand_gesture_recognition

class HandGestureCls : public dl::cls::ClsWrapper {
public:
    typedef enum { MOBILENETV2_0_5_S8_V1 } model_type_t;
    HandGestureCls(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_HAND_GESTURE_CLS_MODEL),
                   bool lazy_load = true);
    std::vector<dl::cls::result_t> run_crop(const dl::image::img_t &img, const std::vector<int> &crop_area);

private:
    hand_gesture_recognition::MobileNetV2 *m_cls_model;
    void load_model();
    model_type_t m_model_type;
};

class HandGestureRecognizer {
private:
    HandGestureCls m_cls;

public:
    HandGestureRecognizer(HandGestureCls::model_type_t model_type =
                              static_cast<HandGestureCls::model_type_t>(CONFIG_DEFAULT_HAND_GESTURE_CLS_MODEL));

    std::vector<dl::cls::result_t> recognize(const dl::image::img_t &img,
                                             const std::list<dl::detect::result_t> &detect_res);
};
