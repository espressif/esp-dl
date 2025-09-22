#pragma once

#include "dl_detect_base.hpp"
#include "dl_detect_pico_postprocessor.hpp"

namespace pedestrian_detect {
class Pico : public dl::detect::DetectImpl {
public:
    static inline constexpr float default_score_thr = 0.7;
    static inline constexpr float default_nms_thr = 0.5;
    Pico(const char *model_name, float score_thr, float nms_thr);
};
} // namespace pedestrian_detect

class PedestrianDetect : public dl::detect::DetectWrapper {
public:
    typedef enum { PICO_S8_V1 } model_type_t;
    PedestrianDetect(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_PEDESTRIAN_DETECT_MODEL),
                     bool lazy_load = true);

private:
    void load_model() override;

    model_type_t m_model_type;
};
