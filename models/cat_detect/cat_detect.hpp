#pragma once
#include "dl_detect_base.hpp"
#include "dl_detect_espdet_postprocessor.hpp"

namespace cat_detect {
class ESPDet : public dl::detect::DetectImpl {
public:
    static inline constexpr float default_score_thr = 0.6;
    static inline constexpr float default_nms_thr = 0.7;
    ESPDet(const char *model_name, float score_thr, float nms_thr);
};
} // namespace cat_detect

class CatDetect : public dl::detect::DetectWrapper {
public:
    typedef enum {
        ESPDET_PICO_224_224_CAT,
        ESPDET_PICO_416_416_CAT,
    } model_type_t;
    CatDetect(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_CAT_DETECT_MODEL),
              bool lazy_load = true);

private:
    void load_model() override;

    model_type_t m_model_type;
};
