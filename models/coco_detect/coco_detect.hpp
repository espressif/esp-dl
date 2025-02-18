#pragma once
#include "dl_detect_base.hpp"
#include "dl_detect_yolo11_postprocessor.hpp"

namespace yolo11_detect {
class Yolo11 : public dl::detect::DetectImpl {
public:
    Yolo11(const char *model_name);
};
} // namespace yolo11_detect

class COCODetect : public dl::detect::DetectWrapper {
public:
    typedef enum {
        YOLO11_n_S8_V1,
        YOLO11_n_S8_V2,
    } model_type_t;
    COCODetect(model_type_t model_type = static_cast<model_type_t>(CONFIG_YOLO11_DETECT_MODEL_TYPE));
};
