#pragma once
#include "dl_detect_base.hpp"
#include "dl_detect_espdet_postprocessor.hpp"

namespace cat_detect {
class ESPDet : public dl::detect::DetectImpl {
public:
    ESPDet(const char *model_name);
};
} // namespace cat_detect

class CatDetect : public dl::detect::DetectWrapper {
public:
    typedef enum {
        ESPDET_PICO_224_224_CAT,
        ESPDET_PICO_416_416_CAT,
    } model_type_t;
    CatDetect(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_CAT_DETECT_MODEL));
};
