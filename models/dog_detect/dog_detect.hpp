#pragma once
#include "dl_detect_base.hpp"
#include "dl_detect_espdet_postprocessor.hpp"

namespace dog_detect {
class ESPDet : public dl::detect::DetectImpl {
public:
    ESPDet(const char *model_name);
};
} // namespace dog_detect

class DogDetect : public dl::detect::DetectWrapper {
public:
    typedef enum {
        ESPDET_PICO_224_224_DOG,
        ESPDET_PICO_416_416_DOG,
    } model_type_t;
    DogDetect(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_DOG_DETECT_MODEL));
};
