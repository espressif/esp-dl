#pragma once

#include "dl_cls_base.hpp"
#include "imagenet_cls_postprocessor.hpp"

namespace imagenet_cls {

class MobileNetV2 : public dl::cls::ClsImpl {
public:
    static inline constexpr int default_topk = 5;
    static inline constexpr float default_score_thr = std::numeric_limits<float>::lowest();
    MobileNetV2(const char *model_name, int topk, float score_thr);
};

} // namespace imagenet_cls

class ImageNetCls : public dl::cls::ClsWrapper {
public:
    typedef enum { MOBILENETV2_S8_V1 } model_type_t;
    ImageNetCls(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_IMAGENET_CLS_MODEL),
                bool lazy_load = true);

private:
    void load_model();

    model_type_t m_model_type;
};
