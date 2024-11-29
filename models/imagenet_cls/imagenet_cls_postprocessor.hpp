#pragma once
#include "dl_cls_postprocessor.hpp"

namespace dl {
namespace cls {
class ImageNetClsPostprocessor : public ClsPostprocessor {
public:
    ImageNetClsPostprocessor(
        Model *model, const float score_thr, const int top_k, bool need_softmax, const std::string &output_name = "");
};
} // namespace cls
} // namespace dl
