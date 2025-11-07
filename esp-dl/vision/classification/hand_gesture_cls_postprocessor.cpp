#include "hand_gesture_cls_postprocessor.hpp"
#include "hand_gesture_category_name.hpp"

namespace dl {
namespace cls {
HandGestureClsPostprocessor::HandGestureClsPostprocessor(
    Model *model, const int top_k, const float score_thr, bool need_softmax, const std::string &output_name) :
    ClsPostprocessor(model, top_k, score_thr, need_softmax, output_name)
{
    m_cat_names = hand_gesture_cat_names;
}
} // namespace cls
} // namespace dl
