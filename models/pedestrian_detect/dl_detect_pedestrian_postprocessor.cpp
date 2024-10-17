#include "dl_detect_pedestrian_postprocessor.hpp"
#include "dl_math.hpp"
#include <algorithm>
#include <cmath>

namespace dl {
namespace detect {
template <typename feature_t>
void PedestrianPostprocessor<feature_t>::parse_stage(TensorBase *score, TensorBase *box, const int stage_index)
{
    int stride_y = this->stages[stage_index].stride_y;
    int stride_x = this->stages[stage_index].stride_x;

    int offset_y = this->stages[stage_index].offset_y;
    int offset_x = this->stages[stage_index].offset_x;

    int H = score->shape[1];
    int W = score->shape[2];
    int C = score->shape[3];
    feature_t *score_element = (feature_t *)score->get_element_ptr();
    feature_t score_threshold_quant;
    if (std::is_same<feature_t, int8_t>::value) {
        score_threshold_quant = (feature_t)DL_CLIP(
            tool::round(this->score_threshold * this->score_threshold / DL_SCALE(score->exponent)), -128, 127);
    } else {
        score_threshold_quant = (feature_t)DL_CLIP(
            tool::round(this->score_threshold * this->score_threshold / DL_SCALE(score->exponent)), -32768, 32767);
    }

    feature_t *box_element = (feature_t *)box->get_element_ptr();

    for (size_t y = 0; y < H; y++) // height
    {
        for (size_t x = 0; x < W; x++) // width
        {
            for (size_t c = 0; c < C; c++) // category number
            {
                if (*score_element > score_threshold_quant) {
                    int center_y = y * stride_y + offset_y;
                    int center_x = x * stride_x + offset_x;

                    float box_element_f[32];
                    for (int i = 0; i < 32; i++) {
                        box_element_f[i] = box_element[i] * DL_SCALE(box->exponent);
                    }

                    result_t new_box = {
                        (int)c,
                        (float)sqrt(*score_element * DL_SCALE(score->exponent)),
                        {(int)((center_x - dl::math::dfl_integral(box_element_f, 7) * stride_x) / this->resize_scale_x),
                         (int)((center_y - dl::math::dfl_integral(box_element_f + 8, 7) * stride_y) /
                               this->resize_scale_y),
                         (int)((center_x + dl::math::dfl_integral(box_element_f + 16, 7) * stride_x) /
                               this->resize_scale_x),
                         (int)((center_y + dl::math::dfl_integral(box_element_f + 24, 7) * stride_y) /
                               this->resize_scale_y)},
                        {0}};

                    this->box_list.insert(
                        std::upper_bound(this->box_list.begin(), this->box_list.end(), new_box, compare_greater_box),
                        new_box);
                }
            }
            score_element++;
            box_element += 32;
        }
    }
}

template <typename feature_t>
void PedestrianPostprocessor<feature_t>::postprocess()
{
    TensorBase *score0 = this->get_model_output("score0");
    TensorBase *bbox0 = this->get_model_output("bbox0");
    TensorBase *score1 = this->get_model_output("score1");
    TensorBase *bbox1 = this->get_model_output("bbox1");
    TensorBase *score2 = this->get_model_output("score2");
    TensorBase *bbox2 = this->get_model_output("bbox2");

    this->parse_stage(score0, bbox0, 0);
    this->parse_stage(score1, bbox1, 1);
    this->parse_stage(score2, bbox2, 2);
    this->nms();
}

template class PedestrianPostprocessor<int8_t>;
template class PedestrianPostprocessor<int16_t>;
} // namespace detect
} // namespace dl
