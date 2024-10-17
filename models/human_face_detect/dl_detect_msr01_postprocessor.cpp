#include "dl_detect_msr01_postprocessor.hpp"
#include "dl_math.hpp"
#include <algorithm>
#include <cmath>

namespace dl {
namespace detect {
template <typename feature_t>
void MSR01Postprocessor<feature_t>::parse_stage(TensorBase *score, TensorBase *box, const int stage_index)
{
    int stride_y = this->stages[stage_index].stride_y;
    int stride_x = this->stages[stage_index].stride_x;

    int offset_y = this->stages[stage_index].offset_y;
    int offset_x = this->stages[stage_index].offset_x;

    std::vector<std::vector<int>> &anchor_shape = this->stages[stage_index].anchor_shape;

    int H = score->shape[1];
    int W = score->shape[2];
    int A = anchor_shape.size();
    int C = score->shape[3] / A;
    feature_t *score_element = (feature_t *)score->get_element_ptr();
    feature_t score_threshold_quant;
    if (std::is_same<feature_t, int8_t>::value) {
        score_threshold_quant = (feature_t)DL_CLIP(
            tool::round(dl::math::inverse_sigmoid(this->score_threshold) / DL_SCALE(score->exponent)), -128, 127);
    } else {
        score_threshold_quant = (feature_t)DL_CLIP(
            tool::round(dl::math::inverse_sigmoid(this->score_threshold) / DL_SCALE(score->exponent)), -32768, 32767);
    }

    feature_t *box_element = (feature_t *)box->get_element_ptr();

    for (size_t y = 0; y < H; y++) // height
    {
        for (size_t x = 0; x < W; x++) // width
        {
            for (size_t a = 0; a < A; a++) // anchor number
            {
                for (size_t c = 0; c < C; c++) // category number
                {
                    if (*score_element > score_threshold_quant) {
                        int center_y = y * stride_y + offset_y;
                        int center_x = x * stride_x + offset_x;
                        int anchor_h = anchor_shape[a][0];
                        int anchor_w = anchor_shape[a][1];
                        result_t new_box = {
                            (int)c,
                            dl::math::sigmoid(*score_element * DL_SCALE(score->exponent)),
                            {(int)((center_x - (anchor_w >> 1) + anchor_w * box_element[0] * DL_SCALE(box->exponent)) /
                                   this->resize_scale_x),
                             (int)((center_y - (anchor_h >> 1) + anchor_h * box_element[1] * DL_SCALE(box->exponent)) /
                                   this->resize_scale_y),
                             (int)((center_x + anchor_w - (anchor_w >> 1) +
                                    anchor_w * box_element[2] * DL_SCALE(box->exponent)) /
                                   this->resize_scale_x),
                             (int)((center_y + anchor_h - (anchor_h >> 1) +
                                    anchor_h * box_element[3] * DL_SCALE(box->exponent)) /
                                   this->resize_scale_y)},
                            {0}};

                        this->box_list.insert(
                            std::upper_bound(
                                this->box_list.begin(), this->box_list.end(), new_box, compare_greater_box),
                            new_box);
                    }
                    score_element++;
                    box_element += 4;
                }
            }
        }
    }
}

template <typename feature_t>
void MSR01Postprocessor<feature_t>::postprocess()
{
    TensorBase *score0 = this->get_model_output("score0");
    TensorBase *bbox0 = this->get_model_output("box0");
    TensorBase *score1 = this->get_model_output("score1");
    TensorBase *bbox1 = this->get_model_output("box1");

    this->parse_stage(score0, bbox0, 0);
    this->parse_stage(score1, bbox1, 1);
    this->nms();
}

template class MSR01Postprocessor<int8_t>;
template class MSR01Postprocessor<int16_t>;
} // namespace detect
} // namespace dl
