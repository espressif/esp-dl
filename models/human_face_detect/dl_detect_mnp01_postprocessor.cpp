#include "dl_detect_mnp01_postprocessor.hpp"
#include "dl_math.hpp"
#include <algorithm>
#include <cmath>

namespace dl {
namespace detect {
template <typename feature_t>
void MNP01Postprocessor<feature_t>::parse_stage(TensorBase *score,
                                                TensorBase *box,
                                                TensorBase *landmark,
                                                const int stage_index)
{
    std::vector<std::vector<int>> &anchor_shape = this->stages[stage_index].anchor_shape;

    int H = score->shape[1];
    int W = score->shape[2];
    int A = anchor_shape.size();
    int C = score->shape[3] / A;
    feature_t *score_element = (feature_t *)score->get_element_ptr();
    feature_t *box_element = (feature_t *)box->get_element_ptr();
    feature_t *landmark_element = (feature_t *)landmark->get_element_ptr();

    for (size_t y = 0; y < H; y++) // height
    {
        for (size_t x = 0; x < W; x++) // width
        {
            for (size_t a = 0; a < A; a++) // anchor number
            {
                // softmax
                float scores[C];
                scores[0] = score_element[0] * DL_SCALE(score->exponent);
                float max_score = scores[0];
                int max_score_c = 0;
                for (int i = 1; i < C; i++) {
                    scores[i] = score_element[i] * DL_SCALE(score->exponent);
                    if (max_score < scores[i]) {
                        max_score_c = i;
                        max_score = scores[i];
                    }
                }
                float sum = 0;
                for (int i = 0; i < C; i++) {
                    sum += expf(scores[i] - max_score);
                }
                max_score = 1. / sum;

                if (max_score > score_threshold) {
                    int anchor_h = anchor_shape[a][0];
                    int anchor_w = anchor_shape[a][1];
                    result_t new_box = {
                        max_score_c,
                        max_score,
                        {(int)(anchor_w * box_element[0] * DL_SCALE(box->exponent) / this->resize_scale_x +
                               this->top_left_x),
                         (int)(anchor_h * box_element[1] * DL_SCALE(box->exponent) / this->resize_scale_y +
                               this->top_left_y),
                         (int)((anchor_w * box_element[2] * DL_SCALE(box->exponent) + anchor_w) / this->resize_scale_x +
                               this->top_left_x),
                         (int)((anchor_h * box_element[3] * DL_SCALE(box->exponent) + anchor_h) / this->resize_scale_y +
                               this->top_left_y)},
                        {
                            (int)(anchor_w * landmark_element[0] * DL_SCALE(landmark->exponent) / this->resize_scale_x +
                                  this->top_left_x),
                            (int)(anchor_h * landmark_element[1] * DL_SCALE(landmark->exponent) / this->resize_scale_y +
                                  this->top_left_y),
                            (int)(anchor_w * landmark_element[2] * DL_SCALE(landmark->exponent) / this->resize_scale_x +
                                  this->top_left_x),
                            (int)(anchor_h * landmark_element[3] * DL_SCALE(landmark->exponent) / this->resize_scale_y +
                                  this->top_left_y),
                            (int)(anchor_w * landmark_element[4] * DL_SCALE(landmark->exponent) / this->resize_scale_x +
                                  this->top_left_x),
                            (int)(anchor_h * landmark_element[5] * DL_SCALE(landmark->exponent) / this->resize_scale_y +
                                  this->top_left_y),
                            (int)(anchor_w * landmark_element[6] * DL_SCALE(landmark->exponent) / this->resize_scale_x +
                                  this->top_left_x),
                            (int)(anchor_h * landmark_element[7] * DL_SCALE(landmark->exponent) / this->resize_scale_y +
                                  this->top_left_y),
                            (int)(anchor_w * landmark_element[8] * DL_SCALE(landmark->exponent) / this->resize_scale_x +
                                  this->top_left_x),
                            (int)(anchor_h * landmark_element[9] * DL_SCALE(landmark->exponent) / this->resize_scale_y +
                                  this->top_left_y),
                        }};

                    this->box_list.insert(
                        std::upper_bound(this->box_list.begin(), this->box_list.end(), new_box, compare_greater_box),
                        new_box);
                }
                score_element += C;
                box_element += 4;
                landmark_element += 10;
            }
        }
    }
}

template <typename feature_t>
void MNP01Postprocessor<feature_t>::postprocess()
{
    TensorBase *score = this->get_model_output("score");
    TensorBase *bbox = this->get_model_output("box");
    TensorBase *landmark = this->get_model_output("landmark");
    this->parse_stage(score, bbox, landmark, 0);
}

template class MNP01Postprocessor<int8_t>;
template class MNP01Postprocessor<int16_t>;
} // namespace detect
} // namespace dl
