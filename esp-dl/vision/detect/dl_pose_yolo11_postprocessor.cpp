#include "dl_pose_yolo11_postprocessor.hpp"
#include "dl_math.hpp"
#include <algorithm>
#include <cmath>

namespace dl {
namespace detect {
template <typename T>
void yolo11posePostProcessor::parse_stage(TensorBase *score, TensorBase *box, TensorBase *kpt, const int stage_index)
{
    int stride_y = m_stages[stage_index].stride_y;
    int stride_x = m_stages[stage_index].stride_x;

    int offset_y = m_stages[stage_index].offset_y;
    int offset_x = m_stages[stage_index].offset_x;

    int H = score->shape[1];
    int W = score->shape[2];
    int C = score->shape[3];

    int coco_kpt_num = 17;
    int coco_kpt_ch = 3; //(x, y, visibility)
    int coco_kpt_total = coco_kpt_num * coco_kpt_ch;
    int coco_kpt_res_total = coco_kpt_num * 2; //(x, y)
    float coco_kpt_conf_th = 0.5;

    T *score_ptr = (T *)score->data;
    T *box_ptr = (T *)box->data;
    T *kpt_ptr = (T *)kpt->data;

    float score_exp = DL_SCALE(score->exponent);
    float box_exp = DL_SCALE(box->exponent);
    float kpt_exp = DL_SCALE(kpt->exponent);

    T score_thr_quant = quantize<T>(dl::math::inverse_sigmoid(m_score_thr), 1.f / score_exp);
    float inv_resize_scale_x = 1.f / m_resize_scale_x;
    float inv_resize_scale_y = 1.f / m_resize_scale_y;

    int reg_max = 16;

    for (size_t y = 0; y < H; y++) {
        for (size_t x = 0; x < W; x++) {
            for (size_t c = 0; c < C; c++) {
                if (*score_ptr > score_thr_quant) {
                    int center_y = y * stride_y + offset_y;
                    int center_x = x * stride_x + offset_x;

                    float box_data[reg_max * 4];
                    for (int i = 0; i < reg_max * 4; i++) {
                        box_data[i] = dequantize(box_ptr[i], box_exp);
                    }

                    std::vector<int> keypoints_vec(coco_kpt_res_total);
                    for (int k = 0; k < coco_kpt_num; k++) {
                        int idx = k * coco_kpt_ch;
                        float kpt_x = dequantize(kpt_ptr[idx], kpt_exp);
                        float kpt_y = dequantize(kpt_ptr[idx + 1], kpt_exp);
                        float kpt_conf = dequantize(kpt_ptr[idx + 2], kpt_exp);

                        if (kpt_conf >= coco_kpt_conf_th) {
                            keypoints_vec[2 * k] =
                                static_cast<int>((kpt_x * 2.0 * stride_x + (center_x - offset_x)) * inv_resize_scale_x);
                            keypoints_vec[2 * k + 1] =
                                static_cast<int>((kpt_y * 2.0 * stride_y + (center_y - offset_y)) * inv_resize_scale_y);
                        } else {
                            keypoints_vec[2 * k] = 0;
                            keypoints_vec[2 * k + 1] = 0;
                        }
                    }

                    result_t new_box = {
                        (int)c,
                        dl::math::sigmoid(dequantize(*score_ptr, score_exp)),
                        {(int)((center_x - dl::math::dfl_integral(box_data, reg_max - 1) * stride_x) *
                               inv_resize_scale_x),
                         (int)((center_y - dl::math::dfl_integral(box_data + reg_max, reg_max - 1) * stride_y) *
                               inv_resize_scale_y),
                         (int)((center_x + dl::math::dfl_integral(box_data + 2 * reg_max, reg_max - 1) * stride_x) *
                               inv_resize_scale_x),
                         (int)((center_y + dl::math::dfl_integral(box_data + 3 * reg_max, reg_max - 1) * stride_y) *
                               inv_resize_scale_y)},
                        keypoints_vec,
                    };

                    m_box_list.insert(std::upper_bound(m_box_list.begin(), m_box_list.end(), new_box, greater_box),
                                      new_box);
                }
                score_ptr++;
            }

            box_ptr += 4 * reg_max;
            kpt_ptr += coco_kpt_total;
        }
    }
}

template void yolo11posePostProcessor::parse_stage<int8_t>(TensorBase *score,
                                                           TensorBase *box,
                                                           TensorBase *kpt,
                                                           const int stage_index);
template void yolo11posePostProcessor::parse_stage<int16_t>(TensorBase *score,
                                                            TensorBase *box,
                                                            TensorBase *kpt,
                                                            const int stage_index);

void yolo11posePostProcessor::postprocess()
{
    TensorBase *bbox0 = m_model->get_output("box0");
    TensorBase *score0 = m_model->get_output("score0");

    TensorBase *bbox1 = m_model->get_output("box1");
    TensorBase *score1 = m_model->get_output("score1");

    TensorBase *bbox2 = m_model->get_output("box2");
    TensorBase *score2 = m_model->get_output("score2");

    TensorBase *kpt0 = m_model->get_output("kpt0");
    TensorBase *kpt1 = m_model->get_output("kpt1");
    TensorBase *kpt2 = m_model->get_output("kpt2");

    if (bbox0->dtype == DATA_TYPE_INT8) {
        parse_stage<int8_t>(score0, bbox0, kpt0, 0);
        parse_stage<int8_t>(score1, bbox1, kpt1, 1);
        parse_stage<int8_t>(score2, bbox2, kpt2, 2);
    } else {
        parse_stage<int16_t>(score0, bbox0, kpt0, 0);
        parse_stage<int16_t>(score1, bbox1, kpt1, 1);
        parse_stage<int16_t>(score2, bbox2, kpt2, 2);
    }
    nms();
}
} // namespace detect
} // namespace dl
