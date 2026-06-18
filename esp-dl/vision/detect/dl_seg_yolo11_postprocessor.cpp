#include "dl_seg_yolo11_postprocessor.hpp"
#include "dl_math.hpp"
#include <algorithm>
#include <cmath>

namespace dl {
namespace detect {

template <typename T>
void yolo11segPostProcessor::parse_stage(TensorBase *score, TensorBase *box, TensorBase *mc, const int stage_index)
{
    int stride_y = m_stages[stage_index].stride_y;
    int stride_x = m_stages[stage_index].stride_x;
    int offset_y = m_stages[stage_index].offset_y;
    int offset_x = m_stages[stage_index].offset_x;

    int H = score->shape[1];
    int W = score->shape[2];
    int C = score->shape[3];

    T *score_ptr = (T *)score->data;
    T *box_ptr = (T *)box->data;
    T *mc_ptr = (T *)mc->data;

    float score_exp = DL_SCALE(score->exponent);
    float box_exp = DL_SCALE(box->exponent);
    float mc_exp = DL_SCALE(mc->exponent);
    T score_thr_quant = quantize<T>(dl::math::inverse_sigmoid(m_score_thr), 1.f / score_exp);
    float inv_resize_scale_x = m_image_preprocessor->get_resize_scale_x(true);
    float inv_resize_scale_y = m_image_preprocessor->get_resize_scale_y(true);
    int border_left = m_image_preprocessor->get_border_left();
    int border_top = m_image_preprocessor->get_border_top();

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

                    std::vector<float> coeff(m_nm);
                    for (int k = 0; k < m_nm; k++) {
                        coeff[k] = dequantize(mc_ptr[k], mc_exp);
                    }

                    result_t new_box = {
                        (int)c,
                        dl::math::sigmoid(dequantize(*score_ptr, score_exp)),
                        {(int)(((center_x - dl::math::dfl_integral(box_data, reg_max - 1) * stride_x) - border_left) *
                               inv_resize_scale_x),
                         (int)(((center_y - dl::math::dfl_integral(box_data + reg_max, reg_max - 1) * stride_y) -
                                border_top) *
                               inv_resize_scale_y),
                         (int)(((center_x + dl::math::dfl_integral(box_data + 2 * reg_max, reg_max - 1) * stride_x) -
                                border_left) *
                               inv_resize_scale_x),
                         (int)(((center_y + dl::math::dfl_integral(box_data + 3 * reg_max, reg_max - 1) * stride_y) -
                                border_top) *
                               inv_resize_scale_y)},
                        {},
                        std::move(coeff),
                        {},
                    };

                    m_box_list.insert(std::upper_bound(m_box_list.begin(), m_box_list.end(), new_box, greater_box),
                                      new_box);
                }
                score_ptr++;
            }

            box_ptr += 4 * reg_max;
            mc_ptr += m_nm;
        }
    }
}

template <typename T>
void yolo11segPostProcessor::synthesize_masks(TensorBase *proto)
{
    if (m_box_list.empty()) {
        return;
    }

    int proto_h = proto->shape[1];
    int proto_w = proto->shape[2];
    int proto_c = proto->shape[3];
    if (proto_c != m_nm) {
        return;
    }

    TensorBase *model_input = m_image_preprocessor->get_model_input();
    int model_h = model_input->shape[1];
    int model_w = model_input->shape[2];

    float proto_exp = DL_SCALE(proto->exponent);
    T *proto_ptr = (T *)proto->data;

    float inv_resize_scale_x = m_image_preprocessor->get_resize_scale_x(true);
    float inv_resize_scale_y = m_image_preprocessor->get_resize_scale_y(true);
    int border_left = m_image_preprocessor->get_border_left();
    int border_top = m_image_preprocessor->get_border_top();
    float scale_to_model_x = 1.f / inv_resize_scale_x;
    float scale_to_model_y = 1.f / inv_resize_scale_y;
    float ratio_x = (float)proto_w / (float)model_w;
    float ratio_y = (float)proto_h / (float)model_h;

    for (result_t &res : m_box_list) {
        if (res.mask_coeff.size() != (size_t)m_nm) {
            continue;
        }

        int x1_model = (int)(res.box[0] * scale_to_model_x + border_left);
        int y1_model = (int)(res.box[1] * scale_to_model_y + border_top);
        int x2_model = (int)(res.box[2] * scale_to_model_x + border_left);
        int y2_model = (int)(res.box[3] * scale_to_model_y + border_top);

        int x1_proto = DL_CLIP((int)(x1_model * ratio_x), 0, proto_w - 1);
        int y1_proto = DL_CLIP((int)(y1_model * ratio_y), 0, proto_h - 1);
        int x2_proto = DL_CLIP((int)(x2_model * ratio_x), 0, proto_w - 1);
        int y2_proto = DL_CLIP((int)(y2_model * ratio_y), 0, proto_h - 1);

        if (x2_proto <= x1_proto || y2_proto <= y1_proto) {
            continue;
        }

        int mask_w = x2_proto - x1_proto;
        int mask_h = y2_proto - y1_proto;
        res.mask.assign(mask_w * mask_h, 0);

        for (int py = y1_proto; py < y2_proto; py++) {
            for (int px = x1_proto; px < x2_proto; px++) {
                float sum = 0.f;
                int proto_base = (py * proto_w + px) * proto_c;
                for (int k = 0; k < m_nm; k++) {
                    sum += res.mask_coeff[k] * dequantize(proto_ptr[proto_base + k], proto_exp);
                }
                res.mask[(py - y1_proto) * mask_w + (px - x1_proto)] = sum > 0.f ? 1 : 0;
            }
        }

        int out_w = res.box[2] - res.box[0];
        int out_h = res.box[3] - res.box[1];
        if (out_w <= 0 || out_h <= 0) {
            res.mask.clear();
            continue;
        }

        std::vector<uint8_t> upsampled(out_w * out_h);
        for (int oy = 0; oy < out_h; oy++) {
            int sy = oy * mask_h / out_h;
            for (int ox = 0; ox < out_w; ox++) {
                int sx = ox * mask_w / out_w;
                upsampled[oy * out_w + ox] = res.mask[sy * mask_w + sx];
            }
        }
        res.mask = std::move(upsampled);
    }
}

template void yolo11segPostProcessor::parse_stage<int8_t>(TensorBase *score,
                                                          TensorBase *box,
                                                          TensorBase *mc,
                                                          const int stage_index);
template void yolo11segPostProcessor::parse_stage<int16_t>(TensorBase *score,
                                                           TensorBase *box,
                                                           TensorBase *mc,
                                                           const int stage_index);
template void yolo11segPostProcessor::synthesize_masks<int8_t>(TensorBase *proto);
template void yolo11segPostProcessor::synthesize_masks<int16_t>(TensorBase *proto);

void yolo11segPostProcessor::postprocess()
{
    TensorBase *bbox0 = m_model->get_output("box0");
    TensorBase *score0 = m_model->get_output("score0");
    TensorBase *bbox1 = m_model->get_output("box1");
    TensorBase *score1 = m_model->get_output("score1");
    TensorBase *bbox2 = m_model->get_output("box2");
    TensorBase *score2 = m_model->get_output("score2");
    TensorBase *mc0 = m_model->get_output("mc0");
    TensorBase *mc1 = m_model->get_output("mc1");
    TensorBase *mc2 = m_model->get_output("mc2");
    TensorBase *proto = m_model->get_output("p");

    if (bbox0->dtype == DATA_TYPE_INT8) {
        parse_stage<int8_t>(score0, bbox0, mc0, 0);
        parse_stage<int8_t>(score1, bbox1, mc1, 1);
        parse_stage<int8_t>(score2, bbox2, mc2, 2);
    } else {
        parse_stage<int16_t>(score0, bbox0, mc0, 0);
        parse_stage<int16_t>(score1, bbox1, mc1, 1);
        parse_stage<int16_t>(score2, bbox2, mc2, 2);
    }

    nms();

    // Clip boxes to source image before mask synthesis, so res.mask.size() == box_w * box_h
    // matches the box coordinates returned by get_result().
    const dl::image::img_t &src_img = m_image_preprocessor->get_src_img();
    for (result_t &res : m_box_list) {
        res.limit_box(src_img.width, src_img.height);
    }

    if (proto->dtype == DATA_TYPE_INT8) {
        synthesize_masks<int8_t>(proto);
    } else {
        synthesize_masks<int16_t>(proto);
    }
}

} // namespace detect
} // namespace dl
