#pragma once

#include <vector>
#include <list>
#include <algorithm>
#include <math.h>

#include "dl_variable.hpp"
#include "dl_image.hpp"
#include "dl_define.hpp"
#include "dl_tool.hpp"

typedef struct
{
    int category;
    float score;
    std::vector<int> box;
    std::vector<int> keypoint;
} detection_prediction_t;

inline float __fast_exp(double x, int steps)
{
    x = 1.0 + x / (1 << steps);
    for (int i = 0; i < steps; i++)
        x *= x;
    return x;
}

inline bool compare_greater_box(detection_prediction_t a, detection_prediction_t b)
{
    return a.score > b.score;
}

template <typename model_input_t, typename model_output_t> 
class Detector
{
private:
public:
    const std::vector<int> input_shape;
    const float score_threshold;
    const float nms_threshold;
    const bool with_keypoint;
    const int top_k;

    float resize_scale_y;
    float resize_scale_x;
    std::list<detection_prediction_t> box_list;
    dl::Feature<model_input_t> resized_input;
    Detector(std::vector<int> input_shape,
             const float resize_scale,
             const float score_threshold,
             const float nms_threshold,
             const bool with_keypoint,
             const int top_k) : input_shape(input_shape),
                                score_threshold(score_threshold),
                                nms_threshold(nms_threshold),
                                with_keypoint(with_keypoint),
                                top_k(top_k)
    {
        int resized_y = int(input_shape[0] * resize_scale + 0.5);
        int resized_x = int(input_shape[1] * resize_scale + 0.5);

        this->resize_scale_y = (float)input_shape[0] / resized_y;
        this->resize_scale_x = (float)input_shape[1] / resized_x;

        this->resized_input.set_shape({resized_y, resized_x, input_shape[2]});
    }
    ~Detector() {}

    virtual void parse_stage(dl::Feature<model_output_t> &score, dl::Feature<model_output_t> &box, const int stage_index) = 0;
    virtual void parse_stage(dl::Feature<model_output_t> &score, dl::Feature<model_output_t> &box, dl::Feature<model_output_t> &keypoint, const int stage_index) = 0;
    virtual void call() = 0;

    template <typename T>
    std::list<detection_prediction_t> &infer(T *input)
    {
#if CONFIG_PRINT_DETECTOR_LATENCY
        dl::tool::Latency latency;
        latency.start();
#endif
        // resize
        this->resized_input.calloc_element();
        dl::image::resize_image_to_rgb888(this->resized_input.element,
                                          this->resized_input.padding[0],
                                          this->resized_input.padding[0] + this->resized_input.shape[0],
                                          this->resized_input.padding[2],
                                          this->resized_input.padding[2] + this->resized_input.shape[1],
                                          this->resized_input.shape[2],
                                          input,
                                          this->input_shape[0],
                                          this->input_shape[1],
                                          this->resized_input.shape[1],
                                          0,
                                          dl::image::IMAGE_RESIZE_NEAREST);
#if CONFIG_PRINT_DETECTOR_LATENCY
        latency.end();
        latency.print("Resize");
        latency.start();
#endif

        // call
        this->box_list.clear();
        this->call();

        // NMS
#if CONFIG_PRINT_DETECTOR_LATENCY
        latency.end();
        latency.print("Call");
        latency.start();
#endif
        int kept_number = 0;
        for (std::list<detection_prediction_t>::iterator kept = this->box_list.begin(); kept != this->box_list.end(); kept++)
        {
            kept_number++;

            if (kept_number >= this->top_k)
            {
                this->box_list.erase(++kept, this->box_list.end());
                break;
            }

            int kept_area = (kept->box[2] - kept->box[0] + 1) * (kept->box[3] - kept->box[1] + 1);

            std::list<detection_prediction_t>::iterator other = kept;
            other++;
            for (; other != this->box_list.end();)
            {
                int inter_lt_x = DL_MAX(kept->box[0], other->box[0]);
                int inter_lt_y = DL_MAX(kept->box[1], other->box[1]);
                int inter_rb_x = DL_MIN(kept->box[2], other->box[2]);
                int inter_rb_y = DL_MIN(kept->box[3], other->box[3]);

                int inter_height = inter_rb_y - inter_lt_y + 1;
                int inter_width = inter_rb_x - inter_lt_x + 1;

                if (inter_height > 0 && inter_width > 0)
                {
                    int other_area = (other->box[2] - other->box[0] + 1) * (other->box[3] - other->box[1] + 1);
                    int inter_area = inter_height * inter_width;
                    float iou = (float)inter_area / (kept_area + other_area - inter_area);
                    if (iou > nms_threshold)
                    {
                        other = this->box_list.erase(other);
                        continue;
                    }
                }
                other++;
            }
        }
#if CONFIG_PRINT_DETECTOR_LATENCY
        latency.end();
        latency.print("NMS");
#endif
        return this->box_list;
    }
};

typedef struct
{
    int stride_y;
    int stride_x;
    int offset_y;
    int offset_x;
    int min_input_size;
    std::vector<std::vector<int>> anchor_shape;
} detector_anchor_box_stage_t;

template <typename model_input_t, typename model_output_t>
class DetectorAnchorBox : public Detector<model_input_t, model_output_t>
{
private:
public:
    std::vector<detector_anchor_box_stage_t> stages;
    DetectorAnchorBox(std::vector<int> input_shape,
                      const float resize_scale,
                      const float score_threshold,
                      const float nms_threshold,
                      const bool with_keypoint,
                      const int top_k,
                      const std::vector<detector_anchor_box_stage_t> stages) : Detector<model_input_t, model_output_t>(input_shape, resize_scale, score_threshold, nms_threshold, with_keypoint, top_k),
                                                                               stages(stages) {}
    ~DetectorAnchorBox() {}

    void parse_stage(dl::Feature<model_output_t> &score, dl::Feature<model_output_t> &box, const int stage_index)
    {
        int stride_y = this->stages[stage_index].stride_y;
        int stride_x = this->stages[stage_index].stride_x;

        int offset_y = this->stages[stage_index].offset_y;
        int offset_x = this->stages[stage_index].offset_x;

        std::vector<std::vector<int>> &anchor_shape = this->stages[stage_index].anchor_shape;

        int H = score.shape[0];
        int W = score.shape[1];
        int A = anchor_shape.size();
        int C = score.shape[2] / A;
        model_output_t *score_element = score.element;
        float score_precision = 1.0f / (1 << -score.exponent);
        int score_threshold = (int)(0.5 - log(1.0 / this->score_threshold - 1) * (1 << -score.exponent));

        model_output_t *box_element = box.element;
        int box_shift = -box.exponent;

        for (size_t y = 0; y < H; y++) // height
        {
            for (size_t x = 0; x < W; x++) // width
            {
                for (size_t a = 0; a < A; a++) // anchor number
                {
                    model_output_t max_score = *score_element++;
                    int max_score_c = 0;
                    for (size_t c = 1; c < C; c++) // category number
                    {
                        if (max_score < *score_element)
                        {
                            max_score = *score_element;
                            max_score_c = c;
                        }
                        score_element++;
                    }

                    if (max_score > score_threshold)
                    {
                        int center_y = (y * stride_y + offset_y) * this->resize_scale_y;
                        int center_x = (x * stride_x + offset_x) * this->resize_scale_x;
                        int anchor_h = anchor_shape[a][0] * this->resize_scale_y;
                        int anchor_w = anchor_shape[a][1] * this->resize_scale_x;

                        detection_prediction_t new_box = {
                            max_score_c,
                            max_score * score_precision,
                            {center_x - anchor_w / 2 + ((anchor_w * box_element[0]) >> box_shift),
                             center_y - anchor_h / 2 + ((anchor_h * box_element[1]) >> box_shift),
                             center_x + anchor_w / 2 + ((anchor_w * box_element[2]) >> box_shift),
                             center_y + anchor_h / 2 + ((anchor_h * box_element[3]) >> box_shift)},
                            {0}};

                        this->box_list.insert(std::upper_bound(this->box_list.begin(), this->box_list.end(), new_box, compare_greater_box), new_box);
                    }
                    box_element += 4;
                }
            }
        }
    }

    void parse_stage(dl::Feature<model_output_t> &score, dl::Feature<model_output_t> &box, dl::Feature<model_output_t> &keypoint, const int stage_index)
    {
        int stride_y = this->stages[stage_index].stride_y;
        int stride_x = this->stages[stage_index].stride_x;

        int offset_y = this->stages[stage_index].offset_y;
        int offset_x = this->stages[stage_index].offset_x;

        std::vector<std::vector<int>> &anchor_shape = this->stages[stage_index].anchor_shape;

        int H = score.shape[0];
        int W = score.shape[1];
        int A = anchor_shape.size();
        int C = score.shape[2] / A;
        model_output_t *score_element = score.element;
        float score_precision = 1.0f / (1 << -score.exponent);
        int score_threshold = (int)(0.5 - log(1.0 / this->score_threshold - 1) * (1 << -score.exponent));

        model_output_t *box_element = box.element;
        int box_shift = -box.exponent;

        model_output_t *keypoint_element = keypoint.element;
        int keypoint_shift = -keypoint.exponent;

        for (size_t y = 0; y < H; y++) // height
        {
            for (size_t x = 0; x < W; x++) // width
            {
                for (size_t a = 0; a < A; a++) // anchor number
                {
                    model_output_t max_score = *score_element++;
                    int max_score_c = 0;
                    for (size_t c = 1; c < C; c++) // category number
                    {
                        if (max_score < *score_element)
                        {
                            max_score = *score_element;
                            max_score_c = c;
                        }
                        score_element++;
                    }

                    if (max_score > score_threshold)
                    {
                        int center_y = (y * stride_y + offset_y) * this->resize_scale_y;
                        int center_x = (x * stride_x + offset_x) * this->resize_scale_x;
                        int anchor_h = anchor_shape[a][0] * this->resize_scale_y;
                        int anchor_w = anchor_shape[a][1] * this->resize_scale_x;

                        detection_prediction_t new_box = {
                            max_score_c,
                            max_score * score_precision,
                            {center_x - anchor_w / 2 + ((anchor_w * box_element[0]) >> box_shift),
                             center_y - anchor_h / 2 + ((anchor_h * box_element[1]) >> box_shift),
                             center_x + anchor_w / 2 + ((anchor_w * box_element[2]) >> box_shift),
                             center_y + anchor_h / 2 + ((anchor_h * box_element[3]) >> box_shift)},
                            {center_x - anchor_w / 2 + ((anchor_w * keypoint_element[0]) >> keypoint_shift),
                             center_y - anchor_h / 2 + ((anchor_h * keypoint_element[1]) >> keypoint_shift),
                             center_x - anchor_w / 2 + ((anchor_w * keypoint_element[2]) >> keypoint_shift),
                             center_y - anchor_h / 2 + ((anchor_h * keypoint_element[3]) >> keypoint_shift),
                             center_x - anchor_w / 2 + ((anchor_w * keypoint_element[4]) >> keypoint_shift),
                             center_y - anchor_h / 2 + ((anchor_h * keypoint_element[5]) >> keypoint_shift),
                             center_x - anchor_w / 2 + ((anchor_w * keypoint_element[6]) >> keypoint_shift),
                             center_y - anchor_h / 2 + ((anchor_h * keypoint_element[7]) >> keypoint_shift),
                             center_x - anchor_w / 2 + ((anchor_w * keypoint_element[8]) >> keypoint_shift),
                             center_y - anchor_h / 2 + ((anchor_h * keypoint_element[9]) >> keypoint_shift)}};

                        this->box_list.insert(std::upper_bound(this->box_list.begin(), this->box_list.end(), new_box, compare_greater_box), new_box);
                    }
                    box_element += 4;
                    keypoint_element += 10;
                }
            }
        }
    }
};

typedef struct
{
    int stride_y;
    int stride_x;
    int offset_y;
    int offset_x;
    int min_input_size;
} detector_anchor_point_stage_t;

template < typename model_input_t, typename model_output_t>
class DetectorAnchorPoint : public Detector< model_input_t, model_output_t>
{
private:
public:
    std::vector<detector_anchor_point_stage_t> stages;
    DetectorAnchorPoint(std::vector<int> input_shape,
                        const float resize_scale,
                        const float score_threshold,
                        const float nms_threshold,
                        const bool with_keypoint,
                        const int top_k,
                        const std::vector<detector_anchor_point_stage_t> stages) : Detector< model_input_t, model_output_t>(input_shape, resize_scale, score_threshold, nms_threshold, with_keypoint, top_k),
                                                                                   stages(stages) {}
    ~DetectorAnchorPoint() {}

    void parse_stage(dl::Feature<model_output_t> &score, dl::Feature<model_output_t> &box, const int stage_index)
    {
        int stride_y = this->stages[stage_index].stride_y;
        int stride_x = this->stages[stage_index].stride_x;

        int offset_y = this->stages[stage_index].offset_y;
        int offset_x = this->stages[stage_index].offset_x;

        int H = score.shape[0];
        int W = score.shape[1];
        int C = score.shape[2];
        model_output_t *score_element = score.element;
        float score_precision = 1.0f / (1 << -score.exponent);
        int score_threshold = (int)(0.5 - log(1.0 / this->score_threshold - 1) * (1 << -score.exponent));

        model_output_t *box_element = box.element;
        float box_precision = 1.0f / (1 << -box.exponent);

        for (size_t y = 0; y < H; y++) // height
        {
            for (size_t x = 0; x < W; x++) // width
            {
                model_output_t max_score = *score_element++;
                int max_score_c = 0;
                for (size_t c = 1; c < C; c++) // category number
                {
                    if (max_score < *score_element)
                    {
                        max_score = *score_element;
                        max_score_c = c;
                    }
                    score_element++;
                }

                if (max_score > score_threshold)
                {
                    int center_y = y * stride_y + offset_y;
                    int center_x = x * stride_x + offset_x;

                    detection_prediction_t new_box = {
                        max_score_c,
                        max_score * score_precision,
                        {(int)((center_x - __fast_exp(box_element[0] * box_precision, 8)) * this->resize_scale_x),
                         (int)((center_y - __fast_exp(box_element[1] * box_precision, 8)) * this->resize_scale_y),
                         (int)((center_x + __fast_exp(box_element[2] * box_precision, 8)) * this->resize_scale_x),
                         (int)((center_y + __fast_exp(box_element[3] * box_precision, 8)) * this->resize_scale_y)},
                        {0}};
                    this->box_list.insert(std::upper_bound(this->box_list.begin(), this->box_list.end(), new_box, compare_greater_box), new_box);
                }
                box_element += 4;
            }
        }
    }

    void parse_stage(dl::Feature<model_output_t> &score, dl::Feature<model_output_t> &box, dl::Feature<model_output_t> &keypoint, const int stage_index)
    {
        this->parse_stage(score, box, stage_index);
        // int stride_y = stage.stride_y;
        // int stride_x = stage.stride_x;

        // int offset_y = stage.offset_y;
        // int offset_x = stage.offset_x;

        // std::vector<std::vector<int>> &anchor_shape = stage.anchor_shape;

        // int H = score.shape[0];
        // int W = score.shape[1];
        // int A = anchor_shape.size();
        // int C = score.shape[2] / A;
        // model_output_t *score_element = score.element;
        // float score_precision = 1.0f / (1 << -score.exponent);
        // int score_threshold = int(this->score_threshold * (1 << -score.exponent) + 0.5);

        // model_output_t *box_element = box.element;
        // int box_shift = -box.exponent;

        // model_output_t *keypoint_element = keypoint.element;
        // int keypoint_shift = -keypoint.exponent;

        // for (size_t y = 0; y < H; y++) // height
        // {
        //     for (size_t x = 0; x < W; x++) // width
        //     {
        //         for (size_t a = 0; a < A; a++) // anchor number
        //         {
        //             model_output_t max_score = *score_element++;
        //             int max_score_c = 0;
        //             for (size_t c = 1; c < C; c++) // category number
        //             {
        //                 if (max_score < *score_element)
        //                 {
        //                     max_score = *score_element;
        //                     max_score_c = c;
        //                 }
        //                 score_element++;
        //             }

        //             if (max_score > score_threshold)
        //             {
        //                 int center_y = (y * stride_y + offset_y) * this->resize_scale_y;
        //                 int center_x = (x * stride_x + offset_x) * this->resize_scale_x;
        //                 int anchor_h = anchor_shape[a][0] * this->resize_scale_y;
        //                 int anchor_w = anchor_shape[a][1] * this->resize_scale_x;

        //                 detection_prediction_t new_box = {
        //                     max_score_c,
        //                     max_score * score_precision,
        //                     {center_x - anchor_w / 2 + ((anchor_w * box_element[0]) >> box_shift),
        //                      center_y - anchor_h / 2 + ((anchor_h * box_element[1]) >> box_shift),
        //                      center_x + anchor_w / 2 + ((anchor_w * box_element[2]) >> box_shift),
        //                      center_y + anchor_h / 2 + ((anchor_h * box_element[3]) >> box_shift)},
        //                     {center_x - anchor_w / 2 + ((anchor_w * keypoint_element[0]) >> keypoint_shift),
        //                      center_y - anchor_h / 2 + ((anchor_h * keypoint_element[1]) >> keypoint_shift),
        //                      center_x - anchor_w / 2 + ((anchor_w * keypoint_element[2]) >> keypoint_shift),
        //                      center_y - anchor_h / 2 + ((anchor_h * keypoint_element[3]) >> keypoint_shift),
        //                      center_x - anchor_w / 2 + ((anchor_w * keypoint_element[4]) >> keypoint_shift),
        //                      center_y - anchor_h / 2 + ((anchor_h * keypoint_element[5]) >> keypoint_shift),
        //                      center_x - anchor_w / 2 + ((anchor_w * keypoint_element[6]) >> keypoint_shift),
        //                      center_y - anchor_h / 2 + ((anchor_h * keypoint_element[7]) >> keypoint_shift),
        //                      center_x - anchor_w / 2 + ((anchor_w * keypoint_element[8]) >> keypoint_shift),
        //                      center_y - anchor_h / 2 + ((anchor_h * keypoint_element[9]) >> keypoint_shift)}};

        //                 this->box_list.insert(std::upper_bound(this->box_list.begin(), this->box_list.end(), new_box, compare_greater_box), new_box);
        //             }
        //             box_element += 4;
        //             keypoint_element += 10;
        //         }
        //     }
        // }
    }
};
