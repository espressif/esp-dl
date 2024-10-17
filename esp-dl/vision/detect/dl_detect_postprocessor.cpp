#include "dl_detect_postprocessor.hpp"

static const char *TAG = "dl::detect::DetectPostprocessor";

namespace dl {
namespace detect {
void DetectPostprocessor::nms()
{
    dl::tool::Latency latency;
    latency.start();
    int kept_number = 0;
    for (std::list<result_t>::iterator kept = this->box_list.begin(); kept != this->box_list.end(); kept++) {
        kept_number++;

        if (kept_number >= this->top_k) {
            this->box_list.erase(++kept, this->box_list.end());
            break;
        }

        int kept_area = (kept->box[2] - kept->box[0] + 1) * (kept->box[3] - kept->box[1] + 1);

        std::list<result_t>::iterator other = kept;
        other++;
        for (; other != this->box_list.end();) {
            int inter_lt_x = DL_MAX(kept->box[0], other->box[0]);
            int inter_lt_y = DL_MAX(kept->box[1], other->box[1]);
            int inter_rb_x = DL_MIN(kept->box[2], other->box[2]);
            int inter_rb_y = DL_MIN(kept->box[3], other->box[3]);

            int inter_height = inter_rb_y - inter_lt_y + 1;
            int inter_width = inter_rb_x - inter_lt_x + 1;

            if (inter_height > 0 && inter_width > 0) {
                int other_area = (other->box[2] - other->box[0] + 1) * (other->box[3] - other->box[1] + 1);
                int inter_area = inter_height * inter_width;
                float iou = (float)inter_area / (kept_area + other_area - inter_area);
                if (iou > this->nms_threshold) {
                    other = this->box_list.erase(other);
                    continue;
                }
            }
            other++;
        }
    }
    latency.end();
    latency.print("detect", "postprocess::nms");
}

TensorBase *DetectPostprocessor::get_model_output(const char *output_name)
{
    TensorBase *output = nullptr;
    auto iter = this->model_outputs_map.find(output_name);
    if (iter != this->model_outputs_map.end()) {
        output = iter->second;
    } else {
        ESP_LOGE(TAG, "Invalid key: %s, it is not in model outputs map.", output_name);
    }
    return output;
}

std::list<result_t> &DetectPostprocessor::get_result(const std::vector<int> &input_shape)
{
    for (result_t &res : this->box_list) {
        for (int i = 0; i < res.box.size(); i++) {
            if (i % 2 == 0)
                res.box[i] = DL_CLIP(res.box[i], 0, input_shape[1] - 1);
            else
                res.box[i] = DL_CLIP(res.box[i], 0, input_shape[0] - 1);
        }
        for (int i = 0; i < res.keypoint.size(); i++) {
            if (i % 2 == 0)
                res.keypoint[i] = DL_CLIP(res.keypoint[i], 0, input_shape[1] - 1);
            else
                res.keypoint[i] = DL_CLIP(res.keypoint[i], 0, input_shape[0] - 1);
        }
    }
    return this->box_list;
}
} // namespace detect
} // namespace dl
