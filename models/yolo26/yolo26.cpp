#include "yolo26.hpp"
#include "dl_image_jpeg.hpp"
#include "dl_math.hpp"
#include "dl_tool.hpp"
#include "esp_heap_caps.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

// --- Constructor ---

YOLO26::YOLO26(dl::Model *model, int k, float thresh, const char **classes) :
    target_k(k), conf_thresh(thresh), class_names(classes)
{
    // Reserve memory for grids
    grid_sizes.resize(3);

    // Initialize the preprocessor with standard YOLO Mean (0) and Std (255)
    m_image_preprocessor = new dl::image::ImagePreprocessor(model, {0, 0, 0}, {255, 255, 255});
    m_image_preprocessor->enable_letterbox({114, 114, 114});

    // Calculate grid sizes using model inputs
    auto inputs = model->get_inputs();
    if (!inputs.empty()) {
        dl::TensorBase *input_tensor = inputs.begin()->second;
        int input_w = input_tensor->shape[2];
        for (int i = 0; i < 3; i++) {
            grid_sizes[i] = input_w / strides[i];
        }
    }
}

YOLO26::~YOLO26()
{
    delete m_image_preprocessor;
}

// --- Methods ---

dl::image::img_t YOLO26::decode_jpeg(const uint8_t *jpg_data, size_t jpg_len)
{
    dl::image::jpeg_img_t jpeg_img = {.data = (void *)jpg_data, .data_len = jpg_len};
    return dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
}

void YOLO26::preprocess(const dl::image::img_t &img)
{
    m_image_preprocessor->preprocess(img);
}

template <typename T>
void YOLO26::decode_grid(dl::TensorBase *p_box,
                         dl::TensorBase *p_cls,
                         int stride,
                         int grid_h,
                         int grid_w,
                         std::vector<dl::detect::result_t> &candidates)
{
    float box_scale = DL_SCALE(p_box->exponent);
    float cls_scale = DL_SCALE(p_cls->exponent);
    T *raw_box = (T *)p_box->data;
    T *raw_cls = (T *)p_cls->data;

    // Optimization: Calculate Threshold natively in type T
    float raw_thresh_float = dl::math::inverse_sigmoid(conf_thresh);
    T cls_thresh = (T)std::floor(raw_thresh_float / cls_scale);

    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            int pixel_idx = (h * grid_w) + w; // NHWC
            int cls_offset = pixel_idx * num_classes;

            float max_score = -1.0f;
            int best_cls_id = -1;

            for (int c = 0; c < num_classes; c++) {
                T raw_val_T = raw_cls[cls_offset + c];
                if (raw_val_T <= cls_thresh)
                    continue;

                float raw_val = dl::dequantize(raw_val_T, cls_scale);
                float score = dl::math::sigmoid(raw_val);
                if (score > max_score) {
                    max_score = score;
                    best_cls_id = c;
                }
            }

            if (max_score < conf_thresh)
                continue;

            // Decode Box (Int8 or Int16)
            int box_offset = pixel_idx * 4;
            float d_l = dl::dequantize(raw_box[box_offset + 0], box_scale);
            float d_t = dl::dequantize(raw_box[box_offset + 1], box_scale);
            float d_r = dl::dequantize(raw_box[box_offset + 2], box_scale);
            float d_b = dl::dequantize(raw_box[box_offset + 3], box_scale);

            float cx = w + 0.5f;
            float cy = h + 0.5f;
            float x1 = (cx - d_l) * stride;
            float y1 = (cy - d_t) * stride;
            float x2 = (cx + d_r) * stride;
            float y2 = (cy + d_b) * stride;

            candidates.push_back({best_cls_id, max_score, {(int)x1, (int)y1, (int)x2, (int)y2}, {}});
        }
    }
}

std::vector<dl::detect::result_t> YOLO26::postprocess(const std::map<std::string, dl::TensorBase *> &outputs)
{
    // Ensure grid_sizes are ready
    if (grid_sizes.empty() || grid_sizes[0] == 0) {
        printf("[YOLO26] Error: Grid sizes not initialized. Call preprocess() first.\n");
        return {};
    }

    dl::TensorBase *p3_box = outputs.at("one2one_p3_box");
    dl::TensorBase *p4_box = outputs.at("one2one_p4_box");
    dl::TensorBase *p5_box = outputs.at("one2one_p5_box");
    dl::TensorBase *p3_cls = outputs.at("one2one_p3_cls");
    dl::TensorBase *p4_cls = outputs.at("one2one_p4_cls");
    dl::TensorBase *p5_cls = outputs.at("one2one_p5_cls");

    // Auto-detect the class count
    this->num_classes = p3_cls->shape[3];

    std::vector<dl::detect::result_t> candidates;
    candidates.reserve(target_k * 2);

    dl::TensorBase *boxes[] = {p3_box, p4_box, p5_box};
    dl::TensorBase *clss[] = {p3_cls, p4_cls, p5_cls};

    for (int i = 0; i < 3; i++) {
        int stride = strides[i];
        int grid_h = grid_sizes[i]; // Use stored grid size
        int grid_w = grid_sizes[i];

        if (boxes[i]->dtype == dl::DATA_TYPE_INT8 && clss[i]->dtype == dl::DATA_TYPE_INT8) {
            decode_grid<int8_t>(boxes[i], clss[i], stride, grid_h, grid_w, candidates);
        } else if (boxes[i]->dtype == dl::DATA_TYPE_INT16 && clss[i]->dtype == dl::DATA_TYPE_INT16) {
            decode_grid<int16_t>(boxes[i], clss[i], stride, grid_h, grid_w, candidates);
        } else {
            printf("[YOLO26] Error: Unsupported tensor dtype for outputs.\n");
            return {};
        }
    }

    // Top-K Extraction (NMS-Free)
    if (candidates.size() > target_k) {
        std::nth_element(candidates.begin(), candidates.begin() + target_k, candidates.end(), dl::detect::greater_box);
        candidates.resize(target_k);
    }

    return candidates;
}
