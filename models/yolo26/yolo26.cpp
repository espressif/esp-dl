#include "yolo26.hpp"
#include "dl_image_jpeg.hpp"
#include "esp_heap_caps.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include "dl_tool.hpp" 



// --- Helpers ---

float YOLO26::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

template <typename T>
float YOLO26::dequantize_val(T val, float scale) {
    return val * scale;
}

// Explicit instantiation
template float YOLO26::dequantize_val<int8_t>(int8_t val, float scale);
template float YOLO26::dequantize_val<int16_t>(int16_t val, float scale);


// --- Constructor ---

YOLO26::YOLO26(dl::Model* model, int k, float thresh, const char** classes) 
    : target_k(k), conf_thresh(thresh), class_names(classes) {
    
    // Reserve memory for grids
    grid_sizes.resize(3);

    // 1. Initialize the preprocessor with standard YOLO Mean (0) and Std (255)
    // NOTE: This will pull the exponent scale dynamically from the model
    m_image_preprocessor = new dl::image::ImagePreprocessor(model, {0, 0, 0}, {255, 255, 255});
    
    // 2. Enable Letterboxing using gray padding {114, 114, 114} to prevent stretching!
    m_image_preprocessor->enable_letterbox({114, 114, 114});

    // 3. Pre-Calculate grid sizes based on Model Input Shape
    auto inputs = model->get_inputs();
    if (!inputs.empty()) {
        dl::TensorBase* input_tensor = inputs.begin()->second;
        int input_w = input_tensor->shape[2]; 
        for(int i=0; i<3; i++) {
            grid_sizes[i] = input_w / strides[i];
        }
    }
}

YOLO26::~YOLO26() {
    delete m_image_preprocessor;
}


// --- Methods ---

dl::image::img_t YOLO26::decode_jpeg(const uint8_t* jpg_data, size_t jpg_len) {
    dl::image::jpeg_img_t jpeg_img = {
        .data = (void*)jpg_data,
        .data_len = jpg_len
    };
    return dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
}

void YOLO26::preprocess(const dl::image::img_t& img) {
    // This single line performs:
    // 1. Letterbox scaling (no stretching)
    // 2. SIMD color conversion (RGB888 -> RGB888 / RGB565)
    // 3. 256-value Hardware LUT mapped Quantization (Matches exact Exponent)
    // 4. Memory copy directly into the Model's Input Tensor
    m_image_preprocessor->preprocess(img);
}

template <typename T>
void YOLO26::decode_grid(dl::TensorBase* p_box, dl::TensorBase* p_cls, int stride, int grid_h, int grid_w, std::vector<Detection>& candidates) {
    float box_scale = std::pow(2.0f, p_box->exponent);
    float cls_scale = std::pow(2.0f, p_cls->exponent);
    T* raw_box = (T*)p_box->data;
    T* raw_cls = (T*)p_cls->data;

    // Optimization: Calculate Threshold natively in type T
    float raw_thresh_float = -std::log(1.0f / conf_thresh - 1.0f);
    T cls_thresh = (T)std::floor(raw_thresh_float / cls_scale);

    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            int pixel_idx = (h * grid_w) + w; // NHWC
            int cls_offset = pixel_idx * num_classes;
            
            float max_score = -1.0f;
            int best_cls_id = -1;

            // Class Score Loop
            for (int c = 0; c < num_classes; c++) {
                // 1. Fast Integer Check (Skip Float Math)
                T raw_val_T = raw_cls[cls_offset + c];
                if (raw_val_T <= cls_thresh) continue; 

                float raw_val = dequantize_val(raw_val_T, cls_scale);
                
                float score = sigmoid(raw_val);
                if (score > max_score) {
                    max_score = score;
                    best_cls_id = c;
                }
            }

            if (max_score < conf_thresh) continue;

            // Decode Box (Int8 or Int16)
            int box_offset = pixel_idx * 4;
            float d_l = dequantize_val(raw_box[box_offset + 0], box_scale);
            float d_t = dequantize_val(raw_box[box_offset + 1], box_scale);
            float d_r = dequantize_val(raw_box[box_offset + 2], box_scale);
            float d_b = dequantize_val(raw_box[box_offset + 3], box_scale);

            float cx = w + 0.5f;
            float cy = h + 0.5f;
            float x1 = (cx - d_l) * stride;
            float y1 = (cy - d_t) * stride;
            float x2 = (cx + d_r) * stride;
            float y2 = (cy + d_b) * stride;

            candidates.push_back({x1, y1, x2, y2, max_score, best_cls_id});
        }
    }
}

std::vector<Detection> YOLO26::postprocess(const std::map<std::string, dl::TensorBase*>& outputs) {
    // Ensure grid_sizes are ready
    if (grid_sizes.empty() || grid_sizes[0] == 0) {
            printf("[YOLO26] Error: Grid sizes not initialized. Call preprocess() first.\n");
            return {};
    }

    dl::TensorBase* p3_box = outputs.at("one2one_p3_box");
    dl::TensorBase* p4_box = outputs.at("one2one_p4_box");
    dl::TensorBase* p5_box = outputs.at("one2one_p5_box");
    dl::TensorBase* p3_cls = outputs.at("one2one_p3_cls");
    dl::TensorBase* p4_cls = outputs.at("one2one_p4_cls");
    dl::TensorBase* p5_cls = outputs.at("one2one_p5_cls");


    // Auto-detect the class count
    this->num_classes = p3_cls->shape[3];

    std::vector<Detection> candidates;
    candidates.reserve(target_k * 2);

    dl::TensorBase* boxes[] = {p3_box, p4_box, p5_box};
    dl::TensorBase* clss[] = {p3_cls, p4_cls, p5_cls};

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

    // Global Sort
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    if (candidates.size() > target_k) {
        candidates.resize(target_k);
    }

    return candidates;
}
