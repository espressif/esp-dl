#include "yolo26.hpp"
#include "dl_image_jpeg.hpp"
#include "esp_heap_caps.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstdio>



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


// --- Constructor ---

YOLO26::YOLO26(int k, float thresh, const char** classes) 
    : target_k(k), conf_thresh(thresh), class_names(classes) {
    
    // Reserve memory for grids
    grid_sizes.resize(3);
    
    // Calculate Quantization LUT (Lossless)
    // Formula: round( (pixel / 255.0) * 128 ) -> int8
    for (int i = 0; i < 256; i++) {
        float normalized = i / 255.0f;
        float scaled = normalized * 128.0f; 
        int val = (int)std::round(scaled);
        
        // Clamp to int8 range
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        
        quantization_lut[i] = (int8_t)val;
    }
}

YOLO26::~YOLO26() {
    // Vectors clean themselves up
}


// --- Methods ---

dl::image::img_t YOLO26::decode_jpeg(const uint8_t* jpg_data, size_t jpg_len) {
    dl::image::jpeg_img_t jpeg_img = {
        .data = (void*)jpg_data,
        .data_len = jpg_len
    };
    return dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
}

dl::image::img_t YOLO26::resize(dl::image::img_t& img, const std::map<std::string, dl::TensorBase*>& inputs) {
    if (inputs.empty()) return img;
    dl::TensorBase* input_tensor = inputs.begin()->second;
    
    int model_h = input_tensor->shape[1];
    int model_w = input_tensor->shape[2];
    
    if (img.width != model_w || img.height != model_h) {
        dl::image::img_t resized_img;
        resized_img.width = model_w;
        resized_img.height = model_h;
        resized_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
        resized_img.data = heap_caps_malloc(dl::image::get_img_byte_size(resized_img), MALLOC_CAP_SPIRAM);
        if (resized_img.data == NULL) {
            size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
            printf("[YOLO26] Error: Failed to allocate %d bytes for resized image in SPIRAM. (Free PSRAM: %u bytes)\n", 
                    (int)dl::image::get_img_byte_size(resized_img), (unsigned int)free_psram);
            // it will crach in .set_dst_img(resized_img)
        }
        
        dl::image::ImageTransformer transformer;
        transformer.set_src_img(img)
                   .set_dst_img(resized_img)
                   .transform();
        
        return resized_img; 
    }
    
    return img;
}

void YOLO26::preprocess(const dl::image::img_t& img, const std::map<std::string, dl::TensorBase*>& inputs) {
    // 1. Get the first input tensor
    if (inputs.empty()) return;
    dl::TensorBase* input_tensor = inputs.begin()->second;

    // Enforce Int8 Input
    if (input_tensor->dtype != dl::DATA_TYPE_INT8) {
        printf("[YOLO26] Error: Input tensor must be INT8 (required for optimizations).\n");
        return;
    }

    // 2. Validate Exponent for LUT Optimization (Expect -7)
    int shift_check = 8 + input_tensor->exponent;
    if (shift_check != 1) {
            printf("[YOLO26] Warning: Model exponent %d not ideal for fast path (Expected -7)\n", input_tensor->exponent);
    }

    // 3. Calculate and Store Grid Sizes
    int input_w = input_tensor->shape[2]; 
    for(int i=0; i<3; i++) {
        grid_sizes[i] = input_w / strides[i];
    }

    // 4. Quantize using LUT (Fast & Lossless)
    uint8_t* rgb_data = (uint8_t*)img.data;
    int8_t* raw_input = (int8_t*)input_tensor->data;
    int total_pixels = img.width * img.height * 3;

    for (int i = 0; i < total_pixels; i++) {
        // LUT lookup recovers exact floating point precision without the cost
        raw_input[i] = quantization_lut[rgb_data[i]];
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
    
    // Enforce Int8 Output
    if (p3_box->dtype != dl::DATA_TYPE_INT8) {
        printf("[YOLO26] Error: Output tensor must be INT8 (required for optimizations).\n");
        return {};
    }

    for (int i = 0; i < 3; i++) {
        int stride = strides[i];
        int grid_h = grid_sizes[i]; // Use stored grid size
        int grid_w = grid_sizes[i];

        float box_scale = std::pow(2.0f, boxes[i]->exponent);
        float cls_scale = std::pow(2.0f, clss[i]->exponent);
        int8_t* raw_box = (int8_t*)boxes[i]->data;
        int8_t* raw_cls = (int8_t*)clss[i]->data;

        // Optimization: Calculate INT8 Threshold
        float raw_thresh_float = -std::log(1.0f / conf_thresh - 1.0f);
        int8_t cls_thresh_int8 = (int8_t)std::floor(raw_thresh_float / cls_scale);

        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                int pixel_idx = (h * grid_w) + w; // NHWC
                int cls_offset = pixel_idx * num_classes;
                
                float max_score = -1.0f;
                int best_cls_id = -1;

                // Class Score Loop
                for (int c = 0; c < num_classes; c++) {
                    // 1. Fast Integer Check (Skip Float Math)
                    int8_t raw_val_int8 = raw_cls[cls_offset + c];
                    if (raw_val_int8 <= cls_thresh_int8) continue; 

                    float raw_val = dequantize_val(raw_val_int8, cls_scale);
                    
                    float score = sigmoid(raw_val);
                    if (score > max_score) {
                        max_score = score;
                        best_cls_id = c;
                    }
                }

                if (max_score < conf_thresh) continue;

                // Decode Box (Int8 Only)
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

    // Global Sort
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    if (candidates.size() > target_k) {
        candidates.resize(target_k);
    }

    return candidates;
}
