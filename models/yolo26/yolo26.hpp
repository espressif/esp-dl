#pragma once
#include "dl_image_process.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_tensor_base.hpp"
#include "coco_classes.hpp"
#include <vector>
#include <map>
#include <string>

// Default Configuration
#define YOLO_TARGET_K 32
#define YOLO_CONF_THRESH 0.25f

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

class YOLO26 {
private:
    // --- State (Calculated/Configured) ---
    std::vector<int> grid_sizes; // Calculated in preprocess
    int num_classes; // Calculated in postprocess
    int target_k;
    float conf_thresh;
    
    // --- Optimization ---
    // Official ESP-DL Image Preprocessor (Handles Resizing, Letterboxing, and SIMD Quantization)
    dl::image::ImagePreprocessor* m_image_preprocessor;

    // Constants
    const int strides[3] = {8, 16, 32};

    // --- Helpers ---
    float sigmoid(float x);
    
    // Dequantization Helper
    template <typename T>
    float dequantize_val(T val, float scale);

    /**
     * @brief Templated decoding loop to seamlessly handle both INT8 and INT16 tensors
     */
    template <typename T>
    void decode_grid(dl::TensorBase* p_box, dl::TensorBase* p_cls, int stride, int grid_h, int grid_w, std::vector<Detection>& candidates);

public:
    const char** class_names;
    
    /**
     * @brief Constructor.
     * Initializes configuration state and configures the ESP-DL ImagePreprocessor.
     * 
     * @param model Pointer to the loaded dl::Model
     * @param k Max detections (Default: 32)
     * @param thresh Confidence threshold (Default: 0.10f)
     * @param classes Class name array (Default: coco_classes)
     */
    YOLO26(dl::Model* model, int k = YOLO_TARGET_K, float thresh = YOLO_CONF_THRESH, const char** classes = coco_classes);
    
    ~YOLO26();

    /**
     * @brief Decodes JPEG to RGB888.
     */
    dl::image::img_t decode_jpeg(const uint8_t* jpg_data, size_t jpg_len);

    /**
     * @brief Preprocesses image natively using ESP-DL (Letterbox + SIMD Quantization directly into Model RAM).
     * 
     * @param img Input image (RGB888)
     */
    void preprocess(const dl::image::img_t& img);

    /**
     * @brief Post-processes outputs using stored state.
     * 
     * OPTIMIZATION: Uses Integer Thresholding to skip floating point math.
     * 
     * @param outputs Map of model outputs
     * @return std::vector<Detection> 
     */
    std::vector<Detection> postprocess(const std::map<std::string, dl::TensorBase*>& outputs);
};
