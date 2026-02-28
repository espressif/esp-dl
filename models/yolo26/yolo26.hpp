#pragma once
#include "dl_image_process.hpp"
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
    const char** class_names;
    
    // --- Optimization ---
    // Lookup Table for Quantization
    // Stores pre-calculated (pixel / 255.0 * 128) values for all 256 inputs.
    int8_t quantization_lut[256];

    // Constants
    const int strides[3] = {8, 16, 32};

    // --- Helpers ---
    float sigmoid(float x);
    
    // Dequantization Helper
    template <typename T>
    float dequantize_val(T val, float scale);

public:
    /**
     * @brief Constructor.
     * Initializes configuration state and Pre-calculates Quantization LUT.
     * 
     * @param k Max detections (Default: 32)
     * @param thresh Confidence threshold (Default: 0.10f)
     * @param classes Class name array (Default: coco_classes)
     */
    YOLO26(int k = YOLO_TARGET_K, float thresh = YOLO_CONF_THRESH, const char** classes = coco_classes);
    
    ~YOLO26();

    /**
     * @brief Decodes JPEG to RGB888.
     */
    dl::image::img_t decode_jpeg(const uint8_t* jpg_data, size_t jpg_len);

    /**
     * @brief Checks and resizes image to match model input shape if necessary.
     * 
     * @param img Input image
     * @param inputs Model input map
     * @return dl::image::img_t Resized image (or original if no resize needed)
     */
    dl::image::img_t resize(dl::image::img_t& img, const std::map<std::string, dl::TensorBase*>& inputs);

    /**
     * @brief Preprocesses image and updates internal state (grid_sizes).
     * 
     * @param img Input image (RGB888)
     * @param inputs Model input map (used to get tensor data and shape)
     */
    void preprocess(const dl::image::img_t& img, const std::map<std::string, dl::TensorBase*>& inputs);

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
