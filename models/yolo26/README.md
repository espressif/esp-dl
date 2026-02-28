# YOLO26 Models

## Model List
[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | YOLO26n (Int8)          |
|----------|------------------------|
| ESP32-S3 | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] |

## Model Benchmarks

| name               | input(h*w*c)  | Training State   | param_copy | preprocess(ms) | model(ms) | postprocess(ms) | mAP50-95 on COCO val2017 |
|--------------------|---------------|------------------|------------|----------------|-----------|-----------------|--------------------------|
| yolo26n_512_s8_s3   | 512 * 512 * 3 | PTQ (Can continue to QAT) | true       | 30             | 6960      | 30              | 31.6                     |
| yolo26n_640_s8_s3   | 640 * 640 * 3 | PTQ (Can continue to QAT) | false      | 60             | 24000     | 30              | 34.2                     |
| yolo26n_512_s8_p4   | 512 * 512 * 3 | QAT              | true       | 20             | 1780      | 10              | 36.5                     |
| yolo26n_640_s8_p4   | 640 * 640 * 3 | QAT              | true       | 30             | 3030      | 20              | 38.5                     |

*Source: Models generated in the `output/` directory of the [YOLOv26 Quantization Tutorial](../../examples/tutorial/how_to_quantize_model/quantize_yolo26/README.md).*
*Note: Performance values depend on memory configuration (Flash vs PSRAM).*

## Model Usage

### 1. Initialize
```cpp
#include "yolo26.hpp"

// Option 1: Custom Classes (Must match your training labels)
const char* my_classes[] = {
    "person", "bicycle", "car", "motorcycle", "airplane" 
    // ... add all your classes here
};
// Initialize Processor
// Params: max_classes=32, confidence_threshold=0.25f, class_list
// Note: The number of classes (e.g., 80 or 28) is AUTOMATICALLY detected from the model.
YOLO26 processor(32, 0.25f, my_classes);

// Option 2: Default COCO Classes
// YOLO26 processor(32, 0.25f, coco_classes);

// 3. Load Model (User responsibility: Flash, SD, or RAM)
// For details on how to use models from Flash/SDCard/Partition, see:
// https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html
// Example: Embedded in RO-DATA
// Declare the symbol (Check your CMake/Component config for the exact name)
extern const uint8_t model_espdl[] asm("_binary_yolo26n_512_espdl_start");

dl::Model* model = new dl::Model((const char *)model_espdl, 
                                 fbs::MODEL_LOCATION_IN_FLASH_RODATA, 
                                 0,                          // max_internal_size 
                                 dl::MEMORY_MANAGER_GREEDY,  // mm_type
                                 nullptr,                    // key
                                 false);                     // param_copy (Keep false to save RAM)
```

### 2. Run
```cpp
// A. Decode JPEG
auto img = processor.decode_jpeg(jpg_data, jpg_len);

// B. Resize (Auto-adapts to model's input shape relative to image)
auto resized = processor.resize(img, model->get_inputs());

// C. Preprocess (LUT accelerated quantization)
processor.preprocess(resized, model->get_inputs());

// D. Run Hardware Inference
model->run();

// E. Postprocess (NMS-Free decoding)
auto results = processor.postprocess(model->get_outputs());
```

### 3. Result Structure
The results are returned as a standard vector of `Detection` structs:

```cpp
struct Detection {
    float x1, y1, x2, y2; // Box coordinates (pixels)
    float score;          // Confidence (0.0 - 1.0)
    int class_id;         // Index in class_names
};
```

#### Example Usage:
```cpp
for (const auto &res : results) {
    if (res.score > 0.5f) {
        ESP_LOGI("YOLO26", "[category: %s, score: %f, x1: %d, y1: %d, x2: %d, y2: %d]",
                 my_classes[res.class_id], // or coco_classes
                 res.score,
                 (int)res.x1, (int)res.y1, (int)res.x2, (int)res.y2);
    }
}
// Example Output:
// I (4256) YOLO26: [category: person, score: 0.882341, x1: 110, y1: 52, x2: 240, y2: 380]
```

## Quantization Constraints
*   **Input/Output Precision:** This processor relies on **Int8** optimizations (LUT Preprocessing & Integer Thresholding).
    *   Therefore, your model's **Input** and **Output** tensors MUST be **Int8**.
    *   *Internal* layers can use any precision supported by ESP-DL (Int8, Int16, Mixed) — you are free to choose the best balance for your model.
