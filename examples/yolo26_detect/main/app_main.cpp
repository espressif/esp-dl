#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "dl_model_base.hpp" // Required for dl::Model
#include "yolo26.hpp" // Use the Official Component

static const char *TAG = "yolo26_detect";

// =========================================================================
// 🛠️ USER CONFIGURATION: CLASS LABELS
// =========================================================================
/* * INSTRUCTIONS:
 * 1. Default: "coco_classes.hpp" (80 classes) is enabled.
 * 2. To use a CUSTOM MODEL (e.g. Lego detection 28 classes):
 * - Uncomment '#include "lego_classes.hpp"'
 */
//#include "lego_classes.hpp"                  // <--- Uncomment for Lego Model
const char **current_classes = coco_classes; // <--- Set your active classes  coco_classes/lego_classes here
// =========================================================================

// --- Model Embedding ---
// The symbol name is generated AUTOMATICALLY by CMakeLists.txt.
// It is injected here as "MODEL_SYMBOL_STR".
extern const uint8_t model_binary_start[] asm(MODEL_SYMBOL_STR);

// --- Image Embedding ---
extern const uint8_t bus_jpg_start[] asm("_binary_bus_jpg_start");
extern const uint8_t bus_jpg_end[]   asm("_binary_bus_jpg_end");
extern const uint8_t person_jpg_start[] asm("_binary_person_jpg_start");
extern const uint8_t person_jpg_end[]   asm("_binary_person_jpg_end");
extern const uint8_t lego_jpg_start[] asm("_binary_lego_jpg_start");
extern const uint8_t lego_jpg_end[]   asm("_binary_lego_jpg_end");


void test_inference(dl::Model *model, YOLO26 &processor, const uint8_t *jpg_data, size_t jpg_len, const char *name)
{
    ESP_LOGI("image:", "%s", name);

    // 1. Decode JPEG
    auto img = processor.decode_jpeg(jpg_data, jpg_len);

    // a. Resize (Outside measurement as requested)
    auto resized_img = processor.resize(img, model->get_inputs());

    // 2. Preprocess (Measure Time)
    TickType_t start_pre = xTaskGetTickCount();
    processor.preprocess(resized_img, model->get_inputs());
    TickType_t end_pre = xTaskGetTickCount();

    // 3. Inference (Measure Time)
    TickType_t start_inf = xTaskGetTickCount();
    model->run();
    TickType_t end_inf = xTaskGetTickCount();

    // 4. Post-Process (Measure Time)
    TickType_t start_post = xTaskGetTickCount();
    auto results = processor.postprocess(model->get_outputs());
    TickType_t end_post = xTaskGetTickCount();

    // Calculate Latencies
    uint32_t lat_pre = (end_pre - start_pre) * portTICK_PERIOD_MS;
    uint32_t lat_inf = (end_inf - start_inf) * portTICK_PERIOD_MS;
    uint32_t lat_post = (end_post - start_post) * portTICK_PERIOD_MS;

    ESP_LOGI(TAG, "Pre: %lu ms | Inf: %lu ms | Post: %lu ms", lat_pre, lat_inf, lat_post);

    for (const auto& res : results) {
        // The processor automatically maps class_id to the string array you passed
        ESP_LOGI("YOLO26", "[category: %s, score: %.2f, x1: %d, y1: %d, x2: %d, y2: %d]",
                    current_classes[res.class_id], 
                    res.score,
                    (int)res.x1, (int)res.y1, (int)res.x2, (int)res.y2);
    }

    if (resized_img.data != img.data) heap_caps_free(resized_img.data);
    heap_caps_free(img.data);
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "YOLO26 Firmware Example");

    // 1. Load Model (Using the Magic Symbol from CMake)
    ESP_LOGI(TAG, "Loading Model from Flash...");
    dl::Model *model = new dl::Model((const char *)model_binary_start, 
                                     fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                                     0, dl::MEMORY_MANAGER_GREEDY,
                                     nullptr, true);

    // 2. Initialize Processor
    YOLO26 processor(YOLO_TARGET_K, YOLO_CONF_THRESH, current_classes);

    // 3. Run Tests
    test_inference(model, processor, bus_jpg_start, (size_t)(bus_jpg_end - bus_jpg_start), "bus.jpg");
    test_inference(model, processor, person_jpg_start, (size_t)(person_jpg_end - person_jpg_start), "person.jpg");
    test_inference(model, processor, lego_jpg_start, (size_t)(lego_jpg_end - lego_jpg_start), "lego.jpg");

    delete model;
}