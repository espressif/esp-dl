#include "dl_model_base.hpp" // Required for dl::Model
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"
#include "yolo26.hpp" // Use the Official Component
#include <stdio.h>
#include <string.h>
#include "dl_tensor_base.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "yolo26_detect";

// =========================================================================
// 🛠️ CLASS LABELS (Auto-selected by CMake based on MODEL_FILENAME)
// =========================================================================
#ifdef USE_LEGO_MODEL
#include "lego_classes.hpp"
const char **current_classes = lego_classes;
#else
#include "coco_classes.hpp"
const char **current_classes = coco_classes;
#endif
// =========================================================================

// --- Model Embedding ---
// The symbol name is generated AUTOMATICALLY by CMakeLists.txt.
// It is injected here as "MODEL_SYMBOL_STR".
extern const uint8_t model_binary_start[] asm(MODEL_SYMBOL_STR);

// --- Image Embedding ---
#ifdef USE_LEGO_MODEL
extern const uint8_t lego_jpg_start[] asm("_binary_lego_jpg_start");
extern const uint8_t lego_jpg_end[] asm("_binary_lego_jpg_end");
#else
extern const uint8_t bus_jpg_start[] asm("_binary_bus_jpg_start");
extern const uint8_t bus_jpg_end[] asm("_binary_bus_jpg_end");
extern const uint8_t person_jpg_start[] asm("_binary_person_jpg_start");
extern const uint8_t person_jpg_end[] asm("_binary_person_jpg_end");
#endif

// --- Raw RGB Embedding (bit-exact validation, bypasses HW JPEG decoder) ---
#ifdef USE_RAW_RGB
namespace test_data {
#include "images/raw_rgb_bus.h"
}
#endif

void test_inference(dl::Model *model, YOLO26 &processor, const uint8_t *jpg_data, size_t jpg_len, const char *name)
{
    ESP_LOGI("image:", "%s", name);

    // 1. Decode JPEG
    auto img = processor.decode_jpeg(jpg_data, jpg_len);

    // 2. Preprocess (Measure Time)
    // Automatically does letterboxing and fills model inputs natively
    int64_t start_pre = esp_timer_get_time();
    processor.preprocess(img);
    int64_t end_pre = esp_timer_get_time();

    // 3. Inference (Measure Time)
    int64_t start_inf = esp_timer_get_time();
    model->run();
    int64_t end_inf = esp_timer_get_time();

    // 4. Post-Process (Measure Time)
    int64_t start_post = esp_timer_get_time();
    auto results = processor.postprocess(model->get_outputs());
    int64_t end_post = esp_timer_get_time();

    // Calculate Latencies (µs → ms)
    uint32_t lat_pre = (uint32_t)((end_pre - start_pre) / 1000);
    uint32_t lat_inf = (uint32_t)((end_inf - start_inf) / 1000);
    uint32_t lat_post = (uint32_t)((end_post - start_post) / 1000);

    ESP_LOGI(TAG, "Pre: %lu ms | Inf: %lu ms | Post: %lu ms", lat_pre, lat_inf, lat_post);

    for (const auto &res : results) {
        // The processor automatically maps class_id to the string array you passed
        ESP_LOGI("YOLO26",
                 "[category: %s, score: %.2f, x1: %d, y1: %d, x2: %d, y2: %d]",
                 current_classes[res.category],
                 res.score,
                 res.box[0],
                 res.box[1],
                 res.box[2],
                 res.box[3]);
    }

    heap_caps_free(img.data);
}

void test_inference_raw(dl::Model *model, YOLO26 &processor,
                       const uint8_t *rgb_data, int width, int height, const char *name)
{
    ESP_LOGI("image:", "%s (raw RGB)", name);

    // 1. Construct img_t from raw RGB888 (bypasses JPEG decoder)
    dl::image::img_t img;
    img.data     = (uint8_t *)rgb_data;
    img.width    = width;
    img.height   = height;
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;

    // 2. Preprocess (Measure Time)
    int64_t start_pre = esp_timer_get_time();
    processor.preprocess(img);
    int64_t end_pre = esp_timer_get_time();

    // 3. Inference (Measure Time)
    int64_t start_inf = esp_timer_get_time();
    model->run();
    int64_t end_inf = esp_timer_get_time();

    // 4. Post-Process (Measure Time)
    int64_t start_post = esp_timer_get_time();
    auto results = processor.postprocess(model->get_outputs());
    int64_t end_post = esp_timer_get_time();

    // Calculate Latencies
    uint32_t lat_pre = (uint32_t)((end_pre - start_pre) / 1000);
    uint32_t lat_inf = (uint32_t)((end_inf - start_inf) / 1000);
    uint32_t lat_post = (uint32_t)((end_post - start_post) / 1000);

    ESP_LOGI(TAG, "Pre: %lu ms | Inf: %lu ms | Post: %lu ms", lat_pre, lat_inf, lat_post);

    for (const auto &res : results) {
        ESP_LOGI("YOLO26",
                 "[category: %s, score: %.2f, x1: %d, y1: %d, x2: %d, y2: %d]",
                 current_classes[res.category],
                 res.score,
                 res.box[0],
                 res.box[1],
                 res.box[2],
                 res.box[3]);
    }
    // No heap_caps_free — rgb_data is static const in flash
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "YOLO26 Firmware Example");

    // 1. Load Model (Using the Magic Symbol from CMake)
    ESP_LOGI(TAG, "Loading Model from Flash...");
    dl::Model *model = new dl::Model((const char *)model_binary_start,
                                     fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                                     0,
                                     dl::MEMORY_MANAGER_GREEDY,
                                     nullptr,
                                     true);

    // 2. Initialize Processor
    YOLO26 processor(model, YOLO_TARGET_K, YOLO_CONF_THRESH, current_classes);

    // 3. Run Tests
#ifdef USE_RAW_RGB
    test_inference_raw(model, processor,
        test_data::raw_rgb_bus, test_data::raw_rgb_bus_width, test_data::raw_rgb_bus_height,
        "bus.jpg");
#elif defined(USE_LEGO_MODEL)
    test_inference(model, processor, lego_jpg_start, (size_t)(lego_jpg_end - lego_jpg_start), "lego.jpg");
#else
    test_inference(model, processor, bus_jpg_start, (size_t)(bus_jpg_end - bus_jpg_start), "bus.jpg");
    //test_inference(model, processor, person_jpg_start, (size_t)(person_jpg_end - person_jpg_start), "person.jpg");
#endif
    model->profile();
    delete model;
}
