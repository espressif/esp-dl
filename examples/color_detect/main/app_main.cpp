#include "color_detect.hpp"
#include "dl_image_jpeg.hpp"
#include "esp_log.h"

extern const uint8_t color_jpg_start[] asm("_binary_color_jpg_start");
extern const uint8_t color_jpg_end[] asm("_binary_color_jpg_end");
const char *TAG = "color_detect";

void task(void *args)
{
    dl::image::jpeg_img_t jpeg_img = {.data = (void *)color_jpg_start,
                                      .data_len = (size_t)(color_jpg_end - color_jpg_start)};
    auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    ColorDetect *detect = new ColorDetect(224, 224);
    detect->register_color({170, 100, 100}, {10, 255, 255}, "red");
    detect->register_color({20, 100, 100}, {35, 255, 255}, "yellow");
    detect->register_color({35, 100, 100}, {85, 255, 255}, "green");
    detect->register_color({100, 100, 100}, {130, 255, 255}, "blue");

    auto &detect_results = detect->run(img);
    for (const auto &res : detect_results) {
        ESP_LOGI(TAG,
                 "[name: %s, [x1: %d, y1: %d, x2: %d, y2: %d]]",
                 detect->get_color_name(res.category).c_str(),
                 res.box[0],
                 res.box[1],
                 res.box[2],
                 res.box[3]);
    }
    delete detect;
    heap_caps_free(img.data);
    vTaskDelete(NULL);
}

extern "C" void app_main(void)
{
    xTaskCreate(task, nullptr, 4096, nullptr, 5, nullptr);
}
