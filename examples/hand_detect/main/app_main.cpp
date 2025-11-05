#include "dl_image_jpeg.hpp"
#include "esp_log.h"
#include "hand_detect.hpp"
#include "bsp/esp-bsp.h"

extern const uint8_t hand_jpg_start[] asm("_binary_hand_jpg_start");
extern const uint8_t hand_jpg_end[] asm("_binary_hand_jpg_end");
const char *TAG = "hand_detect";

extern "C" void app_main(void)
{
#if CONFIG_HAND_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    dl::image::jpeg_img_t jpeg_img = {.data = (void *)hand_jpg_start,
                                      .data_len = (size_t)(hand_jpg_end - hand_jpg_start)};
    auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    HandDetect *detect = new HandDetect();
    auto &detect_results = detect->run(img);
    for (const auto &res : detect_results) {
        ESP_LOGI(TAG,
                 "[category: %d, score: %f, x1: %d, y1: %d, x2: %d, y2: %d]",
                 res.category,
                 res.score,
                 res.box[0],
                 res.box[1],
                 res.box[2],
                 res.box[3]);
    }
    delete detect;
    heap_caps_free(img.data);

#if CONFIG_HAND_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
