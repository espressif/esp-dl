#include "esp_log.h"
#include "imagenet_cls.hpp"
#include "bsp/esp-bsp.h"

extern const uint8_t cat_jpg_start[] asm("_binary_cat_jpg_start");
extern const uint8_t cat_jpg_end[] asm("_binary_cat_jpg_end");
const char *TAG = "mobilenetv2_cls";

extern "C" void app_main(void)
{
#if CONFIG_IMAGENET_CLS_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    dl::image::jpeg_img_t jpeg_img = {.data = (void *)cat_jpg_start, .data_len = (size_t)(cat_jpg_end - cat_jpg_start)};
    auto img = sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    ImageNetCls *cls = new ImageNetCls();

    auto &results = cls->run(img);
    for (const auto &res : results) {
        ESP_LOGI(TAG, "category: %s, score: %f", res.cat_name, res.score);
    }
    delete cls;
    heap_caps_free(img.data);

#if CONFIG_IMAGENET_CLS_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
