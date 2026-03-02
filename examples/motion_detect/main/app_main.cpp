#include "dl_image_jpeg.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "motion_detect.hpp"

#define MOV_THR 300 * 300 * 0.1

extern const uint8_t dog1_jpg_start[] asm("_binary_dog1_jpg_start");
extern const uint8_t dog1_jpg_end[] asm("_binary_dog1_jpg_end");
extern const uint8_t dog2_jpg_start[] asm("_binary_dog2_jpg_start");
extern const uint8_t dog2_jpg_end[] asm("_binary_dog2_jpg_end");

extern "C" void app_main(void)
{
    dl::image::jpeg_img_t dog1_jpeg = {.data = (void *)dog1_jpg_start,
                                       .data_len = (size_t)(dog1_jpg_end - dog1_jpg_start)};
    auto dog1 = dl::image::sw_decode_jpeg(dog1_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    dl::image::jpeg_img_t dog2_jpeg = {.data = (void *)dog2_jpg_start,
                                       .data_len = (size_t)(dog2_jpg_end - dog2_jpg_start)};
    auto dog2 = dl::image::sw_decode_jpeg(dog2_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    uint32_t n = dl::image::get_moving_point_number(dog1, dog2, 1, 5);
    if (n > MOV_THR) {
        ESP_LOGI("MOTION", "moving with %d moving pts", n);
    } else {
        ESP_LOGI("MOTION", "not moving with %d moving pts", n);
    }
    n = dl::image::get_moving_point_number(dog1, dog1, 1, 5);
    if (n > MOV_THR) {
        ESP_LOGI("MOTION", "moving with %d moving pts", n);
    } else {
        ESP_LOGI("MOTION", "not moving with %d moving pts", n);
    }
    heap_caps_free(dog1.data);
    heap_caps_free(dog2.data);
}
