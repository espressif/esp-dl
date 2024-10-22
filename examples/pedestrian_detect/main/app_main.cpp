#include "esp_log.h"
#include "jpeg_decoder.h"
#include "pedestrian_detect.hpp"

extern const uint8_t pedestrian_jpg_start[] asm("_binary_pedestrian_jpg_start");
extern const uint8_t pedestrian_jpg_end[] asm("_binary_pedestrian_jpg_end");
const char *TAG = "pedestrian_detect";

uint8_t *get_image(const uint8_t *jpg_img, uint32_t jpg_img_size, int height, int width)
{
    uint32_t outbuf_size = height * width * 3;
    uint8_t *outbuf = (uint8_t *)heap_caps_malloc(outbuf_size, MALLOC_CAP_SPIRAM);
    // JPEG decode config
    esp_jpeg_image_cfg_t jpeg_cfg = {.indata = (uint8_t *)jpg_img,
                                     .indata_size = jpg_img_size,
                                     .outbuf = outbuf,
                                     .outbuf_size = outbuf_size,
                                     .out_format = JPEG_IMAGE_FORMAT_RGB888,
                                     .out_scale = JPEG_IMAGE_SCALE_0,
                                     .flags = {
                                         .swap_color_bytes = 1,
                                     }};

    esp_jpeg_image_output_t outimg;
    esp_jpeg_decode(&jpeg_cfg, &outimg);
    assert(outimg.height == height && outimg.width == width);
    return outbuf;
}

extern "C" void app_main(void)
{
    uint8_t *pedestrian = get_image(pedestrian_jpg_start, pedestrian_jpg_end - pedestrian_jpg_start, 480, 640);
    PedestrianDetect *detect = new PedestrianDetect();
    auto &detect_results = detect->run((uint8_t *)pedestrian, {480, 640, 3});
    for (const auto &res : detect_results) {
        ESP_LOGI(TAG,
                 "[score: %f, x1: %d, y1: %d, x2: %d, y2: %d]\n",
                 res.score,
                 res.box[0],
                 res.box[1],
                 res.box[2],
                 res.box[3]);
    }
    delete detect;
    heap_caps_free(pedestrian);
}
