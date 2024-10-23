#include "esp_log.h"
#include "human_face_detect.hpp"
#include "jpeg_decoder.h"

extern const uint8_t human_face_jpg_start[] asm("_binary_human_face_jpg_start");
extern const uint8_t human_face_jpg_end[] asm("_binary_human_face_jpg_end");
const char *TAG = "human_face_detect";

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
    uint8_t *human_face = get_image(human_face_jpg_start, human_face_jpg_end - human_face_jpg_start, 240, 320);
    HumanFaceDetect *detect = new HumanFaceDetect();
    auto &detect_results = detect->run((uint8_t *)human_face, {240, 320, 3});
    for (const auto &res : detect_results) {
        ESP_LOGI(TAG,
                 "[score: %f, x1: %d, y1: %d, x2: %d, y2: %d]\n",
                 res.score,
                 res.box[0],
                 res.box[1],
                 res.box[2],
                 res.box[3]);
        ESP_LOGI(
            TAG,
            "left_eye: [%d, %d], left_mouth: [%d, %d], nose: [%d, %d], right_eye: [%d, %d], right_mouth: [%d, %d]]\n",
            res.keypoint[0],
            res.keypoint[1],
            res.keypoint[2],
            res.keypoint[3],
            res.keypoint[4],
            res.keypoint[5],
            res.keypoint[6],
            res.keypoint[7],
            res.keypoint[8],
            res.keypoint[9]);
    }
    delete detect;
    heap_caps_free(human_face);
}
