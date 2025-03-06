#include "esp_log.h"
#include "human_face_detect.hpp"
#include "bsp/esp-bsp.h"

extern const uint8_t human_face_jpg_start[] asm("_binary_human_face_jpg_start");
extern const uint8_t human_face_jpg_end[] asm("_binary_human_face_jpg_end");
const char *TAG = "human_face_detect";

extern "C" void app_main(void)
{
#if CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    dl::image::jpeg_img_t jpeg_img = {.data = (uint8_t *)human_face_jpg_start,
                                      .width = 320,
                                      .height = 240,
                                      .data_size = (uint32_t)(human_face_jpg_end - human_face_jpg_start)};
    dl::image::img_t img;
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    sw_decode_jpeg(jpeg_img, img, true);

    HumanFaceDetect *detect = new HumanFaceDetect();

    auto &detect_results = detect->run(img);
    for (const auto &res : detect_results) {
        ESP_LOGI(TAG,
                 "[score: %f, x1: %d, y1: %d, x2: %d, y2: %d]",
                 res.score,
                 res.box[0],
                 res.box[1],
                 res.box[2],
                 res.box[3]);
        ESP_LOGI(
            TAG,
            "left_eye: [%d, %d], left_mouth: [%d, %d], nose: [%d, %d], right_eye: [%d, %d], right_mouth: [%d, %d]]",
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
    heap_caps_free(img.data);

#if CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
