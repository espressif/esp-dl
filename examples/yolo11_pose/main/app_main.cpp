#include "coco_pose.hpp"
#include "esp_log.h"
#include "bsp/esp-bsp.h"

extern const uint8_t bus_jpg_start[] asm("_binary_bus_jpg_start");
extern const uint8_t bus_jpg_end[] asm("_binary_bus_jpg_end");
const char *TAG = "yolo11n-pose";

extern "C" void app_main(void)
{
#if CONFIG_COCO_POSE_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    dl::image::jpeg_img_t jpeg_img = {.data = (void *)bus_jpg_start, .data_len = (size_t)(bus_jpg_end - bus_jpg_start)};
    auto img = sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    COCOPose *pose = new COCOPose();
    auto &pose_results = pose->run(img);

    const char *kpt_names[17] = {"nose",
                                 "left eye",
                                 "right eye",
                                 "left ear",
                                 "right ear",
                                 "left shoulder",
                                 "right shoulder",
                                 "left elbow",
                                 "right elbow",
                                 "left wrist",
                                 "right wrist",
                                 "left hip",
                                 "right hip",
                                 "left knee",
                                 "right knee",
                                 "left ankle",
                                 "right ankle"};
    for (const auto &res : pose_results) {
        ESP_LOGI(TAG,
                 "[score: %f, x1: %d, y1: %d, x2: %d, y2: %d]",
                 res.score,
                 res.box[0],
                 res.box[1],
                 res.box[2],
                 res.box[3]);

        char log_buf[512];
        char *p = log_buf;
        for (int i = 0; i < 17; ++i) {
            p += sprintf(p, "%s: [%d, %d] ", kpt_names[i], res.keypoint[2 * i], res.keypoint[2 * i + 1]);
        }
        ESP_LOGI(TAG, "%s", log_buf);
    }
    delete pose;
    heap_caps_free(img.data);

#if CONFIG_COCO_POSE_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
