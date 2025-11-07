#include "dl_image_jpeg.hpp"
#include "hand_detect.hpp"
#include "hand_gesture_recognition.hpp"
#include "bsp/esp-bsp.h"

extern const uint8_t gesture_jpg_start[] asm("_binary_gesture_jpg_start");
extern const uint8_t gesture_jpg_end[] asm("_binary_gesture_jpg_end");
const char *TAG = "hand_gesture_recognition";

extern "C" void app_main(void)
{
#if CONFIG_HAND_DETECT_MODEL_IN_SDCARD || CONFIG_HAND_GESTURE_CLS_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif
    dl::image::jpeg_img_t gesture_jpeg = {.data = (void *)gesture_jpg_start,
                                          .data_len = (size_t)(gesture_jpg_end - gesture_jpg_start)};
    auto gesture = dl::image::sw_decode_jpeg(gesture_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    HandDetect *hand_detect = new HandDetect();
    auto hand_gesture_recognizer = new HandGestureRecognizer(HandGestureCls::MOBILENETV2_0_5_S8_V1);
    std::vector<dl::cls::result_t> results = hand_gesture_recognizer->recognize(gesture, hand_detect->run(gesture));

    for (const auto &res : results) {
        ESP_LOGI(TAG, "category: %s, score: %f", res.cat_name, res.score);
    }

    delete hand_detect;
    delete hand_gesture_recognizer;
    heap_caps_free(gesture.data);

#if CONFIG_HAND_DETECT_MODEL_IN_SDCARD || CONFIG_HAND_GESTURE_CLS_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
