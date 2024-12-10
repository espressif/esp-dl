#include "esp_log.h"
#include "human_face_detect.hpp"
#include "human_face_recognition.hpp"

extern const uint8_t bill1_jpg_start[] asm("_binary_bill1_jpg_start");
extern const uint8_t bill1_jpg_end[] asm("_binary_bill1_jpg_end");
extern const uint8_t bill2_jpg_start[] asm("_binary_bill2_jpg_start");
extern const uint8_t bill2_jpg_end[] asm("_binary_bill2_jpg_end");
extern const uint8_t musk1_jpg_start[] asm("_binary_musk1_jpg_start");
extern const uint8_t musk1_jpg_end[] asm("_binary_musk1_jpg_end");
extern const uint8_t musk2_jpg_start[] asm("_binary_musk2_jpg_start");
extern const uint8_t musk2_jpg_end[] asm("_binary_musk2_jpg_end");
const char *TAG = "human_face_recognition";

extern "C" void app_main(void)
{
    dl::image::jpeg_img_t bill1_jpeg = {.data = (uint8_t *)bill1_jpg_start,
                                        .width = 300,
                                        .height = 300,
                                        .data_size = (uint32_t)(bill1_jpg_end - bill1_jpg_start)};
    dl::image::img_t bill1;
    bill1.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    sw_decode_jpeg(bill1_jpeg, bill1);

    dl::image::jpeg_img_t bill2_jpeg = {.data = (uint8_t *)bill2_jpg_start,
                                        .width = 300,
                                        .height = 300,
                                        .data_size = (uint32_t)(bill2_jpg_end - bill2_jpg_start)};
    dl::image::img_t bill2;
    bill2.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    sw_decode_jpeg(bill2_jpeg, bill2);

    dl::image::jpeg_img_t musk1_jpeg = {.data = (uint8_t *)musk1_jpg_start,
                                        .width = 300,
                                        .height = 300,
                                        .data_size = (uint32_t)(musk1_jpg_end - musk1_jpg_start)};
    dl::image::img_t musk1;
    musk1.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    sw_decode_jpeg(musk1_jpeg, musk1);

    dl::image::jpeg_img_t musk2_jpeg = {.data = (uint8_t *)musk2_jpg_start,
                                        .width = 300,
                                        .height = 300,
                                        .data_size = (uint32_t)(musk2_jpg_end - musk2_jpg_start)};
    dl::image::img_t musk2;
    musk2.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    sw_decode_jpeg(musk2_jpeg, musk2);

    auto human_face_detect = new HumanFaceDetect();
    auto human_face_recognizer = new HumanFaceRecognizer();
    human_face_recognizer->enroll(bill1, human_face_detect->run(bill1));
    human_face_recognizer->enroll(bill2, human_face_detect->run(bill2));
    human_face_recognizer->enroll(musk1, human_face_detect->run(musk1));

    auto res = human_face_recognizer->recognize(musk2, human_face_detect->run(musk2));
    for (const auto &k : res) {
        printf("id: %d, sim: %f\n", k.id, k.similarity);
    }

    human_face_recognizer->clear_all_feats();
    delete human_face_recognizer;
    delete human_face_detect;
    heap_caps_free(bill1.data);
    heap_caps_free(bill2.data);
    heap_caps_free(musk1.data);
    heap_caps_free(musk2.data);
}
