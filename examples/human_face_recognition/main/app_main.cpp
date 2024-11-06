#include "esp_log.h"
#include "human_face_recognition.hpp"
#include "jpeg_decoder.h"

extern const uint8_t bill1_jpg_start[] asm("_binary_bill1_jpg_start");
extern const uint8_t bill1_jpg_end[] asm("_binary_bill1_jpg_end");
extern const uint8_t bill2_jpg_start[] asm("_binary_bill2_jpg_start");
extern const uint8_t bill2_jpg_end[] asm("_binary_bill2_jpg_end");
extern const uint8_t musk1_jpg_start[] asm("_binary_musk1_jpg_start");
extern const uint8_t musk1_jpg_end[] asm("_binary_musk1_jpg_end");
extern const uint8_t musk2_jpg_start[] asm("_binary_musk2_jpg_start");
extern const uint8_t musk2_jpg_end[] asm("_binary_musk2_jpg_end");
const char *TAG = "human_face_recognition";

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
    uint8_t *bill1 = get_image(bill1_jpg_start, bill1_jpg_end - bill1_jpg_start, 300, 300);
    uint8_t *bill2 = get_image(bill2_jpg_start, bill2_jpg_end - bill2_jpg_start, 300, 300);
    uint8_t *musk1 = get_image(musk1_jpg_start, musk1_jpg_end - musk1_jpg_start, 300, 300);
    uint8_t *musk2 = get_image(musk2_jpg_start, musk2_jpg_end - musk2_jpg_start, 300, 300);

    auto face_recognizer = new FaceRecognizer(static_cast<dl::recognition::db_type_t>(CONFIG_DB_FILE_SYSTEM),
                                              static_cast<HumanFaceFeat::model_type_t>(CONFIG_HUMAN_FACE_FEAT_MODEL));
    face_recognizer->enroll(bill1, {300, 300, 3});
    face_recognizer->enroll(bill2, {300, 300, 3});
    face_recognizer->enroll(musk1, {300, 300, 3});

    auto res = face_recognizer->recognize(musk2, {300, 300, 3});
    for (const auto &top_k : res) {
        for (const auto &k : top_k) {
            printf("id: %d, sim: %f\n", k.id, k.similarity);
        }
    }

    face_recognizer->clear_all_feats();
    delete face_recognizer;
    heap_caps_free(bill1);
    heap_caps_free(bill2);
    heap_caps_free(musk1);
    heap_caps_free(musk2);
}
