#include "dl_image_jpeg.hpp"
#include "human_face_detect.hpp"
#include "human_face_recognition.hpp"
#include "spiflash_fatfs.hpp"
#include "bsp/esp-bsp.h"

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
#if CONFIG_DB_FATFS_FLASH
    ESP_ERROR_CHECK(fatfs_flash_mount());
#endif
#if CONFIG_DB_SPIFFS
    ESP_ERROR_CHECK(bsp_spiffs_mount());
#endif
#if CONFIG_DB_FATFS_SDCARD || CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD || CONFIG_HUMAN_FACE_FEAT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    dl::image::jpeg_img_t bill1_jpeg = {.data = (void *)bill1_jpg_start,
                                        .data_len = (size_t)(bill1_jpg_end - bill1_jpg_start)};
    auto bill1 = dl::image::sw_decode_jpeg(bill1_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    dl::image::jpeg_img_t bill2_jpeg = {.data = (void *)bill2_jpg_start,
                                        .data_len = (size_t)(bill2_jpg_end - bill2_jpg_start)};
    auto bill2 = dl::image::sw_decode_jpeg(bill2_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    dl::image::jpeg_img_t musk1_jpeg = {.data = (void *)musk1_jpg_start,
                                        .data_len = (size_t)(musk1_jpg_end - musk1_jpg_start)};
    auto musk1 = dl::image::sw_decode_jpeg(musk1_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    dl::image::jpeg_img_t musk2_jpeg = {.data = (void *)musk2_jpg_start,
                                        .data_len = (size_t)(musk2_jpg_end - musk2_jpg_start)};
    auto musk2 = dl::image::sw_decode_jpeg(musk2_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    HumanFaceDetect *human_face_detect = new HumanFaceDetect();

    char db_path[64];
#if CONFIG_DB_FATFS_FLASH
    snprintf(db_path, sizeof(db_path), "%s/face.db", CONFIG_SPIFLASH_MOUNT_POINT);
#elif CONFIG_DB_SPIFFS
    snprintf(db_path, sizeof(db_path), "%s/face.db", CONFIG_BSP_SPIFFS_MOUNT_POINT);
#else
    snprintf(db_path, sizeof(db_path), "%s/face.db", CONFIG_BSP_SD_MOUNT_POINT);
#endif
    auto human_face_recognizer = new HumanFaceRecognizer(db_path);

    human_face_recognizer->enroll(bill1, human_face_detect->run(bill1));
    human_face_recognizer->enroll(bill2, human_face_detect->run(bill2));
    human_face_recognizer->enroll(musk1, human_face_detect->run(musk1));

    auto res = human_face_recognizer->recognize(musk2, human_face_detect->run(musk2));
    for (const auto &k : res) {
        ESP_LOGI(TAG, "id: %d, sim: %f", k.id, k.similarity);
    }

    human_face_recognizer->clear_all_feats();

    delete human_face_detect;
    delete human_face_recognizer;

    heap_caps_free(bill1.data);
    heap_caps_free(bill2.data);
    heap_caps_free(musk1.data);
    heap_caps_free(musk2.data);

#if CONFIG_DB_FATFS_FLASH
    ESP_ERROR_CHECK(fatfs_flash_unmount());
#endif
#if CONFIG_DB_SPIFFS
    ESP_ERROR_CHECK(bsp_spiffs_unmount());
#endif
#if CONFIG_DB_FATFS_SDCARD || CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD || CONFIG_HUMAN_FACE_FEAT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
