#include "dl_image_jpeg.hpp"
#include "pedestrian_detect.hpp"
#include "person_reid.hpp"
#include "spiflash_fatfs.hpp"
#include <filesystem>
#if CONFIG_DB_FATFS_SDCARD || CONFIG_PEDESTRIAN_DETECT_MODEL_IN_SDCARD || CONFIG_PERSON_REID_FEAT_MODEL_IN_SDCARD
#include "bsp/esp-bsp.h"
#endif

extern const uint8_t hrx1_jpg_start[] asm("_binary_hrx1_jpg_start");
extern const uint8_t hrx1_jpg_end[] asm("_binary_hrx1_jpg_end");
extern const uint8_t hrx2_jpg_start[] asm("_binary_hrx2_jpg_start");
extern const uint8_t hrx2_jpg_end[] asm("_binary_hrx2_jpg_end");
extern const uint8_t musk1_jpg_start[] asm("_binary_musk1_jpg_start");
extern const uint8_t musk1_jpg_end[] asm("_binary_musk1_jpg_end");
extern const uint8_t musk2_jpg_start[] asm("_binary_musk2_jpg_start");
extern const uint8_t musk2_jpg_end[] asm("_binary_musk2_jpg_end");
const char *TAG = "person_reid";

extern "C" void app_main(void)
{
#if CONFIG_DB_FATFS_FLASH
    ESP_ERROR_CHECK(fatfs_flash_mount());
#endif
#if CONFIG_DB_SPIFFS
    ESP_ERROR_CHECK(bsp_spiffs_mount());
#endif
#if CONFIG_DB_FATFS_SDCARD || CONFIG_PEDESTRIAN_DETECT_MODEL_IN_SDCARD || CONFIG_PERSON_REID_FEAT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    dl::image::jpeg_img_t hrx1_jpeg = {.data = (void *)hrx1_jpg_start,
                                       .data_len = (size_t)(hrx1_jpg_end - hrx1_jpg_start)};
    auto hrx1 = dl::image::sw_decode_jpeg(hrx1_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    dl::image::jpeg_img_t hrx2_jpeg = {.data = (void *)hrx2_jpg_start,
                                       .data_len = (size_t)(hrx2_jpg_end - hrx2_jpg_start)};
    auto hrx2 = dl::image::sw_decode_jpeg(hrx2_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    dl::image::jpeg_img_t musk1_jpeg = {.data = (void *)musk1_jpg_start,
                                        .data_len = (size_t)(musk1_jpg_end - musk1_jpg_start)};
    auto musk1 = dl::image::sw_decode_jpeg(musk1_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    dl::image::jpeg_img_t musk2_jpeg = {.data = (void *)musk2_jpg_start,
                                        .data_len = (size_t)(musk2_jpg_end - musk2_jpg_start)};
    auto musk2 = dl::image::sw_decode_jpeg(musk2_jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    PedestrianDetect *pedestrian_detect = new PedestrianDetect();

#if CONFIG_DB_FATFS_FLASH
    auto db_path = std::filesystem::path(CONFIG_SPIFLASH_MOUNT_POINT) / "person.db";
#elif CONFIG_DB_SPIFFS
    auto db_path = std::filesystem::path(CONFIG_BSP_SPIFFS_MOUNT_POINT) / "person.db";
#else
    auto db_path = std::filesystem::path(CONFIG_BSP_SD_MOUNT_POINT) / "person.db";
#endif

    auto person_reid_matcher = new PersonReidMatcher(db_path.string());

    person_reid_matcher->clear_all_feats();

    auto det_hrx1 = pedestrian_detect->run(hrx1);
    person_reid_matcher->enroll(hrx1, det_hrx1);
    auto det_musk1 = pedestrian_detect->run(musk1);
    person_reid_matcher->enroll(musk1, det_musk1);

    auto res = person_reid_matcher->recognize(hrx2, pedestrian_detect->run(hrx2));
    // auto res = person_reid_matcher->recognize(musk2, pedestrian_detect->run(musk2));

    for (const auto &k : res) {
        ESP_LOGI(TAG, "id: %d, sim: %f", k.id, k.similarity);
    }

    person_reid_matcher->clear_all_feats();

    delete pedestrian_detect;
    delete person_reid_matcher;

    heap_caps_free(hrx1.data);
    heap_caps_free(hrx2.data);
    heap_caps_free(musk1.data);
    heap_caps_free(musk2.data);

#if CONFIG_DB_FATFS_FLASH
    ESP_ERROR_CHECK(fatfs_flash_unmount());
#endif
#if CONFIG_DB_SPIFFS
    ESP_ERROR_CHECK(bsp_spiffs_unmount());
#endif
#if CONFIG_DB_FATFS_SDCARD || CONFIG_PEDESTRIAN_DETECT_MODEL_IN_SDCARD || CONFIG_PERSON_REID_FEAT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}
