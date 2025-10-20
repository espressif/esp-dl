#include "cat_detect.hpp"
#include "esp_log.h"
#include <filesystem>

#if CONFIG_CAT_DETECT_MODEL_IN_FLASH_RODATA
extern const uint8_t cat_detect_espdl[] asm("_binary_cat_detect_espdl_start");
static const char *path = (const char *)cat_detect_espdl;
#elif CONFIG_CAT_DETECT_MODEL_IN_FLASH_PARTITION
static const char *path = "cat_det";
#else
#if !defined(CONFIG_BSP_SD_MOUNT_POINT)
#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"
#endif
#endif
namespace cat_detect {
ESPDet::ESPDet(const char *model_name, float score_thr, float nms_thr)
{
#if !CONFIG_CAT_DETECT_MODEL_IN_SDCARD
    m_model =
        new dl::Model(path, model_name, static_cast<fbs::model_location_type_t>(CONFIG_CAT_DETECT_MODEL_LOCATION));
#else
    auto sd_path = std::filesystem::path(CONFIG_BSP_SD_MOUNT_POINT) / CONFIG_CAT_DETECT_MODEL_SDCARD_DIR / model_name;
    m_model = new dl::Model(sd_path.c_str(), fbs::MODEL_LOCATION_IN_SDCARD);
#endif
    m_model->minimize();
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor = new dl::image::ImagePreprocessor(m_model, {0, 0, 0}, {255, 255, 255});
#else
    m_image_preprocessor = new dl::image::ImagePreprocessor(
        m_model, {0, 0, 0}, {255, 255, 255}, dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
#endif
    m_image_preprocessor->enable_letterbox({114, 114, 114});
    m_postprocessor = new dl::detect::ESPDetPostProcessor(
        m_model, m_image_preprocessor, score_thr, nms_thr, 10, {{8, 8, 4, 4}, {16, 16, 8, 8}, {32, 32, 16, 16}});
}

} // namespace cat_detect

CatDetect::CatDetect(model_type_t model_type, bool lazy_load) : m_model_type(model_type)
{
    switch (model_type) {
    case model_type_t::ESPDET_PICO_224_224_CAT:
    case model_type_t::ESPDET_PICO_416_416_CAT:
        m_score_thr[0] = cat_detect::ESPDet::default_score_thr;
        m_nms_thr[0] = cat_detect::ESPDet::default_nms_thr;
        break;
    }
    if (lazy_load) {
        m_model = nullptr;
    } else {
        load_model();
    }
}

void CatDetect::load_model()
{
    switch (m_model_type) {
    case model_type_t::ESPDET_PICO_224_224_CAT:
#if CONFIG_FLASH_ESPDET_PICO_224_224_CAT || CONFIG_CAT_DETECT_MODEL_IN_SDCARD
        m_model = new cat_detect::ESPDet("espdet_pico_224_224_cat.espdl", m_score_thr[0], m_nms_thr[0]);
#else
        ESP_LOGE("cat_detect", "espdet_pico_224_224_cat is not selected in menuconfig.", m_score_thr[0], m_nms_thr[0]);
#endif
        break;
    case model_type_t::ESPDET_PICO_416_416_CAT:
#if CONFIG_FLASH_ESPDET_PICO_416_416_CAT || CONFIG_CAT_DETECT_MODEL_IN_SDCARD
        m_model = new cat_detect::ESPDet("espdet_pico_416_416_cat.espdl", m_score_thr[0], m_nms_thr[0]);
#else
        ESP_LOGE("cat_detect", "espdet_pico_416_416_cat is not selected in menuconfig.");
#endif
        break;
    }
}
