#include "pedestrian_detect.hpp"

#if CONFIG_PEDESTRIAN_DETECT_MODEL_IN_FLASH_RODATA
extern const uint8_t pedestrian_detect_espdl[] asm("_binary_pedestrian_detect_espdl_start");
static const char *path = (const char *)pedestrian_detect_espdl;
#elif CONFIG_PEDESTRIAN_DETECT_MODEL_IN_FLASH_PARTITION
static const char *path = "pedestrian_det";
#else
#if !defined(CONFIG_BSP_SD_MOUNT_POINT)
#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"
#endif
#endif
namespace pedestrian_detect {

Pico::Pico(const char *model_name)
{
#if !CONFIG_PEDESTRIAN_DETECT_MODEL_IN_SDCARD
    m_model = new dl::Model(
        path, model_name, static_cast<fbs::model_location_type_t>(CONFIG_PEDESTRIAN_DETECT_MODEL_LOCATION));
#else
    char sd_path[256];
    snprintf(sd_path,
             sizeof(sd_path),
             "%s/%s/%s",
             CONFIG_BSP_SD_MOUNT_POINT,
             CONFIG_PEDESTRIAN_DETECT_MODEL_SDCARD_DIR,
             model_name);
    m_model = new dl::Model(sd_path, static_cast<fbs::model_location_type_t>(CONFIG_PEDESTRIAN_DETECT_MODEL_LOCATION));
#endif
    m_model->minimize();
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor = new dl::image::ImagePreprocessor(m_model, {0, 0, 0}, {1, 1, 1});
#else
    m_image_preprocessor =
        new dl::image::ImagePreprocessor(m_model, {0, 0, 0}, {1, 1, 1}, dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
#endif
    m_postprocessor = new dl::detect::PicoPostprocessor(
        m_model, m_image_preprocessor, 0.7, 0.5, 10, {{8, 8, 4, 4}, {16, 16, 8, 8}, {32, 32, 16, 16}});
}

} // namespace pedestrian_detect

PedestrianDetect::PedestrianDetect(model_type_t model_type)
{
    switch (model_type) {
    case model_type_t::PICO_S8_V1:
#if CONFIG_FLASH_PEDESTRIAN_DETECT_PICO_S8_V1 || CONFIG_PEDESTRIAN_DETECT_MODEL_IN_SDCARD
        m_model = new pedestrian_detect::Pico("pedestrian_detect_pico_s8_v1.espdl");
#else
        ESP_LOGE("pedestrian_detect", "pedestrian_detect_s8_v1 is not selected in menuconfig.");
#endif
        break;
    }
}
