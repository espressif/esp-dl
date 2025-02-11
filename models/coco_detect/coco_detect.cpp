#include "coco_detect.hpp"

#if CONFIG_YOLO11_DETECT_MODEL_IN_FLASH_RODATA
extern const uint8_t yolo11_detect_espdl[] asm("_binary_yolo11_detect_espdl_start");
static const char *path = (const char *)yolo11_detect_espdl;
#elif CONFIG_YOLO11_DETECT_MODEL_IN_FLASH_PARTITION
static const char *path = "yolo11_det";
#endif
namespace yolo11_detect {

Yolo11::Yolo11(const char *model_name)
{
#if CONFIG_IDF_TARGET_ESP32P4
    m_model =
        new dl::Model(path, model_name, static_cast<fbs::model_location_type_t>(CONFIG_YOLO11_DETECT_MODEL_LOCATION));
    m_image_preprocessor =
        new dl::image::ImagePreprocessor(m_model, {0, 0, 0}, {255, 255, 255}, DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
#else
    m_model = new dl::Model(path,
                            model_name,
                            static_cast<fbs::model_location_type_t>(CONFIG_YOLO11_DETECT_MODEL_LOCATION),
                            0,
                            dl::MEMORY_MANAGER_GREEDY,
                            0,
                            false);
    m_image_preprocessor = new dl::image::ImagePreprocessor(m_model, {0, 0, 0}, {255, 255, 255});
#endif
    m_postprocessor =
        new dl::detect::yolo11PostProcessor(m_model, 0.25, 0.7, 10, {{8, 8, 4, 4}, {16, 16, 8, 8}, {32, 32, 16, 16}});
}

} // namespace yolo11_detect

COCODetect::COCODetect(model_type_t model_type)
{
    switch (model_type) {
    case model_type_t::YOLO11_n_S8_V1:
#if CONFIG_YOLO11_DETECT_YOLO11_n_S8_V1
        m_model = new yolo11_detect::Yolo11("yolo11n_detect_s8_v1.espdl");
#else
        ESP_LOGE("yolo11_detect", "yolo11n_detect_s8_v1 is not selected in menuconfig.");
#endif
        break;
    case model_type_t::YOLO11_n_S8_V2:
#if CONFIG_YOLO11_DETECT_YOLO11_n_S8_V2
        m_model = new yolo11_detect::Yolo11("yolo11n_detect_s8_v2.espdl");
#else
        ESP_LOGE("yolo11_detect", "yolo11n_detect_s8_v2 is not selected in menuconfig.");
#endif
        break;
    }
}
