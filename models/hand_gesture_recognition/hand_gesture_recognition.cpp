#include "hand_gesture_recognition.hpp"
#include <filesystem>

#if CONFIG_HAND_GESTURE_CLS_MODEL_IN_FLASH_RODATA
extern const uint8_t hand_gesture_cls_espdl[] asm("_binary_hand_gesture_cls_espdl_start");
static const char *path = (const char *)hand_gesture_cls_espdl;
#elif CONFIG_HAND_GESTURE_CLS_MODEL_IN_FLASH_PARTITION
static const char *path = "hand_gesture_cls";
#else
#if !defined(CONFIG_BSP_SD_MOUNT_POINT)
#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"
#endif
#endif

namespace hand_gesture_recognition {

MobileNetV2::MobileNetV2(const char *model_name, int topk, float score_thr)
{
#if !CONFIG_HAND_GESTURE_CLS_MODEL_IN_SDCARD
    m_model = new dl::Model(
        path, model_name, static_cast<fbs::model_location_type_t>(CONFIG_HAND_GESTURE_CLS_MODEL_LOCATION));
#else
    auto sd_path =
        std::filesystem::path(CONFIG_BSP_SD_MOUNT_POINT) / CONFIG_HAND_GESTURE_CLS_MODEL_SDCARD_DIR / model_name;
    m_model = new dl::Model(sd_path, fbs::MODEL_LOCATION_IN_SDCARD);
#endif
    m_model->minimize();
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor =
        new dl::image::ImagePreprocessor(m_model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
#else
    m_image_preprocessor = new dl::image::ImagePreprocessor(
        m_model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
#endif
    m_postprocessor =
        new dl::cls::HandGestureClsPostprocessor(m_model, topk, std::numeric_limits<float>::lowest(), true);
}

std::vector<dl::cls::result_t> MobileNetV2::run_crop(const dl::image::img_t &img, const std::vector<int> &crop_area)
{
    DL_LOG_INFER_LATENCY_INIT();
    DL_LOG_INFER_LATENCY_START();
    m_image_preprocessor->preprocess(img, crop_area);
    DL_LOG_INFER_LATENCY_END_PRINT("cls", "pre");

    DL_LOG_INFER_LATENCY_START();
    m_model->run();
    DL_LOG_INFER_LATENCY_END_PRINT("cls", "model");

    DL_LOG_INFER_LATENCY_START();
    std::vector<dl::cls::result_t> results = m_postprocessor->postprocess();
    DL_LOG_INFER_LATENCY_END_PRINT("cls", "post");

    return results;
}

} // namespace hand_gesture_recognition

HandGestureCls::HandGestureCls(model_type_t model_type, bool lazy_load) : m_model_type(model_type)
{
    switch (model_type) {
    case model_type_t::MOBILENETV2_0_5_S8_V1:
        m_topk = hand_gesture_recognition::MobileNetV2::default_topk;
        m_score_thr = hand_gesture_recognition::MobileNetV2::default_score_thr;
        break;
    }
    if (lazy_load) {
        m_cls_model = nullptr;
    } else {
        load_model();
    }
}

void HandGestureCls::load_model()
{
    switch (m_model_type) {
    case model_type_t::MOBILENETV2_0_5_S8_V1:
#if CONFIG_FLASH_HAND_GESTURE_CLS_MOBILENETV2_0_5_S8_V1 || CONFIG_HAND_GESTURE_CLS_MODEL_IN_SDCARD
        m_cls_model =
            new hand_gesture_recognition::MobileNetV2("mobilenetv2_0_5_128_128_gesture.espdl", m_topk, m_score_thr);
#else
        ESP_LOGE("hand_gesture_cls", "mobilenetv2_0_5_128_128_gesture is not selected in menuconfig.");
#endif
        break;
    default:
        ESP_LOGE("hand_gesture_cls", "Unknown model type.");
    }
    this->m_model = m_cls_model;
}

std::vector<dl::cls::result_t> HandGestureCls::run_crop(const dl::image::img_t &img, const std::vector<int> &crop_area)
{
    if (!m_cls_model) {
        ESP_LOGW("hand_gesture_cls", "Model not loaded, loading now...");
        load_model();
    }
    return m_cls_model->run_crop(img, crop_area);
}

HandGestureRecognizer::HandGestureRecognizer(HandGestureCls::model_type_t model_type) : m_cls(model_type)
{
}

std::vector<dl::cls::result_t> HandGestureRecognizer::recognize(const dl::image::img_t &img,
                                                                const std::list<dl::detect::result_t> &detect_res)
{
    std::vector<dl::cls::result_t> results;
    if (detect_res.empty()) {
        ESP_LOGW("HandGestureRecognizer", "Failed to recognize. No hand detected.");
        return {};
    }
    results.reserve(detect_res.size());

    for (const auto &hand : detect_res) {
        std::vector<int> crop_area = {hand.box[0], hand.box[1], hand.box[2], hand.box[3]};
        auto res = m_cls.run_crop(img, crop_area);
        results.insert(results.end(), res.begin(), res.end());
    }
    return results;
}
