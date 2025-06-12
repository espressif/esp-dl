#include "human_face_recognition.hpp"

#if CONFIG_HUMAN_FACE_FEAT_MODEL_IN_FLASH_RODATA
extern const uint8_t human_face_feat_espdl[] asm("_binary_human_face_feat_espdl_start");
static const char *path = (const char *)human_face_feat_espdl;
#elif CONFIG_HUMAN_FACE_FEAT_MODEL_IN_FLASH_PARTITION
static const char *path = "human_face_feat";
#else
#if !defined(CONFIG_BSP_SD_MOUNT_POINT)
#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"
#endif
#endif
namespace human_face_recognition {

MFN::MFN(const char *model_name)
{
#if !CONFIG_HUMAN_FACE_FEAT_MODEL_IN_SDCARD
    m_model =
        new dl::Model(path, model_name, static_cast<fbs::model_location_type_t>(CONFIG_HUMAN_FACE_FEAT_MODEL_LOCATION));
#else
    char sd_path[256];
    snprintf(sd_path,
             sizeof(sd_path),
             "%s/%s/%s",
             CONFIG_BSP_SD_MOUNT_POINT,
             CONFIG_HUMAN_FACE_FEAT_MODEL_SDCARD_DIR,
             model_name);
    m_model = new dl::Model(sd_path, static_cast<fbs::model_location_type_t>(CONFIG_HUMAN_FACE_FEAT_MODEL_LOCATION));
#endif
    m_model->minimize();
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor = new dl::image::FeatImagePreprocessor(
        m_model, {127.5, 127.5, 127.5}, {127.5, 127.5, 127.5}, DL_IMAGE_CAP_RGB_SWAP);
#else
    m_image_preprocessor = new dl::image::FeatImagePreprocessor(
        m_model, {127.5, 127.5, 127.5}, {127.5, 127.5, 127.5}, DL_IMAGE_CAP_RGB_SWAP | DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
#endif
    m_postprocessor = new dl::feat::FeatPostprocessor(m_model);
    m_feat_len = m_model->get_output()->get_size();
}

} // namespace human_face_recognition

HumanFaceFeat::HumanFaceFeat(model_type_t model_type)
{
    switch (model_type) {
    case model_type_t::MFN_S8_V1:
#if CONFIG_FLASH_HUMAN_FACE_FEAT_MFN_S8_V1 || CONFIG_HUMAN_FACE_FEAT_MODEL_IN_SDCARD
        m_model = new human_face_recognition::MFN("human_face_feat_mfn_s8_v1.espdl");
#else
        ESP_LOGE("human_face_feat", "human_face_feat_mfn_s8_v1 is not selected in menuconfig.");
#endif
        break;
    case model_type_t::MBF_S8_V1:
#if CONFIG_HUMAN_FACE_FEAT_MBF_S8_V1 || CONFIG_HUMAN_FACE_FEAT_MODEL_IN_SDCARD
        m_model = new human_face_recognition::MBF("human_face_feat_mbf_s8_v1.espdl");
#else
        ESP_LOGE("human_face_feat", "human_face_feat_mbf_s8_v1 is not selected in menuconfig.");
#endif
        break;
    default:
        ESP_LOGE("human_face_feat", "Unknown model type.");
    }
    m_feat_len = m_model->m_feat_len;
}

HumanFaceRecognizer::HumanFaceRecognizer(char *db_path, HumanFaceFeat::model_type_t model_type, float thr, int top_k) :
    m_feat(model_type), m_db(db_path, m_feat.m_feat_len), m_thr(thr), m_top_k(top_k)
{
}

std::vector<dl::recognition::result_t> HumanFaceRecognizer::recognize(const dl::image::img_t &img,
                                                                      std::list<dl::detect::result_t> &detect_res)
{
    std::vector<std::vector<dl::recognition::result_t>> res;
    if (detect_res.empty()) {
        ESP_LOGW("HumanFaceRecognizer", "Failed to recognize. No face detected.");
        return {};
    } else if (detect_res.size() == 1) {
        auto feat = m_feat.run(img, detect_res.back().keypoint);
        return m_db.query_feat(feat, m_thr, m_top_k);
    } else {
        auto max_detect_res =
            std::max_element(detect_res.begin(),
                             detect_res.end(),
                             [](const dl::detect::result_t &a, const dl::detect::result_t &b) -> bool {
                                 return a.box_area() > b.box_area();
                             });
        auto feat = m_feat.run(img, max_detect_res->keypoint);
        return m_db.query_feat(feat, m_thr, m_top_k);
    }
}

esp_err_t HumanFaceRecognizer::enroll(const dl::image::img_t &img, std::list<dl::detect::result_t> &detect_res)
{
    if (detect_res.empty()) {
        ESP_LOGW("HumanFaceRecognizer", "Failed to enroll. No face detected.");
        return ESP_FAIL;
    } else if (detect_res.size() == 1) {
        auto feat = m_feat.run(img, detect_res.back().keypoint);
        return m_db.enroll_feat(feat);
    } else {
        auto max_detect_res =
            std::max_element(detect_res.begin(),
                             detect_res.end(),
                             [](const dl::detect::result_t &a, const dl::detect::result_t &b) -> bool {
                                 return a.box_area() > b.box_area();
                             });
        auto feat = m_feat.run(img, max_detect_res->keypoint);
        return m_db.enroll_feat(feat);
    }
}

esp_err_t HumanFaceRecognizer::clear_all_feats()
{
    return m_db.clear_all_feats();
}

esp_err_t HumanFaceRecognizer::delete_feat(uint16_t id)
{
    return m_db.delete_feat(id);
}

esp_err_t HumanFaceRecognizer::delete_last_feat()
{
    return m_db.delete_last_feat();
}

int HumanFaceRecognizer::get_num_feats()
{
    return m_db.get_num_feats();
}
