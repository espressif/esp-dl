#include "person_reid.hpp"
#include <filesystem>

#if CONFIG_PERSON_REID_FEAT_MODEL_IN_FLASH_RODATA
extern const uint8_t person_reid_feat_espdl[] asm("_binary_person_reid_feat_espdl_start");
static const char *path = (const char *)person_reid_feat_espdl;
#elif CONFIG_PERSON_REID_FEAT_MODEL_IN_FLASH_PARTITION
static const char *path = "person_reid_feat";
#else
#if !defined(CONFIG_BSP_SD_MOUNT_POINT)
#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"
#endif
#endif
namespace person_reid {

OSN::OSN(const char *model_name)
{
#if !CONFIG_PERSON_REID_FEAT_MODEL_IN_SDCARD
    m_model = new dl::Model(
        path, model_name, static_cast<fbs::model_location_type_t>(CONFIG_PERSON_REID_FEAT_MODEL_LOCATION));
#else
    auto sd_path =
        std::filesystem::path(CONFIG_BSP_SD_MOUNT_POINT) / CONFIG_PERSON_REID_FEAT_MODEL_SDCARD_DIR / model_name;
    m_model = new dl::Model(sd_path.c_str(), fbs::MODEL_LOCATION_IN_SDCARD);
#endif
    m_model->minimize();
    m_image_preprocessor = nullptr;
    m_reid_image_preprocessor =
        new dl::image::ImagePreprocessor(m_model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    m_postprocessor = new dl::feat::FeatPostprocessor(m_model);
}

OSN::~OSN()
{
    delete m_reid_image_preprocessor;
}

dl::TensorBase *OSN::run(const dl::image::img_t &img, const std::vector<int> &box)
{
    if (box.size() != 4) {
        ESP_LOGE("person_reid_feat", "Invalid person crop box, expected [x1, y1, x2, y2].");
        return nullptr;
    }

    DL_LOG_INFER_LATENCY_INIT();
    DL_LOG_INFER_LATENCY_START();
    m_reid_image_preprocessor->preprocess(img, box);
    DL_LOG_INFER_LATENCY_END_PRINT("feat", "pre");

    DL_LOG_INFER_LATENCY_START();
    m_model->run();
    DL_LOG_INFER_LATENCY_END_PRINT("feat", "model");

    DL_LOG_INFER_LATENCY_START();
    dl::TensorBase *feat = m_postprocessor->postprocess();
    DL_LOG_INFER_LATENCY_END_PRINT("feat", "post");

    return feat;
}

} // namespace person_reid

PersonReidFeat::PersonReidFeat(model_type_t model_type, bool lazy_load) : m_model_type(model_type)
{
    if (lazy_load) {
        m_model = nullptr;
    } else {
        load_model();
    }
}

void PersonReidFeat::load_model()
{
    switch (m_model_type) {
    case model_type_t::OSN_S8_V1:
#if CONFIG_FLASH_PERSON_REID_FEAT_OSN_S8_V1 || CONFIG_PERSON_REID_FEAT_MODEL_IN_SDCARD
        m_model = new person_reid::OSN("person_reid_feat_osn_s8_v1.espdl");
#else
        ESP_LOGE("person_reid_feat", "person_reid_feat_osn_s8_v1 is not selected in menuconfig.");
#endif
        break;
    default:
        ESP_LOGE("person_reid_feat", "Unknown model type.");
    }
}

PersonReidMatcher::PersonReidMatcher(const std::string &db_path,
                                     PersonReidFeat::model_type_t model_type,
                                     bool lazy_load) :
    m_feat(model_type, lazy_load), m_db_path(db_path), m_thr(0.001), m_top_k(1)
{
    if (lazy_load) {
        m_db = nullptr;
    } else {
        m_db = new dl::recognition::DataBase(m_db_path, m_feat.get_feat_len());
    }
}

PersonReidMatcher::~PersonReidMatcher()
{
    delete m_db;
}

std::vector<dl::recognition::result_t> PersonReidMatcher::recognize(const dl::image::img_t &img,
                                                                    const std::list<dl::detect::result_t> &detect_res)
{
    if (!m_db) {
        m_db = new dl::recognition::DataBase(m_db_path, m_feat.get_feat_len());
    }

    if (detect_res.empty()) {
        ESP_LOGW("PersonReidMatcher", "Failed to recognize. No person detected.");
        return {};
    } else if (detect_res.size() == 1) {
        auto feat = m_feat.run(img, detect_res.back().box);
        return m_db->query_feat(feat, m_thr, m_top_k);
    } else {
        auto max_detect_res =
            std::max_element(detect_res.begin(),
                             detect_res.end(),
                             [](const dl::detect::result_t &a, const dl::detect::result_t &b) -> bool {
                                 return a.box_area() < b.box_area();
                             });
        auto feat = m_feat.run(img, max_detect_res->box);
        return m_db->query_feat(feat, m_thr, m_top_k);
    }
}

esp_err_t PersonReidMatcher::enroll(const dl::image::img_t &img, const std::list<dl::detect::result_t> &detect_res)
{
    if (!m_db) {
        m_db = new dl::recognition::DataBase(m_db_path, m_feat.get_feat_len());
    }
    if (detect_res.empty()) {
        ESP_LOGW("PersonReidMatcher", "Failed to enroll. No person detected.");
        return ESP_FAIL;
    } else if (detect_res.size() == 1) {
        auto feat = m_feat.run(img, detect_res.back().box);
        return m_db->enroll_feat(feat);
    } else {
        auto max_detect_res =
            std::max_element(detect_res.begin(),
                             detect_res.end(),
                             [](const dl::detect::result_t &a, const dl::detect::result_t &b) -> bool {
                                 return a.box_area() < b.box_area();
                             });
        auto feat = m_feat.run(img, max_detect_res->box);
        return m_db->enroll_feat(feat);
    }
}

esp_err_t PersonReidMatcher::clear_all_feats()
{
    if (!m_db) {
        m_db = new dl::recognition::DataBase(m_db_path, m_feat.get_feat_len());
    }
    return m_db->clear_all_feats();
}

esp_err_t PersonReidMatcher::delete_feat(uint16_t id)
{
    if (!m_db) {
        m_db = new dl::recognition::DataBase(m_db_path, m_feat.get_feat_len());
    }
    return m_db->delete_feat(id);
}

esp_err_t PersonReidMatcher::delete_last_feat()
{
    if (!m_db) {
        m_db = new dl::recognition::DataBase(m_db_path, m_feat.get_feat_len());
    }
    return m_db->delete_last_feat();
}

int PersonReidMatcher::get_num_feats()
{
    if (!m_db) {
        m_db = new dl::recognition::DataBase(m_db_path, m_feat.get_feat_len());
    }
    return m_db->get_num_feats();
}

PersonReidFeat *PersonReidMatcher::get_feat_model()
{
    return &m_feat;
}
