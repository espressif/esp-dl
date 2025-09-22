#pragma once

#include "dl_detect_define.hpp"
#include "dl_feat_base.hpp"
#include "dl_recognition_database.hpp"
#include "dl_tensor_base.hpp"
namespace human_face_recognition {
class MFN : public dl::feat::FeatImpl {
public:
    MFN(const char *model_name);
};

using MBF = MFN;
} // namespace human_face_recognition

class HumanFaceFeat : public dl::feat::FeatWrapper {
public:
    typedef enum {
        MFN_S8_V1,
        MBF_S8_V1,
    } model_type_t;

    HumanFaceFeat(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_HUMAN_FACE_FEAT_MODEL),
                  bool lazy_load = true);

private:
    void load_model() override;

    model_type_t m_model_type;
};

class HumanFaceRecognizer {
private:
    HumanFaceFeat m_feat;
    dl::recognition::DataBase *m_db;
    std::string m_db_path;
    float m_thr;
    int m_top_k;

public:
    HumanFaceRecognizer(const std::string &db_path,
                        HumanFaceFeat::model_type_t model_type =
                            static_cast<HumanFaceFeat::model_type_t>(CONFIG_DEFAULT_HUMAN_FACE_FEAT_MODEL),
                        bool lazy_load = true);
    ~HumanFaceRecognizer();

    std::vector<dl::recognition::result_t> recognize(const dl::image::img_t &img,
                                                     const std::list<dl::detect::result_t> &detect_res);
    esp_err_t enroll(const dl::image::img_t &img, const std::list<dl::detect::result_t> &detect_res);
    esp_err_t clear_all_feats();
    esp_err_t delete_feat(uint16_t id);
    esp_err_t delete_last_feat();
    int get_num_feats();
    HumanFaceFeat *get_feat_model();
};
