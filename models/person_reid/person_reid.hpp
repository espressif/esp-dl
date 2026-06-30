#pragma once

#include "dl_detect_define.hpp"
#include "dl_feat_base.hpp"
#include "dl_recognition_database.hpp"
#include "dl_tensor_base.hpp"
namespace person_reid {
class OSN : public dl::feat::FeatImpl {
public:
    OSN(const char *model_name);
    ~OSN();
    dl::TensorBase *run(const dl::image::img_t &img, const std::vector<int> &box) override;

private:
    dl::image::ImagePreprocessor *m_reid_image_preprocessor;
};

} // namespace person_reid

class PersonReidFeat : public dl::feat::FeatWrapper {
public:
    typedef enum {
        OSN_S8_V1,
    } model_type_t;

    PersonReidFeat(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_PERSON_REID_FEAT_MODEL),
                   bool lazy_load = true);

private:
    void load_model() override;

    model_type_t m_model_type;
};

class PersonReidMatcher {
private:
    PersonReidFeat m_feat;
    dl::recognition::DataBase *m_db;
    std::string m_db_path;
    float m_thr;
    int m_top_k;

public:
    PersonReidMatcher(const std::string &db_path,
                      PersonReidFeat::model_type_t model_type =
                          static_cast<PersonReidFeat::model_type_t>(CONFIG_DEFAULT_PERSON_REID_FEAT_MODEL),
                      bool lazy_load = true);
    ~PersonReidMatcher();

    std::vector<dl::recognition::result_t> recognize(const dl::image::img_t &img,
                                                     const std::list<dl::detect::result_t> &detect_res);
    esp_err_t enroll(const dl::image::img_t &img, const std::list<dl::detect::result_t> &detect_res);
    esp_err_t clear_all_feats();
    esp_err_t delete_feat(uint16_t id);
    esp_err_t delete_last_feat();
    int get_num_feats();
    PersonReidFeat *get_feat_model();
};
