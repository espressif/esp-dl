#pragma once

#include "dl_detect_define.hpp"
#include "dl_model_base.hpp"
#include "dl_recognition_database.hpp"
#include "dl_recognition_human_face_image_preprocessor.hpp"
#include "dl_recognition_postprocessor.hpp"
#include "dl_tensor_base.hpp"

class HumanFaceFeat {
public:
    typedef enum { MODEL_MFN, MODEL_MBF } model_type_t;
    /**
     * @brief Construct a new HumanFaceFeat object
     */
    HumanFaceFeat(model_type_t model_type);

    /**
     * @brief Destroy the HumanFaceFeat object
     */
    ~HumanFaceFeat();

    dl::TensorBase *run(const dl::image::img_t &img, const std::vector<int> &landmarks);

private:
    void *m_model;
    model_type_t m_model_type;
};

class HumanFaceRecognizer : public dl::recognition::DB {
private:
    HumanFaceFeat *m_feat_extract;
    float m_thr;
    int m_top_k;

public:
    HumanFaceRecognizer(
        dl::recognition::db_type_t db_type = static_cast<dl::recognition::db_type_t>(CONFIG_DB_FILE_SYSTEM),
        HumanFaceFeat::model_type_t model_type = static_cast<HumanFaceFeat::model_type_t>(CONFIG_HUMAN_FACE_FEAT_MODEL),
        float thr = 0.5,
        int top_k = 1) :
        dl::recognition::DB(db_type, 512, "face"),
        m_feat_extract(new HumanFaceFeat(model_type)),
        m_thr(thr),
        m_top_k(top_k)
    {
    }

    ~HumanFaceRecognizer();

    std::vector<dl::recognition::result_t> recognize(const dl::image::img_t &img,
                                                     std::list<dl::detect::result_t> &detect_res);
    esp_err_t enroll(const dl::image::img_t &img, std::list<dl::detect::result_t> &detect_res);
};

namespace model_zoo {

class MFN {
private:
    dl::Model *m_model;
    dl::recognition::HumanFaceImagePreprocessor *m_image_preprocessor;
    dl::recognition::RecognitionPostprocessor *m_postprocessor;

public:
    MFN(const std::vector<float> &mean, const std::vector<float> &std);
    ~MFN();

    dl::TensorBase *run(const dl::image::img_t &img, const std::vector<int> &landmarks);
};

using MBF = MFN;
} // namespace model_zoo
