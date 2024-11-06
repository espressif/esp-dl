#pragma once

#include "dl_model_base.hpp"
#include "dl_recognition_database.hpp"
#include "dl_recognition_face_image_preprocessor.hpp"
#include "dl_recognition_postprocessor.hpp"
#include "dl_tensor_base.hpp"
#include "human_face_detect.hpp"

class HumanFaceFeat {
private:
    void *model;

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

    /**
     * @brief Inference.
     *
     * @tparam T supports uint8_t and uint16_t
     *         - uint8_t: input image is RGB888
     *         - uint16_t: input image is RGB565
     * @param input_element pointer of input image
     * @param input_shape   shape of input image
     * @return detection result
     */
    template <typename T>
    dl::TensorBase *run(T *input_element, const std::vector<int> &input_shape, const std::vector<int> &landmarks);
};

class FaceRecognizer : public dl::recognition::DB {
private:
    HumanFaceDetect *detect;
    HumanFaceFeat *feat_extract;
    float thr;
    int top_k;
    int model_type;

public:
    FaceRecognizer(dl::recognition::db_type_t db_type = dl::recognition::DB_FATFS_FLASH,
                   HumanFaceFeat::model_type_t model_type = HumanFaceFeat::model_type_t::MODEL_MFN,
                   float thr = 0.5,
                   int top_k = 1) :
        dl::recognition::DB(db_type, 512, "face"),
        detect(new HumanFaceDetect()),
        feat_extract(new HumanFaceFeat(model_type)),
        thr(thr),
        top_k(top_k)
    {
    }

    ~FaceRecognizer();

    template <typename T>
    std::vector<std::list<dl::recognition::query_info>> recognize(T *input_element,
                                                                  const std::vector<int> &input_shape);
    template <typename T>
    esp_err_t enroll(T *input_element, const std::vector<int> &input_shape);
};

namespace model_zoo {

template <typename feature_t>
class MFN {
private:
    dl::recognition::FaceImagePreprocessor<feature_t> *image_preprocessor;
    dl::Model *model;
    dl::recognition::RecognitionPostprocessor<feature_t> *postprocessor;

public:
    MFN(const std::vector<float> &mean, const std::vector<float> &std);
    ~MFN();

    template <typename T>
    dl::TensorBase *run(T *input_element, const std::vector<int> &input_shape, const std::vector<int> &landmarks);
};

} // namespace model_zoo
