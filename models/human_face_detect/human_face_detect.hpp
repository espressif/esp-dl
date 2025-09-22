#pragma once

#include "dl_detect_base.hpp"
#include "dl_detect_mnp_postprocessor.hpp"
#include "dl_detect_msr_postprocessor.hpp"
namespace human_face_detect {
class MSR : public dl::detect::DetectImpl {
public:
    static inline constexpr float default_score_thr = 0.5;
    static inline constexpr float default_nms_thr = 0.5;
    MSR(const char *model_name, float score_thr, float nms_thr);
};

class MNP {
private:
    dl::Model *m_model;
    dl::image::ImagePreprocessor *m_image_preprocessor;
    dl::detect::MNPPostprocessor *m_postprocessor;

public:
    static inline constexpr float default_score_thr = 0.5;
    static inline constexpr float default_nms_thr = 0.5;
    MNP(const char *model_name, float score_thr, float nms_thr);
    ~MNP();
    MNP &set_score_thr(float score_thr);
    MNP &set_nms_thr(float nms_thr);
    dl::Model *get_raw_model();
    std::list<dl::detect::result_t> &run(const dl::image::img_t &img, std::list<dl::detect::result_t> &candidates);
};

class MSRMNP : public dl::detect::Detect {
private:
    MSR m_msr;
    MNP m_mnp;

public:
    MSRMNP(const char *msr_model_name,
           float msr_score_thr,
           float msr_nms_thr,
           const char *mnp_model_name,
           float mnp_score_thr,
           float mnp_nms_thr) :
        m_msr(msr_model_name, msr_score_thr, msr_nms_thr), m_mnp(mnp_model_name, mnp_score_thr, mnp_nms_thr)
    {
    }

    std::list<dl::detect::result_t> &run(const dl::image::img_t &img) override;
    Detect &set_score_thr(float score_thr, int idx) override;
    Detect &set_nms_thr(float nms_thr, int idx) override;
    dl::Model *get_raw_model(int idx) override;
};

} // namespace human_face_detect

class HumanFaceDetect : public dl::detect::DetectWrapper {
public:
    typedef enum { MSRMNP_S8_V1 } model_type_t;

    HumanFaceDetect(model_type_t model_type = static_cast<model_type_t>(CONFIG_DEFAULT_HUMAN_FACE_DETECT_MODEL),
                    bool lazy_load = true);

private:
    void load_model() override;

    model_type_t m_model_type;
};
