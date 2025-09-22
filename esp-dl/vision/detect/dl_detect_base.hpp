#pragma once

#include "dl_detect_postprocessor.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"

namespace dl {
namespace detect {
class Detect {
public:
    virtual ~Detect() {};
    virtual std::list<dl::detect::result_t> &run(const dl::image::img_t &img) = 0;
    virtual Detect &set_score_thr(float score_thr, int idx) = 0;
    virtual Detect &set_nms_thr(float nms_thr, int idx) = 0;
    virtual dl::Model *get_raw_model(int idx) = 0;
};

class DetectWrapper : public Detect {
protected:
    Detect *m_model;
    float m_score_thr[2];
    float m_nms_thr[2];

    virtual void load_model() = 0;

public:
    ~DetectWrapper();
    std::list<dl::detect::result_t> &run(const dl::image::img_t &img) override;
    Detect &set_score_thr(float score_thr, int idx = 0) override;
    Detect &set_nms_thr(float nms_thr, int idx = 0) override;
    dl::Model *get_raw_model(int idx = 0) override;
};

class DetectImpl : public Detect {
protected:
    dl::Model *m_model;
    dl::image::ImagePreprocessor *m_image_preprocessor;
    dl::detect::DetectPostprocessor *m_postprocessor;

public:
    ~DetectImpl();
    std::list<dl::detect::result_t> &run(const dl::image::img_t &img) override;
    Detect &set_score_thr(float score_thr, int idx = 0) override;
    Detect &set_nms_thr(float nms_thr, int idx = 0) override;
    dl::Model *get_raw_model(int idx = 0) override;
};
} // namespace detect
} // namespace dl
