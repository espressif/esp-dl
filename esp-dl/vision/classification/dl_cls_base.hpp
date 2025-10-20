#pragma once

#include "dl_cls_postprocessor.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"

namespace dl {
namespace cls {
class Cls {
public:
    virtual ~Cls() {};
    virtual std::vector<dl::cls::result_t> &run(const dl::image::img_t &img) = 0;
    virtual Cls &set_topk(int topk) = 0;
    virtual Cls &set_score_thr(float score_thr) = 0;
    virtual dl::Model *get_raw_model() = 0;
};

class ClsWrapper : public Cls {
protected:
    Cls *m_model;
    int m_topk;
    float m_score_thr;

    virtual void load_model() = 0;

public:
    ~ClsWrapper();
    std::vector<dl::cls::result_t> &run(const dl::image::img_t &img);
    Cls &set_topk(int topk) override;
    Cls &set_score_thr(float score_thr) override;
    dl::Model *get_raw_model() override;
};

class ClsImpl : public Cls {
protected:
    dl::Model *m_model;
    dl::image::ImagePreprocessor *m_image_preprocessor;
    dl::cls::ClsPostprocessor *m_postprocessor;

public:
    ~ClsImpl();
    std::vector<dl::cls::result_t> &run(const dl::image::img_t &img) override;
    Cls &set_topk(int topk) override;
    Cls &set_score_thr(float score_thr) override;
    dl::Model *get_raw_model() override;
};
} // namespace cls
} // namespace dl
