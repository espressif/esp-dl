#pragma once

#include "dl_feat_image_preprocessor.hpp"
#include "dl_feat_postprocessor.hpp"
#include "dl_model_base.hpp"

namespace dl {
namespace feat {
class Feat {
public:
    virtual ~Feat() {};
    virtual TensorBase *run(const dl::image::img_t &img, const std::vector<int> &landmarks) = 0;
    virtual int get_feat_len() = 0;
    virtual dl::Model *get_raw_model() = 0;
};

class FeatWrapper : public Feat {
protected:
    Feat *m_model;

    virtual void load_model() = 0;

public:
    ~FeatWrapper();
    TensorBase *run(const dl::image::img_t &img, const std::vector<int> &landmarks) override;
    int get_feat_len() override;
    dl::Model *get_raw_model() override;
};

class FeatImpl : public Feat {
protected:
    dl::Model *m_model;
    dl::image::FeatImagePreprocessor *m_image_preprocessor;
    dl::feat::FeatPostprocessor *m_postprocessor;

public:
    ~FeatImpl();
    TensorBase *run(const dl::image::img_t &img, const std::vector<int> &landmarks) override;
    int get_feat_len() override;
    dl::Model *get_raw_model() override;
};
} // namespace feat
} // namespace dl
