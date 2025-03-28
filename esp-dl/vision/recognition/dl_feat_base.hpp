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
    int m_feat_len;
};

class FeatWrapper : public Feat {
protected:
    Feat *m_model;

public:
    ~FeatWrapper() { delete m_model; }
    TensorBase *run(const dl::image::img_t &img, const std::vector<int> &landmarks)
    {
        return m_model->run(img, landmarks);
    }
};

class FeatImpl : public Feat {
protected:
    dl::Model *m_model;
    dl::image::FeatImagePreprocessor *m_image_preprocessor;
    dl::feat::FeatPostprocessor *m_postprocessor;

public:
    ~FeatImpl();
    TensorBase *run(const dl::image::img_t &img, const std::vector<int> &landmarks) override;
};
} // namespace feat
} // namespace dl
