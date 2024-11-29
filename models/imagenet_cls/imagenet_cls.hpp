#pragma once

#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"
#include "imagenet_cls_postprocessor.hpp"

class ImageNetCls {
private:
    void *m_model;

public:
    /**
     * @brief Construct a new ImageNetCls object
     */
    ImageNetCls();

    /**
     * @brief Destroy the ImageNetCls object
     */
    ~ImageNetCls();

    /**
     * @brief Inference.
     *
     * @param img input image
     * @return detection result
     */
    std::vector<dl::cls::result_t> &run(const dl::image::img_t &img);
};

namespace model_zoo {

class ImageNetClsModel {
private:
    dl::Model *m_model;
    dl::image::ImagePreprocessor *m_image_preprocessor;
    dl::cls::ImageNetClsPostprocessor *m_postprocessor;

public:
    ImageNetClsModel(const int top_k,
                     bool need_softmax,
                     const std::vector<float> &mean,
                     const std::vector<float> &std,
                     const float score_thr = std::numeric_limits<float>::lowest());

    ~ImageNetClsModel();

    std::vector<dl::cls::result_t> &run(const dl::image::img_t &img);
};

} // namespace model_zoo
