#include "imagenet_cls.hpp"

extern const uint8_t imagenet_cls_espdl[] asm("_binary_imagenet_cls_espdl_start");

ImageNetCls::ImageNetCls()
{
    m_model = (void *)new model_zoo::ImageNetClsModel(5, true, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
}

ImageNetCls::~ImageNetCls()
{
    if (m_model) {
        delete (model_zoo::ImageNetClsModel *)m_model;
        m_model = nullptr;
    }
}

std::vector<dl::cls::result_t> &ImageNetCls::run(const dl::image::img_t &img)
{
    return ((model_zoo::ImageNetClsModel *)m_model)->run(img);
}

namespace model_zoo {

ImageNetClsModel::ImageNetClsModel(const int top_k,
                                   bool need_softmax,
                                   const std::vector<float> &mean,
                                   const std::vector<float> &std,
                                   const float score_thr) :
    m_model(new dl::Model((const char *)imagenet_cls_espdl)),
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor(
        new dl::image::ImagePreprocessor(m_model, mean, std, DL_IMAGE_CAP_RGB565_BIG_ENDIAN | DL_IMAGE_CAP_RGB_SWAP)),
#else
    m_image_preprocessor(new dl::image::ImagePreprocessor(m_model, mean, std, DL_IMAGE_CAP_RGB_SWAP)),
#endif
    m_postprocessor(new dl::cls::ImageNetClsPostprocessor(m_model, score_thr, top_k, need_softmax))
{
}

ImageNetClsModel::~ImageNetClsModel()
{
    if (m_model) {
        delete m_model;
        m_model = nullptr;
    }
    if (m_image_preprocessor) {
        delete m_image_preprocessor;
        m_image_preprocessor = nullptr;
    }
    if (m_postprocessor) {
        delete m_postprocessor;
        m_postprocessor = nullptr;
    }
}

std::vector<dl::cls::result_t> &ImageNetClsModel::run(const dl::image::img_t &img)
{
    dl::tool::Latency latency[3] = {dl::tool::Latency(), dl::tool::Latency(), dl::tool::Latency()};
    latency[0].start();
    m_image_preprocessor->preprocess(img);
    latency[0].end();

    latency[1].start();
    m_model->run();
    latency[1].end();

    latency[2].start();
    std::vector<dl::cls::result_t> &result = m_postprocessor->postprocess();
    latency[2].end();

    latency[0].print("recognition", "preprocess");
    latency[1].print("recognition", "forward");
    latency[2].print("recognition", "postprocess");
    return result;
}
} // namespace model_zoo
