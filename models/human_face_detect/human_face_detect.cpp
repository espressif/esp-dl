#include "human_face_detect.hpp"

#if CONFIG_HUMAN_FACE_DETECT_MODEL_IN_FLASH_RODATA
extern const uint8_t human_face_detect_espdl[] asm("_binary_human_face_detect_espdl_start");
static const char *path = (const char *)human_face_detect_espdl;
#elif CONFIG_HUMAN_FACE_DETECT_MODEL_IN_FLASH_PARTITION
static const char *path = "human_face_det";
#else
#if !defined(CONFIG_BSP_SD_MOUNT_POINT)
#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"
#endif
#endif
namespace human_face_detect {

MSR::MSR(const char *model_name)
{
#if !CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD
    m_model = new dl::Model(
        path, model_name, static_cast<fbs::model_location_type_t>(CONFIG_HUMAN_FACE_DETECT_MODEL_LOCATION));
#else
    char path[256];
    snprintf(path,
             sizeof(path),
             "%s/%s/%s",
             CONFIG_BSP_SD_MOUNT_POINT,
             CONFIG_HUMAN_FACE_DETECT_MODEL_SDCARD_DIR,
             model_name);
    m_model = new dl::Model(path, static_cast<fbs::model_location_type_t>(CONFIG_HUMAN_FACE_DETECT_MODEL_LOCATION));
#endif
    m_model->minimize();
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor = new dl::image::ImagePreprocessor(m_model, {0, 0, 0}, {1, 1, 1}, DL_IMAGE_CAP_RGB_SWAP);
#else
    m_image_preprocessor = new dl::image::ImagePreprocessor(
        m_model, {0, 0, 0}, {1, 1, 1}, DL_IMAGE_CAP_RGB_SWAP | DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
#endif
    m_postprocessor = new dl::detect::MSRPostprocessor(
        m_model, 0.5, 0.5, 10, {{8, 8, 9, 9, {{16, 16}, {32, 32}}}, {16, 16, 9, 9, {{64, 64}, {128, 128}}}});
}

MNP::MNP(const char *model_name)
{
#if !CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD
    m_model = new dl::Model(
        path, model_name, static_cast<fbs::model_location_type_t>(CONFIG_HUMAN_FACE_DETECT_MODEL_LOCATION));
#else
    char sd_path[256];
    snprintf(sd_path,
             sizeof(sd_path),
             "%s/%s/%s",
             CONFIG_BSP_SD_MOUNT_POINT,
             CONFIG_HUMAN_FACE_DETECT_MODEL_SDCARD_DIR,
             model_name);
    m_model = new dl::Model(sd_path, static_cast<fbs::model_location_type_t>(CONFIG_HUMAN_FACE_DETECT_MODEL_LOCATION));
#endif
    m_model->minimize();
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor = new dl::image::ImagePreprocessor(m_model, {0, 0, 0}, {1, 1, 1}, DL_IMAGE_CAP_RGB_SWAP);
#else
    m_image_preprocessor = new dl::image::ImagePreprocessor(
        m_model, {0, 0, 0}, {1, 1, 1}, DL_IMAGE_CAP_RGB_SWAP | DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
#endif
    m_postprocessor = new dl::detect::MNPPostprocessor(m_model, 0.5, 0.5, 10, {{1, 1, 0, 0, {{48, 48}}}});
}

MNP::~MNP()
{
    delete m_model;
    delete m_image_preprocessor;
    delete m_postprocessor;
};

std::list<dl::detect::result_t> &MNP::run(const dl::image::img_t &img, std::list<dl::detect::result_t> &candidates)
{
    DL_LOG_INFER_LATENCY_ARRAY_INIT_WITH_SIZE(3, 10);
    m_postprocessor->clear_result();
    for (auto &candidate : candidates) {
        int center_x = (candidate.box[0] + candidate.box[2]) >> 1;
        int center_y = (candidate.box[1] + candidate.box[3]) >> 1;
        int side = DL_MAX(candidate.box[2] - candidate.box[0], candidate.box[3] - candidate.box[1]);
        candidate.box[0] = center_x - (side >> 1);
        candidate.box[1] = center_y - (side >> 1);
        candidate.box[2] = candidate.box[0] + side;
        candidate.box[3] = candidate.box[1] + side;
        candidate.limit_box(img.width, img.height);

        DL_LOG_INFER_LATENCY_ARRAY_START(0);
        m_image_preprocessor->preprocess(img, candidate.box);
        DL_LOG_INFER_LATENCY_ARRAY_END(0);

        DL_LOG_INFER_LATENCY_ARRAY_START(1);
        m_model->run();
        DL_LOG_INFER_LATENCY_ARRAY_END(1);

        DL_LOG_INFER_LATENCY_ARRAY_START(2);
        m_postprocessor->set_resize_scale_x(m_image_preprocessor->get_resize_scale_x());
        m_postprocessor->set_resize_scale_y(m_image_preprocessor->get_resize_scale_y());
        m_postprocessor->set_top_left_x(m_image_preprocessor->get_top_left_x());
        m_postprocessor->set_top_left_y(m_image_preprocessor->get_top_left_y());
        m_postprocessor->postprocess();
        DL_LOG_INFER_LATENCY_ARRAY_END(2);
    }
    m_postprocessor->nms();
    std::list<dl::detect::result_t> &result = m_postprocessor->get_result(img.width, img.height);
    if (candidates.size() > 0) {
        DL_LOG_INFER_LATENCY_ARRAY_PRINT(0, "detect", "pre");
        DL_LOG_INFER_LATENCY_ARRAY_PRINT(1, "detect", "model");
        DL_LOG_INFER_LATENCY_ARRAY_PRINT(2, "detect", "post");
    }
    return result;
}

MSRMNP::~MSRMNP()
{
    delete m_msr;
    delete m_mnp;
}

std::list<dl::detect::result_t> &MSRMNP::run(const dl::image::img_t &img)
{
    std::list<dl::detect::result_t> &candidates = m_msr->run(img);
    return m_mnp->run(img, candidates);
}

} // namespace human_face_detect

HumanFaceDetect::HumanFaceDetect(model_type_t model_type)
{
    switch (model_type) {
    case model_type_t::MSRMNP_S8_V1: {
#if CONFIG_HUMAN_FACE_DETECT_MSRMNP_S8_V1 || CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD
        m_model =
            new human_face_detect::MSRMNP("human_face_detect_msr_s8_v1.espdl", "human_face_detect_mnp_s8_v1.espdl");
#else
        ESP_LOGE("human_face_detect", "human_face_detect_msrmnp_s8_v1 is not selected in menuconfig.");
#endif
        break;
    }
    }
}
