#pragma once
#include "dl_image_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "sdkconfig.h"
#include <array>
#include <string>
#include <vector>

namespace pp_ocr_v6 {
class DetPostprocessor;
class RecCTCPostprocessor;

enum class RecModel : int {
    PP_OCR_V6_REC_S8 = 0,
    PP_OCR_V6_REC_S16 = 1,
};

enum class RecMode : int {
    Short = 0, // 48x320
    Dual = 1,  // 48x320 + 48x640
};

struct TextBox {
    std::array<int, 8> points;
    float score;
};

struct OCRResult {
    TextBox box;
    std::string text;
    float score;
};

class Det {
public:
    explicit Det(const char *model_name = "pp_ocr_v6_det_s8.espdl");
    ~Det();

    Det(const Det &) = delete;
    Det &operator=(const Det &) = delete;
    Det(Det &&) = delete;
    Det &operator=(Det &&) = delete;

    std::vector<TextBox> run(const dl::image::img_t &img);

private:
    dl::Model *m_model;
    dl::image::ImagePreprocessor *m_image_preprocessor;
    DetPostprocessor *m_postprocessor;
    dl::image::img_t m_letterbox_img;
    float m_resize_scale;
};

class Rec {
public:
    explicit Rec(const char *model_name = "pp_ocr_v6_rec_s16.espdl");
    ~Rec();

    Rec(const Rec &) = delete;
    Rec &operator=(const Rec &) = delete;
    Rec(Rec &&) = delete;
    Rec &operator=(Rec &&) = delete;

    std::string run(const dl::image::img_t &img, const TextBox &box, float *score = nullptr);
    float input_aspect_ratio() const;

private:
    dl::Model *m_model;
    dl::image::ImagePreprocessor *m_image_preprocessor;
    RecCTCPostprocessor *m_postprocessor;
};

class PPOCRV6 {
public:
    static constexpr float default_rec_score_threshold = 0.5; // drop_score

    explicit PPOCRV6(RecMode rec_mode = static_cast<RecMode>(CONFIG_PP_OCR_V6_REC_MODE),
                     RecModel rec_type = static_cast<RecModel>(CONFIG_DEFAULT_PP_OCR_V6_REC_MODEL));
    ~PPOCRV6();

    PPOCRV6(const PPOCRV6 &) = delete;
    PPOCRV6 &operator=(const PPOCRV6 &) = delete;
    PPOCRV6(PPOCRV6 &&) = delete;
    PPOCRV6 &operator=(PPOCRV6 &&) = delete;

    std::vector<OCRResult> run(const dl::image::img_t &img);

    float get_rec_score_threshold() const { return m_rec_score_threshold; }
    void set_rec_score_threshold(float v) { m_rec_score_threshold = v; }

private:
    Det m_det;
    Rec m_rec;
    Rec *m_rec_long;
    float m_dual_aspect_ratio_threshold;
    float m_rec_score_threshold;
};

} // namespace pp_ocr_v6
