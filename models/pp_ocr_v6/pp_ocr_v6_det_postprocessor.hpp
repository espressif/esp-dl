#pragma once
#include "dl_image_define.hpp"
#include "dl_model_base.hpp"
#include "pp_ocr_v6.hpp"
#include <vector>

namespace pp_ocr_v6 {
class DetPostprocessor {
public:
    static constexpr float default_thresh = 0.2;
    static constexpr float default_box_thresh = 0.4;
    static constexpr float default_unclip_ratio = 1.4;
    static constexpr int default_min_size = 3;
    static constexpr int default_max_candidates = 3000;

    DetPostprocessor(dl::Model *model,
                     float thresh = default_thresh,
                     float box_thresh = default_box_thresh,
                     float unclip_ratio = default_unclip_ratio,
                     int min_size = default_min_size,
                     int max_candidates = default_max_candidates);

    std::vector<TextBox> postprocess(const dl::image::img_t &img, float resize_scale, int model_w, int model_h);

    void set_thresh(float v) { m_thresh = v; }
    void set_box_thresh(float v) { m_box_thresh = v; }
    void set_unclip_ratio(float v) { m_unclip_ratio = v; }

private:
    dl::Model *m_model;
    float m_thresh;
    float m_box_thresh;
    float m_unclip_ratio;
    int m_min_size;
    int m_max_candidates;
};

} // namespace pp_ocr_v6
