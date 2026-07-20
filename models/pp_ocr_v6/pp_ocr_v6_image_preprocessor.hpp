#pragma once
#include "dl_image_define.hpp"
#include "dl_tensor_base.hpp"
#include "pp_ocr_v6.hpp"
#include "pp_ocr_v6_define.hpp"
#include <array>
#include <cstdint>

namespace pp_ocr_v6 {
std::array<Point, 4> text_box_to_points(const TextBox &box);

// Top-left letterbox into dst (RGB888); pad with bg.
void bilinear_letterbox_top_left(
    const dl::image::img_t &src, const dl::image::img_t &dst, int resize_w, int resize_h, uint8_t bg = 127);

float estimate_crop_aspect_ratio(const TextBox &box);

// utility.get_rotate_crop_image; caller frees data.
dl::image::img_t get_rotate_crop_image(const dl::image::img_t &img, const TextBox &box);

// predict_rec.resize_norm_img -> NHWC quantized input (RGB crop written as BGR).
bool resize_norm_img(const dl::image::img_t &img, dl::TensorBase *model_input);

} // namespace pp_ocr_v6
