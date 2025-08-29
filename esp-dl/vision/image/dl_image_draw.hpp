#pragma once
#include "dl_image_define.hpp"
#include <vector>

namespace dl {
namespace image {
void draw_point(const img_t &img, int x, int y, const std::vector<uint8_t> &color, uint8_t radius);

void draw_hollow_rectangle(
    const img_t &img, int x1, int y1, int x2, int y2, const std::vector<uint8_t> &color, uint8_t line_width);

} // namespace image
} // namespace dl
