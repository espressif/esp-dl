#include "dl_image_draw.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <cstring>

namespace dl {
namespace image {
void draw_point(const img_t &img, int x, int y, const std::vector<uint8_t> &color, uint8_t radius)
{
    assert(x >= 0 && x < img.width && y >= 0 && y < img.height);
    int col_step = get_pix_byte_size(img.pix_type);
    int row_step = col_step * img.width;
    assert(color.size() == col_step);

    uint8_t *p_img = static_cast<uint8_t *>(img.data);
    const uint8_t *p_pix = color.data();

    int radius_pow = radius * radius;
    for (int i = -radius; i < radius + 1; i++) {
        int y_ = y + i;
        if (y_ < 0 || y_ >= img.height) {
            continue;
        }
        int i_pow = i * i;
        for (int j = -radius; j < radius + 1; j++) {
            int x_ = x + j;
            if (x_ < 0 || x_ >= img.width) {
                continue;
            }
            int j_pow = j * j;
            if (i_pow + j_pow <= radius_pow) {
                uint8_t *ptr = p_img + y_ * row_step + x_ * col_step;
                memcpy(ptr, p_pix, col_step);
            }
        }
    }
}

void draw_hollow_rectangle(
    const img_t &img, int x1, int y1, int x2, int y2, const std::vector<uint8_t> &color, uint8_t line_width)
{
    if (is_pix_type_quant(img.pix_type)) {
        ESP_LOGE("dl_image_draw", "Can not draw on a quant img.");
        return;
    }
    assert(x2 > x1 && y2 > y1);
    assert(x1 >= 0 && y1 >= 0);
    assert(x2 < img.width && y2 < img.height);
    int col_step = get_pix_byte_size(img.pix_type);
    int row_step = col_step * img.width;
    assert(color.size() == col_step);

    int line_w1 = line_width / 2;
    int line_w2 = line_width - line_w1;
    int x1_min = std::max(x1 - line_w1, 0);
    int x1_max = x1 + line_w2;
    int x1_len = (x1_max - x1_min) * col_step;
    int x2_min = x2 - line_w1;
    int x2_max = std::min(x2 + line_w2, (int)img.width);
    int x2_len = (x2_max - x2_min) * col_step;
    int y1_min = std::max(y1 - line_w1, 0);
    int y1_max = y1 + line_w2;
    int y2_min = y2 - line_w1;
    int y2_max = std::min(y2 + line_w2, (int)img.height);

    const uint8_t *p_pix = color.data();
    int row_len = col_step * (x2_max - x1_min);
    uint8_t *row = (uint8_t *)heap_caps_malloc(row_len, MALLOC_CAP_DEFAULT);
    uint8_t *p_row = row;
    for (int i = x1_min; i < x2_max; i++, p_row += col_step) {
        memcpy(p_row, p_pix, col_step);
    }

    uint8_t *p_img = static_cast<uint8_t *>(img.data) + row_step * y1_min + col_step * x1_min;
    for (int i = y1_min; i < y1_max; i++, p_img += row_step) {
        memcpy(p_img, row, row_len);
    }
    int gap1 = (x2_min - x1_min) * col_step;
    int gap2 = row_step - gap1;
    for (int i = y1_max; i < y2_min; i++, p_img += gap2) {
        memcpy(p_img, row, x1_len);
        p_img += gap1;
        memcpy(p_img, row, x2_len);
    }
    for (int i = y2_min; i < y2_max; i++, p_img += row_step) {
        memcpy(p_img, row, row_len);
    }
    heap_caps_free(row);
}
} // namespace image
} // namespace dl
