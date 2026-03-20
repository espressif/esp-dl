#include "motion_detect.hpp"
#include "dl_image_color.hpp"
#include "dl_math.hpp"
#include "esp_log.h"

static const char *TAG = "motion";

namespace dl {
namespace image {
struct PixCompareFunctor {
    uint8_t *p1;
    uint8_t *p2;
    int width;
    int height;
    int step;
    int gap;
    int stride;
    uint8_t threshold;

    template <typename PixelCvt>
    uint32_t operator()(const PixelCvt &pixel_cvt)
    {
        uint32_t n_moving_pts = 0;
        uint8_t v1, v2;
        for (int i = 0; i < height; i += stride) {
            for (int j = 0; j < width; j += stride) {
                pixel_cvt(p1, &v1);
                pixel_cvt(p2, &v2);
                if (DL_ABS(v1 - v2) > threshold) {
                    n_moving_pts++;
                }
                p1 += step;
                p2 += step;
            }
            p1 += gap;
            p2 += gap;
        }
        return n_moving_pts;
    }
};

uint32_t get_moving_point_number(const img_t &img1, const img_t &img2, int stride, uint8_t threshold)
{
    if (img1.pix_type != img2.pix_type || img1.width != img2.width || img1.height != img2.height) {
        ESP_LOGE(TAG, "Image different in pix_type/width/height.");
        return -1;
    }

    uint8_t *p1 = (uint8_t *)img1.data;
    uint8_t *p2 = (uint8_t *)img2.data;
    int width = img1.width;
    int height = img1.height;
    int step = img1.col_step() * stride;
    int gap = img1.col_step() * (width % stride);
    PixCompareFunctor pix_compare{p1, p2, width, height, step, gap, stride, threshold};

    switch (img1.pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
        return pix_compare(RGB8882Gray<false>());
    case DL_IMAGE_PIX_TYPE_BGR888:
        return pix_compare(RGB8882Gray<true>());
    case DL_IMAGE_PIX_TYPE_RGB565LE:
        return pix_compare(RGB5652Gray<false, false>());
    case DL_IMAGE_PIX_TYPE_RGB565BE:
        return pix_compare(RGB5652Gray<true, false>());
    case DL_IMAGE_PIX_TYPE_BGR565LE:
        return pix_compare(RGB5652Gray<false, true>());
    case DL_IMAGE_PIX_TYPE_BGR565BE:
        return pix_compare(RGB5652Gray<true, true>());
    case DL_IMAGE_PIX_TYPE_YUYV:
        return pix_compare(YUV2Gray<true>());
    case DL_IMAGE_PIX_TYPE_UYVY:
        return pix_compare(YUV2Gray<false>());
    case DL_IMAGE_PIX_TYPE_GRAY: {
        int n_moving_pts = 0;
        for (int i = 0; i < height; i += stride) {
            for (int j = 0; j < width; j += stride) {
                if (DL_ABS(*p1 - *p2) > threshold) {
                    n_moving_pts++;
                }
                p1 += step;
                p2 += step;
            }
            p1 += gap;
            p2 += gap;
        }
        return n_moving_pts;
    }
    default:
        ESP_LOGE(TAG, "Invalid pix_type %s", pix_type2str(img1.pix_type).c_str());
        return -1;
    }
}
} // namespace image
} // namespace dl
