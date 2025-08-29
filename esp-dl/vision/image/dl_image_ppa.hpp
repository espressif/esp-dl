#pragma once
#include "sdkconfig.h"

#if CONFIG_SOC_PPA_SUPPORTED
#include "dl_image_define.hpp"
#include "esp_log.h"
#include <cmath>
#include <vector>
#include "driver/ppa.h"
#include "hal/cache_hal.h"
#include "hal/cache_ll.h"

namespace dl {
namespace image {
constexpr inline float get_ppa_scale(uint16_t src, uint16_t dst)
{
    if (src == dst) {
        return 1;
    }
    float scale = (float)dst / (float)src;
    // ppa use 8 bit to store int part of scale, 4 bit to store frac part of scale.
    if (!(scale >= 0.0625 && scale < 256)) {
        ESP_LOGE("PPA",
                 "PPA can not scale from %d to %d, expected scale is %.2f, only support [0.0625, 256).",
                 src,
                 dst,
                 scale);
        return -1;
    }
    float scale_int;
    float scale_frac = modff(scale, &scale_int);
    scale_frac = floorf((scale_frac) / 0.0625) * 0.0625;
    return scale_int + scale_frac;
}

constexpr inline esp_err_t convert_pix_type_to_ppa_srm_fmt(pix_type_t type, ppa_srm_color_mode_t *srm_fmt)
{
    if (type == DL_IMAGE_PIX_TYPE_RGB888) {
        *srm_fmt = PPA_SRM_COLOR_MODE_RGB888;
    } else if (type == DL_IMAGE_PIX_TYPE_RGB565) {
        *srm_fmt = PPA_SRM_COLOR_MODE_RGB565;
    } else {
        return ESP_FAIL;
    }
    return ESP_OK;
}

esp_err_t resize_ppa(const img_t &src_img,
                     img_t &dst_img,
                     ppa_client_handle_t ppa_handle,
                     uint32_t caps = 0,
                     const std::vector<int> &crop_area = {},
                     float *scale_x_ret = nullptr,
                     float *scale_y_ret = nullptr);
} // namespace image
} // namespace dl
#endif
