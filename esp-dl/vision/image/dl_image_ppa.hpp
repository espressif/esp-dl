#pragma once
#include "sdkconfig.h"

#if CONFIG_SOC_PPA_SUPPORTED
#include "dl_image_define.hpp"
#include "esp_cache.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <cmath>
#include <vector>
#include "driver/ppa.h"
#include "hal/cache_hal.h"
#include "hal/cache_ll.h"

namespace dl {
namespace image {
inline float get_ppa_scale(uint16_t src, uint16_t dst)
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

inline void *alloc_ppa_outbuf(size_t size)
{
    size_t align = cache_hal_get_cache_line_size(CACHE_LL_LEVEL_EXT_MEM, CACHE_TYPE_DATA);
    size_t outbuf_size = dl::image::align_up(size, align);
    void *outbuf = heap_caps_aligned_calloc(align, 1, outbuf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA);
    if (!outbuf) {
        ESP_LOGE("PPA", "Failed to alloc outbuf memory.");
    }
    esp_cache_msync(outbuf, outbuf_size, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
    return outbuf;
}

inline ppa_client_handle_t register_ppa_srm_client(const ppa_client_config_t &cfg = {
                                                       PPA_OPERATION_SRM, 0, PPA_DATA_BURST_LENGTH_128})
{
    ppa_client_handle_t handle;
    ESP_ERROR_CHECK(ppa_register_client(&cfg, &handle));
    return handle;
}

/**
 * @brief
 *
 * @param src_img
 * @param dst_img The data of the dst_img must be allocated by alloc_ppa_outbuf api for alignment and cache sync.
 * @param ppa_handle
 * @param crop_area
 * @param scale_x_ret The actual resize scale_x. PPA only support specific scales. More details see get_ppa_scale api.
 * The output image may have black borders on the right and bottom.
 * @param scale_y_ret The actual resize scale_y. PPA only support specific scales. More details see get_ppa_scale api.
 * The output image may have black borders on the right and bottom.
 * @return esp_err_t
 */
esp_err_t resize_ppa(const img_t &src_img,
                     img_t &dst_img,
                     ppa_client_handle_t ppa_handle,
                     const std::vector<int> &crop_area = {},
                     float *scale_x_ret = nullptr,
                     float *scale_y_ret = nullptr);
} // namespace image
} // namespace dl
#endif
