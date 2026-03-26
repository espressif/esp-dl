#include "dl_image_ppa.hpp"

#if CONFIG_SOC_PPA_SUPPORTED
static const char *TAG = "PPA";
namespace dl {
namespace image {
esp_err_t resize_ppa(const img_t &src_img,
                     img_t &dst_img,
                     ppa_client_handle_t ppa_handle,
                     const std::vector<int> &crop_area,
                     float *scale_x_ret,
                     float *scale_y_ret)
{
    assert(ppa_handle);
    assert(src_img.data);
    assert(dst_img.data);
    assert(src_img.height > 0 && src_img.width > 0);
    assert(dst_img.height > 0 && dst_img.width > 0);
    ppa_srm_color_mode_t input_srm_color_mode;
    ppa_srm_color_mode_t output_srm_color_mode;
    bool byte_swap = false;
    bool rgb_swap = false;
    bool invalid_pix_cvt = false;
    switch (src_img.pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
    case DL_IMAGE_PIX_TYPE_BGR888:
        input_srm_color_mode = PPA_SRM_COLOR_MODE_RGB888;
        switch (dst_img.pix_type) {
        case DL_IMAGE_PIX_TYPE_RGB888:
            output_srm_color_mode = PPA_SRM_COLOR_MODE_RGB888;
            rgb_swap = (src_img.pix_type == DL_IMAGE_PIX_TYPE_BGR888);
            break;
        case DL_IMAGE_PIX_TYPE_BGR888:
            output_srm_color_mode = PPA_SRM_COLOR_MODE_RGB888;
            rgb_swap = (src_img.pix_type == DL_IMAGE_PIX_TYPE_RGB888);
            break;
        case DL_IMAGE_PIX_TYPE_RGB565LE:
            output_srm_color_mode = PPA_SRM_COLOR_MODE_RGB565;
            rgb_swap = (src_img.pix_type == DL_IMAGE_PIX_TYPE_RGB888);
            break;
        case DL_IMAGE_PIX_TYPE_BGR565LE:
            output_srm_color_mode = PPA_SRM_COLOR_MODE_RGB565;
            rgb_swap = (src_img.pix_type == DL_IMAGE_PIX_TYPE_BGR888);
            break;
        default:
            invalid_pix_cvt = true;
            break;
        }
        break;
    case DL_IMAGE_PIX_TYPE_RGB565LE:
    case DL_IMAGE_PIX_TYPE_RGB565BE:
        input_srm_color_mode = PPA_SRM_COLOR_MODE_RGB565;
        byte_swap = (src_img.pix_type == DL_IMAGE_PIX_TYPE_RGB565BE);
        switch (dst_img.pix_type) {
        case DL_IMAGE_PIX_TYPE_BGR888:
            output_srm_color_mode = PPA_SRM_COLOR_MODE_RGB888;
            break;
        case DL_IMAGE_PIX_TYPE_RGB565LE:
            output_srm_color_mode = PPA_SRM_COLOR_MODE_RGB565;
            break;
        default:
            invalid_pix_cvt = true;
            break;
        }
        break;
    case DL_IMAGE_PIX_TYPE_BGR565LE:
    case DL_IMAGE_PIX_TYPE_BGR565BE:
        input_srm_color_mode = PPA_SRM_COLOR_MODE_RGB565;
        byte_swap = (src_img.pix_type == DL_IMAGE_PIX_TYPE_BGR565BE);
        switch (dst_img.pix_type) {
        case DL_IMAGE_PIX_TYPE_RGB888:
            output_srm_color_mode = PPA_SRM_COLOR_MODE_RGB888;
            break;
        case DL_IMAGE_PIX_TYPE_BGR565LE:
            output_srm_color_mode = PPA_SRM_COLOR_MODE_RGB565;
            break;
        default:
            invalid_pix_cvt = true;
            break;
        }
        break;
    default:
        invalid_pix_cvt = true;
        break;
    }
    if (invalid_pix_cvt) {
        ESP_LOGE(TAG,
                 "Can not do PPA resize from %s to %s.",
                 pix_type2str(src_img.pix_type).c_str(),
                 pix_type2str(dst_img.pix_type).c_str());
        return ESP_FAIL;
    }

    ppa_srm_oper_config_t srm_oper_config = {};
    float ppa_scale_x, ppa_scale_y;
    if (crop_area.empty()) {
        ppa_scale_x = get_ppa_scale(src_img.width, dst_img.width);
        ppa_scale_y = get_ppa_scale(src_img.height, dst_img.height);
        if (scale_x_ret) {
            *scale_x_ret = ppa_scale_x;
        }
        if (scale_y_ret) {
            *scale_y_ret = ppa_scale_y;
        }
        srm_oper_config.in.block_offset_y = 0;
        srm_oper_config.in.block_offset_x = 0;
        srm_oper_config.in.block_h = src_img.height;
        srm_oper_config.in.block_w = src_img.width;
    } else {
        assert(crop_area.size() == 4);
        assert(crop_area[2] > crop_area[0] && crop_area[3] > crop_area[1] && crop_area[0] >= 0 && crop_area[1] >= 0 &&
               crop_area[2] <= src_img.width && crop_area[3] <= src_img.height);
        uint16_t src_img_width = crop_area[2] - crop_area[0];
        uint16_t src_img_height = crop_area[3] - crop_area[1];
        ppa_scale_x = get_ppa_scale(src_img_width, dst_img.width);
        ppa_scale_y = get_ppa_scale(src_img_height, dst_img.height);
        if (scale_x_ret) {
            *scale_x_ret = ppa_scale_x;
        }
        if (scale_y_ret) {
            *scale_y_ret = ppa_scale_y;
        }
        srm_oper_config.in.block_offset_y = crop_area[1];
        srm_oper_config.in.block_offset_x = crop_area[0];
        srm_oper_config.in.block_h = src_img_height;
        srm_oper_config.in.block_w = src_img_width;
    }
    srm_oper_config.in.buffer = (const void *)src_img.data;
    srm_oper_config.in.pic_h = src_img.height;
    srm_oper_config.in.pic_w = src_img.width;
    srm_oper_config.in.srm_cm = input_srm_color_mode;
    srm_oper_config.rgb_swap = rgb_swap;
    srm_oper_config.byte_swap = byte_swap;

    srm_oper_config.out.buffer = dst_img.data;
    size_t align = cache_hal_get_cache_line_size(CACHE_LL_LEVEL_EXT_MEM, CACHE_TYPE_DATA);
    size_t out_buf_size = align_up(dst_img.bytes(), align);
    srm_oper_config.out.buffer_size = out_buf_size;
    srm_oper_config.out.pic_h = dst_img.height;
    srm_oper_config.out.pic_w = dst_img.width;
    srm_oper_config.out.block_offset_x = 0;
    srm_oper_config.out.block_offset_y = 0;

    srm_oper_config.out.srm_cm = output_srm_color_mode;
    srm_oper_config.rotation_angle = PPA_SRM_ROTATION_ANGLE_0;

    srm_oper_config.scale_x = ppa_scale_x;
    srm_oper_config.scale_y = ppa_scale_y;
    srm_oper_config.mirror_x = false;
    srm_oper_config.mirror_y = false;

    return ppa_do_scale_rotate_mirror(ppa_handle, &srm_oper_config);
}
} // namespace image
} // namespace dl
#endif
