#include "dl_image_jpeg.hpp"
#include "dl_image_process.hpp"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_jpeg_common.h"
#include "esp_jpeg_dec.h"
#include "esp_jpeg_enc.h"
#include "esp_log.h"
#include "hal/cache_hal.h"
#include "hal/cache_ll.h"

static const char *TAG = "dl_image_jpeg";
namespace dl {
namespace image {
img_t sw_decode_jpeg(const jpeg_img_t &jpeg_img, pix_type_t pix_type, uint32_t caps)
{
    assert(caps == 0 || caps == DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
    img_t img;
    img.pix_type = pix_type;
    jpeg_pixel_format_t output_type;
    switch (pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
        output_type = JPEG_PIXEL_FORMAT_RGB888;
        break;
    case DL_IMAGE_PIX_TYPE_RGB565:
        output_type =
            (caps & DL_IMAGE_CAP_RGB565_BIG_ENDIAN) ? JPEG_PIXEL_FORMAT_RGB565_BE : JPEG_PIXEL_FORMAT_RGB565_LE;
        break;
    default:
        ESP_LOGE(TAG, "Unsupported img pix format.");
        return {};
    }
    jpeg_dec_config_t cfg = {.output_type = output_type,
                             .scale = {.width = 0, .height = 0},
                             .clipper = {.width = 0, .height = 0},
                             .rotate = JPEG_ROTATE_0D,
                             .block_enable = false};

    jpeg_dec_handle_t jpeg_dec = NULL;
    if (jpeg_dec_open(&cfg, &jpeg_dec) != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to open jpeg decoder.");
        return {};
    }

    jpeg_dec_io_t jpeg_io = {};
    jpeg_io.inbuf = (uint8_t *)jpeg_img.data;
    jpeg_io.inbuf_len = jpeg_img.data_len;
    jpeg_dec_header_info_t out_info = {};
    if (jpeg_dec_parse_header(jpeg_dec, &jpeg_io, &out_info) != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to parse jpeg header.");
        jpeg_dec_close(jpeg_dec);
        return {};
    }

    img.width = out_info.width;
    img.height = out_info.height;
    size_t out_buf_len = get_img_byte_size(img);
    img.data = heap_caps_aligned_alloc(16, out_buf_len, MALLOC_CAP_DEFAULT);
    if (!img.data) {
        ESP_LOGE(TAG, "Failed to alloc output buffer.");
        jpeg_dec_close(jpeg_dec);
        return {};
    }
    jpeg_io.outbuf = (uint8_t *)img.data;

    if (jpeg_dec_process(jpeg_dec, &jpeg_io) != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to decode jpeg.");
        jpeg_dec_close(jpeg_dec);
        heap_caps_free(img.data);
        return {};
    }
    jpeg_dec_close(jpeg_dec);
    return img;
}

jpeg_img_t sw_encode_jpeg_base(const img_t &img, uint8_t quality)
{
    jpeg_img_t jpeg_img;
    jpeg_pixel_format_t src_type;
    switch (img.pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
        src_type = JPEG_PIXEL_FORMAT_RGB888;
        break;
    case DL_IMAGE_PIX_TYPE_GRAY:
        src_type = JPEG_PIXEL_FORMAT_GRAY;
        break;
    default:
        ESP_LOGE(TAG, "Unsupported img pix format.");
        return {};
    }
    jpeg_enc_config_t cfg = {.width = img.width,
                             .height = img.height,
                             .src_type = src_type,
                             .subsampling =
                                 (img.pix_type == DL_IMAGE_PIX_TYPE_GRAY) ? JPEG_SUBSAMPLE_GRAY : JPEG_SUBSAMPLE_420,
                             .quality = quality,
                             .rotate = JPEG_ROTATE_0D,
                             .task_enable = false,
                             .hfm_task_priority = 0,
                             .hfm_task_core = 0};

    jpeg_enc_handle_t jpeg_enc = NULL;
    if (jpeg_enc_open(&cfg, &jpeg_enc) != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to open jpeg encoder.");
        return {};
    }

    int img_size = get_img_byte_size(img);
    // yuv420 use 1.5B to represent a pixel.
    size_t out_buf_len = (img.pix_type == DL_IMAGE_PIX_TYPE_GRAY) ? img_size : img_size / 2 + 1;
    jpeg_img.data = heap_caps_malloc(out_buf_len, MALLOC_CAP_DEFAULT);
    if (!jpeg_img.data) {
        ESP_LOGE(TAG, "Failed to alloc output buffer.");
        jpeg_enc_close(jpeg_enc);
        return {};
    }

    int out_len = 0;
    if (jpeg_enc_process(jpeg_enc, (uint8_t *)img.data, img_size, (uint8_t *)jpeg_img.data, out_buf_len, &out_len) !=
        JPEG_ERR_OK) {
        ESP_LOGE(TAG, "Failed to encode jpeg.");
        jpeg_enc_close(jpeg_enc);
        heap_caps_free(jpeg_img.data);
        return {};
    }
    jpeg_enc_close(jpeg_enc);
    jpeg_img.data_len = out_len;
    return jpeg_img;
}

jpeg_img_t sw_encode_jpeg(const img_t &img, uint32_t caps, uint8_t quality)
{
    if (img.pix_type == DL_IMAGE_PIX_TYPE_RGB565 ||
        (img.pix_type == DL_IMAGE_PIX_TYPE_RGB888 && (caps & DL_IMAGE_CAP_RGB_SWAP))) {
        img_t img_cvt = img;
        img_cvt.pix_type = DL_IMAGE_PIX_TYPE_RGB888;
        img_cvt.data = heap_caps_malloc(get_img_byte_size(img_cvt), MALLOC_CAP_DEFAULT);
        ImageTransformer image_transformer;
        image_transformer.set_src_img(img).set_dst_img(img_cvt).set_caps(caps).transform();
        jpeg_img_t ret = sw_encode_jpeg_base(img_cvt, quality);
        heap_caps_free(img_cvt.data);
        return ret;
    } else {
        return sw_encode_jpeg_base(img, quality);
    }
}

#if CONFIG_SOC_JPEG_CODEC_SUPPORTED
img_t hw_decode_jpeg(const jpeg_img_t &jpeg_img,
                     pix_type_t pix_type,
                     uint32_t caps,
                     int timeout_ms,
                     jpeg_yuv_rgb_conv_std_t yuv_rgb_conv_std)
{
    assert(caps == 0 || caps == DL_IMAGE_CAP_RGB_SWAP || caps == DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
    img_t img;
    img.pix_type = pix_type;
    jpeg_decode_picture_info_t header_info;
    if (jpeg_decoder_get_info((const uint8_t *)jpeg_img.data, jpeg_img.data_len, &header_info) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get jpeg header info.");
        return {};
    }

    jpeg_dec_rgb_element_order_t rgb_order = JPEG_DEC_RGB_ELEMENT_ORDER_BGR;
    jpeg_dec_output_format_t output_type;
    switch (pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
        output_type = JPEG_DECODE_OUT_FORMAT_RGB888;
        rgb_order = (caps & DL_IMAGE_CAP_RGB_SWAP) ? JPEG_DEC_RGB_ELEMENT_ORDER_BGR : JPEG_DEC_RGB_ELEMENT_ORDER_RGB;
        assert(header_info.sample_method != JPEG_DOWN_SAMPLING_GRAY);
        break;
    case DL_IMAGE_PIX_TYPE_RGB565:
        output_type = JPEG_DECODE_OUT_FORMAT_RGB565;
        rgb_order =
            (caps & DL_IMAGE_CAP_RGB565_BIG_ENDIAN) ? JPEG_DEC_RGB_ELEMENT_ORDER_RGB : JPEG_DEC_RGB_ELEMENT_ORDER_BGR;
        assert(header_info.sample_method != JPEG_DOWN_SAMPLING_GRAY);
        break;
    case DL_IMAGE_PIX_TYPE_GRAY:
        output_type = JPEG_DECODE_OUT_FORMAT_GRAY;
        assert(header_info.sample_method == JPEG_DOWN_SAMPLING_GRAY);
        break;
    default:
        ESP_LOGE(TAG, "Unsupported img pix format.");
        return {};
    }

    if (header_info.sample_method == JPEG_DOWN_SAMPLING_YUV420) {
        img.height = align_up(header_info.height, 16);
        img.width = align_up(header_info.width, 16);
    } else if (header_info.sample_method == JPEG_DOWN_SAMPLING_YUV422) {
        img.height = align_up(header_info.height, 8);
        img.width = align_up(header_info.width, 16);
    } else {
        img.height = header_info.height;
        img.width = header_info.width;
    }

    // new engine
    jpeg_decoder_handle_t jpgd_handle;
    jpeg_decode_engine_cfg_t decode_eng_cfg = {
        .intr_priority = 0,
        .timeout_ms = timeout_ms,
    };
    if (jpeg_new_decoder_engine(&decode_eng_cfg, &jpgd_handle) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create decoder engine.");
        return {};
    }

    // alloc output buffer
    size_t out_buf_len = get_img_byte_size(img);
#if CONFIG_SPIRAM
    uint32_t cache_level = CACHE_LL_LEVEL_EXT_MEM;
#else
    uint32_t cache_level = CACHE_LL_LEVEL_INT_MEM;
#endif
    size_t alignment = cache_hal_get_cache_line_size(cache_level, CACHE_TYPE_DATA);
    out_buf_len = align_up(out_buf_len, alignment);
    img.data = heap_caps_aligned_calloc(alignment, 1, out_buf_len, MALLOC_CAP_DEFAULT);
    if (!img.data) {
        ESP_LOGE(TAG, "Failed to alloc output buffer.");
        ESP_ERROR_CHECK(jpeg_del_decoder_engine(jpgd_handle));
        return {};
    }

    // decode
    jpeg_decode_cfg_t decode_cfg = {.output_format = output_type, .rgb_order = rgb_order, .conv_std = yuv_rgb_conv_std};
    uint32_t out_len = 0;
    if (jpeg_decoder_process(jpgd_handle,
                             &decode_cfg,
                             (uint8_t *)jpeg_img.data,
                             jpeg_img.data_len,
                             (uint8_t *)img.data,
                             out_buf_len,
                             &out_len) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to run decoder process.");
        ESP_ERROR_CHECK(jpeg_del_decoder_engine(jpgd_handle));
        heap_caps_free(img.data);
        return {};
    }

    // del engine
    ESP_ERROR_CHECK(jpeg_del_decoder_engine(jpgd_handle));
    return img;
}

jpeg_img_t hw_encode_jpeg_base(const img_t &img,
                               uint8_t quality,
                               int timeout_ms,
                               jpeg_down_sampling_type_t rgb_sub_sample_method)
{
    jpeg_img_t jpeg_img;
    jpeg_enc_input_format_t src_type;
    switch (img.pix_type) {
    case DL_IMAGE_PIX_TYPE_RGB888:
        src_type = JPEG_ENCODE_IN_FORMAT_RGB888;
        break;
    case DL_IMAGE_PIX_TYPE_RGB565:
        src_type = JPEG_ENCODE_IN_FORMAT_RGB565;
        break;
    case DL_IMAGE_PIX_TYPE_GRAY:
        src_type = JPEG_ENCODE_IN_FORMAT_GRAY;
        break;
    default:
        ESP_LOGE(TAG, "Unsupported img pix format.");
        return {};
    }

    // new engine
    jpeg_encoder_handle_t jpeg_handle;
    jpeg_encode_engine_cfg_t encode_eng_cfg = {
        .intr_priority = 0,
        .timeout_ms = timeout_ms,
    };
    if (jpeg_new_encoder_engine(&encode_eng_cfg, &jpeg_handle) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create encoder engine.");
        return {};
    }

    // alloc output buffer
    int img_size = get_img_byte_size(img);
    // yuv420 use 1.5B to represent a pixel.
    size_t out_buf_len = (img.pix_type == DL_IMAGE_PIX_TYPE_GRAY) ? img_size : img_size / 2 + 1;
#if CONFIG_SPIRAM
    uint32_t cache_level = CACHE_LL_LEVEL_EXT_MEM;
#else
    uint32_t cache_level = CACHE_LL_LEVEL_INT_MEM;
#endif
    size_t alignment = cache_hal_get_cache_line_size(cache_level, CACHE_TYPE_DATA);
    out_buf_len = align_up(out_buf_len, alignment);
    jpeg_img.data = heap_caps_aligned_calloc(alignment, 1, out_buf_len, MALLOC_CAP_DEFAULT);
    if (!jpeg_img.data) {
        ESP_LOGE(TAG, "Failed to alloc output buffer.");
        ESP_ERROR_CHECK(jpeg_del_encoder_engine(jpeg_handle));
        return {};
    }

    // encode
    jpeg_encode_cfg_t enc_config = {
        .height = (uint32_t)img.height,
        .width = (uint32_t)img.width,
        .src_type = src_type,
        .sub_sample = (img.pix_type == DL_IMAGE_PIX_TYPE_GRAY) ? JPEG_DOWN_SAMPLING_GRAY : rgb_sub_sample_method,
        .image_quality = quality,
    };

    uint32_t out_len = 0;
    if (jpeg_encoder_process(
            jpeg_handle, &enc_config, (uint8_t *)img.data, img_size, (uint8_t *)jpeg_img.data, out_buf_len, &out_len) !=
        ESP_OK) {
        ESP_LOGE(TAG, "Failed to run encoder process.");
        ESP_ERROR_CHECK(jpeg_del_encoder_engine(jpeg_handle));
        heap_caps_free(jpeg_img.data);
        return {};
    }

    // del engine
    ESP_ERROR_CHECK(jpeg_del_encoder_engine(jpeg_handle));
    jpeg_img.data_len = out_len;
    return jpeg_img;
}

jpeg_img_t hw_encode_jpeg(
    const img_t &img, uint32_t caps, uint8_t quality, int timeout_ms, jpeg_down_sampling_type_t rgb_sub_sample_method)
{
    bool need_cvt = false;
    if (img.pix_type == DL_IMAGE_PIX_TYPE_RGB565 &&
        ((caps & DL_IMAGE_CAP_RGB_SWAP) || (caps & DL_IMAGE_CAP_RGB565_BIG_ENDIAN))) {
        if (caps & DL_IMAGE_CAP_RGB565_BIG_ENDIAN) {
            caps |= DL_IMAGE_CAP_RGB565_BYTE_SWAP;
        }
        need_cvt = true;
    } else if (img.pix_type == DL_IMAGE_PIX_TYPE_RGB888 && !(caps & DL_IMAGE_CAP_RGB_SWAP)) {
        caps |= DL_IMAGE_CAP_RGB_SWAP;
        need_cvt = true;
    }
    if (need_cvt) {
        img_t img_cvt = img;
        img_cvt.data = heap_caps_malloc(get_img_byte_size(img_cvt), MALLOC_CAP_DEFAULT);
        ImageTransformer image_transformer;
        image_transformer.set_src_img(img).set_dst_img(img_cvt).set_caps(caps).transform();
        jpeg_img_t ret = hw_encode_jpeg_base(img_cvt, quality, timeout_ms, rgb_sub_sample_method);
        heap_caps_free(img_cvt.data);
        return ret;
    } else {
        return hw_encode_jpeg_base(img, quality, timeout_ms, rgb_sub_sample_method);
    }
}
#endif

esp_err_t write_jpeg(const jpeg_img_t &img, const char *file_name)
{
    FILE *f = fopen(file_name, "wb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open %s.", file_name);
        return ESP_FAIL;
    }
    size_t size = fwrite(img.data, img.data_len, 1, f);
    if (size != 1) {
        ESP_LOGE(TAG, "Failed to write img data.");
        fclose(f);
        return ESP_FAIL;
    }
    fclose(f);
    return ESP_OK;
}

jpeg_img_t read_jpeg(const char *file_name)
{
    jpeg_img_t img = {};
    FILE *f = fopen(file_name, "rb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open %s.", file_name);
        return img;
    }

    fseek(f, 0, SEEK_END);
    img.data_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    img.data = heap_caps_malloc(img.data_len, MALLOC_CAP_DEFAULT);
    assert(img.data);
    fread(img.data, img.data_len, 1, f);
    fclose(f);
    return img;
}
} // namespace image
} // namespace dl
