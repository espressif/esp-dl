#include "dl_image_jpeg.hpp"
#include "esp_jpeg_common.h"
#include "esp_jpeg_dec.h"
#include "esp_jpeg_enc.h"
#if CONFIG_IDF_TARGET_ESP32P4
#include "driver/jpeg_decode.h"
#include "driver/jpeg_encode.h"
#endif
#include "dl_image_color.hpp"
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
        return {};
    }
    jpeg_enc_close(jpeg_enc);
    jpeg_img.data_len = out_len;
    return jpeg_img;
}

jpeg_img_t sw_encode_jpeg(const img_t &img, uint32_t caps, uint8_t quality)
{
    img_t img2encode = img;
    bool free = false;
    if (img.pix_type == DL_IMAGE_PIX_TYPE_RGB565 ||
        (img.pix_type == DL_IMAGE_PIX_TYPE_RGB888 && (caps & DL_IMAGE_CAP_RGB_SWAP))) {
        img2encode.pix_type = DL_IMAGE_PIX_TYPE_RGB888;
        img2encode.data = heap_caps_malloc(get_img_byte_size(img2encode), MALLOC_CAP_DEFAULT);
        convert_img(img, img2encode, caps);
        free = true;
    };
    jpeg_img_t ret = sw_encode_jpeg_base(img2encode, quality);
    if (free) {
        heap_caps_free(img2encode.data);
    }
    return ret;
}

#if CONFIG_SOC_JPEG_CODEC_SUPPORTED
img_t hw_decode_jpeg(const jpeg_img_t &jpeg_img, pix_type_t pix_type, uint32_t caps)
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

    if (header_info.sample_method == JPEG_DOWN_SAMPLING_YUV422 ||
        header_info.sample_method == JPEG_DOWN_SAMPLING_YUV420) {
        img.height = DL_IMAGE_ALIGN_UP(header_info.height, 16);
        img.width = DL_IMAGE_ALIGN_UP(header_info.width, 16);
    } else {
        img.height = header_info.height;
        img.width = header_info.width;
    }

    // alloc output buffer
    size_t out_buf_len = get_img_byte_size(img);
    size_t alignment = cache_hal_get_cache_line_size(CACHE_LL_LEVEL_EXT_MEM, CACHE_TYPE_DATA);
    img.data = heap_caps_aligned_calloc(alignment, 1, out_buf_len, MALLOC_CAP_DEFAULT);
    if (!img.data) {
        ESP_LOGE(TAG, "Failed to alloc output buffer.");
        return {};
    }

    // new engine
    jpeg_decoder_handle_t jpgd_handle;
    jpeg_decode_engine_cfg_t decode_eng_cfg = {
        .intr_priority = 0,
        .timeout_ms = 40,
    };
    if (jpeg_new_decoder_engine(&decode_eng_cfg, &jpgd_handle) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create decoder engine.");
        return {};
    }

    // decode
    jpeg_decode_cfg_t decode_cfg = {
        .output_format = output_type, .rgb_order = rgb_order, .conv_std = JPEG_YUV_RGB_CONV_STD_BT601};
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
        return {};
    }

    // del engine
    ESP_ERROR_CHECK(jpeg_del_decoder_engine(jpgd_handle));
    return img;
}

jpeg_img_t hw_encode_jpeg_base(const img_t &img, uint8_t quality)
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

    // alloc output buffer
    int img_size = get_img_byte_size(img);
    // yuv420 use 1.5B to represent a pixel.
    size_t out_buf_len = (img.pix_type == DL_IMAGE_PIX_TYPE_GRAY) ? img_size : img_size / 2 + 1;
    size_t alignment = cache_hal_get_cache_line_size(CACHE_LL_LEVEL_EXT_MEM, CACHE_TYPE_DATA);
    jpeg_img.data = heap_caps_aligned_calloc(alignment, 1, out_buf_len, MALLOC_CAP_DEFAULT);
    if (!jpeg_img.data) {
        ESP_LOGE(TAG, "Failed to alloc output buffer.");
        return {};
    }

    // new engine
    jpeg_encoder_handle_t jpeg_handle;
    jpeg_encode_engine_cfg_t encode_eng_cfg = {
        .intr_priority = 0,
        .timeout_ms = 70,
    };
    if (jpeg_new_encoder_engine(&encode_eng_cfg, &jpeg_handle) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create encoder engine.");
        return {};
    }

    // encode
    jpeg_encode_cfg_t enc_config = {
        .height = (uint32_t)img.height,
        .width = (uint32_t)img.width,
        .src_type = src_type,
        .sub_sample = (img.pix_type == DL_IMAGE_PIX_TYPE_GRAY) ? JPEG_DOWN_SAMPLING_GRAY : JPEG_DOWN_SAMPLING_YUV420,
        .image_quality = quality,
    };

    uint32_t out_len = 0;
    if (jpeg_encoder_process(
            jpeg_handle, &enc_config, (uint8_t *)img.data, img_size, (uint8_t *)jpeg_img.data, out_buf_len, &out_len) !=
        ESP_OK) {
        ESP_LOGE(TAG, "Failed to run encoder process.");
        ESP_ERROR_CHECK(jpeg_del_encoder_engine(jpeg_handle));
        return {};
    }

    // del engine
    ESP_ERROR_CHECK(jpeg_del_encoder_engine(jpeg_handle));
    jpeg_img.data_len = out_len;
    return jpeg_img;
}

jpeg_img_t hw_encode_jpeg(const img_t &img, uint32_t caps, uint8_t quality)
{
    img_t img2encode = img;
    bool free = false;
    if (img.pix_type == DL_IMAGE_PIX_TYPE_RGB565 &&
        ((caps & DL_IMAGE_CAP_RGB_SWAP) || (caps & DL_IMAGE_CAP_RGB565_BIG_ENDIAN))) {
        if (caps & DL_IMAGE_CAP_RGB565_BIG_ENDIAN) {
            caps |= DL_IMAGE_CAP_RGB565_BYTE_SWAP;
        }
        img2encode.data = heap_caps_malloc(get_img_byte_size(img2encode), MALLOC_CAP_DEFAULT);
        convert_img(img, img2encode, caps);
        free = true;
    };
    if (img.pix_type == DL_IMAGE_PIX_TYPE_RGB888 && !(caps & DL_IMAGE_CAP_RGB_SWAP)) {
        caps |= DL_IMAGE_CAP_RGB_SWAP;
        img2encode.data = heap_caps_malloc(get_img_byte_size(img2encode), MALLOC_CAP_DEFAULT);
        convert_img(img, img2encode, caps);
        free = true;
    }
    jpeg_img_t ret = hw_encode_jpeg_base(img2encode, quality);
    if (free) {
        heap_caps_free(img2encode.data);
    }
    return ret;
}
#endif

esp_err_t write_jpeg(jpeg_img_t &img, const char *file_name)
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
} // namespace image
} // namespace dl
