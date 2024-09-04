/*
 * SPDX-FileCopyrightText: 2015-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "esp_system.h"
#include "esp_rom_caps.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_check.h"
#include "jpeg_decoder.h"

#if CONFIG_JD_USE_ROM
/* When supported in ROM, use ROM functions */
#if defined(ESP_ROM_HAS_JPEG_DECODE)
#include "rom/tjpgd.h"
#else
#error Using JPEG decoder from ROM is not supported for selected target. Please select external code in menuconfig.
#endif

/* The ROM code of TJPGD is older and has different return type in decode callback */
typedef unsigned int jpeg_decode_out_t;
#else
/* When Tiny JPG Decoder is not in ROM or selected external code */
#include "tjpgd.h"

/* The TJPGD outside the ROM code is newer and has different return type in decode callback */
typedef int jpeg_decode_out_t;
#endif

static const char *TAG = "JPEG";

#define LOBYTE(u16)     ((uint8_t)(((uint16_t)(u16)) & 0xff))
#define HIBYTE(u16)     ((uint8_t)((((uint16_t)(u16))>>8) & 0xff))

#if defined(JD_FASTDECODE) && (JD_FASTDECODE == 2)
#define JPEG_WORK_BUF_SIZE  65472
#else
#define JPEG_WORK_BUF_SIZE  3100    /* Recommended buffer size; Independent on the size of the image */
#endif

/* If not set JD_FORMAT, it is set in ROM to RGB888, otherwise, it can be set in config */
#ifndef JD_FORMAT
#define JD_FORMAT 0
#endif

/* Output color bytes from tjpgd (depends on JD_FORMAT) */
#if (JD_FORMAT==0)
#define ESP_JPEG_COLOR_BYTES    3
#elif  (JD_FORMAT==1)
#define ESP_JPEG_COLOR_BYTES    2
#elif  (JD_FORMAT==2)
#error Grayscale image output format is not supported
#define ESP_JPEG_COLOR_BYTES    1
#endif

/*******************************************************************************
* Function definitions
*******************************************************************************/
static uint8_t jpeg_get_div_by_scale(esp_jpeg_image_scale_t scale);
static uint8_t jpeg_get_color_bytes(esp_jpeg_image_format_t format);

static unsigned int jpeg_decode_in_cb(JDEC *jd, uint8_t *buff, unsigned int nbyte);
static jpeg_decode_out_t jpeg_decode_out_cb(JDEC *jd, void *bitmap, JRECT *rect);
/*******************************************************************************
* Public API functions
*******************************************************************************/

esp_err_t esp_jpeg_decode(esp_jpeg_image_cfg_t *cfg, esp_jpeg_image_output_t *img)
{
    esp_err_t ret = ESP_OK;
    uint8_t *workbuf = NULL;
    JRESULT res;
    JDEC JDEC;

    assert(cfg != NULL);
    assert(img != NULL);

    workbuf = heap_caps_malloc(JPEG_WORK_BUF_SIZE, MALLOC_CAP_DEFAULT);
    ESP_GOTO_ON_FALSE(workbuf, ESP_ERR_NO_MEM, err, TAG, "no mem for JPEG work buffer");

    cfg->priv.read = 0;

    /* Prepare image */
    res = jd_prepare(&JDEC, jpeg_decode_in_cb, workbuf, JPEG_WORK_BUF_SIZE, cfg);
    ESP_GOTO_ON_FALSE((res == JDR_OK), ESP_FAIL, err, TAG, "Error in preparing JPEG image!");

    uint8_t scale_div = jpeg_get_div_by_scale(cfg->out_scale);
    uint8_t out_color_bytes = jpeg_get_color_bytes(cfg->out_format);

    /* Size of output image */
    uint32_t outsize = (JDEC.height / scale_div) * (JDEC.width / scale_div) * out_color_bytes;
    ESP_GOTO_ON_FALSE((outsize <= cfg->outbuf_size), ESP_ERR_NO_MEM, err, TAG, "Not enough size in output buffer!");

    /* Size of output image */
    img->height = JDEC.height / scale_div;
    img->width = JDEC.width / scale_div;

    /* Decode JPEG */
    res = jd_decomp(&JDEC, jpeg_decode_out_cb, cfg->out_scale);
    ESP_GOTO_ON_FALSE((res == JDR_OK), ESP_FAIL, err, TAG, "Error in decoding JPEG image!");

err:
    if (workbuf) {
        free(workbuf);
    }

    return ret;
}

/*******************************************************************************
* Private API functions
*******************************************************************************/

static unsigned int jpeg_decode_in_cb(JDEC *dec, uint8_t *buff, unsigned int nbyte)
{
    assert(dec != NULL);

    uint32_t to_read = nbyte;
    esp_jpeg_image_cfg_t *cfg = (esp_jpeg_image_cfg_t *)dec->device;
    assert(cfg != NULL);

    if (buff) {
        if (cfg->priv.read + to_read > cfg->indata_size) {
            to_read = cfg->indata_size - cfg->priv.read;
        }

        /* Copy data from JPEG image */
        memcpy(buff, &cfg->indata[cfg->priv.read], to_read);
        cfg->priv.read += to_read;
    } else if (buff == NULL) {
        /* Skip data */
        cfg->priv.read += to_read;
    }

    return to_read;
}

static jpeg_decode_out_t jpeg_decode_out_cb(JDEC *dec, void *bitmap, JRECT *rect)
{
    uint16_t color = 0;
    assert(dec != NULL);

    esp_jpeg_image_cfg_t *cfg = (esp_jpeg_image_cfg_t *)dec->device;
    assert(cfg != NULL);
    assert(bitmap != NULL);
    assert(rect != NULL);

    uint8_t scale_div = jpeg_get_div_by_scale(cfg->out_scale);
    uint8_t out_color_bytes = jpeg_get_color_bytes(cfg->out_format);

    /* Copy decoded image data to output buffer */
    uint8_t *in = (uint8_t *)bitmap;
    uint32_t line = dec->width / scale_div;
    uint8_t *dst = (uint8_t *)cfg->outbuf;
    for (int y = rect->top; y <= rect->bottom; y++) {
        for (int x = rect->left; x <= rect->right; x++) {
            if ( (JD_FORMAT == 0 && cfg->out_format == JPEG_IMAGE_FORMAT_RGB888) ||
                    (JD_FORMAT == 1 && cfg->out_format == JPEG_IMAGE_FORMAT_RGB565) ) {
                /* Output image format is same as set in TJPGD */
                for (int b = 0; b < ESP_JPEG_COLOR_BYTES; b++) {
                    if (cfg->flags.swap_color_bytes) {
                        dst[(y * line * out_color_bytes) + x * out_color_bytes + b] = in[out_color_bytes - b - 1];
                    } else {
                        dst[(y * line * out_color_bytes) + x * out_color_bytes + b] = in[b];
                    }
                }
            } else if (JD_FORMAT == 0 && cfg->out_format == JPEG_IMAGE_FORMAT_RGB565) {
                /* Output image format is not same as set in TJPGD */
                /* We need to convert the 3 bytes in `in` to a rgb565 value */
                color = ((in[0] & 0xF8) << 8);
                color |= ((in[1] & 0xFC) << 3);
                color |= (in[2] >> 3);

                if (cfg->flags.swap_color_bytes) {
                    dst[(y * line * out_color_bytes) + (x * out_color_bytes)] = HIBYTE(color);
                    dst[(y * line * out_color_bytes) + (x * out_color_bytes) + 1] = LOBYTE(color);
                } else {
                    dst[(y * line * out_color_bytes) + (x * out_color_bytes) + 1] = HIBYTE(color);
                    dst[(y * line * out_color_bytes) + (x * out_color_bytes)] = LOBYTE(color);
                }
            } else {
                ESP_LOGE(TAG, "Selected output format is not supported!");
                assert(0);
            }
            in += ESP_JPEG_COLOR_BYTES;
        }
    }

    return 1;
}

static uint8_t jpeg_get_div_by_scale(esp_jpeg_image_scale_t scale)
{
    switch (scale) {
    /* Not scaled */
    case JPEG_IMAGE_SCALE_0:
        return 1;
    /* Scaled 1:2 */
    case JPEG_IMAGE_SCALE_1_2:
        return 2;
    /* Scaled 1:4 */
    case JPEG_IMAGE_SCALE_1_4:
        return 4;
    /* Scaled 1:8 */
    case JPEG_IMAGE_SCALE_1_8:
        return 8;
    }

    return 1;
}

static uint8_t jpeg_get_color_bytes(esp_jpeg_image_format_t format)
{
    switch (format) {
    /* RGB888 (24-bit/pix) */
    case JPEG_IMAGE_FORMAT_RGB888:
        return 3;
    /* RGB565 (16-bit/pix) */
    case JPEG_IMAGE_FORMAT_RGB565:
        return 2;
    }

    return 1;
}
