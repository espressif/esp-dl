/*
 * SPDX-FileCopyrightText: 2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale of output image
 *
 */
typedef enum {
    JPEG_IMAGE_SCALE_0 = 0, /*!< No scale */
    JPEG_IMAGE_SCALE_1_2,   /*!< Scale 1:2 */
    JPEG_IMAGE_SCALE_1_4,   /*!< Scale 1:4 */
    JPEG_IMAGE_SCALE_1_8,   /*!< Scale 1:8 */
} esp_jpeg_image_scale_t;

/**
 * @brief Format of output image
 *
 */
typedef enum {
    JPEG_IMAGE_FORMAT_RGB888 = 0,   /*!< Format RGB888 */
    JPEG_IMAGE_FORMAT_RGB565,       /*!< Format RGB565 */
} esp_jpeg_image_format_t;

/**
 * @brief JPEG Configuration Type
 *
 */
typedef struct esp_jpeg_image_cfg_s {
    uint8_t *indata;        /*!< Input JPEG image */
    uint32_t indata_size;   /*!< Size of input image  */
    uint8_t *outbuf;        /*!< Output buffer */
    uint32_t outbuf_size;   /*!< Output buffer size */
    esp_jpeg_image_format_t out_format; /*!< Output image format */
    esp_jpeg_image_scale_t  out_scale; /*!< Output scale */

    struct {
        uint8_t swap_color_bytes: 1; /*!< Swap first and last color bytes */
    } flags;

    struct {
        uint32_t read;  /*!< Internal count of read bytes */
    } priv;
} esp_jpeg_image_cfg_t;

/**
 * @brief JPEG output info
 *
 */
typedef struct esp_jpeg_image_output_s {
    uint16_t width;    /*!< Width of the output image */
    uint16_t height;   /*!< Height of the output image */
} esp_jpeg_image_output_t;

/**
 * @brief Decode JPEG image
 *
 * @note This function is blocking.
 *
 * @param cfg: Configuration structure
 * @param img: Output image info
 *
 * @return
 *      - ESP_OK            on success
 *      - ESP_ERR_NO_MEM    if there is no memory for allocating main structure
 *      - ESP_FAIL          if there is an error in decoding JPEG
 */
esp_err_t esp_jpeg_decode(esp_jpeg_image_cfg_t *cfg, esp_jpeg_image_output_t *img);

#ifdef __cplusplus
}
#endif
