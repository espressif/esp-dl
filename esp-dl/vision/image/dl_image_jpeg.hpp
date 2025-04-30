#pragma once
#include "dl_image_define.hpp"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "sdkconfig.h"

namespace dl {
namespace image {
/**
 * @brief Softawre jpeg decode.
 * @note Support decoding color image into RGB888/RGB565. Do not support Gray image. When pix_type is
 * DL_IMAGE_PIX_TYPE_RGB565, the decoded image is in little endian, set paramter caps to DL_IMAGE_CAP_RGB565_BIG_ENDIAN
 * to get a big endian image. When pix_type is DL_IMAGE_PIX_TYPE_RGB888, the decoded image is RGB.
 *
 * @param jpeg_img The jpeg image.
 * @param pix_type The pixel type of the decoded image.
 * @param caps Only useful if the pix_type is DL_IMAGE_PIX_TYPE_RGB565.
 * @return img_t
 */
img_t sw_decode_jpeg(const jpeg_img_t &jpeg_img, pix_type_t pix_type, uint32_t caps = 0);

/**
 * @brief Softawre jpeg encode.
 * @note Support encoding RGB565/RGB888/GRAY image.
 *
 * @param img The image to encode.
 * @param quality Compression quality.
 * @return jpeg_img_t
 */
jpeg_img_t sw_encode_jpeg(const img_t &img, uint32_t caps = 0, uint8_t quality = 80);
#if CONFIG_IDF_TARGET_ESP32P4
/**
 * @brief Hardware jpeg decode.
 * @note Support decoding into GRAY/RGB888/RGB565. Unable to decode a image if w * h is not divided by 8. The decoded
 * image may be a little bigger than the original jpeg image. The w and h will be aligned up to a multiple of 16.
 *
 * @param jpeg_img The jpeg image.
 * @param pix_type The pixel type of the decoded image.
 * @param swap_color_bytes When pix_type is DL_IMAGE_PIX_TYPE_RGB565, the decoded image is RGB in big endian if
 * swap_color_bytes set to false, otherwise it will be RGB in little endian. When pix_type is DL_IMAGE_PIX_TYPE_RGB888,
 * the decoded image is BGR, otherwise it will be RGB.
 * @return img_t
 */
img_t hw_decode_jpeg(const jpeg_img_t &jpeg_img, pix_type_t pix_type, bool swap_color_bytes = false);

/**
 * @brief Hardware jpeg encode
 * @note  Support encoding GRAY/RGB888/RGB565. If The image to encode is RGB565, it should be RGB in big endian. If it
 * is RGB888, it should be BGR.
 *
 * @param img The image to encode.
 * @param quality Compression quality.
 * @return jpeg_img_t
 */
jpeg_img_t hw_encode_jpeg(const img_t &img, uint8_t quality = 80);
#endif
esp_err_t write_jpeg(jpeg_img_t &img, const char *file_name);
} // namespace image
} // namespace dl
