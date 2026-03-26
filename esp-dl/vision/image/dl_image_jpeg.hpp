#pragma once
#include "dl_image_define.hpp"
#include "sdkconfig.h"
#if CONFIG_SOC_JPEG_CODEC_SUPPORTED
#include "driver/jpeg_decode.h"
#include "driver/jpeg_encode.h"
#endif
#include "esp_err.h"

namespace dl {
namespace image {
/**
 * @brief Softawre jpeg decode.
 * @note Support decoding color image into RGB888/RGB565LE/RGB565BE/UYVY. Do not support Gray image.
 *
 * @param jpeg_img The jpeg image.
 * @param pix_type The pixel type of the decoded image.
 * @return img_t
 */
img_t sw_decode_jpeg(const jpeg_img_t &jpeg_img, pix_type_t pix_type);

/**
 * @brief Softawre jpeg encode.
 * @note Support encoding RGB888/GRAY/YUVY/UYVY image.
 *
 * @param img The image to encode.
 * @param quality Compression quality.
 * @return jpeg_img_t
 */
jpeg_img_t sw_encode_jpeg_base(const img_t &img, uint8_t quality = 80);

/**
 * @brief Extend sw_encode_jpeg_base, add support to encode
 * RGB565LE/RGB565BE/BGR565LE/BGR565BE/BGR888/HSV_MASK/YUVY/UYVY with additional convert.
 *
 * @param img The image to encode.
 * @param quality Compression quality.
 * @return jpeg_img_t
 */
jpeg_img_t sw_encode_jpeg(const img_t &img, uint8_t quality = 80);
#if CONFIG_SOC_JPEG_CODEC_SUPPORTED
/**
 * @brief Hardware jpeg decode.
 * @note Support decoding into GRAY/RGB888/RGB565. Unable to decode a image if w * h is not divided by 8. The decoded
 * image may be a little bigger than the original jpeg image. The w and h will be aligned up to a multiple of 16.
 *
 * @param jpeg_img The jpeg image.
 * @param pix_type The pixel type of the decoded image.
 * @param timeout_ms Timeout in ms, if decode process takes more than this time, it will be considered as a failure.
 * @param yuv_rgb_conv_std JPEG decoder yuv->rgb standard
 * @return img_t
 */
img_t hw_decode_jpeg(const jpeg_img_t &jpeg_img,
                     pix_type_t pix_type,
                     int timeout_ms = 60,
                     jpeg_yuv_rgb_conv_std_t yuv_rgb_conv_std = JPEG_YUV_RGB_CONV_STD_BT601);

/**
 * @brief Hardware jpeg encode
 * @note  Support encoding GRAY/BGR888/RGB565LE.
 *
 * @param img The image to encode.
 * @param quality Compression quality.
 * @param timeout_ms Timeout in ms, if encode process takes more than this time, it will be considered as a failure.
 * @param rgb_sub_sample_method JPEG down sampling method for RGB img, can be one of JPEG_DOWN_SAMPLING_YUV444,
 * JPEG_DOWN_SAMPLING_YUV422 or JPEG_DOWN_SAMPLING_YUV420
 * @return jpeg_img_t
 */
jpeg_img_t hw_encode_jpeg_base(const img_t &img,
                               uint8_t quality = 80,
                               int timeout_ms = 70,
                               jpeg_down_sampling_type_t rgb_sub_sample_method = JPEG_DOWN_SAMPLING_YUV420);

/**
 * @brief Extend hw_encode_jpeg_base, add support to encode RGB565BE/BGR565LE/BGR565BE/RGB888/HSV_MASK with additional
 * convert.
 *
 * @param img The image to encode.
 * @param quality Compression quality.
 * @param timeout_ms Timeout in ms, if encode process takes more than this time, it will be considered as a failure.
 * @param rgb_sub_sample_method JPEG down sampling method for RGB img, can be one of JPEG_DOWN_SAMPLING_YUV444,
 * JPEG_DOWN_SAMPLING_YUV422 or JPEG_DOWN_SAMPLING_YUV420
 * @return jpeg_img_t
 */
jpeg_img_t hw_encode_jpeg(const img_t &img,
                          uint8_t quality = 80,
                          int timeout_ms = 70,
                          jpeg_down_sampling_type_t rgb_sub_sample_method = JPEG_DOWN_SAMPLING_YUV420);
#endif
esp_err_t write_jpeg(const jpeg_img_t &img, const char *file_name);
jpeg_img_t read_jpeg(const char *file_name);
} // namespace image
} // namespace dl
