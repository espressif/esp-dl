#pragma once
#include "dl_image_color.hpp"
#include "dl_image_define.hpp"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

namespace dl {
namespace image {
/**
 * @brief Save a BGR or GRAY image to BMP.
 *
 * @param img Image to save.
 * @param file_name Save path of the BMP image.
 * @return esp_err_t
 */
esp_err_t write_bmp_base(const img_t &img, const char *file_name);
/**
 * @brief Extend write_bmp_base, add support to save RGB565/RGB888 in RGB to BMP with additional convert.
 *
 * @param img Image to save.
 * @param file_name Save path of the BMP image.
 * @param caps Default to 0, save RGB565 in little endian RGB/ RGB888 in BGR.
 * @return esp_err_t
 */
esp_err_t write_bmp(const img_t &img, const char *file_name, uint32_t caps = 0);
/**
 * @brief Read a BMP.
 *
 * @param file_name Read path of the BMP image.
 * @return esp_err_t
 */
img_t read_bmp(const char *file_name);
} // namespace image
} // namespace dl
