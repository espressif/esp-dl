#include "dl_image_define.hpp"

namespace dl {
namespace image {

/**
 * @brief Detect target moving by activated detection point number. Each cross in the figure below is a detection point.
 * Once abs(frame_1_detection_point[i] - frame_2_detection_point[i]) > threshold, this detection point is activated.
 * This function will return the number of activated detection point.
 *
 *         __stride__________________________
 *         |        |        |        |   |
 *  stride |        |        |        |   |
 *         |        |        |        |   |
 *         |________|________|________|   |
 *         |        |        |        |   |
 *         |        |        |        |   |
 *         |        |        |        |   |
 *         |________|________|________| height
 *         |        |        |        |   |
 *         |        |        |        |   |
 *         |        |        |        |   |
 *         |________|________|________|   |
 *         |        |        |        |   |
 *         |        |        |        |   |
 *         |        |        |        |   |
 *         |________|________|________|___|___
 *         |                          |
 *         |__________width___________|
 *         |                          |
 *
 *
 * In a application, outside this function, threshold of activated detection point number is needed.
 * Once activated detection point number > number_threshold, this two frame are judged target moved.
 * How to determine the number_threshold?
 * Let's assume that the minimize shape of target is (target_min_height, target_max_width).
 * Then, the number_threshold = [target_min_height / stride] * [target_max_width / stride] * ratio,
 * where ratio is in (0, 1), the smaller the ratio is, the more sensitive the detector is, the more false detected.
 *
 *
 * @param img1      one img
 * @param img2      another img
 * @param stride    stride of detection point, the smaller the stride is, the more reliable the detector is.
 * @param threshold activation threshold of each detection point
 * @return activated detection point number
 */
uint32_t get_moving_point_number(const img_t &img1, const img_t &img2, const int stride, const uint8_t threshold = 5);
} // namespace image
} // namespace dl
