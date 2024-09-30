#pragma once

#include "dl_base.hpp"

namespace dl {
namespace base {
/**
 * @brief multidirectional broadcasting
 *        refer to https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
 *
 * @param shape1 Shape of input1
 * @param shape2 Shape of input2
 *
 * @return Shape of output
 */
std::vector<int> get_multidirectional_broadcasting_shape(const std::vector<int> &shape1,
                                                         const std::vector<int> &shape2);

/**
 * @brief unidirectional broadcasting
 *        refer to https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
 *
 * @param shape1 Shape of input1
 * @param shape2 Shape of input2
 *
 * @return Shape of output
 */
std::vector<int> get_unidirectional_broadcasting_shape(const std::vector<int> &shape1, const std::vector<int> &shape2);

} // namespace base
} // namespace dl
