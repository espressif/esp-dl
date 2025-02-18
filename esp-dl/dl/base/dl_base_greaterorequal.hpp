#pragma once

#include "dl_base.hpp"
#include "dl_base_elemwise.hpp"

namespace dl {
namespace base {
/**
 * @brief int8 element-wise add, support greaterorequaltidirectional broadcasting from 1D to 4D
 *
 * @param args elemwiseArgsType
 */
void elemwise_greaterorequal(elemwiseArgsType<int8_t, bool> *args);

/**
 * @brief int16 element-wise add, support greaterorequaltidirectional broadcasting from 1D to 4D
 *
 * @param args elemwiseArgsType
 */
void elemwise_greaterorequal(elemwiseArgsType<int16_t, bool> *args);

} // namespace base
} // namespace dl
