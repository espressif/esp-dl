#pragma once

#include "dl_base.hpp"

namespace dl {
namespace base {
/**
 * @brief conv2d
 *
 * @tparam feature_t
 * @tparam bias_t
 * @tparam buffer_t
 * @tparam filter_t  filter element type, only differs from feature_t for mixed precision (W8A16)
 * @param args_ptr
 */
template <typename feature_t, typename bias_t, typename buffer_t, typename filter_t = feature_t>
void conv2d(void *const args_ptr);
#if CONFIG_IDF_TARGET_ESP32P4 && CONFIG_PIE_V2_BOOST
// Run a packed MatMul with a fixed filter tile budget.
template <typename T, typename W = T>
void packed_matmul_tiled(ArgsType<T> &args);
#endif
} // namespace base
} // namespace dl
