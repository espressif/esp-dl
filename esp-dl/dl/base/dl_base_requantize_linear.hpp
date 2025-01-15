#pragma once

#include "dl_base.hpp"
#include "dl_tensor_base.hpp"

namespace dl {
namespace base {

std::vector<requantizeArgsType> get_requantize_operation_args(TensorBase *output,
                                                              TensorBase *input,
                                                              const runtime_mode_t runtime_mode = RUNTIME_MODE_AUTO);

/**
 * @brief requantize_linear
 *
 * @tparam out_feature_t
 * @tparam in_feature_t
 * @param args_ptr
 */
template <typename out_feature_t, typename in_feature_t>
void requantize_linear(void *const args_ptr);
} // namespace base
} // namespace dl
