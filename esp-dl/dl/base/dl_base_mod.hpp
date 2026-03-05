#pragma once

#include "dl_base.hpp"
#include "dl_base_elemwise.hpp"

namespace dl {
namespace base {
/**
 * @brief int8/int16/float element-wise mod (C-style fmod), support multidirectional broadcasting
 *        from 1D to 4D. Per ONNX spec, fmod=1 is required for float types. Quantized types use
 *        fmod=1 semantics (dequantize to float, compute fmodf, requantize).
 *
 * @param args elemwiseArgsType
 */
template <typename T>
void elemwise_mod(elemwiseArgsType<T> *args);

} // namespace base
} // namespace dl
