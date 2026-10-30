#pragma once

#include "dl_base_conv_args.hpp"
#include "dl_kernel.hpp"

namespace dl {
namespace base {
/**
 * NOTE: support [H, W, C, 1] only by now
 * NOTE: in tensorflow, when dilation > 1 the stride must be 1. Our api has no such limitation. But we didn't test this
 * opposite situation. https://tensorflow.google.cn/api_docs/python/tf/nn/depthwise_conv2d
 */
void depthwise_conv2d(void *args_ptr,
                      quant_type_t quant,
                      dl_kernel_erased_t fn0,
                      dl_kernel_erased_t fn1,
                      dl_kernel_erased_t fn2 = nullptr);
} // namespace base
} // namespace dl
