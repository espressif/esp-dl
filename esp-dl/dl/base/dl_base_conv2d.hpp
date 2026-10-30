#pragma once

#include "dl_define.hpp"
#include "dl_kernel.hpp"

namespace dl {
namespace base {
void conv2d(void *const args_ptr,
            quant_type_t quant,
            dl_kernel_erased_t fn0,
            dl_kernel_erased_t fn1,
            dl_kernel_erased_t fn2 = nullptr);
} // namespace base
} // namespace dl
