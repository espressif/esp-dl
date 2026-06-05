#pragma once

#include "dl_base.hpp"

namespace dl {
namespace base {

int32_t reduce_l2(int8_t *input, int32_t size, int32_t stride);

int64_t reduce_l2(int16_t *input, int32_t size, int32_t stride);

float reduce_l2(float *input, int32_t size, int32_t stride);

} // namespace base
} // namespace dl
