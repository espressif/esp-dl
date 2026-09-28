#pragma once

// Helpers shared by the conv fast paths in dl_c_conv.cpp, dl_c_dwconv.cpp and
// isa/xtensa/dl_xtensa_conv.cpp. The requantization is bit-exact with buffer_bias_* /
// buffer_0000_* (per-tensor shift, round half up). Only included by those implementation files.

#include "dl_base_conv_args.hpp"
#include "esp_heap_caps.h"
#include <cstdint>

namespace dl {
namespace base {

inline void *conv_c_scratch_alloc(size_t bytes)
{
    void *p = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    return p ? p : heap_caps_malloc(bytes, MALLOC_CAP_DEFAULT);
}

inline int16_t conv_c_requant_s16(int64_t v, int64_t half, int shift, bool relu)
{
    const int32_t lo = (int32_t)v;
    if (shift >= 1 && shift <= 30 && v == lo && lo >= -(1 << 30) && lo <= (1 << 30)) {
        int32_t r = (lo + (int32_t)half) >> shift;
        if (relu && r < 0) {
            r = 0;
        }
        return (int16_t)DL_CLIP(r, INT16_MIN, INT16_MAX);
    }
    if (shift > 0) {
        v = (v + half) >> shift;
    } else {
        v <<= -shift;
    }
    if (relu && v < 0) {
        v = 0;
    }
    return (int16_t)DL_CLIP(v, INT16_MIN, INT16_MAX);
}

inline int16_t conv_c_requant_s16_i32(int32_t v, int32_t half, int shift, bool relu)
{
    v = (v + half) >> shift;
    if (relu && v < 0) {
        v = 0;
    }
    return (int16_t)DL_CLIP(v, INT16_MIN, INT16_MAX);
}

// int16 x int16: a product needs up to 31 bits, so the exact int64 sum is rebuilt from the
// wrapping 32-bit sum S and the sum of the high halves H = sum(p >> 16):
// sum = H * 2^16 + (uint32)(S - H * 2^16), exact for up to 65536 terms.
inline int64_t conv_c_sum_from_sh(int32_t s, int32_t h)
{
    return (int64_t)h * 65536 + (int64_t)((uint32_t)s - ((uint32_t)h << 16));
}

} // namespace base
} // namespace dl
