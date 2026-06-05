#include "dl_base_reduce.hpp"
#include "dl_base_isa.hpp"
#include "dl_math.hpp"
#include "dl_tool.hpp"
#include <cmath>
#include <cstdint>

namespace dl {
namespace base {

int32_t reduce_l2(int8_t *input, int32_t size, int32_t stride)
{
    int32_t sum = 0;

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            int32_t v = input[i];
            sum += v * v;
            i++;
        }
        int32_t aligned_size = (size - i) & ~0xF; // largest multiple of 16 elements
        if (aligned_size > 0) {
            sum += dl_tie728_reduce_l2_s8_aligned(input + i, aligned_size);
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            int32_t v0 = input[i];
            int32_t v1 = input[i + 1];
            int32_t v2 = input[i + 2];
            int32_t v3 = input[i + 3];
            sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        }
#endif
        for (; i < size; i++) {
            int32_t v = input[i];
            sum += v * v;
        }
    } else {
        int8_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            int32_t v = *ptr;
            sum += v * v;
            ptr += stride;
        }
    }

    return sum;
}

int64_t reduce_l2(int16_t *input, int32_t size, int32_t stride)
{
    int64_t sum = 0;

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            int64_t v = input[i];
            sum += v * v;
            i++;
        }
        int32_t aligned_size = (size - i) & ~0x7; // largest multiple of 8 elements (16 bytes)
        if (aligned_size > 0) {
            sum += dl_tie728_reduce_l2_s16_aligned(input + i, aligned_size);
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            int64_t v0 = input[i];
            int64_t v1 = input[i + 1];
            int64_t v2 = input[i + 2];
            int64_t v3 = input[i + 3];
            sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        }
#endif
        for (; i < size; i++) {
            int64_t v = input[i];
            sum += v * v;
        }
    } else {
        int16_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            int64_t v = *ptr;
            sum += v * v;
            ptr += stride;
        }
    }

    return sum;
}

float reduce_l2(float *input, int32_t size, int32_t stride)
{
    float sum = 0.0f;

    if (stride == 1) {
        int32_t i = 0;
        for (; i + 3 < size; i += 4) {
            float v0 = input[i];
            float v1 = input[i + 1];
            float v2 = input[i + 2];
            float v3 = input[i + 3];
            sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        }
        for (; i < size; i++) {
            float v = input[i];
            sum += v * v;
        }
    } else {
        float *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            float v = *ptr;
            sum += v * v;
            ptr += stride;
        }
    }

    return sum;
}

} // namespace base
} // namespace dl
