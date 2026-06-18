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
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            int32_t v = input[i];
            sum += v * v;
            i++;
        }
        int32_t aligned_size = (size - i) & ~0xF; // largest multiple of 16 elements
        // The 40-bit ACCX/XACC cannot saturate before the int32 result itself overflows
        // (each int8 square <= 2^14, so ~2^25 elements would be needed), so the whole
        // aligned region is reduced in a single call.
        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            sum += dl_esp32p4_reduce_l2_s8_aligned(input + i, aligned_size);
#else
            sum += dl_tie728_reduce_l2_s8_aligned(input + i, aligned_size);
#endif
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
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            int64_t v = input[i];
            sum += v * v;
            i++;
        }
        int32_t aligned_size = (size - i) & ~0x7; // largest multiple of 8 elements (16 bytes)
        // Both PIE V1 (tie728) and PIE V2 (esp32p4) accumulate the int16 sum of squares
        // into a 40-bit saturating ACCX/XACC. Each int16 square can reach ~2^30, so more
        // than ~512 elements may overflow it. Process the aligned region in chunks of 256
        // elements (a multiple of 8) and fold each partial result into the 64-bit sum.
        constexpr int32_t kChunkElems = 256;
        for (int32_t processed = 0; processed < aligned_size; processed += kChunkElems) {
            int32_t chunk = aligned_size - processed;
            if (chunk > kChunkElems) {
                chunk = kChunkElems;
            }
#if CONFIG_PIE_V2_BOOST
            sum += dl_esp32p4_reduce_l2_s16_aligned(input + i, chunk);
#else
            sum += dl_tie728_reduce_l2_s16_aligned(input + i, chunk);
#endif
            i += chunk;
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

int8_t reduce_max(int8_t *input, int32_t size, int32_t stride)
{
    int8_t result = INT8_MIN;

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            if (input[i] > result)
                result = input[i];
            i++;
        }
        int32_t aligned_size = (size - i) & ~0xF; // largest multiple of 16 elements
        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            int8_t simd_result = dl_esp32p4_reduce_max_s8_aligned(input + i, aligned_size);
#else
            int8_t simd_result = dl_tie728_reduce_max_s8_aligned(input + i, aligned_size);
#endif
            if (simd_result > result)
                result = simd_result;
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            if (input[i] > result)
                result = input[i];
            if (input[i + 1] > result)
                result = input[i + 1];
            if (input[i + 2] > result)
                result = input[i + 2];
            if (input[i + 3] > result)
                result = input[i + 3];
        }
#endif
        for (; i < size; i++) {
            if (input[i] > result)
                result = input[i];
        }
    } else {
        int8_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            if (*ptr > result)
                result = *ptr;
            ptr += stride;
        }
    }

    return result;
}

int16_t reduce_max(int16_t *input, int32_t size, int32_t stride)
{
    int16_t result = INT16_MIN;

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            if (input[i] > result)
                result = input[i];
            i++;
        }
        int32_t aligned_size = (size - i) & ~0x7; // largest multiple of 8 elements
        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            int16_t simd_result = dl_esp32p4_reduce_max_s16_aligned(input + i, aligned_size);
#else
            int16_t simd_result = dl_tie728_reduce_max_s16_aligned(input + i, aligned_size);
#endif
            if (simd_result > result)
                result = simd_result;
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            if (input[i] > result)
                result = input[i];
            if (input[i + 1] > result)
                result = input[i + 1];
            if (input[i + 2] > result)
                result = input[i + 2];
            if (input[i + 3] > result)
                result = input[i + 3];
        }
#endif
        for (; i < size; i++) {
            if (input[i] > result)
                result = input[i];
        }
    } else {
        int16_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            if (*ptr > result)
                result = *ptr;
            ptr += stride;
        }
    }

    return result;
}

int8_t reduce_min(int8_t *input, int32_t size, int32_t stride)
{
    int8_t result = INT8_MAX;

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            if (input[i] < result)
                result = input[i];
            i++;
        }
        int32_t aligned_size = (size - i) & ~0xF; // largest multiple of 16 elements
        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            int8_t simd_result = dl_esp32p4_reduce_min_s8_aligned(input + i, aligned_size);
#else
            int8_t simd_result = dl_tie728_reduce_min_s8_aligned(input + i, aligned_size);
#endif
            if (simd_result < result)
                result = simd_result;
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            if (input[i] < result)
                result = input[i];
            if (input[i + 1] < result)
                result = input[i + 1];
            if (input[i + 2] < result)
                result = input[i + 2];
            if (input[i + 3] < result)
                result = input[i + 3];
        }
#endif
        for (; i < size; i++) {
            if (input[i] < result)
                result = input[i];
        }
    } else {
        int8_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            if (*ptr < result)
                result = *ptr;
            ptr += stride;
        }
    }

    return result;
}

int16_t reduce_min(int16_t *input, int32_t size, int32_t stride)
{
    int16_t result = INT16_MAX;

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            if (input[i] < result)
                result = input[i];
            i++;
        }
        int32_t aligned_size = (size - i) & ~0x7; // largest multiple of 8 elements
        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            int16_t simd_result = dl_esp32p4_reduce_min_s16_aligned(input + i, aligned_size);
#else
            int16_t simd_result = dl_tie728_reduce_min_s16_aligned(input + i, aligned_size);
#endif
            if (simd_result < result)
                result = simd_result;
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            if (input[i] < result)
                result = input[i];
            if (input[i + 1] < result)
                result = input[i + 1];
            if (input[i + 2] < result)
                result = input[i + 2];
            if (input[i + 3] < result)
                result = input[i + 3];
        }
#endif
        for (; i < size; i++) {
            if (input[i] < result)
                result = input[i];
        }
    } else {
        int16_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            if (*ptr < result)
                result = *ptr;
            ptr += stride;
        }
    }

    return result;
}

float reduce_max(float *input, int32_t size, int32_t stride)
{
    float result = -INFINITY;

    if (stride == 1) {
        int32_t i = 0;
        for (; i + 3 < size; i += 4) {
            if (input[i] > result)
                result = input[i];
            if (input[i + 1] > result)
                result = input[i + 1];
            if (input[i + 2] > result)
                result = input[i + 2];
            if (input[i + 3] > result)
                result = input[i + 3];
        }
        for (; i < size; i++) {
            if (input[i] > result)
                result = input[i];
        }
    } else {
        float *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            if (*ptr > result)
                result = *ptr;
            ptr += stride;
        }
    }

    return result;
}

float reduce_min(float *input, int32_t size, int32_t stride)
{
    float result = INFINITY;

    if (stride == 1) {
        int32_t i = 0;
        for (; i + 3 < size; i += 4) {
            if (input[i] < result)
                result = input[i];
            if (input[i + 1] < result)
                result = input[i + 1];
            if (input[i + 2] < result)
                result = input[i + 2];
            if (input[i + 3] < result)
                result = input[i + 3];
        }
        for (; i < size; i++) {
            if (input[i] < result)
                result = input[i];
        }
    } else {
        float *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            if (*ptr < result)
                result = *ptr;
            ptr += stride;
        }
    }

    return result;
}

int8_t reduce_abs_max(int8_t *input, int32_t size, int32_t stride)
{
    int8_t result = 0; // abs is always >= 0

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            int8_t abs_val = input[i] < 0 ? (input[i] == INT8_MIN ? INT8_MAX : (int8_t)(-input[i])) : input[i];
            if (abs_val > result)
                result = abs_val;
            i++;
        }
        int32_t aligned_size = (size - i) & ~0xF; // largest multiple of 16 elements
        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            int8_t simd_result = dl_esp32p4_reduce_abs_max_s8_aligned(input + i, aligned_size);
#else
            int8_t simd_result = dl_tie728_reduce_abs_max_s8_aligned(input + i, aligned_size);
#endif
            if (simd_result > result)
                result = simd_result;
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            int8_t a0 = input[i] < 0 ? (input[i] == INT8_MIN ? INT8_MAX : (int8_t)(-input[i])) : input[i];
            int8_t a1 =
                input[i + 1] < 0 ? (input[i + 1] == INT8_MIN ? INT8_MAX : (int8_t)(-input[i + 1])) : input[i + 1];
            int8_t a2 =
                input[i + 2] < 0 ? (input[i + 2] == INT8_MIN ? INT8_MAX : (int8_t)(-input[i + 2])) : input[i + 2];
            int8_t a3 =
                input[i + 3] < 0 ? (input[i + 3] == INT8_MIN ? INT8_MAX : (int8_t)(-input[i + 3])) : input[i + 3];
            if (a0 > result)
                result = a0;
            if (a1 > result)
                result = a1;
            if (a2 > result)
                result = a2;
            if (a3 > result)
                result = a3;
        }
#endif
        for (; i < size; i++) {
            int8_t abs_val = input[i] < 0 ? (input[i] == INT8_MIN ? INT8_MAX : (int8_t)(-input[i])) : input[i];
            if (abs_val > result)
                result = abs_val;
        }
    } else {
        int8_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            int8_t abs_val = *ptr < 0 ? (*ptr == INT8_MIN ? INT8_MAX : (int8_t)(-*ptr)) : *ptr;
            if (abs_val > result)
                result = abs_val;
            ptr += stride;
        }
    }

    return result;
}

int16_t reduce_abs_max(int16_t *input, int32_t size, int32_t stride)
{
    int16_t result = 0; // abs is always >= 0

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            int16_t abs_val = input[i] < 0 ? (input[i] == INT16_MIN ? INT16_MAX : (int16_t)(-input[i])) : input[i];
            if (abs_val > result)
                result = abs_val;
            i++;
        }
        int32_t aligned_size = (size - i) & ~0x7; // largest multiple of 8 elements
        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            int16_t simd_result = dl_esp32p4_reduce_abs_max_s16_aligned(input + i, aligned_size);
#else
            int16_t simd_result = dl_tie728_reduce_abs_max_s16_aligned(input + i, aligned_size);
#endif
            if (simd_result > result)
                result = simd_result;
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            int16_t a0 = input[i] < 0 ? (input[i] == INT16_MIN ? INT16_MAX : (int16_t)(-input[i])) : input[i];
            int16_t a1 =
                input[i + 1] < 0 ? (input[i + 1] == INT16_MIN ? INT16_MAX : (int16_t)(-input[i + 1])) : input[i + 1];
            int16_t a2 =
                input[i + 2] < 0 ? (input[i + 2] == INT16_MIN ? INT16_MAX : (int16_t)(-input[i + 2])) : input[i + 2];
            int16_t a3 =
                input[i + 3] < 0 ? (input[i + 3] == INT16_MIN ? INT16_MAX : (int16_t)(-input[i + 3])) : input[i + 3];
            if (a0 > result)
                result = a0;
            if (a1 > result)
                result = a1;
            if (a2 > result)
                result = a2;
            if (a3 > result)
                result = a3;
        }
#endif
        for (; i < size; i++) {
            int16_t abs_val = input[i] < 0 ? (input[i] == INT16_MIN ? INT16_MAX : (int16_t)(-input[i])) : input[i];
            if (abs_val > result)
                result = abs_val;
        }
    } else {
        int16_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            int16_t abs_val = *ptr < 0 ? (*ptr == INT16_MIN ? INT16_MAX : (int16_t)(-*ptr)) : *ptr;
            if (abs_val > result)
                result = abs_val;
            ptr += stride;
        }
    }

    return result;
}

float reduce_abs_max(float *input, int32_t size, int32_t stride)
{
    float result = 0.0f;

    if (stride == 1) {
        int32_t i = 0;
        for (; i + 3 < size; i += 4) {
            float a0 = fabsf(input[i]);
            float a1 = fabsf(input[i + 1]);
            float a2 = fabsf(input[i + 2]);
            float a3 = fabsf(input[i + 3]);
            if (a0 > result)
                result = a0;
            if (a1 > result)
                result = a1;
            if (a2 > result)
                result = a2;
            if (a3 > result)
                result = a3;
        }
        for (; i < size; i++) {
            float abs_val = fabsf(input[i]);
            if (abs_val > result)
                result = abs_val;
        }
    } else {
        float *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            float abs_val = fabsf(*ptr);
            if (abs_val > result)
                result = abs_val;
            ptr += stride;
        }
    }

    return result;
}

int32_t reduce_sum(int8_t *input, int32_t size, int32_t stride)
{
    int32_t sum = 0;

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            sum += input[i];
            i++;
        }
        int32_t aligned_size = (size - i) & ~0xF; // largest multiple of 16 elements
        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            sum += dl_esp32p4_reduce_sum_s8_aligned(input + i, aligned_size);
#else
            sum += dl_tie728_reduce_sum_s8_aligned(input + i, aligned_size);
#endif
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            sum += (int32_t)input[i] + (int32_t)input[i + 1] + (int32_t)input[i + 2] + (int32_t)input[i + 3];
        }
#endif
        for (; i < size; i++) {
            sum += input[i];
        }
    } else {
        int8_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            sum += *ptr;
            ptr += stride;
        }
    }

    return sum;
}

int64_t reduce_sum(int16_t *input, int32_t size, int32_t stride)
{
    int64_t sum = 0;

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            sum += input[i];
            i++;
        }
        int32_t aligned_size = (size - i) & ~0x7; // largest multiple of 8 elements (16 bytes)

        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            sum += dl_esp32p4_reduce_sum_s16_aligned(input + i, aligned_size);
#else
            sum += dl_tie728_reduce_sum_s16_aligned(input + i, aligned_size);
#endif
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            sum += (int64_t)input[i] + (int64_t)input[i + 1] + (int64_t)input[i + 2] + (int64_t)input[i + 3];
        }
#endif
        for (; i < size; i++) {
            sum += input[i];
        }
    } else {
        int16_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            sum += *ptr;
            ptr += stride;
        }
    }

    return sum;
}

float reduce_sum(float *input, int32_t size, int32_t stride)
{
    float sum = 0.0f;

    if (stride == 1) {
        for (int32_t i = 0; i < size; i++) {
            sum += input[i];
        }
    } else {
        float *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            sum += *ptr;
            ptr += stride;
        }
    }

    return sum;
}

int32_t reduce_l1(int8_t *input, int32_t size, int32_t stride)
{
    int32_t sum = 0;

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            int32_t v = input[i];
            sum += v < 0 ? -v : v; // int32 accumulator: |INT8_MIN| = 128 is exact, no saturation
            i++;
        }
        int32_t aligned_size = (size - i) & ~0xF; // largest multiple of 16 elements
        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            sum += dl_esp32p4_reduce_l1_s8_aligned(input + i, aligned_size);
#else
            sum += dl_tie728_reduce_l1_s8_aligned(input + i, aligned_size);
#endif
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            int32_t v0 = input[i];
            int32_t v1 = input[i + 1];
            int32_t v2 = input[i + 2];
            int32_t v3 = input[i + 3];
            sum += (v0 < 0 ? -v0 : v0) + (v1 < 0 ? -v1 : v1) + (v2 < 0 ? -v2 : v2) + (v3 < 0 ? -v3 : v3);
        }
#endif
        for (; i < size; i++) {
            int32_t v = input[i];
            sum += v < 0 ? -v : v;
        }
    } else {
        int8_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            int32_t v = *ptr;
            sum += v < 0 ? -v : v;
            ptr += stride;
        }
    }

    return sum;
}

int64_t reduce_l1(int16_t *input, int32_t size, int32_t stride)
{
    int64_t sum = 0;

    if (stride == 1) {
        int32_t i = 0;
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
        // Scalar prologue to reach a 16-byte aligned address required by the SIMD load.
        while (i < size && (reinterpret_cast<uintptr_t>(input + i) & 0xF)) {
            int32_t v = input[i];
            sum += v < 0 ? -v : v; // int64 accumulator: |INT16_MIN| = 32768 is exact, no saturation
            i++;
        }
        int32_t aligned_size = (size - i) & ~0x7; // largest multiple of 8 elements (16 bytes)

        if (aligned_size > 0) {
#if CONFIG_PIE_V2_BOOST
            sum += dl_esp32p4_reduce_l1_s16_aligned(input + i, aligned_size);
#else
            sum += dl_tie728_reduce_l1_s16_aligned(input + i, aligned_size);
#endif
            i += aligned_size;
        }
#else
        for (; i + 3 < size; i += 4) {
            int32_t v0 = input[i];
            int32_t v1 = input[i + 1];
            int32_t v2 = input[i + 2];
            int32_t v3 = input[i + 3];
            sum += (v0 < 0 ? -v0 : v0) + (v1 < 0 ? -v1 : v1) + (v2 < 0 ? -v2 : v2) + (v3 < 0 ? -v3 : v3);
        }
#endif
        for (; i < size; i++) {
            int32_t v = input[i];
            sum += v < 0 ? -v : v;
        }
    } else {
        int16_t *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            int32_t v = *ptr;
            sum += v < 0 ? -v : v;
            ptr += stride;
        }
    }

    return sum;
}

float reduce_l1(float *input, int32_t size, int32_t stride)
{
    float sum = 0.0f;

    if (stride == 1) {
        for (int32_t i = 0; i < size; i++) {
            sum += fabsf(input[i]);
        }
    } else {
        float *ptr = input;
        for (int32_t i = 0; i < size; i++) {
            sum += fabsf(*ptr);
            ptr += stride;
        }
    }

    return sum;
}

int8_t reduce_mean(int8_t *input, int32_t size, int32_t stride)
{
    if (size == 0) {
        return 0;
    }
    int32_t sum = reduce_sum(input, size, stride);
    // Use dl::tool::round for platform-appropriate rounding:
    //   ESP32-P4: half to even   ESP32-S3: half up
    return (int8_t)dl::tool::round((float)sum / (float)size);
}

int16_t reduce_mean(int16_t *input, int32_t size, int32_t stride)
{
    if (size == 0) {
        return 0;
    }
    int64_t sum = reduce_sum(input, size, stride);
    return (int16_t)dl::tool::round((float)sum / (float)size);
}

float reduce_mean(float *input, int32_t size, int32_t stride)
{
    if (size == 0) {
        return 0.0f;
    }
    float sum = reduce_sum(input, size, stride);
    return sum / (float)size;
}

} // namespace base
} // namespace dl
