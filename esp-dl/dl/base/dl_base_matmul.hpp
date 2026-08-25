#pragma once

#include <stdint.h>

namespace dl {
namespace base {

/**
 * @brief Arguments for a native row-major matrix multiplication.
 *
 * Computes C[m, n] = requantize(sum_k A[m, k] * B[k, n]) for rows in
 * [row_start, row_end). All strides are expressed in elements, not bytes.
 * A, B and C may point at batch/broadcast-specific matrix bases; callers can
 * split row_start/row_end between cores without changing those base pointers.
 *
 * mac_shift follows Conv: output_exp - a_exp - b_exp. Positive values shift
 * right with the target rounding rule; negative values shift left. ReLU and
 * other fused activations are intentionally not supported.
 */
template <typename feature_t>
struct MatMulArgs {
    const feature_t *a = nullptr;
    const feature_t *b = nullptr;
    feature_t *c = nullptr;

    int32_t m = 0;
    int32_t n = 0;
    int32_t k = 0;

    int32_t a_stride = 0;
    int32_t b_stride = 0;
    int32_t c_stride = 0;

    int32_t row_start = 0;
    int32_t row_end = 0;
    int32_t mac_shift = 0;
};

template <typename feature_t>
void matmul(const MatMulArgs<feature_t> &args);

template <>
void matmul<int8_t>(const MatMulArgs<int8_t> &args);

template <>
void matmul<int16_t>(const MatMulArgs<int16_t> &args);

} // namespace base
} // namespace dl
