#pragma once

#include "dl_base.hpp"

namespace dl {
namespace base {

/**
 * @brief RMS normalization for int8
 *
 * @param output output tensor
 * @param input input tensor
 * @param scale scale tensor (float)
 * @param rms rms tensor
 * @param n number of elements
 */
void rms_norm(int8_t *output, int8_t *input, float *scale, float *rms, int n);

/**
 * @brief RMS normalization for int16
 *
 * @param output output tensor
 * @param input input tensor
 * @param scale scale tensor (float)
 * @param rms rms tensor
 * @param n number of elements
 */
void rms_norm(int16_t *output, int16_t *input, float *scale, float *rms, int n);

/**
 * @brief output[i] = truncate(round(input[i] * scale)), with the product computed in float.
 *
 * @param output output tensor, may be the same as input
 * @param input input tensor
 * @param scale scale factor
 * @param n number of elements
 */
void scale_round(int8_t *output, const int8_t *input, float scale, int n);
void scale_round(int16_t *output, const int16_t *input, float scale, int n);

} // namespace base
} // namespace dl
