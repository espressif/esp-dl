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

} // namespace base
} // namespace dl
