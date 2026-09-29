#pragma once

#include <stdint.h>

namespace dl {
namespace audio {

/**
 * @brief Natural logarithm of an unsigned 64-bit integer.
 *
 * @param v Input, must be > 0.
 * @return int64_t ln(v) in Q36. The absolute error is below 2^-28.
 */
int64_t ln_u64_q36(uint64_t v);

/**
 * @brief Square root of an unsigned 64-bit integer.
 *
 * @param v Input.
 * @return uint64_t round(sqrt(v) * 2^8). The relative error is below 2^-24.
 */
uint64_t sqrt_u64_q8(uint64_t v);

/** ln(2) in Q36. */
constexpr int64_t LN2_Q36 = 47632711549LL;

} // namespace audio
} // namespace dl
