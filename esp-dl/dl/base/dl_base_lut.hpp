#pragma once

#include "dl_base.hpp"

namespace dl {
namespace base {

/**
 * @brief Apply an int8 lookup table.
 *
 * Converts each signed input to the unsigned domain and reads the corresponding
 * int8 entry from @p table. ESP32-P4 and ESP32-S3 use their PIE kernels when
 * alignment and size permit; other cases use the C implementation.
 *
 * @param output Output buffer.
 * @param input Input buffer.
 * @param size Number of int8 elements.
 * @param table 256-entry LUT base address. The SIMD path requires 16-byte
 *              alignment.
 */
void lut_s8(int8_t *output, const int8_t *input, int32_t size, const int8_t *table);

/**
 * @brief Apply an int16 nearest-neighbor lookup table.
 *
 * Converts each signed input to the unsigned domain, divides it by
 * @p step using the target's rounding convention, and reads the corresponding
 * int16 entry from @p table. ESP32-P4 and ESP32-S3 use their PIE kernels when
 * alignment and size permit; other cases use the C implementation.
 *
 * @param output Output buffer.
 * @param input Input buffer.
 * @param size Number of int16 elements.
 * @param table LUT base address. SIMD paths require 16-byte alignment and one
 *              padded int16 after the final logical entry.
 * @param step LUT sampling step. Must be a positive power of two.
 */
void lut_s16_nearest_neighbor(int16_t *output, const int16_t *input, int32_t size, const int16_t *table, int32_t step);

} // namespace base
} // namespace dl
