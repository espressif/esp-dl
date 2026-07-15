#include "dl_base_lut.hpp"
#include "dl_base_isa.hpp"
#include "dl_tool.hpp"
#include <cassert>
#include <cstdint>

namespace dl {
namespace base {

static void lut_s8_c(int8_t *output, const int8_t *input, int32_t size, const int8_t *table)
{
    for (int32_t i = 0; i < size; i++) {
        output[i] = table[static_cast<uint8_t>(input[i]) ^ 0x80u];
    }
}

void lut_s8(int8_t *output, const int8_t *input, int32_t size, const int8_t *table)
{
    assert(output);
    assert(input);
    assert(table);

    if (size <= 0) {
        return;
    }

    int32_t processed = 0;
#if CONFIG_PIE_V2_BOOST || CONFIG_PIE_V1_BOOST
    bool aligned = !(reinterpret_cast<uintptr_t>(output) & 0xf) && !(reinterpret_cast<uintptr_t>(input) & 0xf) &&
        !(reinterpret_cast<uintptr_t>(table) & 0xf);
#if CONFIG_PIE_V2_BOOST
    int32_t simd_size = size & ~0x7;
#else
    int32_t simd_size = size & ~0xf;
#endif

    if (aligned && simd_size > 0) {
#if CONFIG_PIE_V2_BOOST
        dl_esp32p4_s8_lut(output, const_cast<int8_t *>(input), simd_size / 8, const_cast<int8_t *>(table));
#else
        dl_tie728_s8_lut(output, const_cast<int8_t *>(input), simd_size / 16, const_cast<int8_t *>(table));
#endif
        processed = simd_size;
    }
#endif

    lut_s8_c(output + processed, input + processed, size - processed, table);
}

static inline uint32_t lut_index(int16_t input, int32_t shift)
{
    uint32_t value = static_cast<uint16_t>(input) ^ 0x8000u;
    if (shift == 0) {
        return value;
    }

    uint32_t index = value >> shift;
    uint32_t remainder = value & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);

#if CONFIG_PIE_V2_BOOST
    if (remainder > half || (remainder == half && (index & 1u))) {
#else
    if (remainder >= half) {
#endif
        index++;
    }
    return index;
}

static void lut_s16_nearest_neighbor_c(
    int16_t *output, const int16_t *input, int32_t size, const int16_t *table, int32_t shift)
{
    for (int32_t i = 0; i < size; i++) {
        output[i] = table[lut_index(input[i], shift)];
    }
}

void lut_s16_nearest_neighbor(int16_t *output, const int16_t *input, int32_t size, const int16_t *table, int32_t step)
{
    if (size <= 0) {
        return;
    }

    int32_t shift = __builtin_ctz(static_cast<uint32_t>(step));
    int32_t processed = 0;

#if CONFIG_PIE_V2_BOOST || CONFIG_PIE_V1_BOOST
    bool aligned = !(reinterpret_cast<uintptr_t>(output) & 0xf) && !(reinterpret_cast<uintptr_t>(input) & 0xf) &&
        !(reinterpret_cast<uintptr_t>(table) & 0xf);
    int32_t simd_size = size & ~0x7;

    if (aligned && simd_size > 0) {
#if CONFIG_PIE_V2_BOOST
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
        dl_esp32p4_s16_lut_nearest_neighbor(
            output, const_cast<int16_t *>(input), simd_size / 8, const_cast<int16_t *>(table), shift);
#else
        dl_tie728_s16_lut_nearest_neighbor(
            output, const_cast<int16_t *>(input), simd_size / 8, const_cast<int16_t *>(table), shift);
#endif
        processed = simd_size;
    }
#endif

    lut_s16_nearest_neighbor_c(output + processed, input + processed, size - processed, table, shift);
}

} // namespace base
} // namespace dl
