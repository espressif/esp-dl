#include "dl_base_dotprod.hpp"
#include "dl_base_isa.hpp"
#include "dl_tool.hpp"

namespace dl {
namespace base {

/**
 * @brief Scale up the accumulator for a negative shift.
 *
 * PIE dot products support only right shifts. For negative shifts, use the
 * vector path with shift 0, then scale the result by 2^(-shift).
 * All callers pass shift < 0 and truncate the result to int16. Shifts below
 * -15 can return the saturated int16 value directly, avoiding overflow.
 */
static inline int64_t dotprod_scale_up(int32_t accumulated, int shift)
{
    if (shift < -15) {
        return accumulated > 0 ? INT16_MAX :
               accumulated < 0 ? INT16_MIN : 0;
    }

    return (int64_t)accumulated * ((int64_t)1 << (-shift));
}

void dotprod_c(int8_t *input0_ptr, int8_t *input1_ptr, int16_t *output_ptr, int length, int shift)
{
    int32_t result = 0;
    float scale = DL_RESCALE(shift);

    for (int i = 0; i < length; i++) {
        result += (int32_t)input0_ptr[i] * (int32_t)input1_ptr[i];
    }

    dl::tool::truncate(*output_ptr, tool::round(result * scale));
}

void dotprod_c(int8_t *input0_ptr, int16_t *input1_ptr, int16_t *output_ptr, int length, int shift)
{
    int32_t result = 0;
    float scale = DL_RESCALE(shift);

    for (int i = 0; i < length; i++) {
        result += (int32_t)input0_ptr[i] * (int32_t)input1_ptr[i];
    }

    dl::tool::truncate(*output_ptr, tool::round(result * scale));
}

void dotprod_c(int16_t *input0_ptr, int16_t *input1_ptr, int16_t *output_ptr, int length, int shift)
{
    int64_t result = 0;
    float scale = DL_RESCALE(shift);

    for (int i = 0; i < length; i++) {
        result += (int32_t)input0_ptr[i] * (int32_t)input1_ptr[i];
    }

    dl::tool::truncate(*output_ptr, tool::round(result * scale));
}

void dotprod(int8_t *input0_ptr, int8_t *input1_ptr, int16_t *output_ptr, int length, int shift)
{
    if (length <= 0) {
        *output_ptr = 0;
        return;
    }
    if (length % 16 != 0) {
        dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
        return;
    }
#if CONFIG_PIE_V2_BOOST
    dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
    if (shift >= 0) {
        dl_esp32p4_dotprod_i8k8o16(output_ptr, input0_ptr, input1_ptr, shift, length);
    } else {
        int16_t accumulated = 0;
        dl_esp32p4_dotprod_i8k8o16(&accumulated, input0_ptr, input1_ptr, 0, length);
        dl::tool::truncate(*output_ptr, dotprod_scale_up(accumulated, shift));
    }
#elif CONFIG_PIE_V1_BOOST
    __attribute__((aligned(16))) int64_t rounding_offset[1];
    rounding_offset[0] = shift > 0 ? (1 << (shift - 1)) : 0;
    if (shift >= 0) {
        dl_tie728_dotprod_i8k8o16(output_ptr, input0_ptr, input1_ptr, shift, length, rounding_offset);
    } else {
        int16_t accumulated = 0;
        dl_tie728_dotprod_i8k8o16(&accumulated, input0_ptr, input1_ptr, 0, length, rounding_offset);
        dl::tool::truncate(*output_ptr, dotprod_scale_up(accumulated, shift));
    }
#else
    dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
#endif
}

void dotprod(int8_t *input0_ptr, int16_t *input1_ptr, int16_t *output_ptr, int length, int shift)
{
    if (length <= 0) {
        *output_ptr = 0;
        return;
    }
    // vldext.s8 loads 16 int8 values. An 8-element tail would read 8 bytes
    // beyond input0, so keep that shape on the scalar path.
    if (length % 16 != 0) {
        dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
        return;
    }
#if CONFIG_PIE_V2_BOOST
    dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
    if (shift >= 0) {
        dl_esp32p4_dotprod_i16k8o16(output_ptr, input0_ptr, input1_ptr, shift, length);
    } else {
        int16_t accumulated = 0;
        dl_esp32p4_dotprod_i16k8o16(&accumulated, input0_ptr, input1_ptr, 0, length);
        dl::tool::truncate(*output_ptr, dotprod_scale_up(accumulated, shift));
    }
#else
    dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
#endif
}

void dotprod(int16_t *input0_ptr, int16_t *input1_ptr, int16_t *output_ptr, int length, int shift)
{
    if (length <= 0) {
        *output_ptr = 0;
        return;
    }
    if (length % 8 != 0) {
        dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
        return;
    }
#if CONFIG_PIE_V2_BOOST
    dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
    if (shift >= 0) {
        dl_esp32p4_dotprod_i16k16o16(output_ptr, input0_ptr, input1_ptr, shift, length);
    } else {
        // o32 keeps the full accumulator instead of saturating to int16 first.
        int32_t accumulated = 0;
        dl_esp32p4_dotprod_i16k16o32(&accumulated, input0_ptr, input1_ptr, 0, length);
        dl::tool::truncate(*output_ptr, dotprod_scale_up(accumulated, shift));
    }
#elif CONFIG_PIE_V1_BOOST
    __attribute__((aligned(16))) int64_t rounding_offset[1];
    rounding_offset[0] = shift > 0 ? (1 << (shift - 1)) : 0;
    if (shift >= 0) {
        dl_tie728_dotprod_i16k16o16(output_ptr, input0_ptr, input1_ptr, shift, length, rounding_offset);
    } else {
        int16_t accumulated = 0;
        dl_tie728_dotprod_i16k16o16(&accumulated, input0_ptr, input1_ptr, 0, length, rounding_offset);
        dl::tool::truncate(*output_ptr, dotprod_scale_up(accumulated, shift));
    }
#else
    dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
#endif
}

void mat_vecs_dotprod(
    int8_t *matrix, int8_t *vectors, int16_t *result, int rows, int cols, int vectors_num, int shift)
{
#if CONFIG_PIE_V2_BOOST
    if (cols > 0 && cols % 16 == 0 && shift >= 0 && rows > 0 && vectors_num > 0) {
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
        dl_esp32p4_mat_vecs_i8k8o16(result, matrix, vectors, shift, cols, rows, vectors_num);
        return;
    }
#endif
    mat_vecs_dotprod<int8_t, int8_t, int16_t>(matrix, vectors, result, rows, cols, vectors_num, cols, rows, shift);
}

void dotprod(float *input0_ptr, float *input1_ptr, float *output_ptr, int length, int shift)
{
#if CONFIG_PIE_V2_BOOST
    dl_esp32p4_dotprod_f32(output_ptr, input0_ptr, input1_ptr, length);
#elif CONFIG_PIE_V1_BOOST
    dl_tie728_dotprod_f32(output_ptr, input0_ptr, input1_ptr, length);
#else
    float result = 0.0f;
    for (int i = 0; i < length; i++) {
        result += input0_ptr[i] * input1_ptr[i];
    }
    *output_ptr = result;
#endif
}

} // namespace base
} // namespace dl
