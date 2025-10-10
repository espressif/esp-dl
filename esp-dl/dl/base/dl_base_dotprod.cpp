#include "dl_base_dotprod.hpp"
#include "dl_base_isa.hpp"
#include "dl_tool.hpp"
#include "esp_dsp.h"

namespace dl {
namespace base {

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
    if (length % 16 == 0 && shift >= 0) {
#if CONFIG_ESP32P4_BOOST
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
        dl_esp32p4_dotprod_i8k8o16(output_ptr, input0_ptr, input1_ptr, shift, length);
#elif CONFIG_TIE728_BOOST
        __attribute__((aligned(16))) int64_t rounding_offset[1];
        rounding_offset[0] = 0;
        if (shift == 0) {
            rounding_offset[0] = 0;
        } else {
            rounding_offset[0] = (1 << (shift - 1));
        }
        dl_tie728_dotprod_i8k8o16(output_ptr, input0_ptr, input1_ptr, shift, length, rounding_offset);
#else
        dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
#endif
    } else {
        dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
    }
}

void dotprod(int8_t *input0_ptr, int16_t *input1_ptr, int16_t *output_ptr, int length, int shift)
{
    if (length % 8 == 0 && shift >= 0) {
#if CONFIG_ESP32P4_BOOST
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
        dl_esp32p4_dotprod_i16k8o16(output_ptr, input0_ptr, input1_ptr, shift, length);
#else
        dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
#endif
    } else {
        dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
    }
}

void dotprod(int16_t *input0_ptr, int16_t *input1_ptr, int16_t *output_ptr, int length, int shift)
{
    if (length % 8 == 0 && shift >= 0) {
#if CONFIG_ESP32P4_BOOST
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
        dl_esp32p4_dotprod_i16k16o16(output_ptr, input0_ptr, input1_ptr, shift, length);
#elif CONFIG_TIE728_BOOST
        __attribute__((aligned(16))) int64_t rounding_offset[1];
        if (shift == 0) {
            rounding_offset[0] = 0;
        } else {
            rounding_offset[0] = (1 << (shift - 1));
        }
        dl_tie728_dotprod_i16k16o16(output_ptr, input0_ptr, input1_ptr, shift, length, rounding_offset);
#else
        dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
#endif
    } else {
        dotprod_c(input0_ptr, input1_ptr, output_ptr, length, shift);
    }
}

void dotprod(float *input0_ptr, float *input1_ptr, float *output_ptr, int length, int shift)
{
    dsps_dotprod_f32(input0_ptr, input1_ptr, output_ptr, length);
}

} // namespace base
} // namespace dl
