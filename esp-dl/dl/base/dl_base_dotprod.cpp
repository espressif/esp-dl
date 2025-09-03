#include "dl_base_dotprod.hpp"
#include "dl_tool.hpp"

namespace dl {
namespace base {

void dotprod_c(int8_t *input0_ptr, int8_t *input1_ptr, int16_t *output_ptr, int length, int bias, int shift)
{
    int32_t result = (int32_t)bias;

    for (int i = 0; i < length; i++) {
        result += (int32_t)input0_ptr[i] * (int32_t)input1_ptr[i];
    }

    result = DL_RIGHT_SHIFT(result, shift);

    dl::tool::truncate(*output_ptr, result);
}

void dotprod_c(int16_t *input0_ptr, int16_t *input1_ptr, int16_t *output_ptr, int length, int bias, int shift)
{
    int64_t result = (int64_t)bias;

    for (int i = 0; i < length; i++) {
        result += (int32_t)input0_ptr[i] * (int32_t)input1_ptr[i];
    }

    result = DL_RIGHT_SHIFT(result, shift);

    dl::tool::truncate(*output_ptr, result);
}

void dotprod_c(float *input0_ptr, float *input1_ptr, float *output_ptr, int length, float bias)
{
    float result = bias;

    for (int i = 0; i < length; i++) {
        result += input0_ptr[i] * input1_ptr[i];
    }

    *output_ptr = result;
}

void dotprod(int8_t *input0_ptr, int8_t *input1_ptr, int16_t *output_ptr, int length, int bias, int shift)
{
    dotprod_c(input0_ptr, input1_ptr, output_ptr, length, bias, shift);
}

void dotprod(int16_t *input0_ptr, int16_t *input1_ptr, int16_t *output_ptr, int length, int bias, int shift)
{
    dotprod_c(input0_ptr, input1_ptr, output_ptr, length, bias, shift);
}

void dotprod(float *input0_ptr, float *input1_ptr, float *output_ptr, int length, float bias, int shift)
{
    dotprod_c(input0_ptr, input1_ptr, output_ptr, length, bias);
}

} // namespace base
} // namespace dl
