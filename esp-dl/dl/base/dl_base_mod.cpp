#include "dl_base.hpp"
#include "dl_base_elemwise.hpp"
#include "dl_base_isa.hpp"
#include <math.h>

namespace dl {
namespace base {

// C-style fmod: result = a - trunc(a/b) * b, sign follows dividend
template <typename feature_t>
void c_impl_mod_n_1(feature_t *output_ptr, feature_t *input0_ptr, feature_t *input1_ptr, void *args)
{
    elemwiseArgsType<feature_t> *elem_args = static_cast<elemwiseArgsType<feature_t> *>(args);
    int32_t length = elem_args->output_d0;
    float input0_scale = elem_args->input0_scale;
    float input1_scale = elem_args->input1_scale;
    float rescale = elem_args->output_rescale;

    float b_val = input1_ptr[0] * input1_scale;

    for (int i = 0; i < length; i++) {
        float a_val = input0_ptr[i] * input0_scale;
        float result = (b_val != 0.0f) ? fmodf(a_val, b_val) : 0.0f;
        tool::truncate<int32_t>(output_ptr[i], tool::round(result * rescale));
    }
}

template <typename feature_t>
void c_impl_mod_1_n(feature_t *output_ptr, feature_t *input0_ptr, feature_t *input1_ptr, void *args)
{
    elemwiseArgsType<feature_t> *elem_args = static_cast<elemwiseArgsType<feature_t> *>(args);
    int32_t length = elem_args->output_d0;
    float input0_scale = elem_args->input0_scale;
    float input1_scale = elem_args->input1_scale;
    float rescale = elem_args->output_rescale;

    float a_val = input0_ptr[0] * input0_scale;

    for (int i = 0; i < length; i++) {
        float b_val = input1_ptr[i] * input1_scale;
        float result = (b_val != 0.0f) ? fmodf(a_val, b_val) : 0.0f;
        tool::truncate<int32_t>(output_ptr[i], tool::round(result * rescale));
    }
}

template <typename feature_t>
void c_impl_mod_n_n(feature_t *output_ptr, feature_t *input0_ptr, feature_t *input1_ptr, void *args)
{
    elemwiseArgsType<feature_t> *elem_args = static_cast<elemwiseArgsType<feature_t> *>(args);
    int32_t length = elem_args->output_d0;
    float input0_scale = elem_args->input0_scale;
    float input1_scale = elem_args->input1_scale;
    float rescale = elem_args->output_rescale;

    for (int i = 0; i < length; i++) {
        float a_val = input0_ptr[i] * input0_scale;
        float b_val = input1_ptr[i] * input1_scale;
        float result = (b_val != 0.0f) ? fmodf(a_val, b_val) : 0.0f;
        tool::truncate<int32_t>(output_ptr[i], tool::round(result * rescale));
    }
}

template <>
void c_impl_mod_n_1<float>(float *output_ptr, float *input0_ptr, float *input1_ptr, void *args)
{
    elemwiseArgsType<float> *elem_args = static_cast<elemwiseArgsType<float> *>(args);
    int32_t length = elem_args->output_d0;
    float b_val = input1_ptr[0];

    for (int i = 0; i < length; i++) {
        output_ptr[i] = (b_val != 0.0f) ? fmodf(input0_ptr[i], b_val) : 0.0f;
    }
}

template <>
void c_impl_mod_1_n<float>(float *output_ptr, float *input0_ptr, float *input1_ptr, void *args)
{
    elemwiseArgsType<float> *elem_args = static_cast<elemwiseArgsType<float> *>(args);
    int32_t length = elem_args->output_d0;
    float a_val = input0_ptr[0];

    for (int i = 0; i < length; i++) {
        output_ptr[i] = (input1_ptr[i] != 0.0f) ? fmodf(a_val, input1_ptr[i]) : 0.0f;
    }
}

template <>
void c_impl_mod_n_n<float>(float *output_ptr, float *input0_ptr, float *input1_ptr, void *args)
{
    elemwiseArgsType<float> *elem_args = static_cast<elemwiseArgsType<float> *>(args);
    int32_t length = elem_args->output_d0;

    for (int i = 0; i < length; i++) {
        output_ptr[i] = (input1_ptr[i] != 0.0f) ? fmodf(input0_ptr[i], input1_ptr[i]) : 0.0f;
    }
}

template <typename T>
void elemwise_mod(elemwiseArgsType<T> *args)
{
    ImplFunc_t<T, T, T> elemwise_func = c_impl_mod_n_n<T>;

    if (args->input1_d0 == 1) {
        elemwise_func = c_impl_mod_n_1<T>;
    } else if (args->input0_d0 == 1) {
        elemwise_func = c_impl_mod_1_n<T>;
    }

    switch (args->dims) {
    case 1:
        elemwise_loop_1d(args, elemwise_func);
        break;
    case 2:
        elemwise_loop_2d(args, elemwise_func);
        break;
    case 3:
        elemwise_loop_3d(args, elemwise_func);
        break;
    case 4:
        elemwise_loop_4d(args, elemwise_func);
        break;
    default:
        break;
    }
}

template void elemwise_mod(elemwiseArgsType<int8_t> *args);
template void elemwise_mod(elemwiseArgsType<int16_t> *args);
template void elemwise_mod(elemwiseArgsType<float> *args);

} // namespace base
} // namespace dl
