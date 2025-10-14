#include "dl_base.hpp"
#include "dl_base_elemwise.hpp"
#include "dl_base_isa.hpp"
#include <math.h>

namespace dl {
namespace base {

// input0_ptr:vector, input1_ptr:scalar
template <typename feature_t>
void c_impl_pow_n_1(feature_t *output_ptr, feature_t *input0_ptr, feature_t *input1_ptr, void *args)
{
    elemwiseArgsType<feature_t> *elem_args = static_cast<elemwiseArgsType<feature_t> *>(args);
    int32_t length = elem_args->output_d0;
    float base_scale = elem_args->input0_scale;
    float exp_scale = elem_args->input1_scale;
    float rescale = elem_args->output_rescale;

    float exp_val = input1_ptr[0] * exp_scale;

    for (int i = 0; i < length; i++) {
        float base_val = input0_ptr[i] * base_scale;
        float result = powf(base_val, exp_val);
        tool::truncate<int32_t>(output_ptr[i], tool::round(result * rescale));
    }
}

// input0_ptr:scalar, input1_ptr:vector
template <typename feature_t>
void c_impl_pow_1_n(feature_t *output_ptr, feature_t *input0_ptr, feature_t *input1_ptr, void *args)
{
    elemwiseArgsType<feature_t> *elem_args = static_cast<elemwiseArgsType<feature_t> *>(args);
    int32_t length = elem_args->output_d0;
    float base_scale = elem_args->input0_scale;
    float exp_scale = elem_args->input1_scale;
    float rescale = elem_args->output_rescale;

    float base_val = input0_ptr[0] * base_scale;

    for (int i = 0; i < length; i++) {
        float exp_val = input1_ptr[i] * exp_scale;
        float result = powf(base_val, exp_val);
        tool::truncate<int32_t>(output_ptr[i], tool::round(result * rescale));
    }
}

// input0_ptr:vector, input1_ptr:vector
template <typename feature_t>
void c_impl_pow_n_n(feature_t *output_ptr, feature_t *input0_ptr, feature_t *input1_ptr, void *args)
{
    elemwiseArgsType<feature_t> *elem_args = static_cast<elemwiseArgsType<feature_t> *>(args);
    int32_t length = elem_args->output_d0;
    float base_scale = elem_args->input0_scale;
    float exp_scale = elem_args->input1_scale;
    float rescale = elem_args->output_rescale;

    for (int i = 0; i < length; i++) {
        float base_val = input0_ptr[i] * base_scale;
        float exp_val = input1_ptr[i] * exp_scale;
        float result = powf(base_val, exp_val);
        tool::truncate<int32_t>(output_ptr[i], tool::round(result * rescale));
    }
}

template <>
void c_impl_pow_n_1<float>(float *output_ptr, float *input0_ptr, float *input1_ptr, void *args)
{
    elemwiseArgsType<float> *elem_args = static_cast<elemwiseArgsType<float> *>(args);
    int32_t length = elem_args->output_d0;

    float exp_val = input1_ptr[0];

    for (int i = 0; i < length; i++) {
        output_ptr[i] = powf(input0_ptr[i], exp_val);
    }
}

// input0_ptr:scalar, input1_ptr:vector
template <>
void c_impl_pow_1_n<float>(float *output_ptr, float *input0_ptr, float *input1_ptr, void *args)
{
    elemwiseArgsType<float> *elem_args = static_cast<elemwiseArgsType<float> *>(args);
    int32_t length = elem_args->output_d0;

    float base_val = input0_ptr[0];

    for (int i = 0; i < length; i++) {
        output_ptr[i] = powf(base_val, input1_ptr[i]);
    }
}

// input0_ptr:vector, input1_ptr:vector
template <>
void c_impl_pow_n_n<float>(float *output_ptr, float *input0_ptr, float *input1_ptr, void *args)
{
    elemwiseArgsType<float> *elem_args = static_cast<elemwiseArgsType<float> *>(args);
    int32_t length = elem_args->output_d0;

    for (int i = 0; i < length; i++) {
        output_ptr[i] = powf(input0_ptr[i], input1_ptr[i]);
    }
}

void elemwise_pow(elemwiseArgsType<int8_t> *args)
{
    int ilen = 16 / sizeof(int8_t);
    ImplFunc_t<int8_t, int8_t, int8_t> elemwise_func = c_impl_pow_n_n<int8_t>; // default impl

    if (args->output_d0 >= ilen) {
        if (args->input1_d0 == 1) {
            elemwise_func = c_impl_pow_n_1<int8_t>;
        } else if (args->input0_d0 == 1) {
            elemwise_func = c_impl_pow_1_n<int8_t>;
        }
    } else {
        if (args->input1_d0 == 1) {
            elemwise_func = c_impl_pow_n_1<int8_t>;
        } else if (args->input0_d0 == 1) {
            elemwise_func = c_impl_pow_1_n<int8_t>;
        }
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

void elemwise_pow(elemwiseArgsType<int16_t> *args)
{
    int ilen = 16 / sizeof(int16_t);
    ImplFunc_t<int16_t, int16_t, int16_t> elemwise_func = c_impl_pow_n_n<int16_t>;

    if (args->output_d0 >= ilen) {
        if (args->input1_d0 == 1) {
            elemwise_func = c_impl_pow_n_1<int16_t>;
        } else if (args->input0_d0 == 1) {
            elemwise_func = c_impl_pow_1_n<int16_t>;
        }
    } else {
        if (args->input1_d0 == 1) {
            elemwise_func = c_impl_pow_n_1<int16_t>;
        } else if (args->input0_d0 == 1) {
            elemwise_func = c_impl_pow_1_n<int16_t>;
        }
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

void elemwise_pow(elemwiseArgsType<float> *args)
{
    int ilen = 16 / sizeof(float);
    ImplFunc_t<float, float, float> elemwise_func = c_impl_pow_n_n<float>;

    if (args->output_d0 >= ilen) {
        if (args->input1_d0 == 1) {
            elemwise_func = c_impl_pow_n_1<float>;
        } else if (args->input0_d0 == 1) {
            elemwise_func = c_impl_pow_1_n<float>;
        }
    } else {
        if (args->input1_d0 == 1) {
            elemwise_func = c_impl_pow_n_1<float>;
        } else if (args->input0_d0 == 1) {
            elemwise_func = c_impl_pow_1_n<float>;
        }
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

} // namespace base
} // namespace dl
