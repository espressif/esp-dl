#include "dl_base_c.hpp"
#include "dl_base_conv_args.hpp"
#include "dl_compile_config.h"
#include "dl_define.hpp"
#include "dl_tool.hpp"
#include <cstddef>

namespace dl {
namespace base {

template <typename feature_t, typename bias_t, typename buffer_t>
void buffer_bias_linear(feature_t *output_ptr, buffer_t *buffer_ptr, const ConvArgsType &args)
{
    bias_t *bias_ptr = (bias_t *)args.bias_element;
    if (args.mac_shift == INT_MIN) // per-channel
    {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], 4);
            // Bias
            buffer_ptr[output_c] += bias_ptr[output_c];
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.filter_channel_factor[output_c] - 4);

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    } else {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // Bias
            buffer_ptr[output_c] += bias_ptr[output_c];
            // right shift
            buffer_ptr[output_c] = tool::shift_and_round(buffer_ptr[output_c], args.mac_shift);

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    }
}

template <typename feature_t, typename bias_t, typename buffer_t>
void buffer_bias_relu(feature_t *output_ptr, buffer_t *buffer_ptr, const ConvArgsType &args)
{
    bias_t *bias_ptr = (bias_t *)args.bias_element;
    if (args.mac_shift == INT_MIN) // per-channel
    {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], 4);
            // Bias
            buffer_ptr[output_c] += bias_ptr[output_c];
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.filter_channel_factor[output_c] - 4);
            // Activation
            if (buffer_ptr[output_c] < 0)
                buffer_ptr[output_c] = 0;

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    } else {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // Bias
            buffer_ptr[output_c] += bias_ptr[output_c];
            // right shift
            buffer_ptr[output_c] = tool::shift_and_round(buffer_ptr[output_c], args.mac_shift);
            // Activation
            if (buffer_ptr[output_c] < 0)
                buffer_ptr[output_c] = 0;

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    }
}

template <typename feature_t, typename bias_t, typename buffer_t>
void buffer_bias_leakyrelu(feature_t *output_ptr, buffer_t *buffer_ptr, const ConvArgsType &args)
{
    bias_t *bias_ptr = (bias_t *)args.bias_element;
    if (args.mac_shift == INT_MIN) // per-channel
    {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], 4);
            // Bias
            buffer_ptr[output_c] += bias_ptr[output_c];
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.filter_channel_factor[output_c] - 4);
            // Activation
            if (buffer_ptr[output_c] < 0) {
                buffer_ptr[output_c] *= args.activation_alpha;
                buffer_ptr[output_c] >>= args.activation_shift;
            }

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    } else {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.mac_shift);
            // Bias
            buffer_ptr[output_c] += bias_ptr[output_c];
            // Activation
            if (buffer_ptr[output_c] < 0) {
                buffer_ptr[output_c] *= args.activation_alpha;
                buffer_ptr[output_c] >>= args.activation_shift;
            }

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    }
}

template <typename feature_t, typename bias_t, typename buffer_t>
void buffer_bias_prelu(feature_t *output_ptr, buffer_t *buffer_ptr, const ConvArgsType &args)
{
    bias_t *bias_ptr = (bias_t *)args.bias_element;
    feature_t *alpha_ptr = (feature_t *)args.activation_alpha_ptr;
    if (args.mac_shift == INT_MIN) // per-channel
    {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], 4);
            // Bias
            buffer_ptr[output_c] += bias_ptr[output_c];
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.filter_channel_factor[output_c] - 4);

            // Activation
            if (buffer_ptr[output_c] < 0) {
                buffer_ptr[output_c] *= alpha_ptr[output_c];
                buffer_ptr[output_c] >>= args.activation_shift;
            }
            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);

            buffer_ptr[output_c] = 0;
        }
    } else {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.mac_shift);
            // Bias
            buffer_ptr[output_c] += bias_ptr[output_c];
            // Activation
            if (buffer_ptr[output_c] < 0) {
                buffer_ptr[output_c] *= alpha_ptr[output_c];
                buffer_ptr[output_c] >>= args.activation_shift;
            }
            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);

            buffer_ptr[output_c] = 0;
        }
    }
}

/**
 * @brief without bias
 *
 */
template <typename feature_t, typename buffer_t>
void buffer_0000_linear(feature_t *output_ptr, buffer_t *buffer_ptr, const ConvArgsType &args)
{
    if (args.mac_shift == INT_MIN) // per-channel
    {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.filter_channel_factor[output_c]);

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    } else {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = tool::shift_and_round(buffer_ptr[output_c], args.mac_shift);

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    }
}

template <typename feature_t, typename buffer_t>
void buffer_0000_relu(feature_t *output_ptr, buffer_t *buffer_ptr, const ConvArgsType &args)
{
    if (args.mac_shift == INT_MIN) // per-channel
    {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.filter_channel_factor[output_c]);
            // Activation
            if (buffer_ptr[output_c] < 0)
                buffer_ptr[output_c] = 0;

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    } else {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = tool::shift_and_round(buffer_ptr[output_c], args.mac_shift);
            // Activation
            if (buffer_ptr[output_c] < 0)
                buffer_ptr[output_c] = 0;

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    }
}

template <typename feature_t, typename buffer_t>
void buffer_0000_leakyrelu(feature_t *output_ptr, buffer_t *buffer_ptr, const ConvArgsType &args)
{
    if (args.mac_shift == INT_MIN) // per-channel
    {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.filter_channel_factor[output_c]);
            // Activation
            if (buffer_ptr[output_c] < 0) {
                buffer_ptr[output_c] *= args.activation_alpha;
                buffer_ptr[output_c] >>= args.activation_shift;
            }

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    } else {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.mac_shift);
            // Activation
            if (buffer_ptr[output_c] < 0) {
                buffer_ptr[output_c] *= args.activation_alpha;
                buffer_ptr[output_c] >>= args.activation_shift;
            }

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    }
}

template <typename feature_t, typename buffer_t>
void buffer_0000_prelu(feature_t *output_ptr, buffer_t *buffer_ptr, const ConvArgsType &args)
{
    feature_t *alpha_ptr = (feature_t *)args.activation_alpha_ptr;
    if (args.mac_shift == INT_MIN) // per-channel
    {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.filter_channel_factor[output_c]);
            // Activation
            if (buffer_ptr[output_c] < 0) {
                buffer_ptr[output_c] *= alpha_ptr[output_c];
                buffer_ptr[output_c] >>= args.activation_shift;
            }

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    } else {
        for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
            // right shift
            buffer_ptr[output_c] = DL_RIGHT_SHIFT(buffer_ptr[output_c], args.mac_shift);
            // Activation
            if (buffer_ptr[output_c] < 0) {
                buffer_ptr[output_c] *= alpha_ptr[output_c];
                buffer_ptr[output_c] >>= args.activation_shift;
            }

            tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
            buffer_ptr[output_c] = 0;
        }
    }
}

#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S16_TAIL_LINEAR
template void buffer_0000_linear<int16_t, DL_S16_BUFFER_TYPE>(int16_t *, DL_S16_BUFFER_TYPE *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S16_TAIL_RELU
template void buffer_0000_relu<int16_t, DL_S16_BUFFER_TYPE>(int16_t *, DL_S16_BUFFER_TYPE *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S16_TAIL_BIAS_LINEAR
template void buffer_bias_linear<int16_t, DL_S16_BUFFER_TYPE, DL_S16_BUFFER_TYPE>(int16_t *,
                                                                                  DL_S16_BUFFER_TYPE *,
                                                                                  const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S16_TAIL_BIAS_RELU
template void buffer_bias_relu<int16_t, DL_S16_BUFFER_TYPE, DL_S16_BUFFER_TYPE>(int16_t *,
                                                                                DL_S16_BUFFER_TYPE *,
                                                                                const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_TAIL_LINEAR
template void buffer_0000_linear<int8_t, int32_t>(int8_t *, int32_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_TAIL_RELU
template void buffer_0000_relu<int8_t, int32_t>(int8_t *, int32_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_TAIL_LEAKYRELU
template void buffer_0000_leakyrelu<int8_t, int32_t>(int8_t *, int32_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_TAIL_PRELU
template void buffer_0000_prelu<int8_t, int32_t>(int8_t *, int32_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_TAIL_BIAS_LINEAR
template void buffer_bias_linear<int8_t, int32_t, int32_t>(int8_t *, int32_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_TAIL_BIAS_RELU
template void buffer_bias_relu<int8_t, int32_t, int32_t>(int8_t *, int32_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_TAIL_BIAS_PER_CH_LINEAR
template void buffer_bias_linear<int8_t, int16_t, int32_t>(int8_t *, int32_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_TAIL_BIAS_PER_CH_RELU
template void buffer_bias_relu<int8_t, int16_t, int32_t>(int8_t *, int32_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_TAIL_BIAS_PER_CH_LEAKYRELU
template void buffer_bias_leakyrelu<int8_t, int16_t, int32_t>(int8_t *, int32_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_TAIL_BIAS_PER_CH_PRELU || DL_KERNEL_DL_C_S8_DW_TAIL_BIAS_PER_CH_PRELU
template void buffer_bias_prelu<int8_t, int8_t, int32_t>(int8_t *, int32_t *, const ConvArgsType &);
#endif

} // namespace base
} // namespace dl
