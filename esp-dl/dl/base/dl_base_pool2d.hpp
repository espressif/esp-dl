#pragma once

#include "dl_base.hpp"

namespace dl {
namespace base {
/**
 * @brief
 *
 * @tparam feature_t
 */
template <typename feature_t>
struct PoolArgsType {
    feature_t *input_element;  /*!<  0 */
    int input_channel;         /*!<  1 */
    int input_stride_y_offset; /*!<  2 input_width_with_padding * input_channel_with_padding * stride_y */
    int input_stride_x_offset; /*!<  3 input_channel_with_padding * stride_x */
    int input_y_offset_bytes;  /*!<  4 input_width_with_padding * input_channel_with_padding * sizeof(feature_t) */
    int input_x_offset_bytes;  /*!<  5 input_channel_with_padding * sizeof(feature_t) */
                               //
    feature_t *output_element; /*!<  6 */
    int output_height;         /*!<  7 */
    int output_width;          /*!<  8 */
    int output_channel;        /*!<  9 */
    int output_y_offset;       /*!< 10 output_width_with_padding * output_channel_with_padding */
    int output_x_offset;       /*!< 11 output_channel_with_padding */
                               //
    int filter_height;         /*!< 12 */
    int filter_width;          /*!< 13 */
    int mac_shift;             /*!< 14 mac_shift = output.exponent - filter.exponent - input.exponent */
                               //
    int c_remainder;           /*!< 15 input_channel % (vector_width / element_width) */

    int avg_pool_area_inv; /*!< 16 2^n/(filter_h * filter_w) */

    int input_height;
    int input_width;
    int padding_h_head;
    int padding_h_tail;
    int padding_w_head;
    int padding_w_tail;

    int stride_x;
    int stride_y;
    int pool_exponent;  /*!< 25 exponent of 1.0 / (filter_height * filter_width)*/
    int c_div_x_1;      /*!< 26 */
    int input_y_offset; /*!<  27 input_width_with_padding * input_channel_with_padding */
    int input_x_offset; /*!<  28 input_channel_with_padding */
    int input_exponent;
    int output_exponent;
    int avg_pool_area;
};

/**
 * @brief Get the pool args object
 *
 * @tparam feature_t
 * @param output
 * @param input
 * @param padding
 * @param filter_shape
 * @param stride_y
 * @param stride_x
 * @param runtime_mode
 * @return std::vector<PoolArgsType<feature_t>>
 */
template <typename feature_t>
std::vector<PoolArgsType<feature_t>> get_pool_args(TensorBase *output,
                                                   TensorBase *input,
                                                   const std::vector<int> &padding,
                                                   const std::vector<int> &filter_shape,
                                                   const std::vector<int> &strides,
                                                   const runtime_mode_t runtime_mode = RUNTIME_MODE_AUTO)
{
    PoolArgsType<feature_t> args;

    if (input->shape.size() == 3) {
        args.input_height = 1;
        args.input_width = input->shape[1];
        args.input_channel = input->shape[2];

        args.output_height = 1;
        args.output_width = output->shape[1];
        args.output_channel = output->shape[2];
        args.filter_height = 1;
        args.filter_width = filter_shape[0];

        args.padding_h_head = 0;
        args.padding_h_tail = 0;
        args.padding_w_head = padding[0];
        args.padding_w_tail = padding[1];

        args.stride_x = strides[0];
        args.stride_y = 1;
    } else if (input->shape.size() == 4) {
        args.input_height = input->shape[1];
        args.input_width = input->shape[2];
        args.input_channel = input->shape[3];

        args.output_height = output->shape[1];
        args.output_width = output->shape[2];
        args.output_channel = output->shape[3];
        args.filter_height = filter_shape[0];
        args.filter_width = filter_shape[1];

        args.padding_h_head = padding[0];
        args.padding_h_tail = padding[1];
        args.padding_w_head = padding[2];
        args.padding_w_tail = padding[3];

        args.stride_x = strides[1];
        args.stride_y = strides[0];
    } else {
        ESP_LOGE(__FUNCTION__, "Do not support input shape.");
        return {};
    }

    args.input_element = (feature_t *)input->get_element_ptr();
    args.input_stride_y_offset = args.input_width * args.input_channel * args.stride_y;
    args.input_stride_x_offset = args.input_channel * args.stride_x;
    args.input_y_offset = args.input_width * args.input_channel;
    args.input_x_offset = args.input_channel;
    args.input_y_offset_bytes = args.input_y_offset * sizeof(feature_t);
    args.input_x_offset_bytes = args.input_x_offset * sizeof(feature_t);
    args.input_exponent = input->exponent;

    args.output_element = (feature_t *)output->get_element_ptr();
    args.output_y_offset = args.output_width * args.output_channel;
    args.output_x_offset = args.output_channel;
    args.output_exponent = output->exponent;

    args.avg_pool_area = args.filter_height * args.filter_width;
    int max_value = INT_MAX;
#if CONFIG_ESP32P4_BOOST
    if (sizeof(feature_t) == 1) {
        max_value = 127;
    } else if (sizeof(feature_t) == 2) {
        max_value = 32767;
    }
#else
    if (sizeof(feature_t) == 1) {
        max_value = 64;
    } else if (sizeof(feature_t) == 2) {
        max_value = 16384;
    }
#endif
    args.pool_exponent = -tool::calculate_exponent(args.filter_height * args.filter_width, max_value);
    args.mac_shift = output->exponent - args.pool_exponent - input->exponent;

    // for ISA
    int u = 16 / sizeof(feature_t);
    args.c_remainder = (args.input_channel % u) * sizeof(feature_t);
#if CONFIG_ESP32P4_BOOST
    args.avg_pool_area_inv = tool::round(1.f / (args.filter_height * args.filter_width) * (1 << (-args.pool_exponent)));
#else
    args.avg_pool_area_inv = (1 << (-args.pool_exponent)) / (args.filter_height * args.filter_width);
#endif
    int c_div_x = args.input_channel / u;
    if (args.c_remainder != 0 && args.input_x_offset % u == 0 && args.output_x_offset % u == 0 &&
        !((unsigned)&args.input_element[0] & 15) && !((unsigned)&args.output_element[0] & 15)) {
        c_div_x += 1;
    }
    args.c_div_x_1 = c_div_x - 1;

    // slice
    std::vector<PoolArgsType<feature_t>> m_args(1, args);
    if (runtime_mode == RUNTIME_MODE_MULTI_CORE) {
    }

    return m_args;
}

typedef void (*avg_pool_c_impl_func_s16_t)(float *, int16_t *, int16_t *, PoolArgsType<int16_t> &);
typedef void (*avg_pool_c_impl_func_s8_t)(float *, int8_t *, int8_t *, PoolArgsType<int8_t> &);

typedef void (*max_pool_c_impl_func_s16_t)(int16_t *, int16_t *, PoolArgsType<int16_t> &);
typedef void (*max_pool_c_impl_func_s8_t)(int8_t *, int8_t *, PoolArgsType<int8_t> &);

/**
 * @brief
 *
 * @tparam feature_t
 * @tparam buffer_t
 * @param args
 * @param i_impl_func
 * @param i_impl_func_sp
 * @param c_impl_func
 * @param n_wise_tail
 */
template <typename feature_t, typename buffer_t>
void avg_pool_shell(PoolArgsType<feature_t> &args,
                    ImplFunc_t<feature_t, feature_t> i_impl_func,
                    ImplFunc_t<feature_t, feature_t> i_impl_func_sp,
                    void (*c_impl_func)(buffer_t *, feature_t *, feature_t *, PoolArgsType<feature_t> &))
{
    feature_t *input_ptr_real = (feature_t *)args.input_element;
    feature_t *output_ptr = (feature_t *)args.output_element;
    int n_h_head = (args.padding_h_head + args.stride_y - 1) / args.stride_y;
    int n_w_head = (args.padding_w_head + args.stride_x - 1) / args.stride_x;
    int n_h_body = (args.input_height + args.padding_h_head - args.filter_height) / args.stride_y + 1 - n_h_head;
    if (n_h_body < 0)
        n_h_body = 0;
    int n_w_body = (args.input_width + args.padding_w_head - args.filter_width) / args.stride_x + 1 - n_w_head;
    if (n_w_body < 0)
        n_w_body = 0;
    int n_h_tail = args.output_height - n_h_head - n_h_body;
    int n_w_tail = args.output_width - n_w_head - n_w_body;
    int filter_h = args.filter_height;
    int filter_w = args.filter_width;
    int new_pool_exponent = args.pool_exponent;
    int max_value = INT_MAX;
#if CONFIG_ESP32P4_BOOST
    if (sizeof(feature_t) == 1) {
        max_value = 127;
    } else if (sizeof(feature_t) == 2) {
        max_value = 32767;
    }
#else
    if (sizeof(feature_t) == 1) {
        max_value = 64;
    } else if (sizeof(feature_t) == 2) {
        max_value = 16384;
    }
#endif

#if !CONFIG_ACCURATE_INFER
    if (i_impl_func && i_impl_func_sp) {
        feature_t *input_syx_real = input_ptr_real;
        feature_t *output_yx = output_ptr;
        for (size_t output_y = 0; output_y < n_h_head; output_y++) {
            output_yx = output_ptr;
            input_syx_real = input_ptr_real;
            args.filter_height = filter_h - args.padding_h_head + output_y * args.stride_y;
            // Fix for filter_size > input_size
            int filter_height_excess =
                filter_h - (args.input_height + (args.padding_h_head - output_y * args.stride_y));
            if (filter_height_excess > 0) {
                args.filter_height -= filter_height_excess;
            }

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                new_pool_exponent = -tool::calculate_exponent(args.filter_height * args.filter_width, max_value);
                args.mac_shift = args.mac_shift + args.pool_exponent - new_pool_exponent;
                args.pool_exponent = new_pool_exponent;
#if CONFIG_ESP32P4_BOOST
                args.avg_pool_area_inv =
                    tool::round(1.f / (args.filter_height * args.filter_width) * (1 << (-args.pool_exponent)));
#else
                args.avg_pool_area_inv = (1 << (-args.pool_exponent)) / (args.filter_height * args.filter_width);
#endif
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            new_pool_exponent = -tool::calculate_exponent(args.filter_height * args.filter_width, max_value);
            args.mac_shift = args.mac_shift + args.pool_exponent - new_pool_exponent;
            args.pool_exponent = new_pool_exponent;
#if CONFIG_ESP32P4_BOOST
            args.avg_pool_area_inv =
                tool::round(1.f / (args.filter_height * args.filter_width) * (1 << (-args.pool_exponent)));
#else
            args.avg_pool_area_inv = (1 << (-args.pool_exponent)) / (args.filter_height * args.filter_width);
#endif
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = filter_w - args.padding_w_tail + (n_w_tail - 1 - output_x) * args.stride_x;
                new_pool_exponent = -tool::calculate_exponent(args.filter_height * args.filter_width, max_value);
                args.mac_shift = args.mac_shift + args.pool_exponent - new_pool_exponent;
                args.pool_exponent = new_pool_exponent;
#if CONFIG_ESP32P4_BOOST
                args.avg_pool_area_inv =
                    tool::round(1.f / (args.filter_height * args.filter_width) * (1 << (-args.pool_exponent)));
#else
                args.avg_pool_area_inv = (1 << (-args.pool_exponent)) / (args.filter_height * args.filter_width);
#endif
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }
            output_ptr += args.output_y_offset;
        }

        input_ptr_real += (args.stride_y * n_h_head - args.padding_h_head) * args.input_y_offset;
        args.filter_height = filter_h;

        for (size_t output_y = 0; output_y < n_h_body; output_y++) {
            input_syx_real = input_ptr_real;
            output_yx = output_ptr;

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                new_pool_exponent = -tool::calculate_exponent(args.filter_height * args.filter_width, max_value);
                args.mac_shift = args.mac_shift + args.pool_exponent - new_pool_exponent;
                args.pool_exponent = new_pool_exponent;
#if CONFIG_ESP32P4_BOOST
                args.avg_pool_area_inv =
                    tool::round(1.f / (args.filter_height * args.filter_width) * (1 << (-args.pool_exponent)));
#else
                args.avg_pool_area_inv = (1 << (-args.pool_exponent)) / (args.filter_height * args.filter_width);
#endif
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            new_pool_exponent = -tool::calculate_exponent(args.filter_height * args.filter_width, max_value);
            args.mac_shift = args.mac_shift + args.pool_exponent - new_pool_exponent;
            args.pool_exponent = new_pool_exponent;
#if CONFIG_ESP32P4_BOOST
            args.avg_pool_area_inv =
                tool::round(1.f / (args.filter_height * args.filter_width) * (1 << (-args.pool_exponent)));
#else
            args.avg_pool_area_inv = (1 << (-args.pool_exponent)) / (args.filter_height * args.filter_width);
#endif
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                i_impl_func_sp(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = filter_w - args.padding_w_tail + (n_w_tail - 1 - output_x) * args.stride_x;
                new_pool_exponent = -tool::calculate_exponent(args.filter_height * args.filter_width, max_value);
                args.mac_shift = args.mac_shift + args.pool_exponent - new_pool_exponent;
                args.pool_exponent = new_pool_exponent;
#if CONFIG_ESP32P4_BOOST
                args.avg_pool_area_inv =
                    tool::round(1.f / (args.filter_height * args.filter_width) * (1 << (-args.pool_exponent)));
#else
                args.avg_pool_area_inv = (1 << (-args.pool_exponent)) / (args.filter_height * args.filter_width);
#endif
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            input_ptr_real += args.input_stride_y_offset;
            output_ptr += args.output_y_offset;
        }

        for (size_t output_y = 0; output_y < n_h_tail; output_y++) {
            input_syx_real = input_ptr_real;
            output_yx = output_ptr;
            args.filter_height = filter_h - args.padding_h_tail + (n_h_tail - 1 - output_y) * args.stride_y;

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                new_pool_exponent = -tool::calculate_exponent(args.filter_height * args.filter_width, max_value);
                args.mac_shift = args.mac_shift + args.pool_exponent - new_pool_exponent;
                args.pool_exponent = new_pool_exponent;
#if CONFIG_ESP32P4_BOOST
                args.avg_pool_area_inv =
                    tool::round(1.f / (args.filter_height * args.filter_width) * (1 << (-args.pool_exponent)));
#else
                args.avg_pool_area_inv = (1 << (-args.pool_exponent)) / (args.filter_height * args.filter_width);
#endif
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            new_pool_exponent = -tool::calculate_exponent(args.filter_height * args.filter_width, max_value);
            args.mac_shift = args.mac_shift + args.pool_exponent - new_pool_exponent;
            args.pool_exponent = new_pool_exponent;
#if CONFIG_ESP32P4_BOOST
            args.avg_pool_area_inv =
                tool::round(1.f / (args.filter_height * args.filter_width) * (1 << (-args.pool_exponent)));
#else
            args.avg_pool_area_inv = (1 << (-args.pool_exponent)) / (args.filter_height * args.filter_width);
#endif
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = filter_w - args.padding_w_tail + (n_w_tail - 1 - output_x) * args.stride_x;
                new_pool_exponent = -tool::calculate_exponent(args.filter_height * args.filter_width, max_value);
                args.mac_shift = args.mac_shift + args.pool_exponent - new_pool_exponent;
                args.pool_exponent = new_pool_exponent;
#if CONFIG_ESP32P4_BOOST
                args.avg_pool_area_inv =
                    tool::round(1.f / (args.filter_height * args.filter_width) * (1 << (-args.pool_exponent)));
#else
                args.avg_pool_area_inv = (1 << (-args.pool_exponent)) / (args.filter_height * args.filter_width);
#endif
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            input_ptr_real += args.input_stride_y_offset;
            output_ptr += args.output_y_offset;
        }
    } else // run c_impl_func
#endif
    {
        buffer_t *buffer = (buffer_t *)heap_caps_calloc(args.output_channel, sizeof(buffer_t), MALLOC_CAP_DEFAULT);
        feature_t *input_syx_real = input_ptr_real;
        feature_t *output_yx = output_ptr;
        for (size_t output_y = 0; output_y < n_h_head; output_y++) {
            output_yx = output_ptr;
            input_syx_real = input_ptr_real;
            args.filter_height = filter_h - args.padding_h_head + output_y * args.stride_y;
            // Fix for filter_size > input_size
            int filter_height_excess =
                filter_h - (args.input_height + (args.padding_h_head - output_y * args.stride_y));
            if (filter_height_excess > 0) {
                args.filter_height -= filter_height_excess;
            }

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                c_impl_func(buffer, input_syx_real, output_yx, args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                c_impl_func(buffer, input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = args.padding_w_head + args.input_width - (n_w_head + n_w_body) * args.stride_x -
                    output_x * args.stride_x;
                c_impl_func(buffer, input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }
            output_ptr += args.output_y_offset;
        }

        input_ptr_real += (args.stride_y * n_h_head - args.padding_h_head) * args.input_y_offset;
        args.filter_height = filter_h;
        for (size_t output_y = 0; output_y < n_h_body; output_y++) {
            input_syx_real = input_ptr_real;
            output_yx = output_ptr;

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                c_impl_func(buffer, input_syx_real, output_yx, args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                c_impl_func(buffer, input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = args.padding_w_head + args.input_width - (n_w_head + n_w_body) * args.stride_x -
                    output_x * args.stride_x;
                c_impl_func(buffer, input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            input_ptr_real += args.input_stride_y_offset;
            output_ptr += args.output_y_offset;
        }

        for (size_t output_y = 0; output_y < n_h_tail; output_y++) {
            input_syx_real = input_ptr_real;
            output_yx = output_ptr;
            args.filter_height = args.padding_h_head + args.input_height - (n_h_head + n_h_body) * args.stride_y -
                output_y * args.stride_y;

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                c_impl_func(buffer, input_syx_real, output_yx, args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                c_impl_func(buffer, input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = args.padding_w_head + args.input_width - (n_w_head + n_w_body) * args.stride_x -
                    output_x * args.stride_x;
                c_impl_func(buffer, input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            input_ptr_real += args.input_stride_y_offset;
            output_ptr += args.output_y_offset;
        }
        heap_caps_free(buffer);
    }
    return;
}

/**
 * @brief
 *
 * @tparam feature_t
 * @tparam buffer_t
 * @param args
 * @param i_impl_func
 * @param i_impl_func_sp
 * @param c_impl_func
 * @param n_wise_tail
 */
template <typename feature_t>
void max_pool_shell(PoolArgsType<feature_t> &args,
                    ImplFunc_t<feature_t, feature_t> i_impl_func,
                    ImplFunc_t<feature_t, feature_t> i_impl_func_sp,
                    void (*c_impl_func)(feature_t *, feature_t *, PoolArgsType<feature_t> &))
{
    feature_t *input_ptr_real = (feature_t *)args.input_element;
    feature_t *output_ptr = (feature_t *)args.output_element;
    int n_h_head = (args.padding_h_head + args.stride_y - 1) / args.stride_y;
    int n_w_head = (args.padding_w_head + args.stride_x - 1) / args.stride_x;
    int n_h_body = ((args.input_height + args.padding_h_head - args.filter_height) / args.stride_y + 1) - n_h_head;
    if (n_h_body < 0)
        n_h_body = 0;
    int n_w_body = ((args.input_width + args.padding_w_head - args.filter_width) / args.stride_x + 1) - n_w_head;
    if (n_w_body < 0)
        n_w_body = 0;
    int n_h_tail = args.output_height - n_h_head - n_h_body;
    int n_w_tail = args.output_width - n_w_head - n_w_body;

    int filter_h = args.filter_height;
    int filter_w = args.filter_width;

    if (i_impl_func && i_impl_func_sp) {
        feature_t *input_syx_real = input_ptr_real;
        feature_t *output_yx = output_ptr;
        for (size_t output_y = 0; output_y < n_h_head; output_y++) {
            output_yx = output_ptr;
            input_syx_real = input_ptr_real;
            args.filter_height = filter_h - args.padding_h_head + output_y * args.stride_y;
            // Fix for filter_size > input_size
            int filter_height_excess =
                filter_h - (args.input_height + (args.padding_h_head - output_y * args.stride_y));
            if (filter_height_excess > 0) {
                args.filter_height -= filter_height_excess;
            }

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = args.padding_w_head + args.input_width - (n_w_head + n_w_body) * args.stride_x -
                    output_x * args.stride_x;
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }
            output_ptr += args.output_y_offset;
        }

        input_ptr_real += (args.stride_y * n_h_head - args.padding_h_head) * args.input_y_offset;
        args.filter_height = filter_h;

        for (size_t output_y = 0; output_y < n_h_body; output_y++) {
            input_syx_real = input_ptr_real;
            output_yx = output_ptr;

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                i_impl_func_sp(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = args.padding_w_head + args.input_width - (n_w_head + n_w_body) * args.stride_x -
                    output_x * args.stride_x;
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            input_ptr_real += args.input_stride_y_offset;
            output_ptr += args.output_y_offset;
        }

        for (size_t output_y = 0; output_y < n_h_tail; output_y++) {
            input_syx_real = input_ptr_real;
            output_yx = output_ptr;
            args.filter_height = args.padding_h_head + args.input_height - (n_h_head + n_h_body) * args.stride_y -
                output_y * args.stride_y;

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = args.padding_w_head + args.input_width - (n_w_head + n_w_body) * args.stride_x -
                    output_x * args.stride_x;
                i_impl_func(output_yx, input_syx_real, (void *const)&args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            input_ptr_real += args.input_stride_y_offset;
            output_ptr += args.output_y_offset;
        }
    } else // run c_impl_func
    {
        feature_t *input_syx_real = input_ptr_real;
        feature_t *output_yx = output_ptr;
        for (size_t output_y = 0; output_y < n_h_head; output_y++) {
            output_yx = output_ptr;
            input_syx_real = input_ptr_real;
            args.filter_height = filter_h - args.padding_h_head + output_y * args.stride_y;
            // Fix for filter_size > input_size
            int filter_height_excess =
                filter_h - (args.input_height + (args.padding_h_head - output_y * args.stride_y));
            if (filter_height_excess > 0) {
                args.filter_height -= filter_height_excess;
            }

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                c_impl_func(input_syx_real, output_yx, args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                c_impl_func(input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = args.padding_w_head + args.input_width - (n_w_head + n_w_body) * args.stride_x -
                    output_x * args.stride_x;
                c_impl_func(input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }
            output_ptr += args.output_y_offset;
        }

        input_ptr_real += (args.stride_y * n_h_head - args.padding_h_head) * args.input_y_offset;
        args.filter_height = filter_h;
        for (size_t output_y = 0; output_y < n_h_body; output_y++) {
            input_syx_real = input_ptr_real;
            output_yx = output_ptr;

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                c_impl_func(input_syx_real, output_yx, args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                c_impl_func(input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = args.padding_w_head + args.input_width - (n_w_head + n_w_body) * args.stride_x -
                    output_x * args.stride_x;
                c_impl_func(input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            input_ptr_real += args.input_stride_y_offset;
            output_ptr += args.output_y_offset;
        }

        for (size_t output_y = 0; output_y < n_h_tail; output_y++) {
            input_syx_real = input_ptr_real;
            output_yx = output_ptr;
            args.filter_height = args.padding_h_head + args.input_height - (n_h_head + n_h_body) * args.stride_y -
                output_y * args.stride_y;

            for (size_t output_x = 0; output_x < n_w_head; output_x++) {
                args.filter_width = filter_w - args.padding_w_head + output_x * args.stride_x;
                // Fix for filter_size > input_size
                int filter_width_excess =
                    filter_w - (args.input_width + (args.padding_w_head - output_x * args.stride_x));
                if (filter_width_excess > 0) {
                    args.filter_width -= filter_width_excess;
                }

                c_impl_func(input_syx_real, output_yx, args);
                output_yx += args.output_x_offset;
            }

            input_syx_real += (args.stride_x * n_w_head - args.padding_w_head) * args.input_x_offset;
            args.filter_width = filter_w;
            for (size_t output_x = 0; output_x < n_w_body; output_x++) {
                c_impl_func(input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            for (size_t output_x = 0; output_x < n_w_tail; output_x++) {
                args.filter_width = args.padding_w_head + args.input_width - (n_w_head + n_w_body) * args.stride_x -
                    output_x * args.stride_x;
                c_impl_func(input_syx_real, output_yx, args);
                input_syx_real += args.input_stride_x_offset;
                output_yx += args.output_x_offset;
            }

            input_ptr_real += args.input_stride_y_offset;
            output_ptr += args.output_y_offset;
        }
    }
    return;
}
} // namespace base
} // namespace dl
