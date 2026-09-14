#pragma once

#include "dl_base.hpp"

namespace dl {
namespace base {

// Use the original path for small convolutions.
// The limits come from P4 tests for the three quantization types.
template <typename T, typename filter_t = T>
constexpr int64_t conv_width_minimum_macs()
{
    return sizeof(T) == 1 ? 384 * 1024 : (sizeof(filter_t) == 1 ? 112 * 1024 : 160 * 1024);
}

// Use only with batch 1, group 1 and MULTI_CORE mode.
// The height split must return one task.
template <typename T, typename filter_t = T>
inline bool can_split_conv_width(const ArgsType<T> &a, int64_t minimum_macs = conv_width_minimum_macs<T, filter_t>())
{
    constexpr int lanes = 16 / sizeof(T);
    return a.input_height >= 1 && a.output_height >= 1 && a.filter_height >= 1 && a.filter_width >= 1 &&
        (a.filter_height > 1 || a.filter_width > 1) && a.dilation_h > 0 && a.dilation_w > 0 && a.stride_x == 1 &&
        a.stride_y == 1 && !a.padding_h_head && !a.padding_h_tail && !a.padding_w_head && !a.padding_w_tail &&
        a.output_width >= 2 &&
        static_cast<int64_t>(a.output_width) + static_cast<int64_t>(a.filter_width - 1) * a.dilation_w ==
        a.input_width &&
        static_cast<int64_t>(a.output_height) + static_cast<int64_t>(a.filter_height - 1) * a.dilation_h ==
        a.input_height &&
        a.input_channel >= lanes && a.output_channel >= lanes && a.input_channel % lanes == 0 &&
        a.output_channel % lanes == 0 &&
        static_cast<int64_t>(a.output_height) * a.output_width * a.input_channel * a.output_channel * a.filter_height *
            a.filter_width >=
        minimum_macs &&
        a.input_channel_with_padding == a.input_channel && a.input_stride_x_offset == a.input_channel &&
        a.input_stride_y_offset == a.input_width * a.input_channel && a.output_x_offset == a.output_channel &&
        a.output_y_offset == a.output_width * a.output_channel &&
        !(reinterpret_cast<uintptr_t>(a.input_element) & 15) && !(reinterpret_cast<uintptr_t>(a.output_element) & 15) &&
        (a.activation_type == Linear || a.activation_type == ReLU);
}

// Keep the original row strides and input dimensions.
// Each task writes separate columns and reads the original input.
template <typename T>
inline void set_conv_width_interval(ArgsType<T> &a, int begin, int count)
{
    a.input_element += begin * a.input_stride_x_offset;
    a.output_element += begin * a.output_x_offset;
    a.output_width = count;
}

} // namespace base
} // namespace dl
