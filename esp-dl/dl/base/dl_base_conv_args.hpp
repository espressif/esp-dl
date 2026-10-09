#pragma once

#include "dl_define.hpp"
#include "dl_tensor_base.hpp"
#include "dl_tool.hpp"
#include "esp_log.h"
#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dl {
namespace base {

inline int conv_feature_bytes(quant_type_t q)
{
    return (q == QUANT_TYPE_SYMM_8BIT) ? 1 : 2;
}

inline int conv_filter_bytes(quant_type_t q)
{
    return (q == QUANT_TYPE_SYMM_16BIT) ? 2 : 1;
}

inline const void *conv_ptr_add(const void *p, int n_elems, int elem_bytes)
{
    return (const char *)p + (size_t)n_elems * (size_t)elem_bytes;
}

inline void *conv_ptr_add(void *p, int n_elems, int elem_bytes)
{
    return const_cast<void *>(conv_ptr_add(static_cast<const void *>(p), n_elems, elem_bytes));
}

inline int conv_buffer_bytes(quant_type_t q)
{
    return (q == QUANT_TYPE_SYMM_8BIT) ? (int)sizeof(int32_t) : (int)sizeof(DL_S16_BUFFER_TYPE);
}

struct ConvArgsType {
    void *input_element = nullptr;        /*!<  0 */
    int input_channel;                    /*!<  1 */
    int input_stride_y_offset;            /*!<  2 input_width_with_padding * input_channel_with_padding * stride_y */
    int input_stride_x_offset;            /*!<  3 input_channel_with_padding * stride_x */
    int input_dilation_y_offset;          /*!<  4 input_width_with_padding * input_channel_with_padding * dilation_y */
    int input_dilation_x_offset;          /*!<  5 input_channel_with_padding * dilation_x */
                                          //
    void *output_element = nullptr;       /*!<  6 */
    int output_height;                    /*!<  7 */
    int output_width;                     /*!<  8 */
    int output_channel;                   /*!<  9 */
    int output_y_offset;                  /*!< 10 output_width_with_padding * output_channel_with_padding */
    int output_x_offset;                  /*!< 11 output_channel_with_padding */
                                          //
    const void *filter_element = nullptr; /*!< 12 */
    int filter_height;                    /*!< 13 */
    int filter_width;                     /*!< 14 */
    int filter_y_offset;                  /*!< 15 filter_width * input_channel */
    int mac_shift;                        /*!< 16 mac_shift = output.exponent - filter.exponent - input.exponent */
                                          //
    const void *bias_element = nullptr;   /*!< 17 */
                                          //
    activation_type_t activation_type;    /*!< 18 */
    int activation_alpha;                 /*!< 19 */
    const void *activation_alpha_ptr = nullptr; /*!< 20 */
    int activation_shift;                       /*!< 21 */
                                                //
    int c_rs1_1;                                /*!< 22 (input_channel >> 1) - 1 */
    int c_rs2_1;                                /*!< 23 (input_channel >> 2) - 1 */
    int n_div_x;                                /*!< 24 output_channel / (vector_width / element_width) */
    int c_div_x_1;                              /*!< 25 input_channel / (vector_width / element_width) - 1 */
    void *tie_filter_channel_factor = nullptr;  /*!< 26 */

    //
    int xtensa_dilation_x_offset;     /*!< 27 (dilation_x * input_channel_with_padding - input_channel) *
                                         sizeof(feature_t)*/
    int xtensa_dilation_y_offset;     /*!< 28 */
                                      //
    int tie_conv2d_dilation_x_offset; /*!< 29 TODO: not used, to be deleted. | dilation_x * input_channel_with_padding *
                                         sizeof(feature_t) - (c_div_x_1 + 1) * 16 */
    int tie_conv2d_dilation_y_offset; /*!< 30 TODO: not used, to be deleted. | */
                                      //
    int tie_depth2d_dilation_x_offset; /*!< 31 */
    int tie_depth2d_dilation_y_offset; /*!< 32 */
    int tie_depth2d_next_hwx1;         /*!< 33 */
                                       //
    int c_remainder;                   /*!< 34 input_channel % (vector_width / element_width) * sizeof(feature_t) */
    int n_remainder;                   /*!< 35 */
    int filter_n_offset;               /*!< 36 filter_height * filter_width * input_channel */
    int filter_w_rs1_1;                /*!< 37 (filter_width >> 1) - 1 */
    int16_t *filter_channel_factor = nullptr; /*!< 38 */
    int input_channel_with_padding;           /*!< 39 */

    int filter_y_offset_unaligned;                  /*!< 40 */
    int filter_n_offset_unaligned;                  /*!< 41 */
    const void *filter_element_unaligned = nullptr; /*!< 42 */

    int output_shift; /*!< 43 */
    int output_scale; /*!< 44 */

    int padding_h_head;                       /*!< 45 */
    int padding_h_tail;                       /*!< 46 */
    int padding_w_head;                       /*!< 47 */
    int padding_w_tail;                       /*!< 48 */
    int dilation_h;                           /*!< 49 */
    int dilation_w;                           /*!< 50 */
    int stride_x;                             /*!< 51 */
    int stride_y;                             /*!< 52 */
    int input_y_offset;                       /*!< 53 */
    int filter_c;                             /*!< 54 */
    int xtensa_dilation_y_offset_stable;      /*!< 55 */
    int tie_depth2d_dilation_y_offset_stable; /*!< 56 */
    int input_width;                          /*!< 57 */

    int filter_y_offset_c; /*!< 58 */
    int filter_n_offset_c; /*!< 59 */

    int element_num;             /*!< 60 */
    int input_height;            /*!< 61 */
    void *debug_value = nullptr; /*!< 62 It will malloc 16 bytes memory if malloc_debug_memory = true */
    bool auto_split;
};

using conv_pie_fn_t = void (*)(void *output, void *input, void *args);
using conv_c_mac_fn_t = void (*)(void *buffer, void *input, const ConvArgsType &args);
using conv_c_tail_fn_t = void (*)(void *output, void *buffer, const ConvArgsType &args);

struct ConvRegions {
    int n_h_head;
    int n_h_body;
    int n_h_tail;
    int n_w_head;
    int n_w_body;
    int n_w_tail;
};

// Outputs [0, end) whose whole window fits before the input's far edge. A
// negative reach must give 0: C division truncates it up to an extra column.
inline int conv_full_window_end(int input_size, int padding_head, int dilation, int filter_size, int stride)
{
    const int reach = input_size + padding_head - dilation * (filter_size - 1) - 1;
    return reach < 0 ? 0 : reach / stride + 1;
}

inline ConvRegions conv_regions(const ConvArgsType &args)
{
    ConvRegions r;
    r.n_h_head = (args.padding_h_head + args.stride_y - 1) / args.stride_y;
    if (r.n_h_head > args.output_height)
        r.n_h_head = args.output_height;
    r.n_w_head = (args.padding_w_head + args.stride_x - 1) / args.stride_x;
    if (r.n_w_head > args.output_width)
        r.n_w_head = args.output_width;

    r.n_h_body = conv_full_window_end(
                     args.input_height, args.padding_h_head, args.dilation_h, args.filter_height, args.stride_y) -
        r.n_h_head;
    if (r.n_h_body < 0)
        r.n_h_body = 0;
    r.n_w_body =
        conv_full_window_end(args.input_width, args.padding_w_head, args.dilation_w, args.filter_width, args.stride_x) -
        r.n_w_head;
    if (r.n_w_body < 0)
        r.n_w_body = 0;

    r.n_h_tail = args.output_height - r.n_h_head - r.n_h_body;
    if (r.n_h_tail < 0)
        r.n_h_tail = 0;
    r.n_w_tail = args.output_width - r.n_w_head - r.n_w_body;
    if (r.n_w_tail < 0)
        r.n_w_tail = 0;
    return r;
}

inline bool conv_has_padding(const ConvArgsType &args)
{
    return args.padding_h_head || args.padding_w_head || args.padding_h_tail || args.padding_w_tail;
}

inline bool conv_is_1x1(const ConvArgsType &args)
{
    return args.filter_height == 1 && args.filter_width == 1;
}

// Small convolutions stay on the original path. The limits come from P4
// measurements of W8A8, W8A16, and W16A16.
inline int64_t conv_width_minimum_macs(quant_type_t quant)
{
    if (quant == QUANT_TYPE_SYMM_8BIT) {
        return 512 * 1024;
    }
    if (quant == QUANT_TYPE_SYMM_W8A16) {
        return 112 * 1024;
    }
    return 160 * 1024;
}

// Width split after the height split returned one task. Each half must keep at
// least two columns so a width-1 slice does not leave the filter-tile loop.
inline bool can_split_conv_width(const ConvArgsType &a, quant_type_t quant)
{
    if (a.output_width < 4 || a.output_height < 1) {
        return false;
    }
    const int64_t macs = static_cast<int64_t>(a.output_height) * a.output_width * a.input_channel * a.output_channel *
        a.filter_height * a.filter_width;
    return macs >= conv_width_minimum_macs(quant);
}

// Output columns [begin, begin + count) as a narrower Conv, like the height
// split: input_width and the w padding describe only the columns this slice
// reads, while the precomputed row strides keep the original width.
inline void set_conv_width_interval(ConvArgsType &a, int begin, int count, int feat_bytes)
{
    const int first = begin * a.stride_x - a.padding_w_head;
    const int last = (begin + count - 1) * a.stride_x - a.padding_w_head + a.dilation_w * (a.filter_width - 1) + 1;
    const int input_begin = first > 0 ? first : 0;
    const int input_end = last < a.input_width ? last : a.input_width;
    a.input_element = conv_ptr_add(a.input_element, input_begin * a.input_channel, feat_bytes);
    a.padding_w_head = input_begin - first;
    a.padding_w_tail = last - input_end;
    a.input_width = input_end - input_begin;
    a.output_element = conv_ptr_add(a.output_element, begin * a.output_x_offset, feat_bytes);
    a.output_width = count;
}

template <typename Fn>
inline void conv_each_1x1_pad(const ConvArgsType &args, const ConvRegions &r, int feat_bytes, Fn &&fn)
{
    for (int y = 0; y < args.output_height; ++y) {
        const bool full_row = y < r.n_h_head || y >= r.n_h_head + r.n_h_body;
        void *out = conv_ptr_add(args.output_element, y * args.output_y_offset, feat_bytes);
        for (int x = 0; x < args.output_width; ++x) {
            if (full_row || x < r.n_w_head || x >= r.n_w_head + r.n_w_body) {
                fn(out);
            }
            out = conv_ptr_add(out, args.output_x_offset, feat_bytes);
        }
    }
}

inline void conv_1x1_body_view(const ConvArgsType &args,
                               const ConvRegions &r,
                               int feat_bytes,
                               void **input,
                               void **output,
                               int *height,
                               int *width)
{
    const int in_y0 = r.n_h_head * args.stride_y - args.padding_h_head;
    const int in_x0 = r.n_w_head * args.stride_x - args.padding_w_head;
    *input = conv_ptr_add(args.input_element, in_y0 * args.input_y_offset + in_x0 * args.input_channel, feat_bytes);
    *output = conv_ptr_add(
        args.output_element, r.n_h_head * args.output_y_offset + r.n_w_head * args.output_x_offset, feat_bytes);
    *height = r.n_h_body;
    *width = r.n_w_body;
}

// Modifications:
// 1. Tensor, Filter, Bias, Activation -> TensorBase pointer
// 2. move dilations from Filter into function's argument
// 3. activation_alpha is used for PReLU and leaky ReLU

/**
 * @brief Owner of the conv-operation args and the per-channel / debug memory they reference.
 *
 * Replaces the former get_conv_operation_args() free function. In the dual-core split case the
 * internal m_args holds two ConvArgsType entries that share the same tie_filter_channel_factor and
 * debug_value allocations (both copied from the first entry via push_back). Freeing those inside
 * the per-args shell therefore freed each block twice. This RAII class allocates them once and
 * releases them once in its destructor, after module_forward_dual_core() has rejoined both cores.
 */
class ConvOpArgs {
public:
    ConvOpArgs(TensorBase *output,
               TensorBase *input,
               std::vector<int> &padding,
               TensorBase *filter,
               const std::vector<int> &strides,
               const std::vector<int> &dilations,
               const int group,
               TensorBase *bias,
               const activation_type_t activate,
               TensorBase *activation_alpha,
               const runtime_mode_t runtime_mode,
               quant_type_t quant,
               bool malloc_debug_memory = false)
    {
        const int act_bytes = conv_feature_bytes(quant);
        const int filter_bytes = conv_filter_bytes(quant);
        ConvArgsType args = {};
        args.input_element = input->get_element_ptr();
        args.output_element = output->get_element_ptr();
        args.filter_element = filter->get_element_ptr();

        if (input->shape.size() == 3) {
            args.input_height = 1;
            args.input_width = input->shape[1];
            args.input_channel = input->shape[2];
            args.dilation_h = 1;
            args.dilation_w = dilations[0];
            args.stride_y = 1;
            args.stride_x = strides[0];

            args.output_height = 1;
            args.output_width = output->shape[1];
            args.output_channel = output->shape[2];

            args.filter_height = 1;
            args.filter_width = filter->shape[0];
            if (group == 1) {
                // conv
                args.filter_y_offset = 0;
                args.filter_c = filter->shape[1]; // dw: filter->shape[2]. conv: filter->shape[1].
            } else {
                // depthwise
                args.filter_y_offset = 16;
                args.filter_c = filter->shape[2]; // dw: filter->shape[2]. conv: filter->shape[1].
            }
            /* It's for c. We need to confirm whether the following definitions conform to the C logical implementation.
             */
            args.filter_y_offset_c = args.filter_width * filter->shape[1];

            args.padding_h_head = 0;
            args.padding_h_tail = 0;
            args.padding_w_head = padding[0];
            args.padding_w_tail = padding[1];
        } else if (input->shape.size() == 4) {
            args.input_height = input->shape[1];
            args.input_width = input->shape[2];
            args.input_channel = input->shape[3];
            args.dilation_h = dilations[0];
            args.dilation_w = dilations[1];
            args.stride_y = strides[0];
            args.stride_x = strides[1];

            args.output_height = output->shape[1];
            args.output_width = output->shape[2];
            args.output_channel = output->shape[3];

            args.filter_height = filter->shape[0];
            args.filter_width = filter->shape[1];
            if (group == 1) {
                // conv
                args.filter_y_offset = 0;
                args.filter_c = filter->shape[2]; // dw: filter->shape[3]. conv: filter->shape[2].
            } else {
                // depthwise
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
                args.filter_y_offset = 16;
#else
                args.filter_y_offset = 0;
#endif
                args.filter_c = filter->shape[3]; // dw: filter->shape[3]. conv: filter->shape[2].
            }
            /* It's for c. We need to confirm whether the following definitions conform to the C logical implementation.
             */
            args.filter_y_offset_c = args.filter_width * filter->shape[2];

            args.padding_h_head = padding[0];
            args.padding_h_tail = padding[1];
            args.padding_w_head = padding[2];
            args.padding_w_tail = padding[3];
        } else {
            ESP_LOGE(__FUNCTION__, "Do not support input shape.");
            return;
        }

        args.filter_n_offset = 0;
        args.filter_n_offset_c = args.filter_y_offset_c * args.filter_height;

        args.input_stride_y_offset = args.input_width * args.input_channel * args.stride_y;
        args.input_stride_x_offset = args.input_channel * args.stride_x;
        args.input_dilation_y_offset = args.input_width * args.input_channel * args.dilation_h;
        args.input_dilation_x_offset = args.input_channel * args.dilation_w;

        args.output_y_offset = args.output_width * args.output_channel;
        args.output_x_offset = args.output_channel;

        args.input_y_offset = args.input_width * args.input_channel;
        args.input_channel_with_padding = args.input_channel;
        args.auto_split = true;
        // printf("input: %d, %d, %d, output: %d, %d, %d\n", input->shape[1], args.input_width, args.input_channel,
        // output->shape[1], args.output_width, args.output_channel);

        if (filter->exponent.is_per_channel()) {
#if CONFIG_PIE_V2_BOOST
            // per-channel quantization
            args.mac_shift = INT_MIN;
            args.tie_filter_channel_factor = tool::calloc_aligned(
                dl::tool::get_aligned_size(args.output_channel * act_bytes), 1, MALLOC_CAP_DEFAULT);
            if (act_bytes == 1) {
                int8_t *p = (int8_t *)args.tie_filter_channel_factor;
                for (int i = 0; i < args.output_channel; i++) {
                    p[i] = (int8_t)(output->exponent - filter->exponent.get(i) - input->exponent);
                }
            } else {
                int16_t *p = (int16_t *)args.tie_filter_channel_factor;
                for (int i = 0; i < args.output_channel; i++) {
                    p[i] = (int16_t)(output->exponent - filter->exponent.get(i) - input->exponent);
                }
            }
#else
            // Don't support per-channel quantization
            args.mac_shift = INT_MIN;
            args.tie_filter_channel_factor = NULL;
#endif
        } else {
            // per-tensor quantization
            args.mac_shift = output->exponent - filter->exponent - input->exponent;
        }

        args.bias_element = bias ? bias->get_element_ptr() : NULL; // TODO: auto_split
        args.activation_type = activate;

        switch (args.activation_type) {
        case ReLU:
            args.activation_alpha = 0;
            args.activation_shift = 0;
            args.activation_alpha_ptr = NULL;
            break;
        case LeakyReLU:
            // ESP_LOGE(__FUNCTION__, "Do not support Leaky ReLU");
            //     args.activation_alpha = activation_alpha->get_element_ptr()[0];
            //     args.activation_shift = -activation_alpha->exponent;
            //     args.activation_alpha_ptr = NULL;
            break;
        case PReLU:
            // ESP_LOGE(__FUNCTION__, "Do not support PReLU");
            // args.activation_alpha_ptr = activation_alpha->get_element_ptr(); //TODO: auto_split
            // args.activation_shift = -activation_alpha->exponent;
            break;
        default:
            args.activation_alpha_ptr = NULL;
            args.activation_shift = -1;
            break;
        }

        // for ISA
        args.c_rs1_1 = (args.input_channel >> 1) - 1;
        args.c_rs2_1 = (args.input_channel >> 2) - 1;
        int u = 16 / act_bytes;
        args.n_div_x = args.output_channel / u; // TODO: auto_split
        args.c_div_x_1 = args.input_channel / u - 1;

        args.c_remainder = args.input_channel % u * act_bytes;
        args.n_remainder = args.output_channel % u;

        args.xtensa_dilation_x_offset = (args.dilation_w * args.input_channel - args.input_channel) * act_bytes;
        args.xtensa_dilation_y_offset_stable = args.dilation_h * args.input_channel * args.input_width;
        args.xtensa_dilation_y_offset = (args.xtensa_dilation_y_offset_stable - args.input_channel -
                                         (args.filter_width - 1) * args.dilation_w * args.input_channel) *
            act_bytes;

        args.filter_y_offset_unaligned = 0;
        args.filter_n_offset_unaligned = 0;
        args.filter_element_unaligned = args.n_remainder
            ? conv_ptr_add(args.filter_element,
                           args.n_div_x * args.filter_height * args.filter_width * args.filter_c * u,
                           filter_bytes)
            : args.filter_element;

        if (group > 1) {
            args.filter_w_rs1_1 = (args.filter_width >> 1) - 1;
            args.tie_depth2d_dilation_x_offset = args.dilation_w * args.input_channel * act_bytes;
            args.tie_depth2d_dilation_y_offset_stable = args.dilation_h * args.input_channel * args.input_width;
            args.tie_depth2d_dilation_y_offset = (args.tie_depth2d_dilation_y_offset_stable -
                                                  (args.filter_width - 1) * args.dilation_w * args.input_channel) *
                act_bytes;

            args.tie_depth2d_next_hwx1 = (args.filter_width - 1) * args.dilation_w +
                (args.filter_height - 1) * args.dilation_h * args.input_width;
            args.tie_depth2d_next_hwx1 = 16 - args.tie_depth2d_next_hwx1 * args.input_channel * act_bytes;
        }
        if (malloc_debug_memory) {
            args.debug_value = tool::calloc_aligned(16, 1, MALLOC_CAP_DEFAULT);
        }
        m_args.assign(1, args);
        if (args.input_height > 4 * args.dilation_h * args.filter_height) {
            if (runtime_mode == RUNTIME_MODE_MULTI_CORE ||
                (runtime_mode == RUNTIME_MODE_AUTO && args.input_height >= 100 && args.input_width >= 50)) {
                m_args.push_back(args);

                // Divide this convolution into two tasks by splitting the input height.
                // up
                int dilation_filter_height = args.dilation_h * (args.filter_height - 1) + 1;
                int half_step =
                    (args.padding_h_head + args.padding_h_tail + args.input_height - dilation_filter_height) /
                    args.stride_y / 2;
                m_args[0].input_height = dilation_filter_height - args.padding_h_head + half_step * args.stride_y;
                m_args[0].padding_h_tail = 0;
                m_args[0].output_height = half_step + 1;
                // bottom
                m_args[1].padding_h_head = 0;
                m_args[1].input_height =
                    dilation_filter_height - args.stride_y + args.input_height - m_args[0].input_height;
                m_args[1].input_element =
                    conv_ptr_add(m_args[1].input_element,
                                 (args.input_height - m_args[1].input_height) * args.input_width * args.input_channel,
                                 act_bytes);
                m_args[1].output_height = args.output_height - m_args[0].output_height;
                m_args[1].output_element = conv_ptr_add(m_args[1].output_element,
                                                        (args.output_height - m_args[1].output_height) *
                                                            args.output_width * args.output_channel,
                                                        act_bytes);
            }
        }
        // Maps too short for the height split are divided by output columns instead.
        if (m_args.size() == 1 && runtime_mode == RUNTIME_MODE_MULTI_CORE && group == 1 && input->shape[0] == 1 &&
            can_split_conv_width(args, quant)) {
            const int left = (args.output_width + 1) / 2;
            m_args.push_back(args);
            set_conv_width_interval(m_args[0], 0, left, act_bytes);
            set_conv_width_interval(m_args[1], left, args.output_width - left, act_bytes);
        }
    }

    ~ConvOpArgs()
    {
        // tie_filter_channel_factor and debug_value are allocated once and shared by both split
        // args (the second entry is push_back'd from the first, copying the pointers). Free the
        // single shared allocation once here. m_args is empty only on the unsupported-shape
        // early-return path.
        if (m_args.empty())
            return;
        // tie_filter_channel_factor is allocated only on the per-channel path (mac_shift == INT_MIN).
        if (m_args[0].mac_shift == INT_MIN)
            heap_caps_free(m_args[0].tie_filter_channel_factor);
        // debug_value defaults to nullptr (see ConvArgsType); free is a no-op when not allocated.
        heap_caps_free(m_args[0].debug_value);
    }

    size_t size() const { return m_args.size(); }
    ConvArgsType &get_args(int i) { return m_args[i]; }

    // This class owns heap allocations (tie_filter_channel_factor / debug_value); copying would
    // share the pointers and double-free on destruction.
    ConvOpArgs(const ConvOpArgs &) = delete;
    ConvOpArgs &operator=(const ConvOpArgs &) = delete;

private:
    std::vector<ConvArgsType> m_args;
};

} // namespace base
} // namespace dl
