#pragma once

#include "dl_define.hpp"
#include <cstddef>
#include <cstdint>

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

inline ConvRegions conv_regions(const ConvArgsType &args)
{
    ConvRegions r;
    r.n_h_head = (args.padding_h_head + args.stride_y - 1) / args.stride_y;
    if (r.n_h_head > args.output_height)
        r.n_h_head = args.output_height;
    r.n_w_head = (args.padding_w_head + args.stride_x - 1) / args.stride_x;
    if (r.n_w_head > args.output_width)
        r.n_w_head = args.output_width;

    r.n_h_body =
        ((args.input_height + args.padding_h_head - args.dilation_h * (args.filter_height - 1) - 1) / args.stride_y +
         1) -
        r.n_h_head;
    if (r.n_h_body < 0)
        r.n_h_body = 0;
    r.n_w_body =
        ((args.input_width + args.padding_w_head - args.dilation_w * (args.filter_width - 1) - 1) / args.stride_x + 1) -
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

template <typename Fn>
inline void conv_each_1x1_pad(const ConvArgsType &args, const ConvRegions &r, int feat_bytes, Fn &&fn)
{
    void *out = args.output_element;
    for (int y = 0; y < r.n_h_head; ++y) {
        for (int x = 0; x < args.output_width; ++x) {
            fn(out);
            out = conv_ptr_add(out, args.output_x_offset, feat_bytes);
        }
    }
    for (int y = 0; y < r.n_h_body; ++y) {
        for (int x = 0; x < r.n_w_head; ++x) {
            fn(out);
            out = conv_ptr_add(out, args.output_x_offset, feat_bytes);
        }
        out = conv_ptr_add(out, r.n_w_body * args.output_x_offset, feat_bytes);
        for (int x = 0; x < r.n_w_tail; ++x) {
            fn(out);
            out = conv_ptr_add(out, args.output_x_offset, feat_bytes);
        }
    }
    for (int y = 0; y < r.n_h_tail; ++y) {
        for (int x = 0; x < args.output_width; ++x) {
            fn(out);
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

} // namespace base
} // namespace dl
