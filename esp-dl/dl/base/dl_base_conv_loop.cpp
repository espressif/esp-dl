#include "dl_base_conv2d.hpp"
#include "dl_base_conv_args.hpp"

#include "dl_base_isa.hpp"
#include "dl_compile_config.h"
#include "dl_tool.hpp"
#include "esp_log.h"

#include <algorithm>
#include <cstdint>

namespace dl {
namespace base {

#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
static void conv_loop_pie_body(
    ConvArgsType &args, conv_pie_fn_t body, int feat_bytes, void *input_ptr, void *output_ptr, int height, int width)
{
    for (int output_y = 0; output_y < height; output_y++) {
        void *input_syx = input_ptr;
        void *output_yx = output_ptr;
        for (int output_x = 0; output_x < width; output_x++) {
            body(output_yx, input_syx, (void *)&args);
            input_syx = conv_ptr_add(input_syx, args.input_stride_x_offset, feat_bytes);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
        }
        input_ptr = conv_ptr_add(input_ptr, args.input_stride_y_offset, feat_bytes);
        output_ptr = conv_ptr_add(output_ptr, args.output_y_offset, feat_bytes);
    }
}

static bool conv_loop_pie_tiled(ConvArgsType &args,
                                conv_pie_fn_t body,
                                int feat_bytes,
                                int filt_bytes,
                                void *input_ptr,
                                void *output_ptr,
                                int height,
                                int width)
{
#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32S31
    constexpr int filter_bytes = 16 * 1024;
#else
    constexpr int filter_bytes = 32 * 1024;
#endif
    constexpr int spatial_tile = 32;

    // height/width/pointers are the body rectangle: the full map when there is
    // no pad, or conv_1x1_body_view() after 1x1 pad fill. Stride is already in
    // input_stride_*_offset; do not require stride==1 or reject padding.
    if (!body || !conv_is_1x1(args) || feat_bytes <= 0 || filt_bytes <= 0 || args.input_channel <= 0 || width <= 0 ||
        height <= 0) {
        return false;
    }
    const int lanes = 16 / feat_bytes;
    if (lanes <= 0 || args.input_channel < lanes || args.input_channel % lanes || args.output_channel % lanes ||
        args.c_remainder || args.n_remainder) {
        return false;
    }
    if ((reinterpret_cast<uintptr_t>(input_ptr) & 15) || (reinterpret_cast<uintptr_t>(output_ptr) & 15)) {
        return false;
    }
    if (args.activation_type != Linear && args.activation_type != ReLU) {
        return false;
    }
#if CONFIG_IDF_TARGET_ESP32S3
    if (args.mac_shift < 0) {
        return false;
    }
#endif

    const int channel_tile = (filter_bytes / filt_bytes / args.input_channel) / lanes * lanes;
    const int pixels = height * width;
    if (channel_tile < lanes || args.output_channel < 2 * channel_tile || pixels < 2) {
        return false;
    }

    const int bias_bytes = (feat_bytes == 1) ? (int)sizeof(int32_t) : (int)sizeof(int64_t);
    const bool per_channel = args.mac_shift == INT_MIN;
    for (int begin = 0; begin < pixels; begin += spatial_tile) {
        const int end = std::min(pixels, begin + spatial_tile);
        for (int channel = 0; channel < args.output_channel; channel += channel_tile) {
            ConvArgsType tile = args;
            tile.output_channel = std::min(channel_tile, args.output_channel - channel);
            tile.n_div_x = tile.output_channel / lanes;
            tile.filter_element = conv_ptr_add(args.filter_element, channel * args.input_channel, filt_bytes);
            if (args.bias_element) {
                tile.bias_element = conv_ptr_add(args.bias_element, channel, bias_bytes);
            }
            if (per_channel && args.tie_filter_channel_factor) {
                tile.tie_filter_channel_factor = conv_ptr_add(args.tie_filter_channel_factor, channel, feat_bytes);
            }
            for (int pixel = begin; pixel < end; ++pixel) {
                const int y = pixel / width;
                const int x = pixel % width;
                void *in = conv_ptr_add(
                    input_ptr, y * args.input_stride_y_offset + x * args.input_stride_x_offset, feat_bytes);
                void *out =
                    conv_ptr_add(output_ptr, y * args.output_y_offset + x * args.output_x_offset + channel, feat_bytes);
                body(out, in, (void *)&tile);
            }
        }
    }
    return true;
}
#endif

#if !(CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST)
static void conv_loop_c_body(ConvArgsType &args,
                             conv_c_mac_fn_t mac_body,
                             conv_c_tail_fn_t tail,
                             void *buffer,
                             int feat_bytes,
                             void *input_ptr,
                             void *output_ptr,
                             int height,
                             int width)
{
    for (int output_y = 0; output_y < height; output_y++) {
        void *input_syx = input_ptr;
        void *output_yx = output_ptr;
        for (int output_x = 0; output_x < width; output_x++) {
            mac_body(buffer, input_syx, args);
            tail(output_yx, buffer, args);
            input_syx = conv_ptr_add(input_syx, args.input_stride_x_offset, feat_bytes);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
        }
        input_ptr = conv_ptr_add(input_ptr, args.input_stride_y_offset, feat_bytes);
        output_ptr = conv_ptr_add(output_ptr, args.output_y_offset, feat_bytes);
    }
}
#endif

#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
static void conv_loop_pie_grid(
    ConvArgsType &args, conv_pie_fn_t body, conv_pie_fn_t border, int feat_bytes, int filt_bytes)
{
    void *input_ptr = args.input_element;
    void *output_ptr = args.output_element;
    ConvRegions r = conv_regions(args);

    int filter_h = args.filter_height;
    int filter_w = args.filter_width;
    const void *filter_ptr = args.filter_element;
    void *input_y_real;
    void *input_x_real;
    const void *filter_ptr_y;
    void *output_yx = output_ptr;
    const void *filter_ptr_unaligned = args.filter_element_unaligned;
    const void *filter_ptr_y_unaligned;
    int unaligned_filter_c_n_offset = args.filter_c * filt_bytes;
    int filter_c_n_offset = args.n_div_x ? args.filter_c * (16 / feat_bytes) * filt_bytes : unaligned_filter_c_n_offset;
    int filter_c_n_ptr_offset = filter_c_n_offset / filt_bytes;

    for (size_t output_y = 0; output_y < (size_t)r.n_h_head; output_y++) {
        args.filter_height =
            filter_h - ((args.padding_h_head - (int)output_y * args.stride_y) + args.dilation_h - 1) / args.dilation_h;
        int filter_height_excess = filter_h -
            (args.input_height + (args.padding_h_head - (int)output_y * args.stride_y) + args.dilation_h - 1) /
                args.dilation_h;
        if (filter_height_excess > 0) {
            args.filter_height -= filter_height_excess;
        } else {
            filter_height_excess = 0;
        }

        input_y_real = conv_ptr_add(input_ptr,
                                    args.input_y_offset *
                                        ((args.stride_y * (int)output_y +
                                          (filter_h - args.filter_height - filter_height_excess) * args.dilation_h) -
                                         args.padding_h_head),
                                    feat_bytes);
        filter_ptr_y =
            conv_ptr_add(filter_ptr,
                         (filter_h - args.filter_height - filter_height_excess) * filter_w * filter_c_n_ptr_offset,
                         filt_bytes);
        filter_ptr_y_unaligned =
            conv_ptr_add(filter_ptr_unaligned,
                         (filter_h - args.filter_height - filter_height_excess) * filter_w * args.filter_c,
                         filt_bytes);
        args.filter_n_offset = (filter_w * (filter_h - args.filter_height)) * filter_c_n_offset;
        args.filter_n_offset_unaligned = (filter_w * (filter_h - args.filter_height)) * unaligned_filter_c_n_offset;

        for (size_t output_x = 0; output_x < (size_t)r.n_w_head; output_x++) {
            args.filter_width = filter_w -
                ((args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) / args.dilation_w;
            int filter_width_excess = filter_w -
                (args.input_width + (args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) /
                    args.dilation_w;
            if (filter_width_excess > 0) {
                args.filter_width -= filter_width_excess;
            } else {
                filter_width_excess = 0;
            }

            input_x_real = conv_ptr_add(input_y_real,
                                        args.input_channel *
                                            ((args.stride_x * (int)output_x +
                                              (filter_w - args.filter_width - filter_width_excess) * args.dilation_w) -
                                             args.padding_w_head),
                                        feat_bytes);
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            args.filter_y_offset_unaligned = (filter_w - args.filter_width) * unaligned_filter_c_n_offset;
            args.xtensa_dilation_y_offset =
                (args.xtensa_dilation_y_offset_stable - args.input_channel -
                 (args.filter_width - 1) * args.dilation_w * args.input_channel_with_padding) *
                feat_bytes;
            args.filter_element = conv_ptr_add(
                filter_ptr_y, (filter_w - args.filter_width - filter_width_excess) * filter_c_n_ptr_offset, filt_bytes);
            args.filter_element_unaligned =
                conv_ptr_add(filter_ptr_y_unaligned,
                             (filter_w - args.filter_width - filter_width_excess) * args.filter_c,
                             filt_bytes);

            border(output_yx, input_x_real, (void *)&args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
        }

        input_x_real = conv_ptr_add(
            input_y_real, args.input_channel * (args.stride_x * r.n_w_head - args.padding_w_head), feat_bytes);
        args.filter_width = filter_w;
        args.xtensa_dilation_y_offset = (args.xtensa_dilation_y_offset_stable - args.input_channel -
                                         (args.filter_width - 1) * args.dilation_w * args.input_channel_with_padding) *
            feat_bytes;
        args.filter_y_offset = 0;
        args.filter_y_offset_unaligned = 0;
        args.filter_element = filter_ptr_y;
        args.filter_element_unaligned = filter_ptr_y_unaligned;
        for (size_t output_x = 0; output_x < (size_t)r.n_w_body; output_x++) {
            border(output_yx, input_x_real, (void *)&args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }

        for (size_t output_x = 0; output_x < (size_t)r.n_w_tail; output_x++) {
            args.filter_width = (args.padding_w_head + args.input_width -
                                 (r.n_w_head + r.n_w_body + (int)output_x) * args.stride_x + args.dilation_w - 1) /
                args.dilation_w;
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            args.filter_y_offset_unaligned = (filter_w - args.filter_width) * unaligned_filter_c_n_offset;
            args.xtensa_dilation_y_offset =
                (args.xtensa_dilation_y_offset_stable - args.input_channel -
                 (args.filter_width - 1) * args.dilation_w * args.input_channel_with_padding) *
                feat_bytes;
            border(output_yx, input_x_real, (void *)&args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }
    }

    args.filter_height = filter_h;
    input_y_real =
        conv_ptr_add(input_ptr, args.input_y_offset * ((args.stride_y * r.n_h_head) - args.padding_h_head), feat_bytes);
    filter_ptr_y = filter_ptr;
    filter_ptr_y_unaligned = filter_ptr_unaligned;
    args.filter_n_offset = 0;
    args.filter_n_offset_unaligned = 0;

    for (size_t output_y = 0; output_y < (size_t)r.n_h_body; output_y++) {
        for (size_t output_x = 0; output_x < (size_t)r.n_w_head; output_x++) {
            args.filter_width = filter_w -
                ((args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) / args.dilation_w;
            int filter_width_excess = filter_w -
                (args.input_width + (args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) /
                    args.dilation_w;
            if (filter_width_excess > 0) {
                args.filter_width -= filter_width_excess;
            } else {
                filter_width_excess = 0;
            }

            input_x_real = conv_ptr_add(input_y_real,
                                        args.input_channel *
                                            ((args.stride_x * (int)output_x +
                                              (filter_w - args.filter_width - filter_width_excess) * args.dilation_w) -
                                             args.padding_w_head),
                                        feat_bytes);
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            args.filter_y_offset_unaligned = (filter_w - args.filter_width) * unaligned_filter_c_n_offset;
            args.xtensa_dilation_y_offset =
                (args.xtensa_dilation_y_offset_stable - args.input_channel -
                 (args.filter_width - 1) * args.dilation_w * args.input_channel_with_padding) *
                feat_bytes;
            args.filter_element = conv_ptr_add(
                filter_ptr_y, (filter_w - args.filter_width - filter_width_excess) * filter_c_n_ptr_offset, filt_bytes);
            args.filter_element_unaligned =
                conv_ptr_add(filter_ptr_y_unaligned,
                             (filter_w - args.filter_width - filter_width_excess) * args.filter_c,
                             filt_bytes);
            border(output_yx, input_x_real, (void *)&args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
        }

        input_x_real = conv_ptr_add(
            input_y_real, args.input_channel * (args.stride_x * r.n_w_head - args.padding_w_head), feat_bytes);
        args.filter_width = filter_w;
        args.xtensa_dilation_y_offset = (args.xtensa_dilation_y_offset_stable - args.input_channel -
                                         (args.filter_width - 1) * args.dilation_w * args.input_channel_with_padding) *
            feat_bytes;
        args.filter_y_offset = 0;
        args.filter_y_offset_unaligned = 0;
        args.filter_element = filter_ptr_y;
        args.filter_element_unaligned = filter_ptr_y_unaligned;
        for (size_t output_x = 0; output_x < (size_t)r.n_w_body; output_x++) {
            body(output_yx, input_x_real, (void *)&args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }

        for (size_t output_x = 0; output_x < (size_t)r.n_w_tail; output_x++) {
            args.filter_width = (args.padding_w_head + args.input_width -
                                 (r.n_w_head + r.n_w_body + (int)output_x) * args.stride_x + args.dilation_w - 1) /
                args.dilation_w;
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            args.filter_y_offset_unaligned = (filter_w - args.filter_width) * unaligned_filter_c_n_offset;
            args.xtensa_dilation_y_offset =
                (args.xtensa_dilation_y_offset_stable - args.input_channel -
                 (args.filter_width - 1) * args.dilation_w * args.input_channel_with_padding) *
                feat_bytes;
            border(output_yx, input_x_real, (void *)&args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }
        input_y_real = conv_ptr_add(input_y_real, args.input_stride_y_offset, feat_bytes);
    }

    for (size_t output_y = 0; output_y < (size_t)r.n_h_tail; output_y++) {
        args.filter_height = (args.padding_h_head + args.input_height -
                              (r.n_h_head + r.n_h_body + (int)output_y) * args.stride_y + args.dilation_h - 1) /
            args.dilation_h;
        args.filter_n_offset = (filter_w * (filter_h - args.filter_height)) * filter_c_n_offset;
        args.filter_n_offset_unaligned = (filter_w * (filter_h - args.filter_height)) * unaligned_filter_c_n_offset;

        for (size_t output_x = 0; output_x < (size_t)r.n_w_head; output_x++) {
            args.filter_width = filter_w -
                ((args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) / args.dilation_w;
            int filter_width_excess = filter_w -
                (args.input_width + (args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) /
                    args.dilation_w;
            if (filter_width_excess > 0) {
                args.filter_width -= filter_width_excess;
            } else {
                filter_width_excess = 0;
            }

            input_x_real = conv_ptr_add(input_y_real,
                                        args.input_channel *
                                            ((args.stride_x * (int)output_x +
                                              (filter_w - args.filter_width - filter_width_excess) * args.dilation_w) -
                                             args.padding_w_head),
                                        feat_bytes);
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            args.filter_y_offset_unaligned = (filter_w - args.filter_width) * unaligned_filter_c_n_offset;
            args.xtensa_dilation_y_offset =
                (args.xtensa_dilation_y_offset_stable - args.input_channel -
                 (args.filter_width - 1) * args.dilation_w * args.input_channel_with_padding) *
                feat_bytes;
            args.filter_element = conv_ptr_add(
                filter_ptr_y, (filter_w - args.filter_width - filter_width_excess) * filter_c_n_ptr_offset, filt_bytes);
            args.filter_element_unaligned =
                conv_ptr_add(filter_ptr_y_unaligned,
                             (filter_w - args.filter_width - filter_width_excess) * args.filter_c,
                             filt_bytes);
            border(output_yx, input_x_real, (void *)&args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
        }

        input_x_real = conv_ptr_add(
            input_y_real, args.input_channel * (args.stride_x * r.n_w_head - args.padding_w_head), feat_bytes);
        args.filter_width = filter_w;
        args.filter_y_offset = 0;
        args.filter_y_offset_unaligned = 0;
        args.xtensa_dilation_y_offset = (args.xtensa_dilation_y_offset_stable - args.input_channel -
                                         (args.filter_width - 1) * args.dilation_w * args.input_channel_with_padding) *
            feat_bytes;
        args.filter_element = filter_ptr_y;
        args.filter_element_unaligned = filter_ptr_y_unaligned;
        for (size_t output_x = 0; output_x < (size_t)r.n_w_body; output_x++) {
            border(output_yx, input_x_real, (void *)&args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }

        for (size_t output_x = 0; output_x < (size_t)r.n_w_tail; output_x++) {
            args.filter_width = (args.padding_w_head + args.input_width -
                                 (r.n_w_head + r.n_w_body + (int)output_x) * args.stride_x + args.dilation_w - 1) /
                args.dilation_w;
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            args.filter_y_offset_unaligned = (filter_w - args.filter_width) * unaligned_filter_c_n_offset;
            args.xtensa_dilation_y_offset =
                (args.xtensa_dilation_y_offset_stable - args.input_channel -
                 (args.filter_width - 1) * args.dilation_w * args.input_channel_with_padding) *
                feat_bytes;
            border(output_yx, input_x_real, (void *)&args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }
        input_y_real = conv_ptr_add(input_y_real, args.input_stride_y_offset, feat_bytes);
    }
}

static void conv_loop_pie(ConvArgsType &args, conv_pie_fn_t body, conv_pie_fn_t border, int feat_bytes, int filt_bytes)
{
    if (conv_is_1x1(args) || !conv_has_padding(args)) {
        void *input_ptr = args.input_element;
        void *output_ptr = args.output_element;
        int height = args.output_height;
        int width = args.output_width;
        if (conv_is_1x1(args) && conv_has_padding(args)) {
            ConvRegions r = conv_regions(args);
            void *zeros = tool::calloc_aligned(
                tool::get_aligned_size((size_t)args.input_channel * (size_t)feat_bytes), 1, MALLOC_CAP_DEFAULT);
            conv_each_1x1_pad(args, r, feat_bytes, [&](void *out) { body(out, zeros, (void *)&args); });
            heap_caps_free(zeros);
            conv_1x1_body_view(args, r, feat_bytes, &input_ptr, &output_ptr, &height, &width);
        }
        if (!conv_loop_pie_tiled(args, body, feat_bytes, filt_bytes, input_ptr, output_ptr, height, width)) {
            conv_loop_pie_body(args, body, feat_bytes, input_ptr, output_ptr, height, width);
        }
        return;
    }
    conv_loop_pie_grid(args, body, border, feat_bytes, filt_bytes);
}
#endif

#if !(CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST)
static void conv_loop_c_grid(ConvArgsType &args,
                             conv_c_mac_fn_t mac_border,
                             conv_c_mac_fn_t mac_body,
                             conv_c_tail_fn_t tail,
                             void *buffer,
                             int feat_bytes,
                             int filt_bytes)
{
    void *input_ptr = args.input_element;
    void *output_ptr = args.output_element;
    ConvRegions r = conv_regions(args);

    int filter_h = args.filter_height;
    int filter_w = args.filter_width;
    const void *filter_ptr = args.filter_element;
    void *input_y_real;
    void *input_x_real;
    const void *filter_ptr_y;
    void *output_yx = output_ptr;
    int filter_c_n_offset = args.input_channel;
    int filter_c_n_ptr_offset = filter_c_n_offset;

    for (size_t output_y = 0; output_y < (size_t)r.n_h_head; output_y++) {
        args.filter_height =
            filter_h - ((args.padding_h_head - (int)output_y * args.stride_y) + args.dilation_h - 1) / args.dilation_h;
        int filter_height_excess = filter_h -
            (args.input_height + (args.padding_h_head - (int)output_y * args.stride_y) + args.dilation_h - 1) /
                args.dilation_h;
        if (filter_height_excess > 0) {
            args.filter_height -= filter_height_excess;
        } else {
            filter_height_excess = 0;
        }

        input_y_real = conv_ptr_add(input_ptr,
                                    args.input_y_offset *
                                        ((args.stride_y * (int)output_y +
                                          (filter_h - args.filter_height - filter_height_excess) * args.dilation_h) -
                                         args.padding_h_head),
                                    feat_bytes);
        filter_ptr_y =
            conv_ptr_add(filter_ptr,
                         (filter_h - args.filter_height - filter_height_excess) * filter_w * filter_c_n_ptr_offset,
                         filt_bytes);
        args.filter_n_offset = (filter_w * (filter_h - args.filter_height)) * filter_c_n_offset;

        for (size_t output_x = 0; output_x < (size_t)r.n_w_head; output_x++) {
            args.filter_width = filter_w -
                ((args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) / args.dilation_w;
            int filter_width_excess = filter_w -
                (args.input_width + (args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) /
                    args.dilation_w;
            if (filter_width_excess > 0) {
                args.filter_width -= filter_width_excess;
            } else {
                filter_width_excess = 0;
            }

            input_x_real = conv_ptr_add(input_y_real,
                                        args.input_channel *
                                            ((args.stride_x * (int)output_x +
                                              (filter_w - args.filter_width - filter_width_excess) * args.dilation_w) -
                                             args.padding_w_head),
                                        feat_bytes);
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            args.filter_element = conv_ptr_add(
                filter_ptr_y, (filter_w - args.filter_width - filter_width_excess) * filter_c_n_ptr_offset, filt_bytes);
            mac_border(buffer, input_x_real, args);
            tail(output_yx, buffer, args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
        }

        input_x_real = conv_ptr_add(
            input_y_real, args.input_channel * (args.stride_x * r.n_w_head - args.padding_w_head), feat_bytes);
        args.filter_width = filter_w;
        args.filter_y_offset = 0;
        args.filter_element = filter_ptr_y;
        for (size_t output_x = 0; output_x < (size_t)r.n_w_body; output_x++) {
            mac_border(buffer, input_x_real, args);
            tail(output_yx, buffer, args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }

        for (size_t output_x = 0; output_x < (size_t)r.n_w_tail; output_x++) {
            args.filter_width = (args.padding_w_head + args.input_width -
                                 (r.n_w_head + r.n_w_body + (int)output_x) * args.stride_x + args.dilation_w - 1) /
                args.dilation_w;
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            mac_border(buffer, input_x_real, args);
            tail(output_yx, buffer, args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }
    }

    args.filter_height = filter_h;
    input_y_real =
        conv_ptr_add(input_ptr, args.input_y_offset * ((args.stride_y * r.n_h_head) - args.padding_h_head), feat_bytes);
    filter_ptr_y = filter_ptr;
    args.filter_n_offset = 0;

    for (size_t output_y = 0; output_y < (size_t)r.n_h_body; output_y++) {
        for (size_t output_x = 0; output_x < (size_t)r.n_w_head; output_x++) {
            args.filter_width = filter_w -
                ((args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) / args.dilation_w;
            int filter_width_excess = filter_w -
                (args.input_width + (args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) /
                    args.dilation_w;
            if (filter_width_excess > 0) {
                args.filter_width -= filter_width_excess;
            } else {
                filter_width_excess = 0;
            }

            input_x_real = conv_ptr_add(input_y_real,
                                        args.input_channel *
                                            ((args.stride_x * (int)output_x +
                                              (filter_w - args.filter_width - filter_width_excess) * args.dilation_w) -
                                             args.padding_w_head),
                                        feat_bytes);
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            args.filter_element = conv_ptr_add(
                filter_ptr_y, (filter_w - args.filter_width - filter_width_excess) * filter_c_n_ptr_offset, filt_bytes);
            mac_border(buffer, input_x_real, args);
            tail(output_yx, buffer, args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
        }

        input_x_real = conv_ptr_add(
            input_y_real, args.input_channel * (args.stride_x * r.n_w_head - args.padding_w_head), feat_bytes);
        args.filter_width = filter_w;
        args.filter_y_offset = 0;
        args.filter_element = filter_ptr_y;
        for (size_t output_x = 0; output_x < (size_t)r.n_w_body; output_x++) {
            mac_body(buffer, input_x_real, args);
            tail(output_yx, buffer, args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }

        for (size_t output_x = 0; output_x < (size_t)r.n_w_tail; output_x++) {
            args.filter_width = (args.padding_w_head + args.input_width -
                                 (r.n_w_head + r.n_w_body + (int)output_x) * args.stride_x + args.dilation_w - 1) /
                args.dilation_w;
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            mac_border(buffer, input_x_real, args);
            tail(output_yx, buffer, args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }
        input_y_real = conv_ptr_add(input_y_real, args.input_stride_y_offset, feat_bytes);
    }

    for (size_t output_y = 0; output_y < (size_t)r.n_h_tail; output_y++) {
        args.filter_height = (args.padding_h_head + args.input_height -
                              (r.n_h_head + r.n_h_body + (int)output_y) * args.stride_y + args.dilation_h - 1) /
            args.dilation_h;
        args.filter_n_offset = (filter_w * (filter_h - args.filter_height)) * filter_c_n_offset;

        for (size_t output_x = 0; output_x < (size_t)r.n_w_head; output_x++) {
            args.filter_width = filter_w -
                ((args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) / args.dilation_w;
            int filter_width_excess = filter_w -
                (args.input_width + (args.padding_w_head - (int)output_x * args.stride_x) + args.dilation_w - 1) /
                    args.dilation_w;
            if (filter_width_excess > 0) {
                args.filter_width -= filter_width_excess;
            } else {
                filter_width_excess = 0;
            }

            input_x_real = conv_ptr_add(input_y_real,
                                        args.input_channel *
                                            ((args.stride_x * (int)output_x +
                                              (filter_w - args.filter_width - filter_width_excess) * args.dilation_w) -
                                             args.padding_w_head),
                                        feat_bytes);
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            args.filter_element = conv_ptr_add(
                filter_ptr_y, (filter_w - args.filter_width - filter_width_excess) * filter_c_n_ptr_offset, filt_bytes);
            mac_border(buffer, input_x_real, args);
            tail(output_yx, buffer, args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
        }

        input_x_real = conv_ptr_add(
            input_y_real, args.input_channel * (args.stride_x * r.n_w_head - args.padding_w_head), feat_bytes);
        args.filter_width = filter_w;
        args.filter_y_offset = 0;
        args.filter_element = filter_ptr_y;
        for (size_t output_x = 0; output_x < (size_t)r.n_w_body; output_x++) {
            mac_border(buffer, input_x_real, args);
            tail(output_yx, buffer, args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }

        for (size_t output_x = 0; output_x < (size_t)r.n_w_tail; output_x++) {
            args.filter_width = (args.padding_w_head + args.input_width -
                                 (r.n_w_head + r.n_w_body + (int)output_x) * args.stride_x + args.dilation_w - 1) /
                args.dilation_w;
            args.filter_y_offset = (filter_w - args.filter_width) * filter_c_n_offset;
            mac_border(buffer, input_x_real, args);
            tail(output_yx, buffer, args);
            output_yx = conv_ptr_add(output_yx, args.output_x_offset, feat_bytes);
            input_x_real = conv_ptr_add(input_x_real, args.input_stride_x_offset, feat_bytes);
        }
        input_y_real = conv_ptr_add(input_y_real, args.input_stride_y_offset, feat_bytes);
    }
}

static void conv_loop_c(ConvArgsType &args,
                        conv_c_mac_fn_t mac_border,
                        conv_c_mac_fn_t mac_body,
                        conv_c_tail_fn_t tail,
                        int feat_bytes,
                        int filt_bytes,
                        int buf_bytes)
{
    bool is_1x1 = conv_is_1x1(args);
    bool has_padding = conv_has_padding(args);

    if (is_1x1 || !has_padding) {
        void *input_ptr = args.input_element;
        void *output_ptr = args.output_element;
        int height = args.output_height;
        int width = args.output_width;

        if (is_1x1 && has_padding) {
            ConvRegions r = conv_regions(args);
            void *zeros = tool::calloc_aligned(args.output_channel, buf_bytes, MALLOC_CAP_DEFAULT);
            conv_each_1x1_pad(args, r, feat_bytes, [&](void *out) { tail(out, zeros, args); });
            heap_caps_free(zeros);
            conv_1x1_body_view(args, r, feat_bytes, &input_ptr, &output_ptr, &height, &width);
        }
        const bool fast = is_1x1 &&
            (feat_bytes == 2 ? conv2d_1x1_s16_fast(args, filt_bytes, input_ptr, output_ptr, height, width)
                             : conv2d_1x1_s8_fast(args, tail, input_ptr, output_ptr, height, width));
        if (!fast) {
            void *buffer = tool::calloc_aligned(args.output_channel, buf_bytes, MALLOC_CAP_DEFAULT);
            conv_loop_c_body(args, mac_body, tail, buffer, feat_bytes, input_ptr, output_ptr, height, width);
            heap_caps_free(buffer);
        }
        return;
    }
    void *buffer = tool::calloc_aligned(args.output_channel, buf_bytes, MALLOC_CAP_DEFAULT);
    conv_loop_c_grid(args, mac_border, mac_body, tail, buffer, feat_bytes, filt_bytes);
    heap_caps_free(buffer);
}
#endif

void conv2d(
    void *const args_ptr, quant_type_t quant, dl_kernel_erased_t fn0, dl_kernel_erased_t fn1, dl_kernel_erased_t fn2)
{
    if (!fn0) {
        ESP_LOGE("conv2d", "unbound Conv body kernel");
        return;
    }
#if CONFIG_PIE_V2_BOOST
    dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
#endif
    ConvArgsType &args = *static_cast<ConvArgsType *>(args_ptr);
    if (!conv_is_1x1(args) && conv_has_padding(args) && !fn1) {
        ESP_LOGE("conv2d", "unbound Conv border kernel");
        return;
    }
    const int feat_bytes = conv_feature_bytes(quant);
    const int filt_bytes = conv_filter_bytes(quant);
#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
    conv_loop_pie(
        args, reinterpret_cast<conv_pie_fn_t>(fn0), reinterpret_cast<conv_pie_fn_t>(fn1), feat_bytes, filt_bytes);
#else
    if (!fn2) {
        ESP_LOGE("conv2d", "unbound Conv tail kernel");
        return;
    }
    conv_loop_c(args,
                reinterpret_cast<conv_c_mac_fn_t>(fn1),
                reinterpret_cast<conv_c_mac_fn_t>(fn0),
                reinterpret_cast<conv_c_tail_fn_t>(fn2),
                feat_bytes,
                filt_bytes,
                conv_buffer_bytes(quant));
#endif
}

} // namespace base
} // namespace dl
