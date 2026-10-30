#include "dl_base_c.hpp"
#include "dl_base_conv_args.hpp"
#include "dl_c_conv_common.hpp"
#include "dl_compile_config.h"
#include "esp_heap_caps.h"

#if CONFIG_XTENSA_MAC16_BOOST
#include "dl_xtensa_conv.hpp"
#endif

namespace dl {
namespace base {
template <typename feature_t, typename buffer_t>
void depthwise_conv2d_33c1(buffer_t *buffer_ptr, feature_t *input_ptr, const ConvArgsType &args)
{
    // static int flag = 1;
    const feature_t *filter_r0 = (feature_t *)args.filter_element;
    const feature_t *filter_r1 = filter_r0 + args.filter_y_offset_c;
    const feature_t *filter_r2 = filter_r1 + args.filter_y_offset_c;
    feature_t *input_row_0 = input_ptr;

    for (size_t filter_x = 0; filter_x < 3; filter_x++)                      // W
    {                                                                        //
        feature_t *input_row_1 = input_row_0 + args.input_dilation_y_offset; //
        feature_t *input_row_2 = input_row_1 + args.input_dilation_y_offset; //
        for (size_t input_c = 0; input_c < args.input_channel; input_c++)    // C
        {                                                                    //
            auto mac = [](buffer_t acc, feature_t in, feature_t w) -> buffer_t {
                return acc + static_cast<buffer_t>(static_cast<int32_t>(in) * static_cast<int32_t>(w));
            };
            buffer_ptr[input_c] = mac(buffer_ptr[input_c], input_row_0[input_c], filter_r0[input_c]);
            buffer_ptr[input_c] = mac(buffer_ptr[input_c], input_row_1[input_c], filter_r1[input_c]);
            buffer_ptr[input_c] = mac(buffer_ptr[input_c], input_row_2[input_c], filter_r2[input_c]);
        }
        filter_r0 += args.input_channel;
        filter_r1 += args.input_channel;
        filter_r2 += args.input_channel;
        input_row_0 += args.input_dilation_x_offset;
    }
    // flag = 0;
}

template <typename feature_t, typename buffer_t>
void depthwise_conv2d_hwc1(buffer_t *buffer_ptr, feature_t *input_ptr, const ConvArgsType &args)
{
    const feature_t *filter_element = (feature_t *)args.filter_element;
    for (size_t filter_y = 0; filter_y < args.filter_height; filter_y++)      // H
    {                                                                         //
        feature_t *input_yx = input_ptr;                                      //
        for (size_t filter_x = 0; filter_x < args.filter_width; filter_x++)   // W
        {                                                                     //
            for (size_t input_c = 0; input_c < args.input_channel; input_c++) // C
            {                                                                 //
                buffer_ptr[input_c] += static_cast<buffer_t>(static_cast<int32_t>(input_yx[input_c]) *
                                                             static_cast<int32_t>(*filter_element));
                filter_element++;
            }
            input_yx += args.input_dilation_x_offset;
        }
        filter_element += args.filter_y_offset;
        input_ptr += args.input_dilation_y_offset;
    }
}

#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S16_DEPTHWISE_CONV2D_33C1
template void depthwise_conv2d_33c1<int16_t, DL_S16_BUFFER_TYPE>(DL_S16_BUFFER_TYPE *, int16_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S16_DEPTHWISE_CONV2D_HWC1
template void depthwise_conv2d_hwc1<int16_t, DL_S16_BUFFER_TYPE>(DL_S16_BUFFER_TYPE *, int16_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_DEPTHWISE_CONV2D_33C1
template void depthwise_conv2d_33c1<int8_t, int32_t>(int32_t *, int8_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_DEPTHWISE_CONV2D_HWC1
template void depthwise_conv2d_hwc1<int8_t, int32_t>(int32_t *, int8_t *, const ConvArgsType &);
#endif

// int16 depthwise conv without padding, requantized in registers and bit-exact with
// depthwise_conv2d_hwc1 + buffer_bias_* / buffer_0000_* (per-tensor shift, round half up).
// Four channels are accumulated at once over all taps. An int16 x int16 product needs up to
// 31 bits, so the exact sum is rebuilt from the wrapping 32-bit sum S and H = sum(p >> 16).
bool depthwise_conv2d_s16_fast(const ConvArgsType &args, void *input_ptr, void *output_ptr, int height, int width)
{
    constexpr int max_taps = 64;
    const int taps = args.filter_height * args.filter_width;
    if (args.mac_shift == INT_MIN || taps > max_taps) {
        return false;
    }
#if CONFIG_XTENSA_MAC16_BOOST
    if (depthwise_conv2d_s16_xtensa(args, input_ptr, output_ptr, height, width)) {
        return true;
    }
#endif
    int offsets[max_taps];
    for (int fy = 0, t = 0; fy < args.filter_height; fy++) {
        for (int fx = 0; fx < args.filter_width; fx++, t++) {
            offsets[t] = (fy * args.input_dilation_y_offset + fx * args.input_dilation_x_offset) * sizeof(int16_t);
        }
    }
    const int C = args.input_channel;
    const int shift = args.mac_shift;
    const int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    const bool relu = args.activation_type == ReLU;
    const int64_t *bias = static_cast<const int64_t *>(args.bias_element);
    const int16_t *filter = static_cast<const int16_t *>(args.filter_element);
    // With |H| < 2^13 and at most max_taps taps the exact sum is within +-2^30 and equals S, so
    // with |bias| <= 2^29 and shift in [1, 29] the requantization stays in int32.
    bool narrow = shift >= 1 && shift <= 29;
    for (int c = 0; narrow && bias && c < C; c++) {
        narrow = bias[c] >= -((int64_t)1 << 29) && bias[c] <= ((int64_t)1 << 29);
    }
    const int32_t half32 = (int32_t)half;
    auto requant = [&](int32_t s, int32_t h, int c) -> int16_t {
        if (narrow && (uint32_t)(h + (1 << 13)) < (1u << 14)) {
            int32_t r = (s + (bias ? (int32_t)bias[c] : 0) + half32) >> shift;
            if (relu && r < 0) {
                r = 0;
            }
            return (int16_t)DL_CLIP(r, INT16_MIN, INT16_MAX);
        }
        return conv_c_requant_s16(conv_c_sum_from_sh(s, h) + (bias ? bias[c] : 0), half, shift, relu);
    };
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int16_t *in = static_cast<const int16_t *>(input_ptr) + y * args.input_stride_y_offset +
                x * args.input_stride_x_offset;
            int16_t *out = static_cast<int16_t *>(output_ptr) + y * args.output_y_offset + x * args.output_x_offset;
            int c = 0;
            for (; c + 4 <= C; c += 4) {
                uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                int32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
                const int16_t *w = filter + c;
                const int16_t *xc = in + c;
                for (int t = 0; t < taps; t++, w += C) {
                    const int16_t *xp =
                        reinterpret_cast<const int16_t *>(reinterpret_cast<const char *>(xc) + offsets[t]);
                    const int32_t p0 = (int32_t)xp[0] * w[0];
                    const int32_t p1 = (int32_t)xp[1] * w[1];
                    const int32_t p2 = (int32_t)xp[2] * w[2];
                    const int32_t p3 = (int32_t)xp[3] * w[3];
                    s0 += p0, s1 += p1, s2 += p2, s3 += p3;
                    h0 += p0 >> 16, h1 += p1 >> 16, h2 += p2 >> 16, h3 += p3 >> 16;
                }
                out[c] = requant((int32_t)s0, h0, c);
                out[c + 1] = requant((int32_t)s1, h1, c + 1);
                out[c + 2] = requant((int32_t)s2, h2, c + 2);
                out[c + 3] = requant((int32_t)s3, h3, c + 3);
            }
            for (; c < C; c++) {
                uint32_t s = 0;
                int32_t h = 0;
                const int16_t *w = filter + c;
                for (int t = 0; t < taps; t++, w += C) {
                    const int32_t p = (int32_t)*reinterpret_cast<const int16_t *>(
                                          reinterpret_cast<const char *>(in + c) + offsets[t]) *
                        w[0];
                    s += p, h += p >> 16;
                }
                out[c] = requant((int32_t)s, h, c);
            }
        }
    }
    return true;
}

// Four channels are accumulated at once over all taps; int8 x int8 products sum exactly in
// int32, the accumulator type of depthwise_conv2d_*<int8_t, int32_t>. The sums of each pixel go
// through the model's own tail, so every s8 bias, activation and per-channel variant keeps its
// exact semantics.
bool depthwise_conv2d_s8_fast(
    const ConvArgsType &args, conv_c_tail_fn_t tail, void *input_ptr, void *output_ptr, int height, int width)
{
    constexpr int max_taps = 64;
    const int taps = args.filter_height * args.filter_width;
    if (taps > max_taps) {
        return false;
    }
    int offsets[max_taps];
    for (int fy = 0, t = 0; fy < args.filter_height; fy++) {
        for (int fx = 0; fx < args.filter_width; fx++, t++) {
            offsets[t] = fy * args.input_dilation_y_offset + fx * args.input_dilation_x_offset;
        }
    }
    const int C = args.input_channel;
    int32_t *acc = static_cast<int32_t *>(heap_caps_malloc(C * sizeof(int32_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!acc) {
        acc = static_cast<int32_t *>(heap_caps_malloc(C * sizeof(int32_t), MALLOC_CAP_DEFAULT));
        if (!acc) {
            return false;
        }
    }
    const int8_t *filter = static_cast<const int8_t *>(args.filter_element);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int8_t *in = static_cast<const int8_t *>(input_ptr) + y * args.input_stride_y_offset +
                x * args.input_stride_x_offset;
            int8_t *out = static_cast<int8_t *>(output_ptr) + y * args.output_y_offset + x * args.output_x_offset;
            int c = 0;
            for (; c + 4 <= C; c += 4) {
                int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                const int8_t *w = filter + c;
                const int8_t *xc = in + c;
                for (int t = 0; t < taps; t++, w += C) {
                    const int8_t *xp = xc + offsets[t];
                    a0 += xp[0] * w[0], a1 += xp[1] * w[1], a2 += xp[2] * w[2], a3 += xp[3] * w[3];
                }
                acc[c] = a0, acc[c + 1] = a1, acc[c + 2] = a2, acc[c + 3] = a3;
            }
            for (; c < C; c++) {
                int32_t a = 0;
                const int8_t *w = filter + c;
                for (int t = 0; t < taps; t++, w += C) {
                    a += in[c + offsets[t]] * w[0];
                }
                acc[c] = a;
            }
            tail(out, acc, args);
        }
    }
    heap_caps_free(acc);
    return true;
}

} // namespace base
} // namespace dl
