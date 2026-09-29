#include "dl_base_c.hpp"
#include "dl_base_conv_args.hpp"
#include "dl_c_conv_common.hpp"
#include "dl_compile_config.h"

#include <algorithm>
#include <type_traits>

#if defined(__riscv)
#include "dl_base_riscv.h"
#endif
#if CONFIG_XTENSA_MAC16_BOOST
#include "dl_xtensa_conv.hpp"
#endif

namespace dl {
namespace base {
template <typename feature_t, typename buffer_t, typename filter_t>
void conv2d_11cn(buffer_t *buffer_ptr, feature_t *input_ptr, const ConvArgsType &args)
{
    const filter_t *filter_element = (const filter_t *)args.filter_element;

    // W8A16: int16 activation x int8 weight. A full dot-product fits in int32 for
    // chunks of 128 channels (128*32767*127 < 2^31). Keep the running sum in int32
    // and widen to the int64 buffer once per chunk.
    if constexpr (std::is_same_v<feature_t, int16_t> && std::is_same_v<filter_t, int8_t>) {
        const size_t channels = args.input_channel;
        const size_t outputs = args.output_channel;
        size_t output_c = 0;
        for (; output_c + 4 <= outputs; output_c += 4) {
            const int8_t *f0 = filter_element;
            const int8_t *f1 = f0 + channels;
            const int8_t *f2 = f1 + channels;
            const int8_t *f3 = f2 + channels;
            int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (size_t base = 0; base < channels; base += 128) {
                int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                size_t end = base + 128;
                if (end > channels) {
                    end = channels;
                }
                for (size_t input_c = base; input_c < end; input_c++) {
                    int32_t x = input_ptr[input_c];
                    a0 += x * static_cast<int32_t>(f0[input_c]);
                    a1 += x * static_cast<int32_t>(f1[input_c]);
                    a2 += x * static_cast<int32_t>(f2[input_c]);
                    a3 += x * static_cast<int32_t>(f3[input_c]);
                }
                s0 += a0;
                s1 += a1;
                s2 += a2;
                s3 += a3;
            }
            buffer_ptr[output_c] = s0;
            buffer_ptr[output_c + 1] = s1;
            buffer_ptr[output_c + 2] = s2;
            buffer_ptr[output_c + 3] = s3;
            filter_element += channels * 4;
        }
        for (; output_c < outputs; output_c++) {
            int64_t sum = 0;
            for (size_t base = 0; base < channels; base += 128) {
                int32_t acc = 0;
                size_t end = base + 128;
                if (end > channels) {
                    end = channels;
                }
                for (size_t input_c = base; input_c < end; input_c++) {
                    acc += static_cast<int32_t>(input_ptr[input_c]) * static_cast<int32_t>(*filter_element++);
                }
                sum += acc;
            }
            buffer_ptr[output_c] = sum;
        }
        return;
    }

    // filter in sequence [H, W, C, N]
    // for (size_t input_c = 0; input_c < args.input_channel; input_c++)
    // {
    //     for (size_t output_c = 0; output_c < args.output_channel; output_c++)
    //     {
    //         buffer_ptr[output_c] += input_ptr[input_c] * (*filter_element++);
    //     }
    // }

    // filter in sequence [N, H, W, C]. int16*int16 and int16*int8 both fit in int32,
    // so the product stays a 32-bit mul. Widening it to buffer_t (often int64) first
    // turns every MAC into a software 64-bit multiply on RV32.
    const size_t oc_aligned = args.output_channel & ~size_t{3};
    size_t output_c = 0;
    for (; output_c < oc_aligned; output_c += 4) {
        buffer_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        const filter_t *f0 = filter_element;
        const filter_t *f1 = f0 + args.input_channel;
        const filter_t *f2 = f1 + args.input_channel;
        const filter_t *f3 = f2 + args.input_channel;
        for (size_t input_c = 0; input_c < args.input_channel; input_c++) {
            int32_t x = static_cast<int32_t>(input_ptr[input_c]);
            a0 += static_cast<buffer_t>(x * static_cast<int32_t>(f0[input_c]));
            a1 += static_cast<buffer_t>(x * static_cast<int32_t>(f1[input_c]));
            a2 += static_cast<buffer_t>(x * static_cast<int32_t>(f2[input_c]));
            a3 += static_cast<buffer_t>(x * static_cast<int32_t>(f3[input_c]));
        }
        filter_element += args.input_channel * 4;
        buffer_ptr[output_c] = a0;
        buffer_ptr[output_c + 1] = a1;
        buffer_ptr[output_c + 2] = a2;
        buffer_ptr[output_c + 3] = a3;
    }
    for (; output_c < args.output_channel; output_c++) {
        buffer_t acc = 0;
        for (size_t input_c = 0; input_c < args.input_channel; input_c++) {
            int32_t x = static_cast<int32_t>(input_ptr[input_c]);
            acc += static_cast<buffer_t>(x * static_cast<int32_t>(*filter_element++));
        }
        buffer_ptr[output_c] = acc;
    }
}

template <typename feature_t, typename buffer_t, typename filter_t>
void conv2d_33cn(buffer_t *buffer_ptr, feature_t *input_ptr, const ConvArgsType &args)
{
    // filter in sequence [H, W, C, N]
    // const filter_t *filter_r0 = args.filter_element;
    // const filter_t *filter_r1 = filter_r0 + args.filter_y_offset;
    // const filter_t *filter_r2 = filter_r1 + args.filter_y_offset;
    // feature_t *&input_syx_d0 = input_ptr;

    // for (size_t filter_x = 0; filter_x < 3; filter_x++)                           // W
    // {                                                                             //
    //     feature_t *input_syx_d1 = input_syx_d0 + args.input_dilation_y_offset;      //
    //     feature_t *input_syx_d2 = input_syx_d1 + args.input_dilation_y_offset;      //
    //     for (size_t input_c = 0; input_c < args.input_channel; input_c++)         // C
    //     {                                                                         //
    //         for (size_t output_c = 0; output_c < args.output_channel; output_c++) // N
    //         {
    //             buffer_ptr[output_c] += input_syx_d0[input_c] * (*filter_r0);
    //             buffer_ptr[output_c] += input_syx_d1[input_c] * (*filter_r1);
    //             buffer_ptr[output_c] += input_syx_d2[input_c] * (*filter_r2);

    //             filter_r0++;
    //             filter_r1++;
    //             filter_r2++;
    //         }
    //     }
    //     input_syx_d0 += args.input_dilation_x_offset;
    // }

    // filter in sequence [N, H, W, C]
    feature_t *input_00 = input_ptr;
    feature_t *input_01 = input_00 + args.input_dilation_x_offset;
    feature_t *input_02 = input_01 + args.input_dilation_x_offset;

    feature_t *input_10 = input_00 + args.input_dilation_y_offset;
    feature_t *input_11 = input_10 + args.input_dilation_x_offset;
    feature_t *input_12 = input_11 + args.input_dilation_x_offset;

    feature_t *input_20 = input_10 + args.input_dilation_y_offset;
    feature_t *input_21 = input_20 + args.input_dilation_x_offset;
    feature_t *input_22 = input_21 + args.input_dilation_x_offset;

    const filter_t *filter_00 = (const filter_t *)args.filter_element;
    const filter_t *filter_01 = filter_00 + args.input_channel;
    const filter_t *filter_02 = filter_01 + args.input_channel;

    const filter_t *filter_10 = filter_00 + args.filter_y_offset_c;
    const filter_t *filter_11 = filter_10 + args.input_channel;
    const filter_t *filter_12 = filter_11 + args.input_channel;

    const filter_t *filter_20 = filter_10 + args.filter_y_offset_c;
    const filter_t *filter_21 = filter_20 + args.input_channel;
    const filter_t *filter_22 = filter_21 + args.input_channel;

    for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
        buffer_t acc = 0;
        for (size_t input_c = 0; input_c < args.input_channel; input_c++) {
            auto mac = [](buffer_t acc, feature_t in, filter_t w) -> buffer_t {
                return acc + static_cast<buffer_t>(static_cast<int32_t>(in) * static_cast<int32_t>(w));
            };
            acc = mac(acc, input_00[input_c], filter_00[input_c]);
            acc = mac(acc, input_01[input_c], filter_01[input_c]);
            acc = mac(acc, input_02[input_c], filter_02[input_c]);
            acc = mac(acc, input_10[input_c], filter_10[input_c]);
            acc = mac(acc, input_11[input_c], filter_11[input_c]);
            acc = mac(acc, input_12[input_c], filter_12[input_c]);
            acc = mac(acc, input_20[input_c], filter_20[input_c]);
            acc = mac(acc, input_21[input_c], filter_21[input_c]);
            acc = mac(acc, input_22[input_c], filter_22[input_c]);
        }
        filter_00 += args.filter_n_offset_c;
        filter_01 += args.filter_n_offset_c;
        filter_02 += args.filter_n_offset_c;
        filter_10 += args.filter_n_offset_c;
        filter_11 += args.filter_n_offset_c;
        filter_12 += args.filter_n_offset_c;
        filter_20 += args.filter_n_offset_c;
        filter_21 += args.filter_n_offset_c;
        filter_22 += args.filter_n_offset_c;

        buffer_ptr[output_c] = acc;
    }
}

template <typename feature_t, typename buffer_t, typename filter_t>
void conv2d_hwcn(buffer_t *buffer_ptr, feature_t *input_ptr, const ConvArgsType &args)
{
    // filter in sequence [H, W, C, N]
    // const filter_t *filter_element = args.filter_element;                             // Reload filter
    // feature_t *&input_syx_dy = input_ptr;                                               //
    // for (size_t filter_y = 0; filter_y < args.filter_height; filter_y++)              // H
    // {                                                                                 //
    //     feature_t *input_syx_dyx = input_syx_dy;                                        //
    //     for (size_t filter_x = 0; filter_x < args.filter_width; filter_x++)           // W
    //     {                                                                             //
    //         for (size_t input_c = 0; input_c < args.input_channel; input_c++)         // C
    //         {                                                                         //
    //             for (size_t output_c = 0; output_c < args.output_channel; output_c++) // N
    //             {
    //                 buffer_ptr[output_c] += input_syx_dyx[input_c] * (*filter_element);
    //                 filter_element++;
    //             }
    //         }
    //         input_syx_dyx += args.input_dilation_x_offset;
    //     }
    //     input_syx_dy += args.input_dilation_y_offset;
    // }

    // filter in sequence [N, H, W, C]
    const filter_t *filter_element = (const filter_t *)args.filter_element;
    for (size_t output_c = 0; output_c < args.output_channel; output_c++)         // N
    {                                                                             //
        feature_t *input_syx_dy = input_ptr;                                      //
        buffer_t acc = 0;                                                         //
        for (size_t filter_y = 0; filter_y < args.filter_height; filter_y++)      // H
        {                                                                         //
            feature_t *input_syx_dyx = input_syx_dy;                              //
            for (size_t filter_x = 0; filter_x < args.filter_width; filter_x++)   // W
            {                                                                     //
                for (size_t input_c = 0; input_c < args.input_channel; input_c++) // C
                {
                    acc += static_cast<buffer_t>(static_cast<int32_t>(input_syx_dyx[input_c]) *
                                                 static_cast<int32_t>(*filter_element++));
                }
                input_syx_dyx += args.input_dilation_x_offset;
            }
            filter_element += args.filter_y_offset;
            input_syx_dy += args.input_dilation_y_offset;
        }
        filter_element += args.filter_n_offset;
        buffer_ptr[output_c] = acc;
    }
}

#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S16_CONV2D_11CN
template void conv2d_11cn<int16_t, DL_S16_BUFFER_TYPE, int16_t>(DL_S16_BUFFER_TYPE *, int16_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S16_CONV2D_33CN
template void conv2d_33cn<int16_t, DL_S16_BUFFER_TYPE, int16_t>(DL_S16_BUFFER_TYPE *, int16_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S16_CONV2D_HWCN
template void conv2d_hwcn<int16_t, DL_S16_BUFFER_TYPE, int16_t>(DL_S16_BUFFER_TYPE *, int16_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_CONV2D_11CN
template void conv2d_11cn<int8_t, int32_t, int8_t>(int32_t *, int8_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_CONV2D_33CN
template void conv2d_33cn<int8_t, int32_t, int8_t>(int32_t *, int8_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_S8_CONV2D_HWCN
template void conv2d_hwcn<int8_t, int32_t, int8_t>(int32_t *, int8_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_W8A16_CONV2D_11CN
template void conv2d_11cn<int16_t, DL_S16_BUFFER_TYPE, int8_t>(DL_S16_BUFFER_TYPE *, int16_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_W8A16_CONV2D_33CN
template void conv2d_33cn<int16_t, DL_S16_BUFFER_TYPE, int8_t>(DL_S16_BUFFER_TYPE *, int16_t *, const ConvArgsType &);
#endif
#if DL_COMPILE_ALL || DL_KERNEL_DL_C_W8A16_CONV2D_HWCN
template void conv2d_hwcn<int16_t, DL_S16_BUFFER_TYPE, int8_t>(DL_S16_BUFFER_TYPE *, int16_t *, const ConvArgsType &);
#endif

// acc[j * 4 + p] = sum_i f_j[i] * xin[i * 4 + p] over [xp, xe), 3 output channels x 4 pixels.
static inline void conv_c_w8a16_mac_3x4(
    const int8_t *f0, const int8_t *f1, const int8_t *f2, const int16_t *xp, const int16_t *xe, int32_t *acc)
{
#if defined(__riscv)
    dl_riscv_w8a16_mac_3x4(f0, f1, f2, xp, xe, acc);
#else
    int32_t a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;
    int32_t a20 = 0, a21 = 0, a22 = 0, a23 = 0;
    for (; xp < xe; xp += 4) {
        const int32_t x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        int32_t w = *f0++;
        a00 += w * x0, a01 += w * x1, a02 += w * x2, a03 += w * x3;
        w = *f1++;
        a10 += w * x0, a11 += w * x1, a12 += w * x2, a13 += w * x3;
        w = *f2++;
        a20 += w * x0, a21 += w * x1, a22 += w * x2, a23 += w * x3;
    }
    acc[0] = a00, acc[1] = a01, acc[2] = a02, acc[3] = a03;
    acc[4] = a10, acc[5] = a11, acc[6] = a12, acc[7] = a13;
    acc[8] = a20, acc[9] = a21, acc[10] = a22, acc[11] = a23;
#endif
}

// int16 x int8: |x * w| <= 2^22, so 256 terms fit an int32 accumulator. Up to 128 terms
// |acc| <= 2^29, which leaves room for a bias within +-2^29 and the rounding half in int32.
static void conv_c_1x1_w8a16_block(const ConvArgsType &args, const int16_t *xin, int16_t *const out[4], int n_px)
{
    const int C = args.input_channel;
    const int N = args.output_channel;
    const int shift = args.mac_shift;
    const int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    const bool relu = args.activation_type == ReLU;
    const int64_t *bias = static_cast<const int64_t *>(args.bias_element);
    const int8_t *f = static_cast<const int8_t *>(args.filter_element);
    const bool narrow = C <= 128 && shift >= 1 && shift <= 30;
    constexpr int64_t bias_limit = (int64_t)1 << 29;

    int oc = 0;
    for (; oc + 3 <= N; oc += 3, f += 3 * C) {
        int64_t s[12] = {};
        for (int base = 0; base < C; base += 256) {
            const int end = std::min(C, base + 256);
            int32_t acc[12];
            conv_c_w8a16_mac_3x4(f + base, f + C + base, f + 2 * C + base, xin + base * 4, xin + end * 4, acc);
            if (C <= 256) {
                for (int j = 0; j < 3; j++) {
                    const int64_t b = bias ? bias[oc + j] : 0;
                    if (narrow && b >= -bias_limit && b <= bias_limit) {
                        for (int p = 0; p < n_px; p++) {
                            out[p][oc + j] =
                                conv_c_requant_s16_i32(acc[j * 4 + p] + (int32_t)b, (int32_t)half, shift, relu);
                        }
                    } else {
                        for (int p = 0; p < n_px; p++) {
                            out[p][oc + j] = conv_c_requant_s16(acc[j * 4 + p] + b, half, shift, relu);
                        }
                    }
                }
                break;
            }
            for (int k = 0; k < 12; k++) {
                s[k] += acc[k];
            }
        }
        if (C > 256) {
            for (int j = 0; j < 3; j++) {
                const int64_t b = bias ? bias[oc + j] : 0;
                for (int p = 0; p < n_px; p++) {
                    out[p][oc + j] = conv_c_requant_s16(s[j * 4 + p] + b, half, shift, relu);
                }
            }
        }
    }
    for (; oc < N; oc++, f += C) {
        int64_t s[4] = {};
        for (int base = 0; base < C; base += 256) {
            const int end = std::min(C, base + 256);
            const int16_t *xp = xin + base * 4;
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int i = base; i < end; i++, xp += 4) {
                const int32_t w = f[i];
                a0 += w * xp[0], a1 += w * xp[1], a2 += w * xp[2], a3 += w * xp[3];
            }
            s[0] += a0, s[1] += a1, s[2] += a2, s[3] += a3;
        }
        const int64_t b = bias ? bias[oc] : 0;
        for (int p = 0; p < n_px; p++) {
            out[p][oc] = conv_c_requant_s16(s[p] + b, half, shift, relu);
        }
    }
}

// sh[j * 4 + p] = wrapping sum S, sh[8 + j * 4 + p] = sum of p >> 16 H, for output channel j of
// 2 and pixel p of 4 over [xp, xe).
static inline void conv_c_s16_mac_2x4(
    const int16_t *f0, const int16_t *f1, const int16_t *xp, const int16_t *xe, int32_t *sh)
{
#if defined(__riscv)
    dl_riscv_s16_mac_2x4(f0, f1, xp, xe, sh);
#else
    uint32_t s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0;
    int32_t h00 = 0, h01 = 0, h02 = 0, h03 = 0, h10 = 0, h11 = 0, h12 = 0, h13 = 0;
    for (; xp < xe; xp += 4) {
        const int32_t x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        int32_t w = *f0++;
        int32_t p0 = w * x0, p1 = w * x1, p2 = w * x2, p3 = w * x3;
        s00 += p0, s01 += p1, s02 += p2, s03 += p3;
        h00 += p0 >> 16, h01 += p1 >> 16, h02 += p2 >> 16, h03 += p3 >> 16;
        w = *f1++;
        p0 = w * x0, p1 = w * x1, p2 = w * x2, p3 = w * x3;
        s10 += p0, s11 += p1, s12 += p2, s13 += p3;
        h10 += p0 >> 16, h11 += p1 >> 16, h12 += p2 >> 16, h13 += p3 >> 16;
    }
    sh[0] = s00, sh[1] = s01, sh[2] = s02, sh[3] = s03, sh[4] = s10, sh[5] = s11, sh[6] = s12, sh[7] = s13;
    sh[8] = h00, sh[9] = h01, sh[10] = h02, sh[11] = h03, sh[12] = h10, sh[13] = h11, sh[14] = h12, sh[15] = h13;
#endif
}

static void conv_c_1x1_s16_block(const ConvArgsType &args, const int16_t *xin, int16_t *const out[4], int n_px)
{
    const int C = args.input_channel;
    const int N = args.output_channel;
    const int shift = args.mac_shift;
    const int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    const bool relu = args.activation_type == ReLU;
    const int64_t *bias = static_cast<const int64_t *>(args.bias_element);
    const int16_t *f = static_cast<const int16_t *>(args.filter_element);

    int oc = 0;
    for (; oc + 2 <= N; oc += 2, f += 2 * C) {
        int32_t sh[16];
        conv_c_s16_mac_2x4(f, f + C, xin, xin + C * 4, sh);
        const int32_t s00 = sh[0], s01 = sh[1], s02 = sh[2], s03 = sh[3];
        const int32_t s10 = sh[4], s11 = sh[5], s12 = sh[6], s13 = sh[7];
        const int32_t h00 = sh[8], h01 = sh[9], h02 = sh[10], h03 = sh[11];
        const int32_t h10 = sh[12], h11 = sh[13], h12 = sh[14], h13 = sh[15];
        const int64_t r[2][4] = {
            {conv_c_sum_from_sh(s00, h00),
             conv_c_sum_from_sh(s01, h01),
             conv_c_sum_from_sh(s02, h02),
             conv_c_sum_from_sh(s03, h03)},
            {conv_c_sum_from_sh(s10, h10),
             conv_c_sum_from_sh(s11, h11),
             conv_c_sum_from_sh(s12, h12),
             conv_c_sum_from_sh(s13, h13)},
        };
        for (int j = 0; j < 2; j++) {
            const int64_t b = bias ? bias[oc + j] : 0;
            for (int p = 0; p < n_px; p++) {
                out[p][oc + j] = conv_c_requant_s16(r[j][p] + b, half, shift, relu);
            }
        }
    }
    for (; oc < N; oc++, f += C) {
        const int16_t *xp = xin;
        uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
        for (int i = 0; i < C; i++, xp += 4) {
            const int32_t w = f[i];
            const int32_t p0 = w * xp[0], p1 = w * xp[1], p2 = w * xp[2], p3 = w * xp[3];
            s0 += p0, s1 += p1, s2 += p2, s3 += p3;
            h0 += p0 >> 16, h1 += p1 >> 16, h2 += p2 >> 16, h3 += p3 >> 16;
        }
        const int64_t r[4] = {conv_c_sum_from_sh((int32_t)s0, h0),
                              conv_c_sum_from_sh((int32_t)s1, h1),
                              conv_c_sum_from_sh((int32_t)s2, h2),
                              conv_c_sum_from_sh((int32_t)s3, h3)};
        const int64_t b = bias ? bias[oc] : 0;
        for (int p = 0; p < n_px; p++) {
            out[p][oc] = conv_c_requant_s16(r[p] + b, half, shift, relu);
        }
    }
}

// acc[p][oc] = sum_i f_oc[i] * xin[i * 4 + p]. int8 x int8 products sum exactly in int32, the
// accumulator type of conv2d_11cn<int8_t, int32_t>, so the int16-activation MAC kernel is reused
// on the widened inputs.
static void conv_c_1x1_s8_block(const ConvArgsType &args, const int16_t *xin, int32_t *const acc[4])
{
    const int C = args.input_channel;
    const int N = args.output_channel;
    const int8_t *f = static_cast<const int8_t *>(args.filter_element);
    const int16_t *xe = xin + C * 4;

    int oc = 0;
    for (; oc + 3 <= N; oc += 3, f += 3 * C) {
        int32_t a[12];
        conv_c_w8a16_mac_3x4(f, f + C, f + 2 * C, xin, xe, a);
        for (int j = 0; j < 3; j++) {
            for (int p = 0; p < 4; p++) {
                acc[p][oc + j] = a[j * 4 + p];
            }
        }
    }
    for (; oc < N; oc++, f += C) {
        const int16_t *xp = xin;
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int i = 0; i < C; i++, xp += 4) {
            const int32_t w = f[i];
            a0 += w * xp[0], a1 += w * xp[1], a2 += w * xp[2], a3 += w * xp[3];
        }
        acc[0][oc] = a0, acc[1][oc] = a1, acc[2][oc] = a2, acc[3][oc] = a3;
    }
}

// acc[oc] = sum_i f_oc[i] * x[i] for a single pixel, where the 4-pixel kernel would waste lanes.
static void conv_c_1x1_s8_px1(const ConvArgsType &args, const int8_t *x, int32_t *acc)
{
    const int C = args.input_channel;
    const int N = args.output_channel;
    const int8_t *f = static_cast<const int8_t *>(args.filter_element);

    int oc = 0;
    for (; oc + 4 <= N; oc += 4, f += 4 * C) {
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int i = 0; i < C; i++) {
            const int32_t v = x[i];
            a0 += v * f[i], a1 += v * f[C + i], a2 += v * f[2 * C + i], a3 += v * f[3 * C + i];
        }
        acc[oc] = a0, acc[oc + 1] = a1, acc[oc + 2] = a2, acc[oc + 3] = a3;
    }
    for (; oc < N; oc++, f += C) {
        int32_t a = 0;
        for (int i = 0; i < C; i++) {
            a += x[i] * f[i];
        }
        acc[oc] = a;
    }
}

// Pixels are processed 4 at a time so each weight load feeds 4 MACs. The inputs of pixels
// [begin, begin + n_px) are interleaved as xin[i * 4 + p], zero for p >= n_px, so that a single
// pointer with immediate offsets reaches all of them.
template <typename feature_t>
static void conv_c_gather_px4(const ConvArgsType &args,
                              const feature_t *input,
                              feature_t *output,
                              int width,
                              int begin,
                              int n_px,
                              int16_t *xin,
                              feature_t *out[4])
{
    const feature_t *in[4] = {};
    for (int p = 0; p < n_px; p++) {
        const int y = (begin + p) / width;
        const int x = (begin + p) % width;
        in[p] = input + y * args.input_stride_y_offset + x * args.input_stride_x_offset;
        out[p] = output + y * args.output_y_offset + x * args.output_x_offset;
    }
    for (int i = 0; i < args.input_channel; i++) {
        for (int p = 0; p < 4; p++) {
            xin[i * 4 + p] = p < n_px ? in[p][i] : 0;
        }
    }
}

// Requantizes in registers, bit-exact with buffer_bias_* / buffer_0000_* (per-tensor shift,
// round half up).
bool conv2d_1x1_s16_fast(
    const ConvArgsType &args, int filt_bytes, void *input_ptr, void *output_ptr, int height, int width)
{
    if (args.mac_shift == INT_MIN || args.input_channel > 65536) {
        return false;
    }
#if CONFIG_XTENSA_MAC16_BOOST
    if (conv2d_1x1_s16_xtensa(args, filt_bytes, input_ptr, output_ptr, height, width)) {
        return true;
    }
#endif
    int16_t *xin = static_cast<int16_t *>(conv_c_scratch_alloc((size_t)args.input_channel * 4 * sizeof(int16_t)));
    if (!xin) {
        return false;
    }
    const int pixels = height * width;
    for (int begin = 0; begin < pixels; begin += 4) {
        const int n_px = std::min(4, pixels - begin);
        int16_t *out[4] = {};
        conv_c_gather_px4(args,
                          static_cast<const int16_t *>(input_ptr),
                          static_cast<int16_t *>(output_ptr),
                          width,
                          begin,
                          n_px,
                          xin,
                          out);
        if (filt_bytes == 1) {
            conv_c_1x1_w8a16_block(args, xin, out, n_px);
        } else {
            conv_c_1x1_s16_block(args, xin, out, n_px);
        }
    }
    heap_caps_free(xin);
    return true;
}

// The int32 sums of each pixel go through the model's own tail, so every s8 bias, activation
// and per-channel variant keeps its exact semantics.
bool conv2d_1x1_s8_fast(
    const ConvArgsType &args, conv_c_tail_fn_t tail, void *input_ptr, void *output_ptr, int height, int width)
{
    const int C = args.input_channel;
    const int N = args.output_channel;
    const size_t xin_bytes = (size_t)C * 4 * sizeof(int16_t);
    int16_t *xin = static_cast<int16_t *>(conv_c_scratch_alloc(xin_bytes + (size_t)N * 4 * sizeof(int32_t)));
    if (!xin) {
        return false;
    }
    int32_t *acc0 = reinterpret_cast<int32_t *>(xin + C * 4);
    int32_t *const acc[4] = {acc0, acc0 + N, acc0 + 2 * N, acc0 + 3 * N};
    const int8_t *input = static_cast<const int8_t *>(input_ptr);
    int8_t *output = static_cast<int8_t *>(output_ptr);
    const int pixels = height * width;
    const int full = pixels & ~3;
    for (int begin = 0; begin < full; begin += 4) {
        int8_t *out[4] = {};
        conv_c_gather_px4(args, input, output, width, begin, 4, xin, out);
        conv_c_1x1_s8_block(args, xin, acc);
        for (int p = 0; p < 4; p++) {
            tail(out[p], acc[p], args);
        }
    }
    for (int pixel = full; pixel < pixels; pixel++) {
        const int y = pixel / width;
        const int x = pixel % width;
        conv_c_1x1_s8_px1(args, input + y * args.input_stride_y_offset + x * args.input_stride_x_offset, acc0);
        tail(output + y * args.output_y_offset + x * args.output_x_offset, acc0, args);
    }
    heap_caps_free(xin);
    return true;
}

} // namespace base
} // namespace dl
