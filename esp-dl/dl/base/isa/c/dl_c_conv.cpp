#include "dl_base_c.hpp"
#include "dl_base_conv_args.hpp"
#include "dl_compile_config.h"
#include <type_traits>

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

} // namespace base
} // namespace dl
