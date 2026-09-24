#include "dl_base_c.hpp"
#include "dl_base_conv_args.hpp"
#include "dl_compile_config.h"

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

} // namespace base
} // namespace dl
