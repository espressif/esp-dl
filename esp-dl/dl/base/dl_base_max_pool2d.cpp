#include "dl_base_max_pool2d.hpp"

#include "dl_base_activate_buffer.hpp"
#include "dl_base_activate_output.hpp"
#include "dl_base_isa.hpp"

namespace dl {
namespace base {
template <typename feature_t, typename filter_t, typename buffer_t>
inline void max_pool2d_hwc1(buffer_t *buffer_ptr,
                            feature_t *input_ptr,
                            feature_t *output_ptr,
                            PoolArgsType<feature_t> &args)
{
    for (size_t input_c = 0; input_c < args.input_channel; input_c++) {
        buffer_ptr[input_c] = input_ptr[input_c];
    }

    for (size_t filter_y = 0; filter_y < args.filter_height; filter_y++) // H
    {                                                                    //
        feature_t *input_yx = input_ptr;
        for (size_t filter_x = 0; filter_x < args.filter_width; filter_x++)   // W
        {                                                                     //
            for (size_t input_c = 0; input_c < args.input_channel; input_c++) // C
            {                                                                 //
                buffer_ptr[input_c] = DL_MAX(input_yx[input_c], buffer_ptr[input_c]);
            }
            input_yx += args.input_x_offset;
        }
        input_ptr += args.input_y_offset;
    }

    for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
        tool::truncate(output_ptr[output_c], buffer_ptr[output_c]);
        buffer_ptr[output_c] = 0;
    }
}

template <typename feature_t, typename filter_t, typename buffer_t>
void max_pool2d(void *args_ptr)
{
    const PoolArgsType<feature_t> &args = *((PoolArgsType<feature_t> *)args_ptr);

    max_pool_shell(args, NULL, NULL, max_pool2d_hwc1<feature_t, filter_t, buffer_t>);
}

template <>
void max_pool2d<int16_t, int16_t, int64_t>(void *args_ptr)
{
    PoolArgsType<int16_t> &args = *((PoolArgsType<int16_t> *)args_ptr);

    i_impl_func_s16_t i_impl_func = NULL;
    i_impl_func_s16_t i_impl_func_sp = NULL;
    pool_c_impl_func_s16_t c_impl_func = NULL;

#if CONFIG_TIE728_BOOST
    if (args.input_x_offset % 8 == 0 && args.output_x_offset % 8 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        i_impl_func = dl_tie728_s16_max_pool2d_hwc1;
        i_impl_func_sp = (args.filter_height == 2 && args.filter_width == 2) ? dl_tie728_s16_max_pool2d_22c1
                                                                             : dl_tie728_s16_max_pool2d_hwc1;
    } else {
        i_impl_func = dl_tie728_s16_unaligned_max_pool2d_hwc1;
        i_impl_func_sp = (args.filter_height == 2 && args.filter_width == 2) ? dl_tie728_s16_unaligned_max_pool2d_22c1
                                                                             : dl_tie728_s16_unaligned_max_pool2d_hwc1;
    }
#else
    c_impl_func = max_pool2d_hwc1<int16_t, int16_t, int64_t>;
#endif

    max_pool_shell<int16_t, int64_t>(args, i_impl_func, i_impl_func_sp, c_impl_func);
}

template <>
void max_pool2d<int8_t, int8_t, int32_t>(void *args_ptr)
{
    PoolArgsType<int8_t> &args = *((PoolArgsType<int8_t> *)args_ptr);

    i_impl_func_s8_t i_impl_func = NULL;
    i_impl_func_s8_t i_impl_func_sp = NULL;
    pool_c_impl_func_s8_t c_impl_func = NULL;

#if CONFIG_TIE728_BOOST
    if (args.input_x_offset % 16 == 0 && args.output_x_offset % 16 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        i_impl_func = dl_tie728_s8_max_pool2d_hwc1;
        i_impl_func_sp = (args.filter_height == 2 && args.filter_width == 2) ? dl_tie728_s8_max_pool2d_22c1
                                                                             : dl_tie728_s8_max_pool2d_hwc1;
    } else {
        i_impl_func = dl_tie728_s8_unaligned_max_pool2d_hwc1;
        i_impl_func_sp = (args.filter_height == 2 && args.filter_width == 2) ? dl_tie728_s8_unaligned_max_pool2d_22c1
                                                                             : dl_tie728_s8_unaligned_max_pool2d_hwc1;
    }
#else
    c_impl_func = max_pool2d_hwc1<int8_t, int8_t, int32_t>;
#endif

    max_pool_shell<int8_t, int32_t>(args, i_impl_func, i_impl_func_sp, c_impl_func);
}

} // namespace base
} // namespace dl
