#include "dl_base_avg_pool2d.hpp"

#include "dl_base_activate_buffer.hpp"
#include "dl_base_activate_output.hpp"
#include "dl_base_isa.hpp"

namespace dl {
namespace base {

template <typename feature_t, typename buffer_t>
inline void avgpool2d_hwc_sum(buffer_t *buffer_ptr, feature_t *input_ptr, PoolArgsType<feature_t> &args)
{
    for (size_t filter_y = 0; filter_y < args.filter_height; filter_y++) // H
    {                                                                    //
        feature_t *input_yx = input_ptr;
        for (size_t filter_x = 0; filter_x < args.filter_width; filter_x++)   // W
        {                                                                     //
            for (size_t input_c = 0; input_c < args.input_channel; input_c++) // C
            {
                buffer_ptr[input_c] += (buffer_t)input_yx[input_c];
            }
            input_yx += args.input_x_offset;
        }
        input_ptr += args.input_y_offset;
    }
}

template <typename feature_t, typename buffer_t>
inline void avgpool2d_hwc(buffer_t *buffer_ptr,
                          feature_t *input_ptr,
                          feature_t *output_ptr,
                          PoolArgsType<feature_t> &args)
{
    float avg_pool_area_inv = 1.f / args.avg_pool_area;
    float scale = DL_SCALE(args.input_exponent) * avg_pool_area_inv * DL_RESCALE(args.output_exponent);

#if CONFIG_ESP32P4_BOOST
    if constexpr (std::is_same_v<feature_t, int8_t>) {
        if (args.input_channel % 16 == 0) {
            dl_esp32p4_s8_avg_pool2d_hwc_sum(buffer_ptr, input_ptr, &args);
        } else {
            avgpool2d_hwc_sum(buffer_ptr, input_ptr, args);
        }
    } else {
        avgpool2d_hwc_sum(buffer_ptr, input_ptr, args);
    }
#else
    avgpool2d_hwc_sum(buffer_ptr, input_ptr, args);
#endif

    for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
        tool::truncate(output_ptr[output_c], tool::round(buffer_ptr[output_c] * scale));
        buffer_ptr[output_c] = 0;
    }
}

template <>
inline void avgpool2d_hwc(float *buffer_ptr, float *input_ptr, float *output_ptr, PoolArgsType<float> &args)
{
    for (size_t filter_y = 0; filter_y < args.filter_height; filter_y++) // H
    {                                                                    //
        float *input_yx = input_ptr;
        for (size_t filter_x = 0; filter_x < args.filter_width; filter_x++)   // W
        {                                                                     //
            for (size_t input_c = 0; input_c < args.input_channel; input_c++) // C
            {
                buffer_ptr[input_c] += input_yx[input_c];
            }
            input_yx += args.input_x_offset;
        }
        input_ptr += args.input_y_offset;
    }

    float inv_avg_pool_area = 1.0 / args.avg_pool_area;
    for (size_t output_c = 0; output_c < args.output_channel; output_c++) {
        output_ptr[output_c] = buffer_ptr[output_c] * inv_avg_pool_area;
        buffer_ptr[output_c] = 0.0;
    }
}

inline void load_avg_pool2d_hwc1_s16(ImplFunc_t<int16_t, int16_t> &i_impl_func,
                                     ImplFunc_t<int16_t, int16_t> &i_impl_func_sp,
                                     avg_pool_c_impl_func_s16_t &c_impl_func,
                                     PoolArgsType<int16_t> &args)
{
#if CONFIG_ACCURATE_INFER
    c_impl_func = avgpool2d_hwc<int16_t, int32_t>;
#else
#if CONFIG_TIE728_BOOST
    if (args.input_x_offset % 8 == 0 && args.output_x_offset % 8 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        i_impl_func = dl_tie728_s16_avg_pool2d_hwc1;
        i_impl_func_sp = (args.filter_height == 2 && args.filter_width == 2) ? dl_tie728_s16_avg_pool2d_22c1
                                                                             : dl_tie728_s16_avg_pool2d_hwc1;
    } else {
        i_impl_func = dl_tie728_s16_unaligned_avg_pool2d_hwc1;
        i_impl_func_sp = (args.filter_height == 2 && args.filter_width == 2) ? dl_tie728_s16_unaligned_avg_pool2d_22c1
                                                                             : dl_tie728_s16_unaligned_avg_pool2d_hwc1;
    }
#else
    c_impl_func = avgpool2d_hwc<int16_t, int32_t>;
#endif
#endif
}

template <>
void avg_pool2d<int16_t>(void *args_ptr)
{
    PoolArgsType<int16_t> &args = *((PoolArgsType<int16_t> *)args_ptr);

    ImplFunc_t<int16_t, int16_t> i_impl_func;
    ImplFunc_t<int16_t, int16_t> i_impl_func_sp;
    avg_pool_c_impl_func_s16_t c_impl_func = NULL;

    load_avg_pool2d_hwc1_s16(i_impl_func, i_impl_func_sp, c_impl_func, args);
    avg_pool_shell<int16_t, int32_t>(args, i_impl_func, i_impl_func_sp, c_impl_func);
}

inline void load_avg_pool2d_hwc1_s8(ImplFunc_t<int8_t, int8_t> &i_impl_func,
                                    ImplFunc_t<int8_t, int8_t> &i_impl_func_sp,
                                    avg_pool_c_impl_func_s8_t &c_impl_func,
                                    PoolArgsType<int8_t> &args)
{
#if CONFIG_ACCURATE_INFER
    c_impl_func = avgpool2d_hwc<int8_t, int32_t>;
#else
#if CONFIG_ESP32P4_BOOST
    if (args.input_x_offset % 16 == 0 && args.output_x_offset % 16 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        i_impl_func = dl_esp32p4_s8_avg_pool2d_hwc1;
        i_impl_func_sp = (args.filter_height == 2 && args.filter_width == 2) ? dl_esp32p4_s8_avg_pool2d_22c1
                                                                             : dl_esp32p4_s8_avg_pool2d_hwc1;
    } else {
        i_impl_func = dl_esp32p4_s8_unaligned_avg_pool2d_hwc1;
        i_impl_func_sp = (args.filter_height == 2 && args.filter_width == 2) ? dl_esp32p4_s8_unaligned_avg_pool2d_22c1
                                                                             : dl_esp32p4_s8_unaligned_avg_pool2d_hwc1;
    }
#elif CONFIG_TIE728_BOOST
    if (args.input_x_offset % 16 == 0 && args.output_x_offset % 16 == 0 && !((unsigned)&args.input_element[0] & 15) &&
        !((unsigned)&args.output_element[0] & 15)) {
        i_impl_func = dl_tie728_s8_avg_pool2d_hwc1;
        i_impl_func_sp = (args.filter_height == 2 && args.filter_width == 2) ? dl_tie728_s8_avg_pool2d_22c1
                                                                             : dl_tie728_s8_avg_pool2d_hwc1;
    } else {
        i_impl_func = dl_tie728_s8_unaligned_avg_pool2d_hwc1;
        i_impl_func_sp = (args.filter_height == 2 && args.filter_width == 2) ? dl_tie728_s8_unaligned_avg_pool2d_22c1
                                                                             : dl_tie728_s8_unaligned_avg_pool2d_hwc1;
    }
#else
    c_impl_func = avgpool2d_hwc<int8_t, int32_t>;
#endif
#endif
}

template <>
void avg_pool2d<int8_t>(void *args_ptr)
{
    PoolArgsType<int8_t> &args = *((PoolArgsType<int8_t> *)args_ptr);

    ImplFunc_t<int8_t, int8_t> i_impl_func;
    ImplFunc_t<int8_t, int8_t> i_impl_func_sp;
    avg_pool_c_impl_func_s8_t c_impl_func = NULL;
#if CONFIG_ESP32P4_BOOST
    dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
#endif

    load_avg_pool2d_hwc1_s8(i_impl_func, i_impl_func_sp, c_impl_func, args);
    avg_pool_shell<int8_t, int32_t>(args, i_impl_func, i_impl_func_sp, c_impl_func);
}

template <>
void avg_pool2d<float>(void *args_ptr)
{
    PoolArgsType<float> &args = *((PoolArgsType<float> *)args_ptr);
    avg_pool_shell<float, float>(args, NULL, NULL, avgpool2d_hwc<float, float>);
}

} // namespace base
} // namespace dl
