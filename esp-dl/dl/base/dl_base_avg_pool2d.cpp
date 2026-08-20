#include "dl_base_avg_pool2d.hpp"

#include "dl_base_activate_buffer.hpp"
#include "dl_base_activate_output.hpp"
#include "dl_base_isa.hpp"

namespace dl {
namespace base {

#if CONFIG_PIE_V1_BOOST
/**
 * @brief Sign-extend a 20-bit value (stored in the low 20 bits) to int32.
 */
inline int32_t sign_extend_20bit(uint32_t value)
{
    return (int32_t)(value << 12) >> 12;
}

/**
 * @brief Scale one 160-bit half of the TIE728 s8 QACC dump written by
 * dl_tie728_s8_avg_pool2d_hwc_sum (8 lanes x 20-bit, lane i at bit [20i, 20i+20)) into 8
 * quantized outputs. The dump layout per 64-byte channel block is:
 *   word 0..3  = QACC_L[127:0],  word 4 = QACC_L[159:128],  words 5..7 = gap (unused),
 *   word 8..11 = QACC_H[127:0],  word 12 = QACC_H[159:128], words 13..15 = unused.
 * so lanes 0..7 start at word 0 and lanes 8..15 at word 8.
 *
 * Scaling straight out of the packed dump keeps the unpacked sums in registers; going through
 * an int32 buffer instead costs a store and a load per channel. Eight lanes at a time is what
 * fits in the register file - unpacking all 16 makes the compiler spill.
 */
template <typename feature_t>
inline void scale_tie728_s8_qacc_8x20(const uint32_t *raw, feature_t *output, float scale)
{
    uint32_t w0 = raw[0], w1 = raw[1], w2 = raw[2], w3 = raw[3], w4 = raw[4];

    tool::truncate(output[0], tool::round(sign_extend_20bit(w0) * scale));
    tool::truncate(output[1], tool::round(sign_extend_20bit((w0 >> 20) | (w1 << 12)) * scale));
    tool::truncate(output[2], tool::round(sign_extend_20bit(w1 >> 8) * scale));
    tool::truncate(output[3], tool::round(sign_extend_20bit((w1 >> 28) | (w2 << 4)) * scale));
    tool::truncate(output[4], tool::round(sign_extend_20bit((w2 >> 16) | (w3 << 16)) * scale));
    tool::truncate(output[5], tool::round(sign_extend_20bit(w3 >> 4) * scale));
    tool::truncate(output[6], tool::round(sign_extend_20bit((w3 >> 24) | (w4 << 8)) * scale));
    tool::truncate(output[7], tool::round(sign_extend_20bit(w4 >> 12) * scale));
}

/**
 * @brief Scale the dump of the TIE728 s16 QACC written by dl_tie728_s16_avg_pool2d_hwc_sum
 * (8 lanes x 40-bit, lane i at bit [40i, 40i+40) of the 320-bit QACC) into 8 quantized
 * outputs. The dump layout (52 bytes used of a 13-word scratch) is the same as above.
 * The caller guarantees |sum| <= 2^31 - 1, so only the low 32 bits of each lane are read.
 */
template <typename feature_t>
inline void scale_tie728_s16_qacc_8x40(const uint32_t *raw, feature_t *output, float scale)
{
    uint32_t w0 = raw[0], w1 = raw[1], w2 = raw[2], w3 = raw[3], w4 = raw[4];

    tool::truncate(output[0], tool::round((int32_t)w0 * scale));
    tool::truncate(output[1], tool::round((int32_t)((w1 >> 8) | (w2 << 24)) * scale));
    tool::truncate(output[2], tool::round((int32_t)((w2 >> 16) | (w3 << 16)) * scale));
    tool::truncate(output[3], tool::round((int32_t)((w3 >> 24) | (w4 << 8)) * scale));

    uint32_t w8 = raw[8], w9 = raw[9], w10 = raw[10], w11 = raw[11], w12 = raw[12];

    tool::truncate(output[4], tool::round((int32_t)w8 * scale));
    tool::truncate(output[5], tool::round((int32_t)((w9 >> 8) | (w10 << 24)) * scale));
    tool::truncate(output[6], tool::round((int32_t)((w10 >> 16) | (w11 << 16)) * scale));
    tool::truncate(output[7], tool::round((int32_t)((w11 >> 24) | (w12 << 8)) * scale));
}
#endif

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

#if CONFIG_PIE_V2_BOOST
    if constexpr (std::is_same_v<feature_t, int8_t>) {
        if (args.input_channel % 16 == 0) {
            dl_esp32p4_s8_avg_pool2d_hwc_sum(buffer_ptr, input_ptr, &args);
        } else {
            avgpool2d_hwc_sum(buffer_ptr, input_ptr, args);
        }
    } else {
        avgpool2d_hwc_sum(buffer_ptr, input_ptr, args);
    }
#elif CONFIG_PIE_V1_BOOST
    if constexpr (std::is_same_v<feature_t, int8_t>) {
        // TIE728 s8 QACC accumulates 16 lanes x 20-bit (signed, saturating). Accumulating the raw
        // sum (multiplier = 1) is exact as long as |sum| <= 2^19 - 1, i.e. 128 * area <= 2^19 - 1.
        // EE.ST.QACC_*.128 forces its address to a 16-byte boundary, so a misaligned buffer_ptr
        // would silently dump the accumulator somewhere else.
        if (args.input_channel % 16 == 0 && args.filter_height * args.filter_width <= 4095 &&
            !((unsigned)input_ptr & 15) && !((unsigned)buffer_ptr & 15)) {
            dl_tie728_s8_avg_pool2d_hwc_sum(buffer_ptr, input_ptr, &args);
            int c_div_x = args.input_channel / 16;
            for (int block = 0; block < c_div_x; block++) {
                const uint32_t *raw = (const uint32_t *)(buffer_ptr + 16 * block);
                scale_tie728_s8_qacc_8x20(raw, output_ptr + 16 * block, scale);
                scale_tie728_s8_qacc_8x20(raw + 8, output_ptr + 16 * block + 8, scale);
            }
            // The buffer holds the raw QACC dump, and the C path accumulates into it, so it has
            // to go back to all zeros. dl_tie728_bzero clears 16 bytes per instruction.
            tool::set_zero(buffer_ptr, args.input_channel * sizeof(buffer_t));
            return;
        }
    } else if constexpr (std::is_same_v<feature_t, int16_t>) {
        // TIE728 s16 QACC accumulates 8 lanes x 40-bit (signed, saturating). Accumulating the raw
        // sum (multiplier = 1) is exact and fits int32 as long as 32768 * area <= 2^31 - 1.
        if (args.input_channel % 8 == 0 && args.filter_height * args.filter_width <= 65535 &&
            !((unsigned)input_ptr & 15)) {
            // The dump goes to the stack, so buffer_ptr keeps the zeros the C path expects.
            uint32_t scratch[13] __attribute__((aligned(16)));
            int c_div_x = args.input_channel / 8;
            for (int block = 0; block < c_div_x; block++) {
                dl_tie728_s16_avg_pool2d_hwc_sum(scratch, input_ptr + 8 * block, &args);
                scale_tie728_s16_qacc_8x40(scratch, output_ptr + 8 * block, scale);
            }
            return;
        }
    }
    avgpool2d_hwc_sum(buffer_ptr, input_ptr, args);
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
#if CONFIG_PIE_V1_BOOST
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
#if CONFIG_PIE_V2_BOOST
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
#elif CONFIG_PIE_V1_BOOST
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
#if CONFIG_PIE_V2_BOOST
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
