#pragma once

// Compile-time names for the C conv tails. Same contract as dl_c_conv.hpp:
// dl_kernel.inc casts these names to dl_kernel_erased_t. Two names may point
// at one instantiation (the s8 per-channel PReLU tail and its depthwise twin).

#include "dl_define_private.hpp"
#include <cstdint>

namespace dl {
namespace base {

struct ConvArgsType;

template <typename feature_t, typename bias_t, typename buffer_t>
void buffer_bias_linear(feature_t *output, buffer_t *buffer, const ConvArgsType &args);

template <typename feature_t, typename bias_t, typename buffer_t>
void buffer_bias_relu(feature_t *output, buffer_t *buffer, const ConvArgsType &args);

template <typename feature_t, typename bias_t, typename buffer_t>
void buffer_bias_leakyrelu(feature_t *output, buffer_t *buffer, const ConvArgsType &args);

template <typename feature_t, typename bias_t, typename buffer_t>
void buffer_bias_prelu(feature_t *output, buffer_t *buffer, const ConvArgsType &args);

template <typename feature_t, typename buffer_t>
void buffer_0000_linear(feature_t *output, buffer_t *buffer, const ConvArgsType &args);

template <typename feature_t, typename buffer_t>
void buffer_0000_relu(feature_t *output, buffer_t *buffer, const ConvArgsType &args);

template <typename feature_t, typename buffer_t>
void buffer_0000_leakyrelu(feature_t *output, buffer_t *buffer, const ConvArgsType &args);

template <typename feature_t, typename buffer_t>
void buffer_0000_prelu(feature_t *output, buffer_t *buffer, const ConvArgsType &args);

} // namespace base
} // namespace dl

inline constexpr auto dl_c_s16_tail_linear = &dl::base::buffer_0000_linear<int16_t, DL_S16_BUFFER_TYPE>;
inline constexpr auto dl_c_s16_tail_relu = &dl::base::buffer_0000_relu<int16_t, DL_S16_BUFFER_TYPE>;
inline constexpr auto dl_c_s16_tail_bias_linear =
    &dl::base::buffer_bias_linear<int16_t, DL_S16_BUFFER_TYPE, DL_S16_BUFFER_TYPE>;
inline constexpr auto dl_c_s16_tail_bias_relu =
    &dl::base::buffer_bias_relu<int16_t, DL_S16_BUFFER_TYPE, DL_S16_BUFFER_TYPE>;

inline constexpr auto dl_c_s8_tail_linear = &dl::base::buffer_0000_linear<int8_t, int32_t>;
inline constexpr auto dl_c_s8_tail_relu = &dl::base::buffer_0000_relu<int8_t, int32_t>;
inline constexpr auto dl_c_s8_tail_leakyrelu = &dl::base::buffer_0000_leakyrelu<int8_t, int32_t>;
inline constexpr auto dl_c_s8_tail_prelu = &dl::base::buffer_0000_prelu<int8_t, int32_t>;
inline constexpr auto dl_c_s8_tail_bias_linear = &dl::base::buffer_bias_linear<int8_t, int32_t, int32_t>;
inline constexpr auto dl_c_s8_tail_bias_relu = &dl::base::buffer_bias_relu<int8_t, int32_t, int32_t>;
inline constexpr auto dl_c_s8_tail_bias_per_ch_linear = &dl::base::buffer_bias_linear<int8_t, int16_t, int32_t>;
inline constexpr auto dl_c_s8_tail_bias_per_ch_relu = &dl::base::buffer_bias_relu<int8_t, int16_t, int32_t>;
inline constexpr auto dl_c_s8_tail_bias_per_ch_leakyrelu = &dl::base::buffer_bias_leakyrelu<int8_t, int16_t, int32_t>;
inline constexpr auto dl_c_s8_tail_bias_per_ch_prelu = &dl::base::buffer_bias_prelu<int8_t, int8_t, int32_t>;
inline constexpr auto dl_c_s8_dw_tail_bias_per_ch_prelu = &dl::base::buffer_bias_prelu<int8_t, int8_t, int32_t>;
