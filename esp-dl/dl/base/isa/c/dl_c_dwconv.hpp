#pragma once

// Compile-time names for the C depthwise conv kernels. Same contract as
// dl_c_conv.hpp: dl_kernel.inc casts these names to dl_kernel_erased_t.

#include "dl_define_private.hpp"
#include <cstdint>

namespace dl {
namespace base {

struct ConvArgsType;

template <typename feature_t, typename buffer_t>
void depthwise_conv2d_33c1(buffer_t *buffer, feature_t *input, const ConvArgsType &args);

template <typename feature_t, typename buffer_t>
void depthwise_conv2d_hwc1(buffer_t *buffer, feature_t *input, const ConvArgsType &args);

/**
 * @brief Whole-layer int16 depthwise conv without padding, including bias, requantization and
 * Linear/ReLU.
 *
 * @return false if the layer is not supported; nothing has been written then.
 */
bool depthwise_conv2d_s16_fast(const ConvArgsType &args, void *input_ptr, void *output_ptr, int height, int width);

/**
 * @brief Whole-layer int8 depthwise conv without padding. The MACs run here; bias,
 * requantization and activation are left to tail, called once per output pixel.
 *
 * @return false if the layer is not supported; nothing has been written then.
 */
bool depthwise_conv2d_s8_fast(const ConvArgsType &args,
                              void (*tail)(void *output, void *buffer, const ConvArgsType &args),
                              void *input_ptr,
                              void *output_ptr,
                              int height,
                              int width);

} // namespace base
} // namespace dl

inline constexpr auto dl_c_s16_depthwise_conv2d_33c1 = &dl::base::depthwise_conv2d_33c1<int16_t, DL_S16_BUFFER_TYPE>;
inline constexpr auto dl_c_s16_depthwise_conv2d_hwc1 = &dl::base::depthwise_conv2d_hwc1<int16_t, DL_S16_BUFFER_TYPE>;

inline constexpr auto dl_c_s8_depthwise_conv2d_33c1 = &dl::base::depthwise_conv2d_33c1<int8_t, int32_t>;
inline constexpr auto dl_c_s8_depthwise_conv2d_hwc1 = &dl::base::depthwise_conv2d_hwc1<int8_t, int32_t>;
