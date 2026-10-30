#pragma once

namespace dl {
namespace base {

struct ConvArgsType;

/**
 * @brief MAC16 version of conv2d_1x1_s16_fast.
 *
 * @return false if the layer is not supported; nothing has been written then.
 */
bool conv2d_1x1_s16_xtensa(
    const ConvArgsType &args, int filt_bytes, void *input_ptr, void *output_ptr, int height, int width);

/**
 * @brief MAC16 version of depthwise_conv2d_s16_fast.
 *
 * @return false if the layer is not supported; nothing has been written then.
 */
bool depthwise_conv2d_s16_xtensa(const ConvArgsType &args, void *input_ptr, void *output_ptr, int height, int width);

} // namespace base
} // namespace dl
