#pragma once

#include "sdkconfig.h"

#if CONFIG_SPIRAM_SUPPORT || CONFIG_ESP32_SPIRAM_SUPPORT || CONFIG_ESP32S3_SPIRAM_SUPPORT
#define DL_SPIRAM_SUPPORT 1
#else
#define DL_SPIRAM_SUPPORT 0
#endif

#define DL_Q16_MIN (-32768)
#define DL_Q16_MAX (32767)
#define DL_Q8_MIN (-128)
#define DL_Q8_MAX (127)

#ifndef DL_MAX
#define DL_MAX(x, y) (((x) < (y)) ? (y) : (x))
#endif

#ifndef DL_MIN
#define DL_MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

#ifndef DL_CLIP
#define DL_CLIP(x, low, high) ((x) < (low)) ? (low) : (((x) > (high)) ? (high) : (x))
#endif

#ifndef DL_ABS
#define DL_ABS(x) ((x) < 0 ? (-(x)) : (x))
#endif

namespace dl
{
    typedef enum
    {
        RELU,
        LEAKY_RELU,
        PRELU,
    } relu_type_t;

    typedef enum
    {
        PADDING_VALID,
        PADDING_SAME,
        PADDING_SAME_MXNET
    } padding_type_t;
} // namespace dl