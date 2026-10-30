#include "dl_base.hpp"
#include "dl_base_elemwise.hpp"
#include "dl_base_isa.hpp"
#include <cstring>

// The integer paths below replace soft-float code; with an FPU the float path is faster.
#if defined(__riscv) && !defined(__riscv_flen) && !CONFIG_ROUND_HALF_EVEN_ENABLED
#define DL_MUL_SOFT_FLOAT_INT_PATH 1
#else
#define DL_MUL_SOFT_FLOAT_INT_PATH 0
#endif

namespace dl {
namespace base {

#if DL_MUL_SOFT_FLOAT_INT_PATH
// Integer versions of round_half_up(float(a * b) * scale) for a power-of-two scale = 2^-shift,
// bit-exact with IEEE single precision: both the int->float conversion and the "+ 0.5f" round
// the exact integer value to 24 significant bits (nearest even).
static inline bool mul_pow2_shift(float scale, int &shift)
{
    uint32_t b;
    memcpy(&b, &scale, sizeof(b));
    const int be = b >> 23;
    if ((b & 0x807fffff) || be == 0 || be == 255) {
        return false;
    }
    shift = 127 - be;
    return shift >= -30 && shift <= 30;
}

static inline int32_t mul_rne24(int32_t v)
{
    uint32_t a = v < 0 ? -(uint32_t)v : (uint32_t)v;
    if (a < (1u << 24)) {
        return v;
    }
    const uint32_t h = a >> 24;
    const int d =
        h >= 16 ? (h >= 64 ? (h >= 128 ? 8 : 7) : (h >= 32 ? 6 : 5)) : (h >= 4 ? (h >= 8 ? 4 : 3) : (h >= 2 ? 2 : 1));
    const uint32_t half = 1u << (d - 1);
    uint32_t q = a >> d;
    const uint32_t rem = a & ((half << 1) - 1);
    q += rem > half || (rem == half && (q & 1));
    a = q << d;
    return v < 0 ? -(int32_t)a : (int32_t)a;
}

// t is the exact integer product, |t| <= 2^30.
static inline int32_t mul_requant_pow2(int32_t t, int shift)
{
    if (shift > 0) {
        return mul_rne24(mul_rne24(t) + (1 << (shift - 1))) >> shift;
    }
    // Values that need rounding here are far outside the int16 range and saturate either way.
    const int64_t v = (int64_t)t << -shift;
    return (int32_t)DL_CLIP(v, INT32_MIN, INT32_MAX);
}
#endif

// input0_ptr:vector, input1_ptr:scalar
template <typename feature_t>
void c_impl_mul_n_1(feature_t *output_ptr, feature_t *input0_ptr, feature_t *input1_ptr, void *args)
{
    elemwiseArgsType<feature_t> *elem_args = static_cast<elemwiseArgsType<feature_t> *>(args);
    int32_t length = elem_args->output_d0;
#if DL_MUL_SOFT_FLOAT_INT_PATH
    int shift;
    if (mul_pow2_shift(elem_args->output_rescale, shift)) {
        const int32_t b = input1_ptr[0];
        for (int i = 0; i < length; i++) {
            tool::truncate<int32_t>(output_ptr[i], mul_requant_pow2(input0_ptr[i] * b, shift));
        }
        return;
    }
#endif
    float scale = input1_ptr[0] * elem_args->output_rescale;
    for (int i = 0; i < length; i++) {
        float out = input0_ptr[i] * scale;
        tool::truncate<int32_t>(output_ptr[i], tool::round(out));
    }
}

// input0_ptr:scalar, input1_ptr:vector
template <typename feature_t>
void c_impl_mul_1_n(feature_t *output_ptr, feature_t *input0_ptr, feature_t *input1_ptr, void *args)
{
    elemwiseArgsType<feature_t> *elem_args = static_cast<elemwiseArgsType<feature_t> *>(args);
    int32_t length = elem_args->output_d0;
#if DL_MUL_SOFT_FLOAT_INT_PATH
    int shift;
    if (mul_pow2_shift(elem_args->output_rescale, shift)) {
        const int32_t a = input0_ptr[0];
        for (int i = 0; i < length; i++) {
            tool::truncate<int32_t>(output_ptr[i], mul_requant_pow2(a * input1_ptr[i], shift));
        }
        return;
    }
#endif
    float scale = input0_ptr[0] * elem_args->output_rescale;
    for (int i = 0; i < length; i++) {
        float out = input1_ptr[i] * scale;
        tool::truncate<int32_t>(output_ptr[i], tool::round(out));
    }
}

// input0_ptr:vector, input1_ptr:vector
template <typename feature_t>
void c_impl_mul_n_n(feature_t *output_ptr, feature_t *input0_ptr, feature_t *input1_ptr, void *args)
{
    elemwiseArgsType<feature_t> *elem_args = static_cast<elemwiseArgsType<feature_t> *>(args);
    int32_t length = elem_args->output_d0;
#if DL_MUL_SOFT_FLOAT_INT_PATH
    int shift;
    if (mul_pow2_shift(elem_args->output_rescale, shift)) {
        for (int i = 0; i < length; i++) {
            tool::truncate<int32_t>(output_ptr[i], mul_requant_pow2(input0_ptr[i] * input1_ptr[i], shift));
        }
        return;
    }
#endif
    float scale = elem_args->output_rescale;
    for (int i = 0; i < length; i++) {
        int temp = input0_ptr[i] * input1_ptr[i];
        float out = temp * scale;
        tool::truncate<int32_t>(output_ptr[i], tool::round(out));
    }
}

// input0_ptr:vector, input1_ptr:scalar
template <>
void c_impl_mul_n_1<float>(float *output_ptr, float *input0_ptr, float *input1_ptr, void *args)
{
    elemwiseArgsType<float> *elem_args = static_cast<elemwiseArgsType<float> *>(args);
    int32_t length = elem_args->output_d0;
    float input1 = input1_ptr[0];
    for (int i = 0; i < length; i++) {
        output_ptr[i] = input0_ptr[i] * input1;
    }
}

// input0_ptr:scalar, input1_ptr:vector
template <>
void c_impl_mul_1_n<float>(float *output_ptr, float *input0_ptr, float *input1_ptr, void *args)
{
    elemwiseArgsType<float> *elem_args = static_cast<elemwiseArgsType<float> *>(args);
    int32_t length = elem_args->output_d0;
    float input0 = input0_ptr[0];
    for (int i = 0; i < length; i++) {
        output_ptr[i] = input1_ptr[i] * input0;
    }
}

// input0_ptr:vector, input1_ptr:vector
template <>
void c_impl_mul_n_n<float>(float *output_ptr, float *input0_ptr, float *input1_ptr, void *args)
{
    elemwiseArgsType<float> *elem_args = static_cast<elemwiseArgsType<float> *>(args);
    int32_t length = elem_args->output_d0;
    for (int i = 0; i < length; i++) {
        output_ptr[i] = input0_ptr[i] * input1_ptr[i];
    }
}

void elemwise_mul(elemwiseArgsType<int8_t> *args)
{
    int ilen = 16 / sizeof(int8_t);
    ImplFunc_t<int8_t, int8_t, int8_t> elemwise_func = c_impl_mul_n_n<int8_t>; // default impl

    if (args->output_d0 >= ilen) {
#if CONFIG_PIE_V2_BOOST
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);

        if (args->input0_d0 % ilen == 0 && args->input1_d0 % ilen == 0) {
            elemwise_func = dl_esp32p4_s8_mul_w1_16_w2_16;
        } else if (args->input1_d0 == 1) {
            if (args->input0_d0 % ilen == 0) {
                elemwise_func = dl_esp32p4_s8_mul_w1_16_w2_1;
            } else {
                elemwise_func = dl_esp32p4_s8_mul_w1_16_w2_1_unaligned;
            }
        } else if (args->input0_d0 == 1) {
            if (args->input1_d0 % ilen == 0) {
                elemwise_func = dl_esp32p4_s8_mul_w1_1_w2_16;
            } else {
                elemwise_func = dl_esp32p4_s8_mul_w1_1_w2_16_unaligned;
            }
        } else {
            elemwise_func = dl_esp32p4_s8_mul_w1_16_w2_16_unaligned;
        }
#elif CONFIG_PIE_V1_BOOST
        if (args->input0_d0 % ilen == 0 && args->input1_d0 % ilen == 0) {
            elemwise_func = dl_tie728_s8_mul_w1_16_w2_16;
        } else if (args->input1_d0 == 1) {
            if (args->input0_d0 % ilen == 0) {
                elemwise_func = dl_tie728_s8_mul_w1_16_w2_1;
            } else {
                elemwise_func = dl_tie728_s8_mul_w1_16_w2_1_unaligned;
            }
        } else if (args->input0_d0 == 1) {
            if (args->input1_d0 % ilen == 0) {
                elemwise_func = dl_tie728_s8_mul_w1_1_w2_16;
            } else {
                elemwise_func = dl_tie728_s8_mul_w1_1_w2_16_unaligned;
            }
        } else {
            elemwise_func = dl_tie728_s8_mul_w1_16_w2_16_unaligned;
        }
#else
        args->output_rescale = args->input0_scale * args->input1_scale * args->output_rescale;
        if (args->input1_d0 == 1) {
            elemwise_func = c_impl_mul_n_1<int8_t>;
        } else if (args->input0_d0 == 1) {
            elemwise_func = c_impl_mul_1_n<int8_t>;
        }
#endif
    } else {
        args->output_rescale = args->input0_scale * args->input1_scale * args->output_rescale;
        if (args->input1_d0 == 1) {
            elemwise_func = c_impl_mul_n_1<int8_t>;
        } else if (args->input0_d0 == 1) {
            elemwise_func = c_impl_mul_1_n<int8_t>;
        }
    }

    switch (args->dims) {
    case 1:
        elemwise_loop_1d(args, elemwise_func);
        break;
    case 2:
        elemwise_loop_2d(args, elemwise_func);
        break;
    case 3:
        elemwise_loop_3d(args, elemwise_func);
        break;
    case 4:
        elemwise_loop_4d(args, elemwise_func);
        break;
    default:
        break;
    }
}

void elemwise_mul(elemwiseArgsType<int16_t> *args)
{
    int ilen = 16 / sizeof(int16_t);
    ImplFunc_t<int16_t, int16_t, int16_t> elemwise_func = c_impl_mul_n_n<int16_t>;

    if (args->output_d0 >= ilen) {
#if CONFIG_PIE_V2_BOOST
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);

        if (args->input0_d0 % ilen == 0 && args->input1_d0 % ilen == 0) {
            elemwise_func = dl_esp32p4_s16_mul_w1_8_w2_8;
        } else if (args->input1_d0 == 1) {
            if (args->input0_d0 % ilen == 0) {
                elemwise_func = dl_esp32p4_s16_mul_w1_8_w2_1;
            } else {
                elemwise_func = dl_esp32p4_s16_mul_w1_8_w2_1_unaligned;
            }
        } else if (args->input0_d0 == 1) {
            if (args->input1_d0 % ilen == 0) {
                elemwise_func = dl_esp32p4_s16_mul_w1_1_w2_8;
            } else {
                elemwise_func = dl_esp32p4_s16_mul_w1_1_w2_8_unaligned;
            }
        } else {
            elemwise_func = dl_esp32p4_s16_mul_w1_8_w2_8_unaligned;
        }
#elif CONFIG_PIE_V1_BOOST
        if (args->input0_d0 % ilen == 0 && args->input1_d0 % ilen == 0) {
            elemwise_func = dl_tie728_s16_mul_w1_8_w2_8;
        } else if (args->input1_d0 == 1) {
            if (args->input0_d0 % ilen == 0) {
                elemwise_func = dl_tie728_s16_mul_w1_8_w2_1;
            } else {
                elemwise_func = dl_tie728_s16_mul_w1_8_w2_1_unaligned;
            }
        } else if (args->input0_d0 == 1) {
            if (args->input1_d0 % ilen == 0) {
                elemwise_func = dl_tie728_s16_mul_w1_1_w2_8;
            } else {
                elemwise_func = dl_tie728_s16_mul_w1_1_w2_8_unaligned;
            }
        } else {
            elemwise_func = dl_tie728_s16_mul_w1_8_w2_8_unaligned;
        }
#else
        args->output_rescale = args->input0_scale * args->input1_scale * args->output_rescale;
        if (args->input1_d0 == 1) {
            elemwise_func = c_impl_mul_n_1<int16_t>;
        } else if (args->input0_d0 == 1) {
            elemwise_func = c_impl_mul_1_n<int16_t>;
        }
#endif
    } else {
        args->output_rescale = args->input0_scale * args->input1_scale * args->output_rescale;
        if (args->input1_d0 == 1) {
            elemwise_func = c_impl_mul_n_1<int16_t>;
        } else if (args->input0_d0 == 1) {
            elemwise_func = c_impl_mul_1_n<int16_t>;
        }
    }

    switch (args->dims) {
    case 1:
        elemwise_loop_1d(args, elemwise_func);
        break;
    case 2:
        elemwise_loop_2d(args, elemwise_func);
        break;
    case 3:
        elemwise_loop_3d(args, elemwise_func);
        break;
    case 4:
        elemwise_loop_4d(args, elemwise_func);
        break;
    default:
        break;
    }
}

void elemwise_mul(elemwiseArgsType<float> *args)
{
    ImplFunc_t<float, float, float> elemwise_func = c_impl_mul_n_n<float>;

    if (args->input1_d0 == 1) {
        elemwise_func = c_impl_mul_n_1<float>;
    } else if (args->input0_d0 == 1) {
        elemwise_func = c_impl_mul_1_n<float>;
    }

    switch (args->dims) {
    case 1:
        elemwise_loop_1d(args, elemwise_func);
        break;
    case 2:
        elemwise_loop_2d(args, elemwise_func);
        break;
    case 3:
        elemwise_loop_3d(args, elemwise_func);
        break;
    case 4:
        elemwise_loop_4d(args, elemwise_func);
        break;
    default:
        break;
    }
}

} // namespace base
} // namespace dl
