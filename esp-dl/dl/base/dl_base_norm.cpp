#include "dl_base_norm.hpp"
#include "dl_base_isa.hpp"
#include "dl_tool.hpp"
#include <cstring>

// The integer paths below replace soft-float code; with an FPU the float path is faster.
#if defined(__riscv) && !defined(__riscv_flen) && !CONFIG_ROUND_HALF_EVEN_ENABLED
#define DL_NORM_SOFT_FLOAT_INT_PATH 1
#else
#define DL_NORM_SOFT_FLOAT_INT_PATH 0
#endif

namespace dl {
namespace base {

#if DL_NORM_SOFT_FLOAT_INT_PATH
// Integer emulation of the float reference below for targets without an FPU. Each step is
// bit-exact with IEEE single precision (round to nearest even), so the results equal
//   truncate(round_half_up(float(x) * (rms * scale[j]))).
// esp-dl is built with -ffast-math, which evaluates the reference loop in exactly that order
// (rms * scale[j] first); the emulation has to follow it to give the same results.
// Values whose intermediate exponents leave the normal float range use the float path.
static const uint8_t rms_bit_len_table[256] = {
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
};

static inline uint32_t rms_float_bits(float f)
{
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

// Rounds the exact product of two 24-bit significands in [2^23, 2^24) back to 24 bits.
static inline uint32_t rms_mul24(uint32_t a, uint32_t b, int &e)
{
    const uint64_t p = (uint64_t)a * b;
    const uint32_t lo = (uint32_t)p;
    const uint32_t t = ((uint32_t)(p >> 32) << 16) | (lo >> 16);
    const uint32_t sticky = (lo & 0xffff) != 0;
    const int sh = (t >> 31) ? 8 : 7;
    uint32_t m = t >> sh;
    const uint32_t rem = t & ((1u << sh) - 1);
    const uint32_t half = 1u << (sh - 1);
    if (rem > half || (rem == half && (sticky | (m & 1)))) {
        m++;
    }
    e += 16 + sh;
    if (m == (1u << 24)) {
        m >>= 1;
        e++;
    }
    return m;
}

// round_half_up(v) for v = +-m * 2^e with m in [2^23, 2^24), saturated beyond the int16 range.
static inline int32_t rms_round_half_up(uint32_t m, int e, bool neg)
{
    if (e >= -8) {
        return neg ? INT16_MIN : INT16_MAX;
    }
    if (e <= -26) {
        return 0;
    }
    // floorf(v + 0.5f): the sum is (+-m + 2^(K-1)) * 2^-K and is rounded to 24 bits.
    const int k = -e;
    int32_t s = (neg ? -(int32_t)m : (int32_t)m) + (1 << (k - 1));
    // From 2^24 on (s < 2^25) the float sum drops bit 0 with round to nearest even. Bit 0 itself
    // is shifted out below, only the carry of rounding up matters.
    if (s >= (1 << 24)) {
        s += s & (s >> 1) & 1;
    }
    return s >> k;
}

// float(x) * r for x != 0, |x| <= 2^15, with r = mr * 2^er.
static inline uint32_t rms_mul_int(uint32_t ax, uint32_t mr, int er, int &e)
{
    const int bl = ax >> 8 ? 8 + rms_bit_len_table[ax >> 8] : rms_bit_len_table[ax];
    e = bl - 24 + er;
    return rms_mul24(ax << (24 - bl), mr, e);
}

template <typename T>
static bool scale_round_int_emulated(T *output, const T *input, float scale, int n)
{
    const uint32_t rb = rms_float_bits(scale);
    const int rbe = (rb >> 23) & 0xff;
    const int er = rbe - 150;
    if (rbe == 0 || rbe == 255 || er < -148 || er + 17 > 103) {
        return false;
    }
    const uint32_t mr = (rb & 0x7fffff) | 0x800000;
    const uint32_t rneg = rb >> 31;
    for (int j = 0; j < n; j++) {
        const int x = input[j];
        if (x == 0) {
            output[j] = 0;
            continue;
        }
        int e;
        const uint32_t m = rms_mul_int(x < 0 ? -x : x, mr, er, e);
        tool::truncate(output[j], rms_round_half_up(m, e, ((uint32_t)(x < 0) ^ rneg) != 0));
    }
    return true;
}

template <typename T>
static bool rms_norm_int_emulated(T *output, const T *input, const float *scale, float rms, int n)
{
    const uint32_t rb = rms_float_bits(rms);
    const int rbe = (rb >> 23) & 0xff;
    if (rbe == 0 || rbe == 255) {
        return false;
    }
    const uint32_t mr = (rb & 0x7fffff) | 0x800000;
    const int er = rbe - 150;
    // rms * scale[j] has exponent in [er + es + 23, er + es + 25], float(x) times it adds up to 17.
    for (int j = 0; j < n; j++) {
        const int es = (int)((rms_float_bits(scale[j]) >> 23) & 0xff) - 150;
        if (es == -150 || es == 105 || er + es + 23 < -148 || er + es + 25 + 17 > 103) {
            return false;
        }
    }
    const uint32_t rneg = rb >> 31;
    for (int j = 0; j < n; j++) {
        const int x = input[j];
        if (x == 0) {
            output[j] = 0;
            continue;
        }
        const uint32_t sb = rms_float_bits(scale[j]);
        int ec = er + (int)((sb >> 23) & 0xff) - 150;
        const uint32_t mc = rms_mul24(mr, (sb & 0x7fffff) | 0x800000, ec);
        int e;
        const uint32_t m = rms_mul_int(x < 0 ? -x : x, mc, ec, e);
        tool::truncate(output[j], rms_round_half_up(m, e, ((uint32_t)(x < 0) ^ rneg ^ (sb >> 31)) != 0));
    }
    return true;
}
#endif

void rms_norm(int8_t *output, int8_t *input, float *scale, float *rms, int n)
{
#if CONFIG_PIE_V1_BOOST
    dl_tie728_rmsnorm_s8(output, input, scale, rms, n);
#elif CONFIG_PIE_V2_BOOST
    dl_esp32p4_rmsnorm_s8(output, input, scale, rms, n);
#else
#if DL_NORM_SOFT_FLOAT_INT_PATH
    if (rms_norm_int_emulated(output, input, scale, *rms, n)) {
        return;
    }
#endif
    float inv_rms = *rms;
    for (int j = 0; j < n; j++) {
        float result = input[j] * inv_rms * scale[j];
        tool::truncate(output[j], tool::round(result));
    }
#endif
}

void rms_norm(int16_t *output, int16_t *input, float *scale, float *rms, int n)
{
#if CONFIG_PIE_V1_BOOST
    dl_tie728_rmsnorm_s16(output, input, scale, rms, n);
#elif CONFIG_PIE_V2_BOOST
    dl_esp32p4_rmsnorm_s16(output, input, scale, rms, n);
#else
#if DL_NORM_SOFT_FLOAT_INT_PATH
    if (rms_norm_int_emulated(output, input, scale, *rms, n)) {
        return;
    }
#endif
    float inv_rms = *rms;
    for (int j = 0; j < n; j++) {
        float result = input[j] * inv_rms * scale[j];
        tool::truncate(output[j], tool::round(result));
    }
#endif
}

template <typename T>
static void scale_round_impl(T *output, const T *input, float scale, int n)
{
#if DL_NORM_SOFT_FLOAT_INT_PATH
    if (scale_round_int_emulated(output, input, scale, n)) {
        return;
    }
#endif
    for (int j = 0; j < n; j++) {
        float result = input[j] * scale;
        tool::truncate(output[j], tool::round(result));
    }
}

void scale_round(int8_t *output, const int8_t *input, float scale, int n)
{
    scale_round_impl(output, input, scale, n);
}

void scale_round(int16_t *output, const int16_t *input, float scale, int n)
{
    scale_round_impl(output, input, scale, n);
}

} // namespace base
} // namespace dl
