#pragma once

/**
 * @brief Private Attention softmax helpers for dl_base_attention.cpp and white-box tests.
 *
 * Internal linkage; see dl_base_attention_common.hpp.
 */

#include "dl_base.hpp"
#include "dl_base_attention_common.hpp"
#include "dl_base_isa.hpp"
#include "dl_tool.hpp"
#include "esp_log.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dl {
namespace base {
namespace {

constexpr int INT8_EXP_SHIFT_MAX = 21;     /*!< largest index shift used by the int8 exp LUT */
constexpr int PV_S16_BITS = 29;            /*!< int16 P·V fixed-point scale */
constexpr int PV_S16_LO_BITS = 15;         /*!< low-half width of the split int16 exp score */
constexpr int INT16_QACC_QK_D32_SHIFT = 5; /*!< must match QK16_D32_SHIFT in dl_esp32p4_s16_attention.S */

/**
 * @brief Build a LUT for exp(-diff * logit_scale), where diff = row_max_dot - dot >= 0.
 *
 * Runtime lookup: index = min(diff, lut.sat) >> lut.shift.
 * Stored value: table[index] = round(127 * exp(-index * 2^shift * logit_scale)).
 */
inline bool build_int8_exp_lut(AttentionInt8ExpLut &lut, float logit_scale)
{
    if (lut.valid && lut.built_for == logit_scale) {
        return true;
    }
    lut.valid = false;
    if (!is_positive_normal(logit_scale)) {
        return false;
    }

    // Group dot-product differences by a power of two so lookup indices use a cheap right shift.
    constexpr float target_lut_step = 16.0f / 255.0f;
    const float target_integer_stride = target_lut_step / logit_scale;
    int index_shift = 0;
    if (target_integer_stride >= 1.0f) {
        while (index_shift < INT8_EXP_SHIFT_MAX &&
               static_cast<float>(1 << (index_shift + 1)) <= target_integer_stride) {
            ++index_shift;
        }
    }

    // Store exp(-difference * logit_scale) as a quantized exp score in [0, 127].
    const float real_step_per_entry = static_cast<float>(1 << index_shift) * logit_scale;
    for (int lut_index = 0; lut_index < 512; ++lut_index) {
        const float exp_value =
            attention_float_exp_lut_data.lookup(-static_cast<float>(lut_index) * real_step_per_entry);
        int quantized_exp_score = static_cast<int>(127.0f * exp_value + 0.5f);
        if (quantized_exp_score < 0)
            quantized_exp_score = 0;
        if (quantized_exp_score > 127)
            quantized_exp_score = 127;
        lut.table[lut_index] = static_cast<int32_t>(quantized_exp_score);
    }
    lut.shift = index_shift;
    lut.sat = 511 << index_shift;
    lut.built_for = logit_scale;
    lut.valid = true;
    return true;
}

#if CONFIG_PIE_V2_BOOST
/**
 * @brief Only the batched int8 Q·K (PIE v2) produces dot rows. masked_slots: the row may hold
 * QK_DOTS_MASKED entries (causal or attn_mask).
 */
template <bool masked_slots>
inline void attention_softmax_quant_row_s8(const int32_t *dots,
                                           int tile_n,
                                           int n_dot,
                                           float logit_scale,
                                           int32_t *dot_row_max,
                                           float *acc,
                                           int D,
                                           int8_t *quantized_exp_scores,
                                           float *row_sum,
                                           float *exp_scale,
                                           float v_scale,
                                           int32_t tile_max,
                                           bool defer_pv_scale,
                                           const AttentionInt8ExpLut &int8_exp_lut,
                                           int32_t *exp_scratch)
{
    const int32_t previous_max = *dot_row_max;
    const int32_t new_max = (tile_max > previous_max) ? tile_max : previous_max;
    float sum = *row_sum;
    if (previous_max != INT32_MIN && new_max != previous_max) {
        const float rescale =
            attention_float_exp_lut_data.lookup(static_cast<float>(previous_max - new_max) * logit_scale);
        sum *= rescale;
        if (defer_pv_scale) {
            int32_t *raw_acc = reinterpret_cast<int32_t *>(acc);
            for (int d = 0; d < D; ++d) {
                raw_acc[d] = static_cast<int32_t>(static_cast<float>(raw_acc[d]) * rescale);
            }
        } else if constexpr (masked_slots) {
            for (int d = 0; d < D; ++d) {
                acc[d] *= rescale;
            }
        } else {
            int d = 0;
            for (; d + 7 < D; d += 8) {
                acc[d] *= rescale;
                acc[d + 1] *= rescale;
                acc[d + 2] *= rescale;
                acc[d + 3] *= rescale;
                acc[d + 4] *= rescale;
                acc[d + 5] *= rescale;
                acc[d + 6] *= rescale;
                acc[d + 7] *= rescale;
            }
            for (; d < D; ++d) {
                acc[d] *= rescale;
            }
        }
    }

    // A masked sentinel may produce any provisional qv in [0, 127]. The
    // cleanup below removes that exact value from both e_sum and the P row.
    int32_t e_sum = dl_esp32p4_s8_attention_exp_lut_shift(quantized_exp_scores,
                                                          dots,
                                                          tile_n,
                                                          new_max,
                                                          int8_exp_lut.sat,
                                                          int8_exp_lut.shift,
                                                          int8_exp_lut.table,
                                                          exp_scratch);

    if constexpr (masked_slots) {
        for (int t2 = 0; t2 < tile_n; ++t2) {
            if (dots[t2] == QK_DOTS_MASKED) {
                e_sum -= quantized_exp_scores[t2];
                quantized_exp_scores[t2] = 0;
            }
        }
    }

    *dot_row_max = new_max;
    *row_sum = sum + static_cast<float>(e_sum) * (1.0f / 127.0f);
    *exp_scale = v_scale * (1.0f / 127.0f);
    if (tile_n < n_dot) {
        for (int t2 = tile_n; t2 < n_dot; ++t2) {
            quantized_exp_scores[t2] = 0;
        }
    }
}
#endif // CONFIG_PIE_V2_BOOST

/**
 * @brief dtype-specific exp used by the fp32 softmax path.
 */
template <dtype_t qkv_dtype>
inline float softmax_expf(float x)
{
    if constexpr (qkv_dtype == DATA_TYPE_INT16) {
        return attention_float_exp_lut().lookup_precise(x);
    } else if constexpr (qkv_dtype == DATA_TYPE_INT8) {
        // The PPQ int8 device emulator reproduces lookup() bit for bit.
        return attention_float_exp_lut().lookup(x);
    } else {
        return std::exp(x);
    }
}

/**
 * @brief fixed_one * lookup_precise(-r) for r in [0, FLOAT_EXP_PRECISE_MAX_R], branch-free.
 * fixed_one is a power of two, so scaling the polynomial coefficients by it is exact.
 */
template <int fixed_one = 1>
inline float exp_neg_precise_clamped(const AttentionFloatExpLut &lut, float r)
{
    constexpr float one = static_cast<float>(fixed_one);
    const int idx = static_cast<int>(r * FLOAT_EXP_LUT_INVERSE_STEP);
    const float d = r - static_cast<float>(idx) * FLOAT_EXP_LUT_STEP;
    const float p = one - d * (one - d * (one * 0.5f - d * (one * (1.0f / 6.0f))));
    return lut.table[idx] * p;
}

/**
 * @brief Clamp m - s into the precise LUT domain [0, FLOAT_EXP_PRECISE_MAX_R].
 */
inline float exp_neg_clamp_arg(float m, float s)
{
    return std::fmin(m - s, FLOAT_EXP_PRECISE_MAX_R);
}

/**
 * @brief e[t] = attention_float_exp_lut().lookup_precise(s[t] - m) and returns l + sum(e); requires
 * m >= s[t]. -inf scores clamp to FLOAT_EXP_PRECISE_MAX_R and come out 0.
 *
 * Four independent exp chains per iteration hide the FPU latency of each one.
 */
inline float softmax_exp_row_precise(float *__restrict e, const float *__restrict s, int n, float m, float l)
{
    if (is_neg_inf(m)) {
        for (int t = 0; t < n; ++t) {
            e[t] = 0.f;
        }
        return l;
    }
    const AttentionFloatExpLut &lut = attention_float_exp_lut();
    int t = 0;
    for (; t + 3 < n; t += 4) {
        const float y0 = exp_neg_precise_clamped(lut, exp_neg_clamp_arg(m, s[t]));
        const float y1 = exp_neg_precise_clamped(lut, exp_neg_clamp_arg(m, s[t + 1]));
        const float y2 = exp_neg_precise_clamped(lut, exp_neg_clamp_arg(m, s[t + 2]));
        const float y3 = exp_neg_precise_clamped(lut, exp_neg_clamp_arg(m, s[t + 3]));
        e[t] = y0;
        e[t + 1] = y1;
        e[t + 2] = y2;
        e[t + 3] = y3;
        l += y0;
        l += y1;
        l += y2;
        l += y3;
    }
    for (; t < n; ++t) {
        const float y = exp_neg_precise_clamped(lut, exp_neg_clamp_arg(m, s[t]));
        e[t] = y;
        l += y;
    }
    return l;
}

/**
 * @brief Split the int16 exp score for QACC P·V: X = trunc(e * 2^29) = hi * 2^15 + lo, so e ~= X *
 * 2^-29 with error < 2^-29. e <= 1 keeps hi in [0, 2^14] and lo in [-2^14, 2^14), so a 64-token
 * lane sum stays below 2^35 on the s16 QACC.
 *
 * y_scaled = y * 2^PV_S16_BITS.
 */
inline void split_scaled_e_s16(float y_scaled, int16_t *hi, int16_t *lo)
{
    const int32_t x = static_cast<int32_t>(y_scaled);
    const int32_t h = (x + (1 << (PV_S16_LO_BITS - 1))) >> PV_S16_LO_BITS;
    *hi = static_cast<int16_t>(h);
    *lo = static_cast<int16_t>(x - (h << PV_S16_LO_BITS));
}

/**
 * @brief Split an unscaled exp score y into the int16 hi/lo pair used by QACC P·V.
 */
inline void split_e_s16(float y, int16_t *hi, int16_t *lo)
{
    split_scaled_e_s16(y * static_cast<float>(1 << PV_S16_BITS), hi, lo);
}

/**
 * @brief softmax_exp_row_precise writing split exp scores: e_hi = e[0..n_pad), e_lo =
 * e[n_pad..2*n_pad); slots in [n, n_pad) are 0.
 *
 * Returns l + sum(e) with the same fp32 values as softmax_exp_row_precise: the row runs at
 * 2^PV_S16_BITS scale, which is exact while l < 2^98.
 */
inline float softmax_exp_row_split_s16(
    int16_t *__restrict e, const float *__restrict s, int n, int n_pad, float m, float l)
{
    constexpr int fixed_one = 1 << PV_S16_BITS;
    int16_t *e_hi = e;
    int16_t *e_lo = e + n_pad;
    if (is_neg_inf(m)) {
        std::memset(e, 0, static_cast<size_t>(2 * n_pad) * sizeof(int16_t));
        return l;
    }
    const AttentionFloatExpLut &lut = attention_float_exp_lut();
    float l_scaled = l * static_cast<float>(fixed_one);
    int t = 0;
    for (; t + 3 < n; t += 4) {
        const float y0 = exp_neg_precise_clamped<fixed_one>(lut, exp_neg_clamp_arg(m, s[t]));
        const float y1 = exp_neg_precise_clamped<fixed_one>(lut, exp_neg_clamp_arg(m, s[t + 1]));
        const float y2 = exp_neg_precise_clamped<fixed_one>(lut, exp_neg_clamp_arg(m, s[t + 2]));
        const float y3 = exp_neg_precise_clamped<fixed_one>(lut, exp_neg_clamp_arg(m, s[t + 3]));
        split_scaled_e_s16(y0, e_hi + t, e_lo + t);
        split_scaled_e_s16(y1, e_hi + t + 1, e_lo + t + 1);
        split_scaled_e_s16(y2, e_hi + t + 2, e_lo + t + 2);
        split_scaled_e_s16(y3, e_hi + t + 3, e_lo + t + 3);
        l_scaled += y0;
        l_scaled += y1;
        l_scaled += y2;
        l_scaled += y3;
    }
    // A plain `t < n` tail trips a spurious -Waggressive-loop-optimizations
    // once inlined into callers that know the e[] object size.
    for (int k = n - t; k > 0; --k, ++t) {
        const float y = exp_neg_precise_clamped<fixed_one>(lut, exp_neg_clamp_arg(m, s[t]));
        split_scaled_e_s16(y, e_hi + t, e_lo + t);
        l_scaled += y;
    }
    for (; t < n_pad; ++t) {
        e_hi[t] = 0;
        e_lo[t] = 0;
    }
    return l_scaled * (1.0f / static_cast<float>(fixed_one));
}

/**
 * @brief Load one Q/K/V element and apply its dequant scale.
 */
template <dtype_t qkv_dtype>
inline float load_qkv_element(const void *data, int idx, float scale);

template <>
inline float load_qkv_element<DATA_TYPE_FLOAT>(const void *data, int idx, float /*scale*/)
{
    return static_cast<const float *>(data)[idx];
}
template <>
inline float load_qkv_element<DATA_TYPE_INT8>(const void *data, int idx, float scale)
{
    return static_cast<float>(static_cast<const int8_t *>(data)[idx]) * scale;
}
template <>
inline float load_qkv_element<DATA_TYPE_INT16>(const void *data, int idx, float scale)
{
    return static_cast<float>(static_cast<const int16_t *>(data)[idx]) * scale;
}

/**
 * @brief Store one Attention output element, rounding and clipping when quantized.
 */
template <dtype_t qkv_dtype>
inline void store_out_element(void *data, int idx, float value, float inv_scale);

template <>
inline void store_out_element<DATA_TYPE_FLOAT>(void *data, int idx, float value, float /*inv*/)
{
    static_cast<float *>(data)[idx] = value;
}
template <>
inline void store_out_element<DATA_TYPE_INT8>(void *data, int idx, float value, float inv_scale)
{
    int q = tool::round(value * inv_scale);
    int8_t clipped;
    tool::truncate(clipped, q);
    static_cast<int8_t *>(data)[idx] = clipped;
}
template <>
inline void store_out_element<DATA_TYPE_INT16>(void *data, int idx, float value, float inv_scale)
{
    int q = tool::round(value * inv_scale);
    int16_t clipped;
    tool::truncate(clipped, q);
    static_cast<int16_t *>(data)[idx] = clipped;
}

/**
 * @brief An int16 dot of length n needs up to 30 + ceil(log2(n)) bits; shifting right by
 * ceil(log2(n)) keeps it within 2^30, so the int32 result never saturates.
 */
constexpr int int16_dot_shift(int n)
{
    int s = 0;
    while ((1 << s) < n) {
        ++s;
    }
    return s;
}

/**
 * @brief Arithmetic right shift of v by s bits, rounding half-to-even.
 */
inline int64_t shift_round_half_even(int64_t v, int s)
{
    if (s <= 0) {
        return v;
    }
    const int64_t floor_q = v >> s;
    const int64_t rem = v - (floor_q << s);
    const int64_t half = int64_t(1) << (s - 1);
    return floor_q + ((rem > half || (rem == half && (floor_q & 1))) ? 1 : 0);
}

static_assert(int16_dot_shift(32) == INT16_QACC_QK_D32_SHIFT, "int16 D=32 paths must agree on the dot shift");

/**
 * @brief Keep int16 QK rounding consistent across cores.
 */
template <dtype_t qkv_dtype>
inline void attention_select_qk_rounding()
{
#if CONFIG_PIE_V2_BOOST
    if (qkv_dtype == DATA_TYPE_INT16) {
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
    }
#endif
}

/**
 * @brief Whether this dtype/D can use the PIE SIMD / QACC QK path.
 */
template <dtype_t qkv_dtype>
inline bool can_use_simd_dot(int D)
{
    (void)D;
    return false;
}
#if CONFIG_PIE_V2_BOOST
template <>
inline bool can_use_simd_dot<DATA_TYPE_INT8>(int D)
{
    // Per-row and batched-Q QACC kernels are D-agnostic (16-lane boundary via D_pad).
    return (D > 0) && (D <= ATTENTION_MAX_HEAD_DIM);
}
template <>
inline bool can_use_simd_dot<DATA_TYPE_INT16>(int D)
{
    // Q/K/V are zero-padded to an 8-lane boundary before the int16 kernels,
    // including one complete 8-lane block when the logical D is smaller than 8.
    return (D > 0) && (D <= ATTENTION_MAX_HEAD_DIM);
}
template <>
inline bool can_use_simd_dot<DATA_TYPE_FLOAT>(int /*D*/)
{
    return true;
}
#endif

/**
 * @brief Scaled Q·K dot of one pair of rows.
 */
template <dtype_t qkv_dtype>
inline float simd_scaled_dot(const void *q_data, int q_base, const void *k_data, int k_base, int D, float logit_scale)
{
    (void)q_data;
    (void)q_base;
    (void)k_data;
    (void)k_base;
    (void)D;
    (void)logit_scale;
    return 0.f;
}
#if CONFIG_PIE_V2_BOOST
template <>
inline float simd_scaled_dot<DATA_TYPE_INT8>(
    const void *q_data, int q_base, const void *k_data, int k_base, int D, float logit_scale)
{
    int32_t result = 0;
    const int8_t *q_ptr = static_cast<const int8_t *>(q_data) + q_base;
    const int8_t *k_ptr = static_cast<const int8_t *>(k_data) + k_base;
    for (int d = 0; d < D; ++d) {
        result += static_cast<int32_t>(q_ptr[d]) * static_cast<int32_t>(k_ptr[d]);
    }
    return static_cast<float>(result) * logit_scale;
}
template <>
inline float simd_scaled_dot<DATA_TYPE_INT16>(
    const void *q_data, int q_base, const void *k_data, int k_base, int D, float logit_scale)
{
    int64_t result = 0;
    const int16_t *q_ptr = static_cast<const int16_t *>(q_data) + q_base;
    const int16_t *k_ptr = static_cast<const int16_t *>(k_data) + k_base;
    for (int d = 0; d < D; ++d) {
        result += static_cast<int64_t>(q_ptr[d]) * static_cast<int64_t>(k_ptr[d]);
    }
    const int shift = int16_dot_shift(D);
    const int32_t shifted = static_cast<int32_t>(shift_round_half_even(result, shift));
    return static_cast<float>(shifted) * (logit_scale * static_cast<float>(1 << shift));
}
template <>
inline float simd_scaled_dot<DATA_TYPE_FLOAT>(
    const void *q_data, int q_base, const void *k_data, int k_base, int D, float logit_scale)
{
    float result = 0.f;
    float *q_ptr = const_cast<float *>(static_cast<const float *>(q_data) + q_base);
    float *k_ptr = const_cast<float *>(static_cast<const float *>(k_data) + k_base);
    dl_esp32p4_dotprod_f32(&result, q_ptr, k_ptr, D);
    return result * logit_scale;
}
#endif

} // namespace
} // namespace base
} // namespace dl
