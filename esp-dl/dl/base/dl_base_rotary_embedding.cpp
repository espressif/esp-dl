#include "dl_base_rotary_embedding.hpp"

#include "dl_base_isa.hpp"
#include "dl_define.hpp"
#include "dl_tool.hpp"
#include "esp_log.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

static const char *TAG = "RotaryEmbedding";

namespace dl {
namespace base {

namespace {

inline float load_float_element(const void *data, int idx, dtype_t dtype, float scale)
{
    switch (dtype) {
    case DATA_TYPE_FLOAT:
        return static_cast<const float *>(data)[idx];
    case DATA_TYPE_INT8:
        return static_cast<float>(static_cast<const int8_t *>(data)[idx]) * scale;
    case DATA_TYPE_INT16:
        return static_cast<float>(static_cast<const int16_t *>(data)[idx]) * scale;
    default:
        return 0.0f;
    }
}

inline void store_float_element(void *data, int idx, dtype_t dtype, float value, float inv_scale)
{
    switch (dtype) {
    case DATA_TYPE_FLOAT:
        static_cast<float *>(data)[idx] = value;
        return;
    case DATA_TYPE_INT8: {
        int8_t clipped;
        tool::truncate(clipped, tool::round(value * inv_scale));
        static_cast<int8_t *>(data)[idx] = clipped;
        return;
    }
    case DATA_TYPE_INT16: {
        int16_t clipped;
        tool::truncate(clipped, tool::round(value * inv_scale));
        static_cast<int16_t *>(data)[idx] = clipped;
        return;
    }
    default:
        return;
    }
}

inline bool rope_supported_dtype(dtype_t dtype)
{
    return dtype == DATA_TYPE_FLOAT || dtype == DATA_TYPE_INT8 || dtype == DATA_TYPE_INT16;
}

inline bool rope_dtype_matches_quant_type(dtype_t xy_dtype, quant_type_t quant_type)
{
    switch (quant_type) {
    case QUANT_TYPE_NONE:
        return rope_supported_dtype(xy_dtype);
    case QUANT_TYPE_FLOAT32:
        return xy_dtype == DATA_TYPE_FLOAT;
    case QUANT_TYPE_SYMM_8BIT:
        return xy_dtype == DATA_TYPE_INT8;
    case QUANT_TYPE_SYMM_16BIT:
        return xy_dtype == DATA_TYPE_INT16;
    default:
        return false;
    }
}

inline bool rope_quantized_dtype(dtype_t dtype)
{
    return dtype == DATA_TYPE_INT8 || dtype == DATA_TYPE_INT16;
}

inline bool checked_positive_mul(int lhs, int rhs, int &result)
{
    if (lhs <= 0 || rhs <= 0 || lhs > INT_MAX / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline bool checked_positive_shape_size(const std::vector<int> &shape, int &result)
{
    result = 1;
    for (int dim : shape) {
        if (!checked_positive_mul(result, dim, result)) {
            return false;
        }
    }
    return true;
}

inline bool rope_check_per_tensor_exponent(const TensorBase *tensor, const char *name)
{
    if (!tensor->exponent.is_valid()) {
        ESP_LOGE(TAG, "%s has an invalid exponent", name);
        return false;
    }
    if (tensor->exponent.is_per_channel()) {
        ESP_LOGE(TAG, "%s uses per-channel exponents; RotaryEmbedding requires per-tensor quantization", name);
        return false;
    }
    if (rope_quantized_dtype(tensor->dtype)) {
        const int exponent = tensor->exponent.get();
        if (exponent <= -31 || exponent >= 31) {
            ESP_LOGE(TAG, "%s exponent %d is out of range (-31, 31)", name, exponent);
            return false;
        }
        const float scale = DL_SCALE(exponent);
        if (!std::isfinite(scale) || scale <= 0.0f) {
            ESP_LOGE(TAG, "%s exponent %d cannot be represented as a finite, positive fp32 scale", name, exponent);
            return false;
        }
    }
    return true;
}

// Cache row offset, in elements, for the sequence position at (b, l).
inline int rope_cache_offset(const RopeArgs &args, int b, int l)
{
    return b * args.cache_stride_batch + l * args.cache_stride_pos;
}

inline uint64_t signed_magnitude(int64_t value)
{
    return value < 0 ? static_cast<uint64_t>(-(value + 1)) + 1U : static_cast<uint64_t>(value);
}

inline int64_t apply_sign(uint64_t magnitude, bool negative)
{
    if (!negative) {
        return static_cast<int64_t>(magnitude);
    }
    if (magnitude == (UINT64_C(1) << 63)) {
        return INT64_MIN;
    }
    return -static_cast<int64_t>(magnitude);
}

// Exact signed power-of-two rescale using the same target policy as
// dl::tool::round(): HALF_EVEN on P4/S31, HALF_UP on S3/C. The
// unsigned-magnitude formulation avoids implementation-defined signed
// shifts and the ``1 << 63`` overflow boundary.
inline int64_t rescale_integer_target(int64_t value, int shift)
{
    if (shift > 0) {
        if (shift >= 64) {
            return 0;
        }
        const bool negative = value < 0;
        const uint64_t magnitude = signed_magnitude(value);
        uint64_t quotient = magnitude >> shift;
        const uint64_t mask = (UINT64_C(1) << shift) - 1U;
        const uint64_t remainder = magnitude & mask;
        const uint64_t halfway = UINT64_C(1) << (shift - 1);
#if CONFIG_ROUND_HALF_EVEN_ENABLED
        const bool increment = remainder > halfway || (remainder == halfway && (quotient & 1U));
#else
        // HALF_UP is floor(x + 0.5): positive ties increment, negative ties
        // move toward zero and therefore leave the magnitude unchanged.
        const bool increment = negative ? (remainder > halfway) : (remainder >= halfway);
#endif
        if (increment) {
            ++quotient;
        }
        return apply_sign(quotient, negative);
    }
    if (shift == 0 || value == 0) {
        return value;
    }

    const bool negative = value < 0;
    const uint64_t magnitude = signed_magnitude(value);
    const int left_shift = -shift;
    const uint64_t limit = negative ? (UINT64_C(1) << 63) : static_cast<uint64_t>(INT64_MAX);
    if (left_shift >= 63 || magnitude > (limit >> left_shift)) {
        return negative ? INT64_MIN : INT64_MAX;
    }
    return apply_sign(magnitude << left_shift, negative);
}

// One rotated pair: X[i], X[j] against cos[k], sin[k]. Split out of the loop
// so the unrolled body below can issue several back to back; each is an
// independent dependency chain, which is the whole point of the unroll.
//
// Both outputs read the same cache entry k — that is the ONNX cache layout,
// where the cache holds one value per *pair* rather than per element.
template <typename XT, typename CT, typename YT, typename acc_t>
inline void rope_one_pair(const XT *xr,
                          YT *yr,
                          const CT *cr,
                          const CT *sr,
                          int i,
                          int j,
                          int k,
                          float kc,
                          float ks,
                          bool same_k,
                          int integer_mac_shift)
{
    const acc_t x1 = xr[i];
    const acc_t x2 = xr[j];
    const acc_t c = cr[k];
    const acc_t s = sr[k];
    if (integer_mac_shift != INT_MIN) {
        const int64_t acc1 = static_cast<int64_t>(x1) * c - static_cast<int64_t>(x2) * s;
        const int64_t acc2 = static_cast<int64_t>(x1) * s + static_cast<int64_t>(x2) * c;
        tool::truncate(yr[i], rescale_integer_target(acc1, integer_mac_shift));
        tool::truncate(yr[j], rescale_integer_target(acc2, integer_mac_shift));
        return;
    }

    int q1, q2;
    if (same_k) {
        q1 = tool::round(static_cast<float>(x1 * c - x2 * s) * kc);
        q2 = tool::round(static_cast<float>(x1 * s + x2 * c) * kc);
    } else {
        q1 = tool::round(static_cast<float>(x1 * c) * kc - static_cast<float>(x2 * s) * ks);
        q2 = tool::round(static_cast<float>(x1 * s) * ks + static_cast<float>(x2 * c) * kc);
    }
    tool::truncate(yr[i], q1);
    tool::truncate(yr[j], q2);
}

// Unrolled by 4.
// Half-split (rotate_half) only: pair (k, k + half). The GPT-J interleaved
// layout is out of scope for the vision subset.
template <typename XT, typename CT, typename YT, typename acc_t>
inline void rope_one_row(
    const XT *xr, YT *yr, const CT *cr, const CT *sr, const RopeArgs &args, float kc, float ks, bool same_k)
{
    const int half = args.half_dim;
    int k = 0;
    for (; k + 4 <= half; k += 4) {
        rope_one_pair<XT, CT, YT, acc_t>(
            xr, yr, cr, sr, k + 0, k + 0 + half, k + 0, kc, ks, same_k, args.integer_mac_shift);
        rope_one_pair<XT, CT, YT, acc_t>(
            xr, yr, cr, sr, k + 1, k + 1 + half, k + 1, kc, ks, same_k, args.integer_mac_shift);
        rope_one_pair<XT, CT, YT, acc_t>(
            xr, yr, cr, sr, k + 2, k + 2 + half, k + 2, kc, ks, same_k, args.integer_mac_shift);
        rope_one_pair<XT, CT, YT, acc_t>(
            xr, yr, cr, sr, k + 3, k + 3 + half, k + 3, kc, ks, same_k, args.integer_mac_shift);
    }
    for (; k < half; ++k) {
        rope_one_pair<XT, CT, YT, acc_t>(xr, yr, cr, sr, k, k + half, k, kc, ks, same_k, args.integer_mac_shift);
    }
}

// Process a small token block for one head before moving to the next head.
// [B,H,L,D] stores adjacent positions of a head contiguously, while a position-major
// walk jumps by L*D elements between heads on every row. Blocking keeps X/Y
// accesses burst-friendly without allocating workspace; the corresponding
// cos/sin rows remain small enough to be reused from cache across heads.
template <typename XT, typename CT, typename YT>
void rope_rows_typed(const RopeArgs &args)
{
    using acc_t = typename std::conditional<sizeof(XT) == 1, int32_t, int64_t>::type;

    const XT *x = static_cast<const XT *>(args.x);
    const CT *cos = static_cast<const CT *>(args.cos);
    const CT *sin = static_cast<const CT *>(args.sin);
    YT *y = static_cast<YT *>(args.y);

    const float kc = args.x_scale * args.cos_scale * args.y_inv_scale;
    const float ks = args.x_scale * args.sin_scale * args.y_inv_scale;
    // cos/sin normally share one calibration range, which collapses the pair
    // of scales to a single multiply.
    const bool same_k = (kc == ks);

    // PIE v2 SIMD fast paths. Three variants are implemented today:
    //   * int8 X + int8  caches + int8  Y  (dl_esp32p4_s8_rope_neox_simd)
    //   * int16 X + int16 caches + int16 Y (dl_esp32p4_s16_rope_neox_simd)
    //   * int8 X + int16 caches + int8  Y  (dl_esp32p4_s8_s16c_rope_neox_simd)
#if CONFIG_PIE_V2_BOOST
    constexpr bool is_s8_uniform =
        std::is_same<XT, int8_t>::value && std::is_same<CT, int8_t>::value && std::is_same<YT, int8_t>::value;
    constexpr bool is_s16_uniform =
        std::is_same<XT, int16_t>::value && std::is_same<CT, int16_t>::value && std::is_same<YT, int16_t>::value;
    constexpr bool is_s8_s16c =
        std::is_same<XT, int8_t>::value && std::is_same<CT, int16_t>::value && std::is_same<YT, int8_t>::value;
    constexpr bool has_simd = is_s8_uniform || is_s16_uniform || is_s8_s16c;
    // Per-iter pair count differs by kernel:
    //   int8×3         → 16 pairs (VMULAS.S8, 16 int8 per QR)
    //   int16×3        →  8 pairs (VMULAS.S16, 8 int16 per QR)
    //   mixed aligned  → 16 pairs (two widened int16 QRs)
    //   mixed 8-byte   →  8 pairs (64-bit x1/x2 loads widened together)
    //
    // The mixed 8-pair variant covers D=48/80-style shapes whose two int8
    // half rows cannot both be 16-byte aligned. X/Y only need 8-byte
    // alignment there; int16 cache rows still need 16-byte alignment.
    const uintptr_t xy_base_bits = reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(y);
    const uintptr_t cache_base_bits = reinterpret_cast<uintptr_t>(cos) | reinterpret_cast<uintptr_t>(sin);
    const bool all_base_aligned_16 = ((xy_base_bits | cache_base_bits) & 0xF) == 0;
    const bool mixed_base_aligned_8 = (xy_base_bits & 0x7) == 0 && (cache_base_bits & 0xF) == 0;
    const bool use_mixed_8_pair = is_s8_s16c && args.half_dim > 0 && (args.half_dim % 16) != 0 &&
        (args.half_dim % 8) == 0 && mixed_base_aligned_8;
    const bool uniform_s8_shape_ok = is_s8_uniform && (args.half_dim % 16) == 0 && all_base_aligned_16;
    const bool uniform_s16_shape_ok = is_s16_uniform && (args.half_dim % 8) == 0 && all_base_aligned_16;
    const bool mixed_shape_ok = is_s8_s16c && (((args.half_dim % 16) == 0 && all_base_aligned_16) || use_mixed_8_pair);
    const bool simd_shape_ok = has_simd && args.simd_mac_shift > 0 && args.simd_same_k && args.half_dim > 0 &&
        (uniform_s8_shape_ok || uniform_s16_shape_ok || mixed_shape_ok);
    // srcmb.{s8,s16}.qacc reads CFG.vxrm for the shift's rounding mode.
    // Neighbouring PIE kernels leave vxrm in whatever state suited them
    // (matmul / conv sets HALF_EVEN, LUT explicitly clears it). We need
    // HALF_EVEN here to agree with tool::round<>() on PIE v2 — otherwise
    // the SIMD lane can round in a different direction than the scalar row
    // it replaces, producing off-by-1 outputs that fail bit-exact
    // comparisons.
    if (simd_shape_ok) {
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
    }
#else
    const bool simd_shape_ok = false;
#endif

    int flat_token = args.token_start;
    int batch_index = args.token_start / args.seq_len;
    int token_index = args.token_start - batch_index * args.seq_len;
    while (flat_token < args.token_end) {
        const int task_remaining = args.token_end - flat_token;
        const int batch_remaining = args.seq_len - token_index;
        int block = task_remaining < batch_remaining ? task_remaining : batch_remaining;
        if (block > args.token_block_size) {
            block = args.token_block_size;
        }

        for (int head_index = 0; head_index < args.head_count; ++head_index) {
            const int first_row =
                batch_index * args.x_stride_batch + head_index * args.x_stride_head + token_index * args.x_stride_pos;
            int cache_off = rope_cache_offset(args, batch_index, token_index);
            for (int token_in_block = 0; token_in_block < block; ++token_in_block) {
                const int row = first_row + token_in_block * args.x_stride_pos;
                const CT *cr = cos + cache_off;
                const CT *sr = sin + cache_off;
#if CONFIG_PIE_V2_BOOST
                if (simd_shape_ok) {
                    // Each branch is a template-time-constant if constexpr,
                    // so the compiler discards the mismatching arms and
                    // only the matching kernel signature is instantiated.
                    if constexpr (is_s8_uniform) {
                        dl_esp32p4_s8_rope_neox_simd(reinterpret_cast<int8_t *>(y + row),
                                                     reinterpret_cast<int8_t *>(y + row) + args.half_dim,
                                                     reinterpret_cast<const int8_t *>(x + row),
                                                     reinterpret_cast<const int8_t *>(x + row) + args.half_dim,
                                                     reinterpret_cast<const int8_t *>(cr),
                                                     reinterpret_cast<const int8_t *>(sr),
                                                     args.half_dim / 16,
                                                     args.simd_mac_shift);
                    } else if constexpr (is_s16_uniform) {
                        dl_esp32p4_s16_rope_neox_simd(reinterpret_cast<int16_t *>(y + row),
                                                      reinterpret_cast<int16_t *>(y + row) + args.half_dim,
                                                      reinterpret_cast<const int16_t *>(x + row),
                                                      reinterpret_cast<const int16_t *>(x + row) + args.half_dim,
                                                      reinterpret_cast<const int16_t *>(cr),
                                                      reinterpret_cast<const int16_t *>(sr),
                                                      args.half_dim / 8,
                                                      args.simd_mac_shift);
                    } else if constexpr (is_s8_s16c) {
                        if (use_mixed_8_pair) {
                            dl_esp32p4_s8_s16c8_rope_neox_simd(reinterpret_cast<int8_t *>(y + row),
                                                               reinterpret_cast<int8_t *>(y + row) + args.half_dim,
                                                               reinterpret_cast<const int8_t *>(x + row),
                                                               reinterpret_cast<const int8_t *>(x + row) +
                                                                   args.half_dim,
                                                               reinterpret_cast<const int16_t *>(cr),
                                                               reinterpret_cast<const int16_t *>(sr),
                                                               args.half_dim / 8,
                                                               args.simd_mac_shift);
                        } else {
                            dl_esp32p4_s8_s16c_rope_neox_simd(reinterpret_cast<int8_t *>(y + row),
                                                              reinterpret_cast<int8_t *>(y + row) + args.half_dim,
                                                              reinterpret_cast<const int8_t *>(x + row),
                                                              reinterpret_cast<const int8_t *>(x + row) + args.half_dim,
                                                              reinterpret_cast<const int16_t *>(cr),
                                                              reinterpret_cast<const int16_t *>(sr),
                                                              args.half_dim / 16,
                                                              args.simd_mac_shift);
                        }
                    }
                    cache_off += args.cache_stride_pos;
                    continue;
                }
#endif
                rope_one_row<XT, CT, YT, acc_t>(x + row, y + row, cr, sr, args, kc, ks, same_k);
                cache_off += args.cache_stride_pos;
            }
        }

        flat_token += block;
        token_index += block;
        if (token_index == args.seq_len) {
            token_index = 0;
            ++batch_index;
        }
    }
}

// Mixed / float dtype combinations: same traversal, per-element dequantize.
void rope_rows_generic(const RopeArgs &args)
{
    int flat_token = args.token_start;
    int batch_index = args.token_start / args.seq_len;
    int token_index = args.token_start - batch_index * args.seq_len;
    while (flat_token < args.token_end) {
        const int task_remaining = args.token_end - flat_token;
        const int batch_remaining = args.seq_len - token_index;
        int block = task_remaining < batch_remaining ? task_remaining : batch_remaining;
        if (block > args.token_block_size) {
            block = args.token_block_size;
        }

        for (int head_index = 0; head_index < args.head_count; ++head_index) {
            const int first_row =
                batch_index * args.x_stride_batch + head_index * args.x_stride_head + token_index * args.x_stride_pos;
            int cache_off = rope_cache_offset(args, batch_index, token_index);
            for (int token_in_block = 0; token_in_block < block; ++token_in_block) {
                const int row = first_row + token_in_block * args.x_stride_pos;
                for (int k = 0; k < args.half_dim; ++k) {
                    const int i = k;
                    const int j = k + args.half_dim;
                    const float x1 = load_float_element(args.x, row + i, args.x_dtype, args.x_scale);
                    const float x2 = load_float_element(args.x, row + j, args.x_dtype, args.x_scale);
                    const float c = load_float_element(args.cos, cache_off + k, args.cos_dtype, args.cos_scale);
                    const float sv = load_float_element(args.sin, cache_off + k, args.sin_dtype, args.sin_scale);
                    store_float_element(args.y, row + i, args.y_dtype, x1 * c - x2 * sv, args.y_inv_scale);
                    store_float_element(args.y, row + j, args.y_dtype, x1 * sv + x2 * c, args.y_inv_scale);
                }
                cache_off += args.cache_stride_pos;
            }
        }

        flat_token += block;
        token_index += block;
        if (token_index == args.seq_len) {
            token_index = 0;
            ++batch_index;
        }
    }
}

void rope_run_task(const RopeArgs &args)
{
    if (args.x_dtype == DATA_TYPE_INT8 && args.y_dtype == DATA_TYPE_INT8 && args.cos_dtype == DATA_TYPE_INT16 &&
        args.sin_dtype == DATA_TYPE_INT16) {
        rope_rows_typed<int8_t, int16_t, int8_t>(args);
    } else if (args.x_dtype == DATA_TYPE_INT8 && args.y_dtype == DATA_TYPE_INT8 && args.cos_dtype == DATA_TYPE_INT8 &&
               args.sin_dtype == DATA_TYPE_INT8) {
        rope_rows_typed<int8_t, int8_t, int8_t>(args);
    } else if (args.x_dtype == DATA_TYPE_INT16 && args.y_dtype == DATA_TYPE_INT16 &&
               args.cos_dtype == DATA_TYPE_INT16 && args.sin_dtype == DATA_TYPE_INT16) {
        rope_rows_typed<int16_t, int16_t, int16_t>(args);
    } else {
        rope_rows_generic(args);
    }
}

} // namespace

void rope_zero_output(TensorBase *y)
{
    if (y == nullptr || y->data == nullptr) {
        return;
    }
    const int elems = y->get_size();
    if (elems <= 0) {
        return;
    }
    size_t elem_size = 0;
    switch (y->dtype) {
    case DATA_TYPE_FLOAT:
        elem_size = sizeof(float);
        break;
    case DATA_TYPE_INT8:
        elem_size = sizeof(int8_t);
        break;
    case DATA_TYPE_INT16:
        elem_size = sizeof(int16_t);
        break;
    case DATA_TYPE_INT32:
        elem_size = sizeof(int32_t);
        break;
    case DATA_TYPE_INT64:
        elem_size = sizeof(int64_t);
        break;
    default:
        return; // unknown dtype — leave the buffer alone
    }
    if (static_cast<size_t>(elems) > SIZE_MAX / elem_size) {
        return;
    }
    std::memset(y->data, 0, static_cast<size_t>(elems) * elem_size);
}

bool rope_check_attributes(int interleaved, int rotary_embedding_dim, int num_heads, const char *who)
{
    const char *label = who != nullptr ? who : "<unnamed>";
    if (interleaved != 0) {
        ESP_LOGE(TAG, "%s: interleaved=%d is unsupported; only half-split layout is available", label, interleaved);
        return false;
    }
    if (num_heads < 0) {
        ESP_LOGE(TAG, "%s: num_heads=%d must be non-negative", label, num_heads);
        return false;
    }
    if (rotary_embedding_dim < 0 || (rotary_embedding_dim & 1)) {
        ESP_LOGE(TAG,
                 "%s: rotary_embedding_dim=%d must be zero or a positive even full head dimension",
                 label,
                 rotary_embedding_dim);
        return false;
    }
    return true;
}

bool rope_check_shapes(const std::vector<int> &x_shape,
                       const std::vector<int> &cos_shape,
                       const std::vector<int> &sin_shape,
                       int interleaved,
                       int rotary_embedding_dim,
                       int num_heads,
                       RopeDims &dims,
                       const char *who)
{
    const char *label = who != nullptr ? who : "<unnamed>";
    if (!rope_check_attributes(interleaved, rotary_embedding_dim, num_heads, label)) {
        return false;
    }
    if (x_shape.size() != 4) {
        ESP_LOGE(TAG, "%s: X must be 4D [B,H,L,D], got rank %d", label, static_cast<int>(x_shape.size()));
        return false;
    }
    if (!checked_positive_shape_size(x_shape, dims.x_elements)) {
        ESP_LOGE(TAG, "%s: X dimensions must be positive and their product must fit in int", label);
        return false;
    }

    dims.batch = x_shape[0];
    dims.head_count = x_shape[1];
    dims.seq_len = x_shape[2];
    dims.head_dim = x_shape[3];
    if (num_heads != 0 && num_heads != dims.head_count) {
        ESP_LOGE(TAG, "%s: num_heads=%d contradicts X's H axis (%d)", label, num_heads, dims.head_count);
        return false;
    }
    if (dims.head_dim & 1) {
        ESP_LOGE(TAG, "%s: head_dim D=%d must be even", label, dims.head_dim);
        return false;
    }
    if (rotary_embedding_dim != 0 && rotary_embedding_dim != dims.head_dim) {
        ESP_LOGE(TAG,
                 "%s: rotary_embedding_dim=%d != head_dim D=%d; partial rotation is unsupported",
                 label,
                 rotary_embedding_dim,
                 dims.head_dim);
        return false;
    }
    dims.half_dim = dims.head_dim / 2;

    if (cos_shape != sin_shape) {
        ESP_LOGE(TAG, "%s: cos_cache and sin_cache must have the same shape", label);
        return false;
    }
    const int cache_rank = static_cast<int>(cos_shape.size());
    if (cache_rank != 2 && cache_rank != 3) {
        ESP_LOGE(TAG, "%s: cos/sin cache must be 2D or 3D, got rank %d", label, cache_rank);
        return false;
    }
    if (!checked_positive_shape_size(cos_shape, dims.cache_elements)) {
        ESP_LOGE(TAG, "%s: cache dimensions must be positive and their product must fit in int", label);
        return false;
    }
    if (cos_shape[cache_rank - 1] != dims.half_dim) {
        ESP_LOGE(TAG, "%s: cache last dim %d != D/2 (%d)", label, cos_shape[cache_rank - 1], dims.half_dim);
        return false;
    }

    dims.cache_stride_pos = dims.half_dim;
    if (cache_rank == 3) {
        const int cache_batch = cos_shape[0];
        if (cache_batch != dims.batch && cache_batch != 1) {
            ESP_LOGE(TAG, "%s: cache batch B=%d matches neither X's B=%d nor 1", label, cache_batch, dims.batch);
            return false;
        }
        if (cos_shape[1] != dims.seq_len) {
            ESP_LOGE(TAG, "%s: cache sequence length L=%d != X's L=%d", label, cos_shape[1], dims.seq_len);
            return false;
        }
        int full_cache_stride_batch = 0;
        if (!checked_positive_mul(dims.seq_len, dims.half_dim, full_cache_stride_batch)) {
            ESP_LOGE(TAG, "%s: cache batch stride overflows int", label);
            return false;
        }
        dims.cache_stride_batch = cache_batch == 1 ? 0 : full_cache_stride_batch;
    } else {
        if (cos_shape[0] != dims.seq_len) {
            ESP_LOGE(TAG, "%s: cache sequence length L=%d != X's L=%d", label, cos_shape[0], dims.seq_len);
            return false;
        }
        dims.cache_stride_batch = 0;
    }

    if (!checked_positive_mul(dims.seq_len, dims.head_dim, dims.x_stride_head) ||
        !checked_positive_mul(dims.head_count, dims.x_stride_head, dims.x_stride_batch) ||
        !checked_positive_mul(dims.batch, dims.seq_len, dims.token_count) ||
        dims.batch > INT_MAX / dims.x_stride_batch || dims.batch * dims.x_stride_batch != dims.x_elements) {
        ESP_LOGE(TAG, "%s: X stride or token-count computation overflows int", label);
        return false;
    }
    return true;
}

std::vector<RopeArgs> get_rotary_embedding_operation_args(TensorBase *y,
                                                          TensorBase *x,
                                                          TensorBase *cos_cache,
                                                          TensorBase *sin_cache,
                                                          int interleaved,
                                                          int rotary_embedding_dim,
                                                          int num_heads,
                                                          quant_type_t quant_type,
                                                          runtime_mode_t mode,
                                                          const char *who)
{
    if (!x || !cos_cache || !sin_cache || !y) {
        ESP_LOGE(TAG,
                 "missing input/output tensor (x=%p cos=%p sin=%p y=%p)",
                 static_cast<const void *>(x),
                 static_cast<const void *>(cos_cache),
                 static_cast<const void *>(sin_cache),
                 static_cast<void *>(y));
        rope_zero_output(y);
        return {};
    }
    if (!x->data || !cos_cache->data || !sin_cache->data || !y->data) {
        ESP_LOGE(TAG,
                 "missing input/output buffer (x=%p cos=%p sin=%p y=%p)",
                 x->data,
                 cos_cache->data,
                 sin_cache->data,
                 y->data);
        rope_zero_output(y);
        return {};
    }
    if (y->data == cos_cache->data || y->data == sin_cache->data) {
        ESP_LOGE(TAG, "output Y buffer must not alias cos_cache or sin_cache");
        return {};
    }
    if (y->shape != x->shape) {
        ESP_LOGE(TAG, "Y shape must equal X shape");
        rope_zero_output(y);
        return {};
    }
    if (x->dtype != y->dtype) {
        ESP_LOGE(TAG, "X dtype %d != Y dtype %d", static_cast<int>(x->dtype), static_cast<int>(y->dtype));
        rope_zero_output(y);
        return {};
    }
    if (!rope_dtype_matches_quant_type(x->dtype, quant_type)) {
        ESP_LOGE(TAG,
                 "X/Y dtype %s does not match quant_type %s",
                 dtype_to_string(x->dtype),
                 quant_type_to_string(quant_type));
        rope_zero_output(y);
        return {};
    }
    if (!rope_supported_dtype(x->dtype) || !rope_supported_dtype(cos_cache->dtype) ||
        !rope_supported_dtype(sin_cache->dtype) || !rope_supported_dtype(y->dtype)) {
        ESP_LOGE(TAG,
                 "unsupported dtype combination X=%d cos=%d sin=%d Y=%d "
                 "(only float32/int8/int16 are supported)",
                 static_cast<int>(x->dtype),
                 static_cast<int>(cos_cache->dtype),
                 static_cast<int>(sin_cache->dtype),
                 static_cast<int>(y->dtype));
        rope_zero_output(y);
        return {};
    }
    if (cos_cache->dtype != sin_cache->dtype) {
        ESP_LOGE(TAG,
                 "cos dtype %d != sin dtype %d",
                 static_cast<int>(cos_cache->dtype),
                 static_cast<int>(sin_cache->dtype));
        rope_zero_output(y);
        return {};
    }
    if (!rope_check_per_tensor_exponent(x, "X") || !rope_check_per_tensor_exponent(cos_cache, "cos_cache") ||
        !rope_check_per_tensor_exponent(sin_cache, "sin_cache") || !rope_check_per_tensor_exponent(y, "Y")) {
        rope_zero_output(y);
        return {};
    }

    RopeDims dims;
    if (!rope_check_shapes(
            x->shape, cos_cache->shape, sin_cache->shape, interleaved, rotary_embedding_dim, num_heads, dims, who)) {
        rope_zero_output(y);
        return {};
    }
    if (x->get_size() != dims.x_elements || y->get_size() != dims.x_elements) {
        ESP_LOGE(TAG,
                 "X/Y storage size does not match shape product (%d/%d vs expected %d)",
                 x->get_size(),
                 y->get_size(),
                 dims.x_elements);
        rope_zero_output(y);
        return {};
    }
    if (cos_cache->get_size() != dims.cache_elements || sin_cache->get_size() != dims.cache_elements) {
        ESP_LOGE(TAG,
                 "cos/sin storage size does not match cache shape product (%d/%d vs expected %d)",
                 cos_cache->get_size(),
                 sin_cache->get_size(),
                 dims.cache_elements);
        rope_zero_output(y);
        return {};
    }

    const int H = dims.head_count;
    const int L = dims.seq_len;
    const int D = dims.head_dim;
    const int D_half = dims.half_dim;
    const int x_elements = dims.x_elements;
    const int token_count = dims.token_count;

    RopeArgs base;
    base.x = x->data;
    base.cos = cos_cache->data;
    base.sin = sin_cache->data;
    base.y = y->data;
    base.x_dtype = x->dtype;
    base.cos_dtype = cos_cache->dtype;
    base.sin_dtype = sin_cache->dtype;
    base.y_dtype = y->dtype;
    base.x_scale = rope_quantized_dtype(x->dtype) ? DL_SCALE(x->exponent) : 1.0f;
    base.cos_scale = rope_quantized_dtype(cos_cache->dtype) ? DL_SCALE(cos_cache->exponent) : 1.0f;
    base.sin_scale = rope_quantized_dtype(sin_cache->dtype) ? DL_SCALE(sin_cache->exponent) : 1.0f;
    base.y_inv_scale = rope_quantized_dtype(y->dtype) ? DL_RESCALE(y->exponent) : 1.0f;
    base.head_dim = D;
    base.half_dim = D_half;
    base.head_count = H;
    base.seq_len = L;
    base.cache_stride_batch = dims.cache_stride_batch;
    base.cache_stride_pos = dims.cache_stride_pos;
    base.x_stride_batch = dims.x_stride_batch;
    base.x_stride_head = dims.x_stride_head;
    base.x_stride_pos = D;

    const bool is_s8_uniform = (x->dtype == DATA_TYPE_INT8 && cos_cache->dtype == DATA_TYPE_INT8 &&
                                sin_cache->dtype == DATA_TYPE_INT8 && y->dtype == DATA_TYPE_INT8);
    const bool is_s16_uniform = (x->dtype == DATA_TYPE_INT16 && cos_cache->dtype == DATA_TYPE_INT16 &&
                                 sin_cache->dtype == DATA_TYPE_INT16 && y->dtype == DATA_TYPE_INT16);
    const bool is_s8_s16c = (x->dtype == DATA_TYPE_INT8 && cos_cache->dtype == DATA_TYPE_INT16 &&
                             sin_cache->dtype == DATA_TYPE_INT16 && y->dtype == DATA_TYPE_INT8);
    const bool typed_integer_combo = is_s8_uniform || is_s16_uniform || is_s8_s16c;

    base.simd_same_k = (cos_cache->exponent == sin_cache->exponent);
    if (base.simd_same_k && typed_integer_combo) {
        const int64_t shift64 = static_cast<int64_t>(y->exponent.get()) - static_cast<int64_t>(x->exponent.get()) -
            static_cast<int64_t>(cos_cache->exponent.get());
        if (shift64 > INT_MIN && shift64 <= INT_MAX) {
            base.integer_mac_shift = static_cast<int>(shift64);
        }

        int shift_max = 0;
        if (is_s8_uniform || is_s8_s16c) {
            shift_max = 31; // SRCMB.S8.QACC
        } else if (is_s16_uniform) {
            shift_max = 63; // SRCMB.S16.QACC
        }
        if (shift64 > 0 && shift64 <= shift_max) {
            base.simd_mac_shift = static_cast<int>(shift64);
        }
    }

    if (base.integer_mac_shift == INT_MIN) {
        const float kc = base.x_scale * base.cos_scale * base.y_inv_scale;
        const float ks = base.x_scale * base.sin_scale * base.y_inv_scale;
        if (!std::isfinite(kc) || !std::isfinite(ks) || kc == 0.0f || ks == 0.0f) {
            ESP_LOGE(TAG, "combined X/cache/Y exponent scale is not finite and non-zero");
            rope_zero_output(y);
            return {};
        }
    }

    const int work_elements = x_elements;
    const bool can_split = token_count >= 2;
    bool auto_dual = false;
#if CONFIG_PIE_V2_BOOST
    const uintptr_t xy_base_bits = reinterpret_cast<uintptr_t>(x->data) | reinterpret_cast<uintptr_t>(y->data);
    const uintptr_t cache_base_bits =
        reinterpret_cast<uintptr_t>(cos_cache->data) | reinterpret_cast<uintptr_t>(sin_cache->data);
    const bool all_base_aligned_16 = ((xy_base_bits | cache_base_bits) & 0xF) == 0;
    const bool mixed_base_aligned_8 = (xy_base_bits & 0x7) == 0 && (cache_base_bits & 0xF) == 0;
    const bool p4_s8_uniform_simd =
        is_s8_uniform && base.simd_mac_shift > 0 && base.simd_same_k && (D_half % 16) == 0 && all_base_aligned_16;
    const bool p4_s16_uniform_simd =
        is_s16_uniform && base.simd_mac_shift > 0 && base.simd_same_k && (D_half % 8) == 0 && all_base_aligned_16;
    const bool p4_s8_s16c_simd = is_s8_s16c && base.simd_mac_shift > 0 && base.simd_same_k &&
        (((D_half % 16) == 0 && all_base_aligned_16) || ((D_half % 8) == 0 && mixed_base_aligned_8));
    const bool p4_simd_candidate = p4_s8_uniform_simd || p4_s16_uniform_simd || p4_s8_s16c_simd;

    if (!p4_simd_candidate) {
        const int scalar_min_elements = (x->dtype == DATA_TYPE_FLOAT) ? 768 : 384;
        auto_dual = work_elements >= scalar_min_elements;
    } else if (x->dtype == DATA_TYPE_INT8) {
        // Below 32 Ki elements launch overhead dominates; above 128 Ki
        // elements the mixed/pure int8 kernels saturate external bandwidth.
        auto_dual = work_elements >= 32768 && work_elements <= 131072;
    } else {
        // int16 SIMD remains compute-heavy at the largest standard case.
        auto_dual = work_elements >= 32768;
    }
#else
    const int auto_min_elements = (x->dtype == DATA_TYPE_INT8) ? 2048 : 1024;
    auto_dual = work_elements >= auto_min_elements;
#endif
    auto_dual = token_count >= 4 && auto_dual;
    const bool dual =
#if CONFIG_FREERTOS_NUMBER_OF_CORES > 1
        can_split && (mode == RUNTIME_MODE_MULTI_CORE || (mode == RUNTIME_MODE_AUTO && auto_dual));
#else
        false;
#endif
    base.token_block_size = H >= 16 ? 16 : 8;
    if (!dual && D >= 128) {
        base.token_block_size = 1;
    }
    std::vector<RopeArgs> tasks(dual ? 2 : 1, base);
    if (dual) {
        tasks[0].token_start = 0;
        tasks[0].token_end = token_count / 2;
        tasks[1].token_start = tasks[0].token_end;
        tasks[1].token_end = token_count;
    } else {
        tasks[0].token_start = 0;
        tasks[0].token_end = token_count;
    }
    return tasks;
}

void rotary_embedding(void *args_ptr)
{
    if (args_ptr != nullptr) {
        rope_run_task(*static_cast<const RopeArgs *>(args_ptr));
    }
}

} // namespace base
} // namespace dl
