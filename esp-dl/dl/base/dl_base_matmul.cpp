#include "dl_base_matmul.hpp"

#include "dl_base_isa.hpp"
#include "dl_define_private.hpp"
#include "dl_tool.hpp"
#include <stddef.h>
#include <stdint.h>
#include <type_traits>

namespace dl {
namespace base {
namespace {

#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
static_assert(sizeof(void *) == 4, "MatMul assembly ABI requires 32-bit pointers");
static_assert(offsetof(MatMulArgs<int8_t>, a) == 0 && offsetof(MatMulArgs<int8_t>, b) == 4 &&
                  offsetof(MatMulArgs<int8_t>, c) == 8 && offsetof(MatMulArgs<int8_t>, m) == 12 &&
                  offsetof(MatMulArgs<int8_t>, n) == 16 && offsetof(MatMulArgs<int8_t>, k) == 20 &&
                  offsetof(MatMulArgs<int8_t>, a_stride) == 24 && offsetof(MatMulArgs<int8_t>, b_stride) == 28 &&
                  offsetof(MatMulArgs<int8_t>, c_stride) == 32 && offsetof(MatMulArgs<int8_t>, row_start) == 36 &&
                  offsetof(MatMulArgs<int8_t>, row_end) == 40 && offsetof(MatMulArgs<int8_t>, mac_shift) == 44 &&
                  offsetof(MatMulArgs<int8_t>, activation) == 48,
              "MatMulArgs layout changed; update the native-KN assembly ABI");
static_assert(sizeof(MatMulArgs<int8_t>) == sizeof(MatMulArgs<int16_t>),
              "MatMul argument layout must be identical for int8 and int16");
#endif

template <typename feature_t>
using accumulator_t = typename std::conditional<sizeof(feature_t) == 1, int32_t, int64_t>::type;

template <typename feature_t>
void matmul_c(const MatMulArgs<feature_t> &args, int32_t n_begin, int32_t n_end)
{
    using acc_t = accumulator_t<feature_t>;

    for (int32_t m = args.row_start; m < args.row_end; ++m) {
        const feature_t *a_row = args.a + m * args.a_stride;
        feature_t *c_row = args.c + m * args.c_stride;
        for (int32_t n = n_begin; n < n_end; ++n) {
            acc_t acc = 0;
            const feature_t *b_col = args.b + n;
            for (int32_t k = 0; k < args.k; ++k) {
                acc += static_cast<acc_t>(a_row[k]) * b_col[k * args.b_stride];
            }
            acc_t shifted = tool::shift_and_round<acc_t>(acc, args.mac_shift);
            if (args.activation == ReLU && shifted < 0) {
                shifted = 0;
            }
            tool::truncate(c_row[n], shifted);
        }
    }
}

template <typename feature_t>
bool valid_args(const MatMulArgs<feature_t> &args)
{
    return args.a && args.b && args.c && args.m >= 0 && args.n >= 0 && args.k >= 0 && args.a_stride >= args.k &&
        args.b_stride >= args.n && args.c_stride >= args.n && args.row_start >= 0 && args.row_start <= args.row_end &&
        args.row_end <= args.m;
}

#if CONFIG_PIE_V1_BOOST || CONFIG_PIE_V2_BOOST
template <typename feature_t>
bool simd_ready(const MatMulArgs<feature_t> &args, int32_t vector_elements)
{
    // Negative mac_shift is a left shift that the TIE728 rounding macros
    // (and the P4 srcmb rounding bit we use for native-KN) do not implement.
    return args.k > 0 && args.n >= vector_elements && args.mac_shift >= 0;
}

#if CONFIG_PIE_V1_BOOST
template <typename feature_t>
bool simd_aligned(const MatMulArgs<feature_t> &args, int32_t vector_elements)
{
    // Aligned kernel uses VLD.128.XP / VST.128.IP. Every K-step B row and
    // every output store must stay 16-byte aligned, so the row strides
    // have to be a multiple of the vector width.
    auto aligned16 = [](const void *ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0; };
    return (args.b_stride % vector_elements) == 0 && (args.c_stride % vector_elements) == 0 && aligned16(args.b) &&
        aligned16(args.c);
}
#endif
#endif

} // namespace

template <>
void matmul<int8_t>(const MatMulArgs<int8_t> &args)
{
    if (!valid_args(args) || args.row_start == args.row_end || args.n == 0) {
        return;
    }

    int32_t simd_n = 0;
#if CONFIG_PIE_V2_BOOST || CONFIG_PIE_V1_BOOST
    constexpr int32_t vector_elements = 16;
    if (simd_ready(args, vector_elements)) {
        simd_n = args.n & -vector_elements;
        MatMulArgs<int8_t> simd_args = args;
        simd_args.n = simd_n;
#if CONFIG_PIE_V2_BOOST
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
        dl_esp32p4_cfg_misalign(HW_MISALIGN, HW_MISALIGN);
        dl_esp32p4_s8_matmul_kn(&simd_args);
        dl_esp32p4_cfg_misalign(FORCE_ALIGN, FORCE_ALIGN);
#else
        if (simd_aligned(args, vector_elements)) {
            dl_tie728_s8_matmul_kn(&simd_args);
        } else {
            dl_tie728_s8_unaligned_matmul_kn(&simd_args);
        }
#endif
    }
#endif
    matmul_c(args, simd_n, args.n);
}

template <>
void matmul<int16_t>(const MatMulArgs<int16_t> &args)
{
    if (!valid_args(args) || args.row_start == args.row_end || args.n == 0) {
        return;
    }

    int32_t simd_n = 0;
#if CONFIG_PIE_V2_BOOST || CONFIG_PIE_V1_BOOST
    constexpr int32_t vector_elements = 8;
    if (simd_ready(args, vector_elements)) {
        simd_n = args.n & -vector_elements;
        MatMulArgs<int16_t> simd_args = args;
        simd_args.n = simd_n;
#if CONFIG_PIE_V2_BOOST
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
        dl_esp32p4_cfg_misalign(HW_MISALIGN, HW_MISALIGN);
        dl_esp32p4_s16_matmul_kn(&simd_args);
        dl_esp32p4_cfg_misalign(FORCE_ALIGN, FORCE_ALIGN);
#else
        if (simd_aligned(args, vector_elements)) {
            dl_tie728_s16_matmul_kn(&simd_args);
        } else {
            dl_tie728_s16_unaligned_matmul_kn(&simd_args);
        }
#endif
    }
#endif
    matmul_c(args, simd_n, args.n);
}

} // namespace base
} // namespace dl
