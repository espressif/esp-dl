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

#if CONFIG_PIE_V2_BOOST
static_assert(sizeof(void *) == 4, "ESP32-P4 MatMul assembly ABI requires RV32 pointers");
static_assert(offsetof(MatMulArgs<int8_t>, a) == 0 && offsetof(MatMulArgs<int8_t>, b) == 4 &&
                  offsetof(MatMulArgs<int8_t>, c) == 8 && offsetof(MatMulArgs<int8_t>, m) == 12 &&
                  offsetof(MatMulArgs<int8_t>, n) == 16 && offsetof(MatMulArgs<int8_t>, k) == 20 &&
                  offsetof(MatMulArgs<int8_t>, a_stride) == 24 && offsetof(MatMulArgs<int8_t>, b_stride) == 28 &&
                  offsetof(MatMulArgs<int8_t>, c_stride) == 32 && offsetof(MatMulArgs<int8_t>, row_start) == 36 &&
                  offsetof(MatMulArgs<int8_t>, row_end) == 40 && offsetof(MatMulArgs<int8_t>, mac_shift) == 44,
              "MatMulArgs layout changed; update the ESP32-P4 assembly ABI");
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
            const acc_t shifted = tool::shift_and_round<acc_t>(acc, args.mac_shift);
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

#if CONFIG_PIE_V2_BOOST
template <typename feature_t>
bool simd_compatible(const MatMulArgs<feature_t> &args, int32_t vector_elements)
{
    return args.k > 0 && args.n >= vector_elements;
}
#endif

} // namespace

template <>
void matmul<int8_t>(const MatMulArgs<int8_t> &args)
{
    if (!valid_args(args) || args.row_start == args.row_end || args.n == 0) {
        return;
    }

    int32_t simd_n = 0;
#if CONFIG_PIE_V2_BOOST
    constexpr int32_t vector_elements = 16;
    if (simd_compatible(args, vector_elements)) {
        simd_n = args.n & -vector_elements;
        MatMulArgs<int8_t> simd_args = args;
        simd_args.n = simd_n;
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
        dl_esp32p4_cfg_misalign(HW_MISALIGN, HW_MISALIGN);
        dl_esp32p4_s8_matmul_kn(&simd_args);
        dl_esp32p4_cfg_misalign(FORCE_ALIGN, FORCE_ALIGN);
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
#if CONFIG_PIE_V2_BOOST
    constexpr int32_t vector_elements = 8;
    if (simd_compatible(args, vector_elements)) {
        simd_n = args.n & -vector_elements;
        MatMulArgs<int16_t> simd_args = args;
        simd_args.n = simd_n;
        dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
        dl_esp32p4_cfg_misalign(HW_MISALIGN, HW_MISALIGN);
        dl_esp32p4_s16_matmul_kn(&simd_args);
        dl_esp32p4_cfg_misalign(FORCE_ALIGN, FORCE_ALIGN);
    }
#endif
    matmul_c(args, simd_n, args.n);
}

} // namespace base
} // namespace dl
