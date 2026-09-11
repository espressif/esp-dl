#include "sdkconfig.h"

#include "dl_base.hpp"
#include "dl_base_attention_kernels.hpp"
#include "dl_base_attention_softmax.hpp"
#include "dl_base_isa.hpp"

/**
 * @brief White-box Attention kernel tests.
 *
 * The kernels only exist for PIE v2 (ESP32-P4 / ESP32-S31).
 */
#if CONFIG_PIE_V2_BOOST

#include "dl_tool.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "unity.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

static const char *TAG = "TEST DL ATTENTION";

/**
 * @brief Tile / shift constants matching the Attention private headers so the tests exercise the
 * same code paths the module drives at runtime.
 *
 * These cannot take the header names directly: those live in an anonymous namespace.
 */
static constexpr int ATTENTION_KEY_TILE_SIZE = 64;
static constexpr int QK16_D32_SHIFT = 5;         /*!< INT16_QACC_QK_D32_SHIFT / QK16_D32_SHIFT */
static constexpr float QK16_DOT_SCALE = 0.3719f; /*!< stands in for logit_scale * 2^QK16_D32_SHIFT */

/**
 * @brief t-block-major repack (dst[g*16*D + d*16 + tl]) used by the QACC Q·K kernels.
 *
 * Mirrors build_k_head_tblk_s8 in the Attention base implementation, without the fast
 * 4x4 transpose (correctness path is enough for the test).
 */
static void build_k_head_tblk_s8_ref(int8_t *dst, const int8_t *src, int Lk, int D)
{
    const int n_group = (Lk + 15) / 16;
    for (int g = 0; g < n_group; ++g) {
        const int t0 = g * 16;
        const int nt = std::min(16, Lk - t0);
        int8_t *dg = dst + static_cast<size_t>(g) * 16 * static_cast<size_t>(D);
        std::memset(dg, 0, static_cast<size_t>(16) * static_cast<size_t>(D));
        for (int tl = 0; tl < nt; ++tl) {
            const int8_t *srow = src + static_cast<size_t>(t0 + tl) * static_cast<size_t>(D);
            for (int d = 0; d < D; ++d) {
                dg[d * 16 + tl] = srow[d];
            }
        }
    }
}

using QkQaccKernel = void (*)(int32_t *, const int8_t *, const int8_t *, int, int, int, int32_t *, int32_t *);

/**
 * @brief Multi-row, exact-sized input buffers plus guarded score rows exercise both the Q/K pointer
 * recurrences and every 16-token tail shape.
 */
static void run_qk_qacc_case(int D, int nq, int Lk, QkQaccKernel kernel)
{
    constexpr int guard = 16;
    constexpr int32_t sentinel = static_cast<int32_t>(0x5aa5c33cu);
    const int n_groups = (Lk + 15) / 16;
    const int padded = (Lk + 15) & ~15;
    const size_t k_blk_bytes = static_cast<size_t>(n_groups) * 16 * D;

    int8_t *k_blk =
        static_cast<int8_t *>(heap_caps_aligned_alloc(16, k_blk_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int8_t *k_src = static_cast<int8_t *>(
        heap_caps_aligned_alloc(16, static_cast<size_t>(Lk) * D, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int8_t *q = static_cast<int8_t *>(
        heap_caps_aligned_alloc(16, static_cast<size_t>(nq) * D, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int32_t *dots_storage = static_cast<int32_t *>(
        heap_caps_aligned_alloc(16,
                                sizeof(int32_t) * (guard + static_cast<size_t>(nq) * ATTENTION_KEY_TILE_SIZE + guard),
                                MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int32_t *row_max = static_cast<int32_t *>(
        heap_caps_aligned_alloc(16, sizeof(int32_t) * nq, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    TEST_ASSERT_NOT_NULL(k_blk);
    TEST_ASSERT_NOT_NULL(k_src);
    TEST_ASSERT_NOT_NULL(q);
    TEST_ASSERT_NOT_NULL(dots_storage);
    TEST_ASSERT_NOT_NULL(row_max);

    for (int r = 0; r < nq; ++r) {
        for (int d = 0; d < D; ++d) {
            q[r * D + d] = static_cast<int8_t>((r * 29 + d * 11 + 7) % 127 - 63);
        }
    }
    for (int t = 0; t < Lk; ++t) {
        for (int d = 0; d < D; ++d) {
            k_src[t * D + d] = static_cast<int8_t>((t * 31 + d * 17 + 5) % 127 - 63);
        }
    }
    build_k_head_tblk_s8_ref(k_blk, k_src, Lk, D);

    const size_t dots_count = guard + static_cast<size_t>(nq) * ATTENTION_KEY_TILE_SIZE + guard;
    std::fill_n(dots_storage, dots_count, sentinel);
    std::fill_n(row_max, nq, sentinel);
    int32_t *dots = dots_storage + guard;
    alignas(16) int32_t tmp16[16];
    kernel(dots, q, k_blk, nq, Lk, ATTENTION_KEY_TILE_SIZE, row_max, tmp16);

    for (int r = 0; r < nq; ++r) {
        int32_t ref_max = std::numeric_limits<int32_t>::min();
        for (int t = 0; t < Lk; ++t) {
            int32_t ref = 0;
            for (int d = 0; d < D; ++d) {
                ref += static_cast<int32_t>(q[r * D + d]) * static_cast<int32_t>(k_src[t * D + d]);
            }
            ref_max = std::max(ref_max, ref);
            TEST_ASSERT_EQUAL_INT32_MESSAGE(ref, dots[r * ATTENTION_KEY_TILE_SIZE + t], "QACC Q.K dot mismatch");
        }
        for (int t = Lk; t < padded; ++t) {
            TEST_ASSERT_EQUAL_INT32_MESSAGE(0, dots[r * ATTENTION_KEY_TILE_SIZE + t], "QACC Q.K padded lane mismatch");
        }
        for (int t = padded; t < ATTENTION_KEY_TILE_SIZE; ++t) {
            TEST_ASSERT_EQUAL_INT32_MESSAGE(
                sentinel, dots[r * ATTENTION_KEY_TILE_SIZE + t], "QACC Q.K wrote past padded row");
        }
        TEST_ASSERT_EQUAL_INT32_MESSAGE(ref_max, row_max[r], "QACC Q.K row_max mismatch");
    }
    for (int i = 0; i < guard; ++i) {
        TEST_ASSERT_EQUAL_INT32_MESSAGE(sentinel, dots_storage[i], "QACC Q.K prefix guard corrupted");
        TEST_ASSERT_EQUAL_INT32_MESSAGE(
            sentinel, dots_storage[guard + nq * ATTENTION_KEY_TILE_SIZE + i], "QACC Q.K suffix guard corrupted");
    }

    heap_caps_free(k_blk);
    heap_caps_free(k_src);
    heap_caps_free(q);
    heap_caps_free(dots_storage);
    heap_caps_free(row_max);
}

TEST_CASE("Test dl base attention s8 qk_qacc: guarded rows and every tail", "[pie][attention]")
{
    for (int Lk : {1, 15, 16, 17, 37, 63, 64}) {
        ESP_LOGI(TAG, "attention_qk_i8_qacc D=64 Lk=%d", Lk);
        run_qk_qacc_case(64, 3, Lk, dl_esp32p4_s8_attention_qk_qacc_d64);
    }
}

TEST_CASE("Test dl base attention s8 qk_qacc_d32: guarded rows and every tail", "[pie][attention]")
{
    for (int Lk : {1, 15, 16, 17, 37, 63, 64}) {
        ESP_LOGI(TAG, "attention_qk_i8_qacc_d32 D=32 Lk=%d", Lk);
        run_qk_qacc_case(32, 3, Lk, dl_esp32p4_s8_attention_qk_qacc_d32);
    }
}

/**
 * @brief extreme=true fills Q and K with -128, the largest |dot| (D * 2^14) per D.
 */
static void run_qk_qacc_generic_case(int D, int nq, int Lk, bool extreme)
{
    const int n_groups = (Lk + 15) / 16;
    int8_t *k_blk = static_cast<int8_t *>(
        heap_caps_aligned_alloc(16, static_cast<size_t>(n_groups) * 16 * D, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int8_t *k_src = static_cast<int8_t *>(
        heap_caps_aligned_alloc(16, static_cast<size_t>(Lk) * D, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int8_t *q = static_cast<int8_t *>(
        heap_caps_aligned_alloc(16, static_cast<size_t>(nq) * D, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int32_t *dots = static_cast<int32_t *>(heap_caps_aligned_alloc(
        16, sizeof(int32_t) * nq * ATTENTION_KEY_TILE_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    TEST_ASSERT_NOT_NULL(k_blk);
    TEST_ASSERT_NOT_NULL(k_src);
    TEST_ASSERT_NOT_NULL(q);
    TEST_ASSERT_NOT_NULL(dots);

    for (int i = 0; i < nq; ++i) {
        for (int d = 0; d < D; ++d) {
            q[i * D + d] = extreme ? int8_t(-128) : static_cast<int8_t>((i * 29 + d * 11 + 7) % 255 - 127);
        }
    }
    for (int t = 0; t < Lk; ++t) {
        for (int d = 0; d < D; ++d) {
            k_src[t * D + d] = extreme ? int8_t(-128) : static_cast<int8_t>((t * 31 + d * 17 + 5) % 255 - 127);
        }
    }
    build_k_head_tblk_s8_ref(k_blk, k_src, Lk, D);

    alignas(16) int32_t tmp16[16];
    int32_t row_max[4] = {0, 0, 0, 0};
    std::memset(dots, 0, sizeof(int32_t) * nq * ATTENTION_KEY_TILE_SIZE);
    dl_esp32p4_s8_attention_qk_qacc_generic(dots, q, k_blk, nq, Lk, ATTENTION_KEY_TILE_SIZE, row_max, tmp16, D);

    for (int i = 0; i < nq; ++i) {
        int32_t ref_max = std::numeric_limits<int32_t>::min();
        for (int t = 0; t < Lk; ++t) {
            int32_t ref = 0;
            for (int d = 0; d < D; ++d) {
                ref += static_cast<int32_t>(q[i * D + d]) * static_cast<int32_t>(k_src[t * D + d]);
            }
            ref_max = std::max(ref_max, ref);
            TEST_ASSERT_EQUAL_INT32_MESSAGE(ref, dots[i * ATTENTION_KEY_TILE_SIZE + t], "QACC generic dot mismatch");
        }
        TEST_ASSERT_EQUAL_INT32_MESSAGE(ref_max, row_max[i], "QACC generic row_max mismatch");
    }

    heap_caps_free(k_blk);
    heap_caps_free(k_src);
    heap_caps_free(q);
    heap_caps_free(dots);
}

TEST_CASE("Test dl base attention s8 qk_qacc_generic: every D_pad, rows and tails", "[pie][attention]")
{
    for (int D = 16; D <= 128; D += 16) {
        ESP_LOGI(TAG, "attention_qk_i8_qacc_generic D=%d", D);
        run_qk_qacc_generic_case(D, 3, 37, false);
        run_qk_qacc_generic_case(D, 2, 64, false);
        run_qk_qacc_generic_case(D, 1, 5, false);
    }
}

TEST_CASE("Test dl base attention s8 qk_qacc: extreme dots fit the QACC lanes", "[pie][attention]")
{
    run_qk_qacc_generic_case(128, 2, 37, true);
    run_qk_qacc_generic_case(48, 2, 37, true);
}

/**
 * @brief int16 t-block repack (dst[g*8*D + d*8 + tl]) matching build_k_head_tblk_s16 in
 * attention_kernels.hpp. 8-token groups because s16 vsmulas.qacc broadcasts one of 8 lanes per MAC
 * and the QACC low half stores 8 int32 dots.
 */
static void build_k_head_tblk_s16_ref(int16_t *dst, const int16_t *src, int Lk, int D)
{
    const int n_group = (Lk + 7) / 8;
    for (int g = 0; g < n_group; ++g) {
        const int t0 = g * 8;
        const int nt = std::min(8, Lk - t0);
        int16_t *dg = dst + static_cast<size_t>(g) * 8 * static_cast<size_t>(D);
        std::memset(dg, 0, static_cast<size_t>(8) * static_cast<size_t>(D) * sizeof(int16_t));
        for (int tl = 0; tl < nt; ++tl) {
            const int16_t *srow = src + static_cast<size_t>(t0 + tl) * static_cast<size_t>(D);
            for (int d = 0; d < D; ++d) {
                dg[d * 8 + tl] = srow[d];
            }
        }
    }
}

static int64_t shift_half_even(int64_t v, int s)
{
    if (s == 0) {
        return v;
    }
    int64_t out = v >> s;
    const int64_t rem = v - (out << s);
    const int64_t half = int64_t(1) << (s - 1);
    if (rem > half || (rem == half && (out & 1))) {
        ++out;
    }
    return out;
}

static round_mode_t current_round_mode()
{
    return static_cast<round_mode_t>((dl_esp32p4_get_cfg() >> 4) & 0xf);
}

static uint32_t float_bits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

static int64_t qk16_exact_dot(const int16_t *q, const int16_t *k_row, int D)
{
    int64_t acc = 0;
    for (int d = 0; d < D; ++d) {
        acc += static_cast<int64_t>(q[d]) * static_cast<int64_t>(k_row[d]);
    }
    return acc;
}

/**
 * @brief Runs the kernel for one Q row.
 *
 * Lanes past Lk are primed with a sentinel so the padding contract (tail lanes up to the next
 * multiple of 8 become 0, nothing further is written) is checked too.
 */
static void run_qk16_qacc(float *scores, const int16_t *q, const int16_t *k_blk, int Lk)
{
    alignas(16) int32_t tmp16[16] = {};
    for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
        scores[t] = -1.0f;
    }
    dl_esp32p4_s16_attention_qk_qacc_d32(
        scores, q, k_blk, /*nq=*/1, Lk, ATTENTION_KEY_TILE_SIZE, QK16_DOT_SCALE, tmp16);
    const int padded = (Lk + 7) & ~7;
    for (int t = Lk; t < ATTENTION_KEY_TILE_SIZE; ++t) {
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(
            float_bits(t < padded ? 0.0f : -1.0f), float_bits(scores[t]), "s16 QACC padding lanes");
    }
}

/**
 * @brief Bit-exact against float(round_half_even(dot / 2^5)) * scale.
 */
static void expect_qk16_scores(const float *scores, const int16_t *q, const int16_t *k_src, int D, int Lk)
{
    for (int t = 0; t < Lk; ++t) {
        const int32_t dot = static_cast<int32_t>(shift_half_even(qk16_exact_dot(q, k_src + t * D, D), QK16_D32_SHIFT));
        const float expect = static_cast<float>(dot) * QK16_DOT_SCALE;
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(float_bits(expect), float_bits(scores[t]), "s16 QACC score mismatch");
    }
}

// Full-range data (many dots exceed int32); Lk=19 = 2 full 8-token groups
// plus a 3-token tail.
TEST_CASE("Test dl base attention s16 qk_qacc_d32: D=32 scores", "[pie][attention]")
{
    ESP_LOGI(TAG, "attention_qk_i16_qacc_d32 D=32");
    constexpr int D = 32;
    constexpr int Lk = 19;
    const int n_groups = (Lk + 7) / 8;
    const size_t k_blk_elems = static_cast<size_t>(n_groups) * 8 * D;

    int16_t *k_blk = static_cast<int16_t *>(
        heap_caps_aligned_alloc(16, k_blk_elems * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *k_src = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(Lk) * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *q =
        static_cast<int16_t *>(heap_caps_aligned_alloc(16, D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    TEST_ASSERT_NOT_NULL(k_blk);
    TEST_ASSERT_NOT_NULL(k_src);
    TEST_ASSERT_NOT_NULL(q);

    for (int d = 0; d < D; ++d) {
        q[d] = static_cast<int16_t>((d * 7919 + 13) % 65536 - 32768);
    }
    for (int t = 0; t < Lk; ++t) {
        for (int d = 0; d < D; ++d) {
            k_src[t * D + d] = static_cast<int16_t>((t * 12289 + d * 4099 + 5) % 65536 - 32768);
        }
    }
    build_k_head_tblk_s16_ref(k_blk, k_src, Lk, D);

    alignas(16) float scores[ATTENTION_KEY_TILE_SIZE];
    run_qk16_qacc(scores, q, k_blk, Lk);
    expect_qk16_scores(scores, q, k_src, D, Lk);

    heap_caps_free(k_blk);
    heap_caps_free(k_src);
    heap_caps_free(q);
}

// int16 dots beyond int32, at the extremes of the int16 range.
TEST_CASE("Test dl base attention s16 qk: dots beyond int32", "[pie][attention]")
{
    constexpr int D = 32;
    constexpr int Lk = 8;
    static const int16_t k_fill[Lk] = {32767, -32767, 2049, -2049, 1, -1, 0, 16384};

    int16_t *k_blk = static_cast<int16_t *>(
        heap_caps_aligned_alloc(16, 8 * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *k_src = static_cast<int16_t *>(
        heap_caps_aligned_alloc(16, (Lk * D + 16) * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *q = static_cast<int16_t *>(
        heap_caps_aligned_alloc(16, (D + 16) * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    TEST_ASSERT_NOT_NULL(k_blk);
    TEST_ASSERT_NOT_NULL(k_src);
    TEST_ASSERT_NOT_NULL(q);
    std::memset(q, 0, (D + 16) * sizeof(int16_t));
    std::memset(k_src, 0, (Lk * D + 16) * sizeof(int16_t));
    for (int d = 0; d < D; ++d) {
        q[d] = 32767;
    }
    for (int t = 0; t < Lk; ++t) {
        for (int d = 0; d < D; ++d) {
            k_src[t * D + d] = k_fill[t];
        }
    }
    build_k_head_tblk_s16_ref(k_blk, k_src, Lk, D);

    alignas(16) float scores[ATTENTION_KEY_TILE_SIZE];
    run_qk16_qacc(scores, q, k_blk, Lk);
    expect_qk16_scores(scores, q, k_src, D, Lk);

    heap_caps_free(k_blk);
    heap_caps_free(k_src);
    heap_caps_free(q);
}

// Exact .5 ties after the >>5: dot = r + 32*m with q = {1, 32, 0, ...}.
TEST_CASE("Test dl base attention s16 qk: QACC rounds half-even", "[pie][attention]")
{
    constexpr int D = 32;
    constexpr int Lk = 8;
    static const int16_t rem_fill[Lk] = {16, 16, -16, -16, 48, -48, 17, 15};
    static const int16_t mul_fill[Lk] = {1001, 1000, -1001, -1000, 0, 0, 0, 0};

    int16_t *k_blk = static_cast<int16_t *>(
        heap_caps_aligned_alloc(16, 8 * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *k_src = static_cast<int16_t *>(
        heap_caps_aligned_alloc(16, (Lk * D + 16) * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *q = static_cast<int16_t *>(
        heap_caps_aligned_alloc(16, (D + 16) * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    TEST_ASSERT_NOT_NULL(k_blk);
    TEST_ASSERT_NOT_NULL(k_src);
    TEST_ASSERT_NOT_NULL(q);
    std::memset(q, 0, (D + 16) * sizeof(int16_t));
    std::memset(k_src, 0, (Lk * D + 16) * sizeof(int16_t));
    q[0] = 1;
    q[1] = 32;
    for (int t = 0; t < Lk; ++t) {
        k_src[t * D] = rem_fill[t];
        k_src[t * D + 1] = mul_fill[t];
    }
    build_k_head_tblk_s16_ref(k_blk, k_src, Lk, D);

    alignas(16) float scores[ATTENTION_KEY_TILE_SIZE];
    run_qk16_qacc(scores, q, k_blk, Lk);
    expect_qk16_scores(scores, q, k_src, D, Lk);

    heap_caps_free(k_blk);
    heap_caps_free(k_src);
    heap_caps_free(q);
}

static int dot_shift_for(int n)
{
    int s = 0;
    while ((1 << s) < n) {
        ++s;
    }
    return s;
}

/**
 * @brief Generic-D kernel: nq rows, Lk tokens (two full groups + tail), score rows padded with a
 * sentinel past the 8-lane round-up.
 */
static void run_qk16_generic_case(int D, int Lk, int nq, bool extremes)
{
    const int n_groups = (Lk + 7) / 8;
    const int shift = dot_shift_for(D);
    const float dot_scale = QK16_DOT_SCALE;
    int16_t *k_blk = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(n_groups) * 8 * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *k_src = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(Lk) * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *q = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(nq) * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *scores = static_cast<float *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(nq) * ATTENTION_KEY_TILE_SIZE * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    TEST_ASSERT_NOT_NULL(k_blk);
    TEST_ASSERT_NOT_NULL(k_src);
    TEST_ASSERT_NOT_NULL(q);
    TEST_ASSERT_NOT_NULL(scores);

    for (int r = 0; r < nq; ++r) {
        for (int d = 0; d < D; ++d) {
            q[r * D + d] = extremes ? static_cast<int16_t>(r & 1 ? 32767 : -32768)
                                    : static_cast<int16_t>(((r * 131 + d) * 7919 + 13) % 65536 - 32768);
        }
    }
    for (int t = 0; t < Lk; ++t) {
        for (int d = 0; d < D; ++d) {
            k_src[t * D + d] = extremes ? static_cast<int16_t>(t & 1 ? -32768 : 32767)
                                        : static_cast<int16_t>((t * 12289 + d * 4099 + 5) % 65536 - 32768);
        }
    }
    build_k_head_tblk_s16_ref(k_blk, k_src, Lk, D);

    for (int i = 0; i < nq * ATTENTION_KEY_TILE_SIZE; ++i) {
        scores[i] = -1.0f;
    }
    alignas(16) int32_t tmp16[16] = {};
    dl_esp32p4_s16_attention_qk_qacc_generic(scores, q, k_blk, nq, Lk, ATTENTION_KEY_TILE_SIZE, tmp16, D, dot_scale);

    const int padded = (Lk + 7) & ~7;
    const round_mode_t saved = current_round_mode();
    dl_esp32p4_cfg_round(ROUND_MODE_HALF_EVEN);
    for (int r = 0; r < nq; ++r) {
        const int16_t *qr = q + r * D;
        const float *sr = scores + r * ATTENTION_KEY_TILE_SIZE;
        for (int t = 0; t < Lk; ++t) {
            const int32_t dot = static_cast<int32_t>(shift_half_even(qk16_exact_dot(qr, k_src + t * D, D), shift));
            const float expect = static_cast<float>(dot) * dot_scale;
            TEST_ASSERT_EQUAL_HEX32_MESSAGE(float_bits(expect), float_bits(sr[t]), "s16 QACC generic score mismatch");
        }
        for (int t = Lk; t < ATTENTION_KEY_TILE_SIZE; ++t) {
            TEST_ASSERT_EQUAL_HEX32_MESSAGE(
                float_bits(t < padded ? 0.0f : -1.0f), float_bits(sr[t]), "s16 QACC generic padding lanes");
        }
    }
    dl_esp32p4_cfg_round(saved);

    heap_caps_free(k_blk);
    heap_caps_free(k_src);
    heap_caps_free(q);
    heap_caps_free(scores);
}

TEST_CASE("Test dl base attention s16 qk_qacc_generic: aligned D scores", "[pie][attention]")
{
    for (int d : {8, 16, 24, 32, 40, 48, 56, 64}) {
        ESP_LOGI(TAG, "attention_qk_i16_qacc_generic D=%d", d);
        run_qk16_generic_case(d, 19, 3, false);
        run_qk16_generic_case(d, 64, 2, false);
    }
}

TEST_CASE("Test dl base attention s16: padded D preserves QK and V", "[pie][attention]")
{
    constexpr int Lk = 11;
    constexpr int nq = 2;
    constexpr int n_tok = 16;
    constexpr float dot_scale = 1.0f / 32.0f;
    for (int D : {1, 2, 3, 4, 5, 6, 7, 10, 31, 50, 63}) {
        const int D_pad = (D + 7) & ~7;
        const int n_groups = (Lk + 7) / 8;
        int16_t *q_src = static_cast<int16_t *>(heap_caps_aligned_alloc(
            16, static_cast<size_t>(nq) * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        int16_t *q_pad = static_cast<int16_t *>(heap_caps_aligned_alloc(
            16, static_cast<size_t>(nq) * D_pad * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        int16_t *k_src = static_cast<int16_t *>(heap_caps_aligned_alloc(
            16, static_cast<size_t>(Lk) * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        int16_t *k_blk = static_cast<int16_t *>(heap_caps_aligned_alloc(
            16, static_cast<size_t>(n_groups) * 8 * D_pad * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        int16_t *v_src = static_cast<int16_t *>(heap_caps_aligned_alloc(
            16, static_cast<size_t>(Lk) * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        int16_t *v_dblk = static_cast<int16_t *>(heap_caps_aligned_alloc(
            16, static_cast<size_t>(n_tok) * D_pad * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        TEST_ASSERT_NOT_NULL(q_src);
        TEST_ASSERT_NOT_NULL(q_pad);
        TEST_ASSERT_NOT_NULL(k_src);
        TEST_ASSERT_NOT_NULL(k_blk);
        TEST_ASSERT_NOT_NULL(v_src);
        TEST_ASSERT_NOT_NULL(v_dblk);

        for (int r = 0; r < nq; ++r) {
            for (int d = 0; d < D; ++d) {
                q_src[r * D + d] = static_cast<int16_t>((r * 131 + d * 17 + 3) % 4093 - 2046);
            }
        }
        for (int t = 0; t < Lk; ++t) {
            for (int d = 0; d < D; ++d) {
                k_src[t * D + d] = static_cast<int16_t>((t * 257 + d * 29 + 5) % 4093 - 2046);
                v_src[t * D + d] = static_cast<int16_t>((t * 193 + d * 31 + 7) % 4093 - 2046);
            }
        }

        dl::base::copy_q_block_to_tile_buf<dl::DATA_TYPE_INT16>(
            reinterpret_cast<uint8_t *>(q_pad), q_src, 0, 0, nq, D, D_pad, D);
        dl::base::build_k_head_tblk_s16(k_blk, k_src, Lk, D, D_pad, D);
        dl::base::load_v_tile_dblk_s16(v_dblk, v_src, 0, 0, Lk, n_tok, D, D_pad, D);

        for (int r = 0; r < nq; ++r) {
            for (int d = D; d < D_pad; ++d) {
                TEST_ASSERT_EQUAL_INT16_MESSAGE(0, q_pad[r * D_pad + d], "padded Q lane is not zero");
            }
        }
        for (int dc = 0; dc < D_pad / 8; ++dc) {
            for (int t = 0; t < n_tok; ++t) {
                for (int i = 0; i < 8; ++i) {
                    const int d = dc * 8 + i;
                    const int16_t expected = (t < Lk && d < D) ? v_src[t * D + d] : 0;
                    TEST_ASSERT_EQUAL_INT16_MESSAGE(
                        expected, v_dblk[(dc * n_tok + t) * 8 + i], "d-block V padding mismatch");
                }
            }
        }

        alignas(16) float scores[nq * ATTENTION_KEY_TILE_SIZE] = {};
        alignas(16) int32_t qacc_buffer[16] = {};
        const int shift = dot_shift_for(D);
        if (D_pad == 32) {
            dl_esp32p4_s16_attention_qk_qacc_d32(
                scores, q_pad, k_blk, nq, Lk, ATTENTION_KEY_TILE_SIZE, dot_scale, qacc_buffer);
        } else {
            dl_esp32p4_s16_attention_qk_qacc_generic(
                scores, q_pad, k_blk, nq, Lk, ATTENTION_KEY_TILE_SIZE, qacc_buffer, D, dot_scale);
        }
        for (int r = 0; r < nq; ++r) {
            for (int t = 0; t < Lk; ++t) {
                const int64_t dot = qk16_exact_dot(q_src + r * D, k_src + t * D, D);
                const float expected = static_cast<float>(shift_half_even(dot, shift)) * dot_scale;
                TEST_ASSERT_EQUAL_HEX32_MESSAGE(float_bits(expected),
                                                float_bits(scores[r * ATTENTION_KEY_TILE_SIZE + t]),
                                                "padded int16 QK score mismatch");
            }
        }

        heap_caps_free(q_src);
        heap_caps_free(q_pad);
        heap_caps_free(k_src);
        heap_caps_free(k_blk);
        heap_caps_free(v_src);
        heap_caps_free(v_dblk);
    }
}

TEST_CASE("Test dl base attention s16: BNHD head-chunk K/V pack matches per-head", "[pie][attention]")
{
    constexpr int chunk_heads = 4;
    constexpr int Lk = 20;
    constexpr int D = 16;
    constexpr int D_pad = 16;
    constexpr int src_stride = chunk_heads * D;
    const int n_group = (Lk + 7) / 8;
    const size_t k_head_elems = static_cast<size_t>(n_group) * 8 * D_pad;
    const int n_tiles = (Lk + ATTENTION_KEY_TILE_SIZE - 1) / ATTENTION_KEY_TILE_SIZE;
    const int tile_n = std::min(ATTENTION_KEY_TILE_SIZE, Lk);
    const int n_tok = (tile_n + 7) & ~7;
    const int v_tile_stride = D_pad * ATTENTION_KEY_TILE_SIZE * static_cast<int>(sizeof(int16_t));

    std::vector<int16_t> src(static_cast<size_t>(Lk) * src_stride);
    for (int t = 0; t < Lk; ++t) {
        for (int h = 0; h < chunk_heads; ++h) {
            for (int d = 0; d < D; ++d) {
                src[t * src_stride + h * D + d] = static_cast<int16_t>(t * 17 + h * 13 + d);
            }
        }
    }

    std::vector<int16_t> packed(chunk_heads * k_head_elems, 1);
    std::vector<int16_t> per_head(chunk_heads * k_head_elems, 2);
    uint8_t staging[4][256];
    dl::base::build_k_heads_tblk_s16(
        packed.data(), k_head_elems, src.data(), chunk_heads, Lk, D, D_pad, src_stride, 0, 100, staging);
    for (int h = 0; h < chunk_heads; ++h) {
        dl::base::build_k_head_tblk_s16(
            per_head.data() + static_cast<size_t>(h) * k_head_elems, src.data() + h * D, Lk, D, D_pad, src_stride);
    }
    TEST_ASSERT_EQUAL_INT16_ARRAY(per_head.data(), packed.data(), static_cast<int>(packed.size()));

    std::vector<int16_t> split = per_head;
    std::vector<int16_t> split_ref = per_head;
    dl::base::build_k_heads_tblk_s16(
        split.data(), k_head_elems, src.data(), chunk_heads, Lk, D, D_pad, src_stride, 1, 2, staging);
    for (int h = 0; h < chunk_heads; ++h) {
        dl::base::build_k_head_tblk_s16(split_ref.data() + static_cast<size_t>(h) * k_head_elems,
                                        src.data() + h * D,
                                        Lk,
                                        D,
                                        D_pad,
                                        src_stride,
                                        1,
                                        2);
    }
    TEST_ASSERT_EQUAL_INT16_ARRAY(split_ref.data(), split.data(), static_cast<int>(split.size()));

    constexpr int D_narrow = 12;
    constexpr int D_pad_narrow = 16;
    constexpr int stride_narrow = chunk_heads * D_narrow;
    std::vector<int16_t> src_narrow(static_cast<size_t>(Lk) * stride_narrow);
    for (int t = 0; t < Lk; ++t) {
        for (int i = 0; i < stride_narrow; ++i) {
            src_narrow[t * stride_narrow + i] = static_cast<int16_t>(t * 3 + i + 1);
        }
    }
    const size_t k_head_narrow = static_cast<size_t>(n_group) * 8 * D_pad_narrow;
    std::vector<int16_t> packed_narrow(chunk_heads * k_head_narrow, 1);
    std::vector<int16_t> per_head_narrow(chunk_heads * k_head_narrow, 2);
    dl::base::build_k_heads_tblk_s16(packed_narrow.data(),
                                     k_head_narrow,
                                     src_narrow.data(),
                                     chunk_heads,
                                     Lk,
                                     D_narrow,
                                     D_pad_narrow,
                                     stride_narrow,
                                     0,
                                     100,
                                     staging);
    for (int h = 0; h < chunk_heads; ++h) {
        dl::base::build_k_head_tblk_s16(per_head_narrow.data() + static_cast<size_t>(h) * k_head_narrow,
                                        src_narrow.data() + h * D_narrow,
                                        Lk,
                                        D_narrow,
                                        D_pad_narrow,
                                        stride_narrow);
    }
    TEST_ASSERT_EQUAL_INT16_ARRAY(per_head_narrow.data(), packed_narrow.data(), static_cast<int>(packed_narrow.size()));
    const int v_stride_narrow = D_pad_narrow * ATTENTION_KEY_TILE_SIZE * static_cast<int>(sizeof(int16_t));
    std::vector<int8_t> v_packed_narrow(static_cast<size_t>(chunk_heads * n_tiles) * v_stride_narrow, 0);
    std::vector<int8_t> v_ref_narrow(static_cast<size_t>(chunk_heads * n_tiles) * v_stride_narrow, 0);
    dl::base::load_v_tile_dblk_heads_s16(v_packed_narrow.data(),
                                         n_tiles,
                                         v_stride_narrow,
                                         chunk_heads,
                                         src_narrow.data(),
                                         0,
                                         0,
                                         tile_n,
                                         n_tok,
                                         D_narrow,
                                         D_pad_narrow,
                                         stride_narrow);
    for (int h = 0; h < chunk_heads; ++h) {
        dl::base::load_v_tile_dblk_s16(reinterpret_cast<int16_t *>(v_ref_narrow.data() + h * n_tiles * v_stride_narrow),
                                       src_narrow.data(),
                                       h * D_narrow,
                                       0,
                                       tile_n,
                                       n_tok,
                                       D_narrow,
                                       D_pad_narrow,
                                       stride_narrow);
    }
    TEST_ASSERT_EQUAL_INT8_ARRAY(v_ref_narrow.data(), v_packed_narrow.data(), static_cast<int>(v_packed_narrow.size()));

    std::vector<int8_t> v_packed(static_cast<size_t>(chunk_heads * n_tiles) * v_tile_stride, 0);
    std::vector<int8_t> v_ref(static_cast<size_t>(chunk_heads * n_tiles) * v_tile_stride, 0);
    dl::base::load_v_tile_dblk_heads_s16(
        v_packed.data(), n_tiles, v_tile_stride, chunk_heads, src.data(), 0, 0, tile_n, n_tok, D, D_pad, src_stride);
    for (int h = 0; h < chunk_heads; ++h) {
        dl::base::load_v_tile_dblk_s16(reinterpret_cast<int16_t *>(v_ref.data() + h * n_tiles * v_tile_stride),
                                       src.data(),
                                       h * D,
                                       0,
                                       tile_n,
                                       n_tok,
                                       D,
                                       D_pad,
                                       src_stride);
    }
    TEST_ASSERT_EQUAL_INT8_ARRAY(v_ref.data(), v_packed.data(), static_cast<int>(v_packed.size()));
}

// |dot| = D * 2^30 (2^36 at D=64): the s16 QACC lanes must carry it.
TEST_CASE("Test dl base attention s16 qk_qacc_generic: extreme dots", "[pie][attention]")
{
    for (int d : {8, 24, 32, 64}) {
        run_qk16_generic_case(d, 16, 2, true);
    }
}

static void load_v_tile_dblk_s16_ref(int16_t *dst, const int16_t *v, int tile_n, int n_tok, int D)
{
    for (int dc = 0; dc < D / 8; ++dc) {
        for (int t = 0; t < n_tok; ++t) {
            for (int i = 0; i < 8; ++i) {
                dst[(dc * n_tok + t) * 8 + i] = (t < tile_n) ? v[t * D + dc * 8 + i] : 0;
            }
        }
    }
}

/**
 * @brief Split weights reproduce e to 2^-29, and the kernel adds on top of acc.
 */
static void run_pv16_case(int D, int tile_n, int nq, bool extremes)
{
    const int n_tok = (tile_n + 7) & ~7;
    int16_t *v = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(tile_n) * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *v_dblk = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(n_tok) * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *e = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(nq) * 2 * n_tok * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *ef = static_cast<float *>(
        heap_caps_malloc(static_cast<size_t>(nq) * n_tok * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *acc = static_cast<float *>(
        heap_caps_aligned_alloc(16,
                                static_cast<size_t>(nq) * dl::base::ATTENTION_MAX_HEAD_DIM * sizeof(float),
                                MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *acc_ref = static_cast<float *>(
        heap_caps_aligned_alloc(16,
                                static_cast<size_t>(nq) * dl::base::ATTENTION_MAX_HEAD_DIM * sizeof(float),
                                MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    TEST_ASSERT_NOT_NULL(v);
    TEST_ASSERT_NOT_NULL(v_dblk);
    TEST_ASSERT_NOT_NULL(e);
    TEST_ASSERT_NOT_NULL(ef);
    TEST_ASSERT_NOT_NULL(acc);
    TEST_ASSERT_NOT_NULL(acc_ref);

    for (int t = 0; t < tile_n; ++t) {
        for (int d = 0; d < D; ++d) {
            v[t * D + d] = extremes ? static_cast<int16_t>((d & 1) ? -32768 : 32767)
                                    : static_cast<int16_t>((t * 12289 + d * 4099 + 5) % 65536 - 32768);
        }
    }
    load_v_tile_dblk_s16_ref(v_dblk, v, tile_n, n_tok, D);

    for (int r = 0; r < nq; ++r) {
        int16_t *hi = e + r * 2 * n_tok;
        int16_t *lo = hi + n_tok;
        for (int t = 0; t < n_tok; ++t) {
            float y = 0.f;
            if (t < tile_n) {
                y = extremes ? 1.0f : std::exp(-0.37f * static_cast<float>((t * 7 + r * 3) % 23));
            }
            ef[r * n_tok + t] = y;
            dl::base::split_e_s16(y, hi + t, lo + t);
        }
        for (int d = 0; d < dl::base::ATTENTION_MAX_HEAD_DIM; ++d) {
            acc[r * dl::base::ATTENTION_MAX_HEAD_DIM + d] = 0.25f * static_cast<float>(d);
            acc_ref[r * dl::base::ATTENTION_MAX_HEAD_DIM + d] = acc[r * dl::base::ATTENTION_MAX_HEAD_DIM + d];
        }
    }

    const float v_scale = 1.0f / 4096.0f;
    alignas(16) int32_t tmp32[32];
    dl::base::attention_pv_s16_qacc_c(
        acc_ref, e, v_dblk, nq, n_tok, D, dl::base::ATTENTION_MAX_HEAD_DIM, v_scale / 536870912.0f);
    dl_esp32p4_s16_attention_pv_qacc(
        acc, e, v_dblk, nq, n_tok, D, dl::base::ATTENTION_MAX_HEAD_DIM, tmp32, v_scale / 536870912.0f);

    for (int r = 0; r < nq; ++r) {
        for (int d = 0; d < D; ++d) {
            TEST_ASSERT_EQUAL_HEX32_MESSAGE(float_bits(acc_ref[r * dl::base::ATTENTION_MAX_HEAD_DIM + d]),
                                            float_bits(acc[r * dl::base::ATTENTION_MAX_HEAD_DIM + d]),
                                            "s16 QACC P.V differs from exact C integer reference");
            double ideal = 0.0;
            double mag = 0.0;
            for (int t = 0; t < tile_n; ++t) {
                const double term = static_cast<double>(ef[r * n_tok + t]) * v[t * D + d] * v_scale;
                ideal += term;
                mag += std::fabs(term);
            }
            const double init = 0.25 * d;
            const double got = static_cast<double>(acc[r * dl::base::ATTENTION_MAX_HEAD_DIM + d]) - init;
            // Split error is below 2^-29 per weight; the rest is fp32
            // rounding of the lane total and the final add.
            const double tol = tile_n * 32768.0 * v_scale * std::ldexp(1.0, -29) +
                (mag + std::fabs(init) + std::fabs(ideal)) * std::ldexp(1.0, -22);
            if (std::fabs(got - ideal) > tol) {
                ESP_LOGE(
                    TAG, "pv16 D=%d n=%d r=%d d=%d got=%.9g ideal=%.9g tol=%.3g", D, tile_n, r, d, got, ideal, tol);
            }
            TEST_ASSERT_TRUE_MESSAGE(std::fabs(got - ideal) <= tol, "s16 QACC P.V mismatch");
        }
        for (int d = D; d < dl::base::ATTENTION_MAX_HEAD_DIM; ++d) {
            TEST_ASSERT_EQUAL_FLOAT_MESSAGE(
                0.25f * d, acc[r * dl::base::ATTENTION_MAX_HEAD_DIM + d], "s16 QACC P.V wrote past D");
        }
    }

    heap_caps_free(v);
    heap_caps_free(v_dblk);
    heap_caps_free(e);
    heap_caps_free(ef);
    heap_caps_free(acc);
    heap_caps_free(acc_ref);
}

TEST_CASE("Test dl base attention s16 pv_qacc: split-weight lanes", "[pie][attention]")
{
    for (int d : {8, 24, 32, 64}) {
        for (int n : {1, 8, 13, 64}) {
            ESP_LOGI(TAG, "attention_pv_i16_qacc D=%d tile_n=%d", d, n);
            run_pv16_case(d, n, 3, false);
        }
        run_pv16_case(d, 64, 2, true);
    }
}

static void benchmark_pv16_case(int D, int nq)
{
    constexpr int n_tok = ATTENTION_KEY_TILE_SIZE;
    constexpr int rounds = 7;
    constexpr int iterations = 200;
    const int acc_stride = std::max(D, dl::base::ATTENTION_MAX_HEAD_DIM);
    int16_t *v_dblk = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(n_tok) * D * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int16_t *e = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(nq) * 2 * n_tok * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *acc = static_cast<float *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(nq) * acc_stride * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    TEST_ASSERT_NOT_NULL(v_dblk);
    TEST_ASSERT_NOT_NULL(e);
    TEST_ASSERT_NOT_NULL(acc);

    for (int i = 0; i < n_tok * D; ++i) {
        v_dblk[i] = static_cast<int16_t>((i * 4099 + 17) % 65536 - 32768);
    }
    for (int r = 0; r < nq; ++r) {
        int16_t *hi = e + r * 2 * n_tok;
        int16_t *lo = hi + n_tok;
        for (int t = 0; t < n_tok; ++t) {
            dl::base::split_e_s16(std::exp(-0.13f * static_cast<float>((t * 7 + r * 3) % 31)), hi + t, lo + t);
        }
    }
    std::memset(acc, 0, static_cast<size_t>(nq) * acc_stride * sizeof(float));

    alignas(16) int32_t tmp32[32];
    constexpr float scale = (1.0f / 4096.0f) / 536870912.0f;
    for (int i = 0; i < 4; ++i) {
        dl_esp32p4_s16_attention_pv_qacc(acc, e, v_dblk, nq, n_tok, D, acc_stride, tmp32, scale);
    }

    int64_t samples[rounds];
    for (int r = 0; r < rounds; ++r) {
        const int64_t start = esp_timer_get_time();
        for (int i = 0; i < iterations; ++i) {
            dl_esp32p4_s16_attention_pv_qacc(acc, e, v_dblk, nq, n_tok, D, acc_stride, tmp32, scale);
        }
        samples[r] = esp_timer_get_time() - start;
    }
    std::sort(samples, samples + rounds);
    const double ns_per_call = static_cast<double>(samples[rounds / 2]) * 1000.0 / iterations;
    ESP_LOGI(TAG, "PV16_BENCH D=%d nq=%d n_tok=%d median_ns=%.1f", D, nq, n_tok, ns_per_call);

    heap_caps_free(v_dblk);
    heap_caps_free(e);
    heap_caps_free(acc);
}

TEST_CASE("Benchmark dl base attention s16 pv_qacc", "[attention_benchmark]")
{
    for (int d : {8, 64, 128}) {
        for (int nq : {1, 8, 32}) {
            benchmark_pv16_case(d, nq);
        }
    }
}

TEST_CASE("Test dl base attention s16 softmax: split row matches the fp32 row", "[pie][attention]")
{
    constexpr int n = 61;
    constexpr int pad = 64;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    alignas(16) float s[n];
    alignas(16) float e_ref[n];
    alignas(16) int16_t e[2 * pad];
    for (float m : {0.0f, 3.25f, -7.5f, 1000.0f}) {
        for (int t = 0; t < n; ++t) {
            s[t] = (t % 11 == 4) ? neg_inf : m - static_cast<float>(t) * (34.0f / n);
        }
        s[0] = m;
        s[1] = m - 16.0f;
        s[2] = m - 32.0f;
        std::memset(e, 0x55, sizeof(e));
        const float l_ref = dl::base::softmax_exp_row_precise(e_ref, s, n, m, 0.75f);
        const float l = dl::base::softmax_exp_row_split_s16(e, s, n, pad, m, 0.75f);
        TEST_ASSERT_EQUAL_HEX32(float_bits(l_ref), float_bits(l));
        for (int t = 0; t < pad; ++t) {
            int16_t hi = 0, lo = 0;
            if (t < n) {
                dl::base::split_e_s16(e_ref[t], &hi, &lo);
            }
            TEST_ASSERT_EQUAL_INT16(hi, e[t]);
            TEST_ASSERT_EQUAL_INT16(lo, e[pad + t]);
        }
    }
}

static size_t pv8_v_index(int d, int t)
{
    return (static_cast<size_t>(d / 16) * ATTENTION_KEY_TILE_SIZE + t) * 16 + (d & 15);
}

static void run_pv8_float_case(int D, int nq)
{
    constexpr int guard = 16;
    constexpr int stride = dl::base::ATTENTION_MAX_HEAD_DIM;
    constexpr float sentinel = -123456.75f;
    const size_t v_bytes = static_cast<size_t>(D) * ATTENTION_KEY_TILE_SIZE;
    int8_t *v_blk = static_cast<int8_t *>(heap_caps_aligned_alloc(16, v_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int8_t *e = static_cast<int8_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(nq) * ATTENTION_KEY_TILE_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *acc_storage = static_cast<float *>(heap_caps_aligned_alloc(
        16, sizeof(float) * (guard + static_cast<size_t>(nq) * stride + guard), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *deq =
        static_cast<float *>(heap_caps_aligned_alloc(16, sizeof(float) * nq, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    TEST_ASSERT_NOT_NULL(v_blk);
    TEST_ASSERT_NOT_NULL(e);
    TEST_ASSERT_NOT_NULL(acc_storage);
    TEST_ASSERT_NOT_NULL(deq);

    for (int r = 0; r < nq; ++r) {
        for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
            e[r * ATTENTION_KEY_TILE_SIZE + t] = static_cast<int8_t>((r * 23 + t * 7 + 3) % 127 - 63);
        }
        deq[r] = r == 0 ? 1.0f : (r == 1 ? -0.125f : 0.03125f);
    }
    for (int d = 0; d < D; ++d) {
        for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
            v_blk[pv8_v_index(d, t)] = static_cast<int8_t>((d * 13 + t * 17 + 5) % 127 - 63);
        }
    }

    const size_t acc_count = guard + static_cast<size_t>(nq) * stride + guard;
    std::fill_n(acc_storage, acc_count, sentinel);
    float *acc = acc_storage + guard;
    for (int r = 0; r < nq; ++r) {
        for (int d = 0; d < D; ++d) {
            acc[r * stride + d] = 0.25f * static_cast<float>(r * 257 + d);
        }
    }
    alignas(16) int32_t tmp16[16];
    dl_esp32p4_s8_attention_pv_qacc(acc, e, v_blk, deq, nq, D, stride, tmp16);

    for (int r = 0; r < nq; ++r) {
        for (int d = 0; d < D; ++d) {
            int32_t ref = 0;
            for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
                ref += static_cast<int32_t>(e[r * ATTENTION_KEY_TILE_SIZE + t]) *
                    static_cast<int32_t>(v_blk[pv8_v_index(d, t)]);
            }
            const float init = 0.25f * static_cast<float>(r * 257 + d);
            const float expected = std::fma(static_cast<float>(ref), deq[r], init);
            TEST_ASSERT_EQUAL_HEX32_MESSAGE(
                float_bits(expected), float_bits(acc[r * stride + d]), "QACC P.V float-acc lane mismatch");
        }
        for (int d = D; d < stride; ++d) {
            TEST_ASSERT_EQUAL_HEX32_MESSAGE(
                float_bits(sentinel), float_bits(acc[r * stride + d]), "QACC P.V float wrote past D");
        }
    }
    for (int i = 0; i < guard; ++i) {
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(
            float_bits(sentinel), float_bits(acc_storage[i]), "QACC P.V float prefix guard corrupted");
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(float_bits(sentinel),
                                        float_bits(acc_storage[guard + nq * stride + i]),
                                        "QACC P.V float suffix guard corrupted");
    }

    heap_caps_free(v_blk);
    heap_caps_free(e);
    heap_caps_free(acc_storage);
    heap_caps_free(deq);
}

static void run_pv8_i32_case(int D, int nq)
{
    constexpr int guard = 16;
    constexpr int stride = dl::base::ATTENTION_MAX_HEAD_DIM;
    constexpr int32_t sentinel = static_cast<int32_t>(0x6b5a4c3du);
    const size_t v_bytes = static_cast<size_t>(D) * ATTENTION_KEY_TILE_SIZE;
    int8_t *v_blk = static_cast<int8_t *>(heap_caps_aligned_alloc(16, v_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int8_t *e = static_cast<int8_t *>(heap_caps_aligned_alloc(
        16, static_cast<size_t>(nq) * ATTENTION_KEY_TILE_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    int32_t *acc_storage = static_cast<int32_t *>(
        heap_caps_aligned_alloc(16,
                                sizeof(int32_t) * (guard + static_cast<size_t>(nq) * stride + guard),
                                MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    TEST_ASSERT_NOT_NULL(v_blk);
    TEST_ASSERT_NOT_NULL(e);
    TEST_ASSERT_NOT_NULL(acc_storage);

    for (int r = 0; r < nq; ++r) {
        for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
            e[r * ATTENTION_KEY_TILE_SIZE + t] = static_cast<int8_t>((r * 31 + t * 5 + 11) % 127 - 63);
        }
    }
    for (int d = 0; d < D; ++d) {
        for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
            v_blk[pv8_v_index(d, t)] = static_cast<int8_t>((d * 19 + t * 23 + 7) % 127 - 63);
        }
    }

    const size_t acc_count = guard + static_cast<size_t>(nq) * stride + guard;
    std::fill_n(acc_storage, acc_count, sentinel);
    int32_t *acc = acc_storage + guard;
    for (int r = 0; r < nq; ++r) {
        for (int d = 0; d < D; ++d) {
            acc[r * stride + d] = r * 10000 + d * 7;
        }
    }
    alignas(16) int32_t tmp16[16];
    dl_esp32p4_s8_attention_pv_qacc_i32(acc, e, v_blk, nq, D, stride, tmp16);
    dl_esp32p4_s8_attention_pv_qacc_i32(acc, e, v_blk, nq, D, stride, tmp16);

    for (int r = 0; r < nq; ++r) {
        for (int d = 0; d < D; ++d) {
            int32_t ref = 0;
            for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
                ref += static_cast<int32_t>(e[r * ATTENTION_KEY_TILE_SIZE + t]) *
                    static_cast<int32_t>(v_blk[pv8_v_index(d, t)]);
            }
            TEST_ASSERT_EQUAL_INT32_MESSAGE(
                r * 10000 + d * 7 + 2 * ref, acc[r * stride + d], "QACC P.V int32 lane mismatch");
        }
        for (int d = D; d < stride; ++d) {
            TEST_ASSERT_EQUAL_INT32_MESSAGE(sentinel, acc[r * stride + d], "QACC P.V int32 wrote past D");
        }
    }
    for (int i = 0; i < guard; ++i) {
        TEST_ASSERT_EQUAL_INT32_MESSAGE(sentinel, acc_storage[i], "QACC P.V int32 prefix guard corrupted");
        TEST_ASSERT_EQUAL_INT32_MESSAGE(
            sentinel, acc_storage[guard + nq * stride + i], "QACC P.V int32 suffix guard corrupted");
    }

    heap_caps_free(v_blk);
    heap_caps_free(e);
    heap_caps_free(acc_storage);
}

TEST_CASE("Test dl base attention s8 pv_qacc: guarded rows and D chunks", "[pie][attention]")
{
    for (int nq : {1, 3}) {
        for (int D : {16, 32, 48, 64}) {
            ESP_LOGI(TAG, "attention_pv_i8_qacc D=%d nq=%d", D, nq);
            run_pv8_float_case(D, nq);
        }
    }
}

// The second call must accumulate on top of the first (tile loop relies on it).
TEST_CASE("Test dl base attention s8 pv_qacc_i32: guarded rows accumulate", "[pie][attention]")
{
    for (int nq : {1, 3}) {
        for (int D : {16, 32, 48, 64}) {
            ESP_LOGI(TAG, "attention_pv_i8_qacc_i32 D=%d nq=%d", D, nq);
            run_pv8_i32_case(D, nq);
        }
    }
}

/**
 * @brief PIE default vxrm is round-half-even.
 *
 * The kernel is written to match whichever mode the silicon uses; the test probes it once and then
 * verifies row-by-row.
 */
static int32_t exp_shift_idx_ref(int32_t diff, int shift, int round_mode)
{
    if (shift <= 0) {
        return diff;
    }
    if (round_mode == 0) {
        return diff >> shift;
    }
    if (round_mode == 1) {
        return (diff + (1 << (shift - 1))) >> shift;
    }
    const int32_t divisor = 1 << shift;
    const int32_t half = 1 << (shift - 1);
    const int32_t rem = diff & (divisor - 1);
    const int32_t quo = diff >> shift;
    if (rem > half) {
        return quo + 1;
    }
    if (rem < half) {
        return quo;
    }
    return quo + (quo & 1);
}

/**
 * @brief Migrated from exp_shift_round_mode in the Attention base implementation.
 */
static int probe_exp_shift_round_mode(int32_t *scratch)
{
    constexpr int shift = 4;
    alignas(16) int32_t ident[128];
    for (int i = 0; i < 128; ++i) {
        ident[i] = i;
    }
    alignas(16) int32_t dots[ATTENTION_KEY_TILE_SIZE];
    alignas(16) int8_t e[ATTENTION_KEY_TILE_SIZE];
    const int32_t m_new = 4096;
    for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
        dots[t] = m_new;
    }
    dots[1] = m_new - (1 << (shift - 1));
    dots[2] = m_new - 3 * (1 << (shift - 1));
    dl_esp32p4_s8_attention_exp_lut_shift(e, dots, 3, m_new, 127 << shift, shift, ident, scratch);
    if (e[0] == 0 && e[1] == 0 && e[2] == 1) {
        return 0;
    }
    if (e[0] == 0 && e[1] == 1 && e[2] == 2) {
        return 1;
    }
    if (e[0] == 0 && e[1] == 0 && e[2] == 2) {
        return 2;
    }
    return -1;
}

// Migrated from exp_shift_verify in the Attention base implementation.
// Sweeps every n in [1, ATTENTION_KEY_TILE_SIZE] so every block/remainder/tail combination in
// the kernel is exercised.
TEST_CASE("Test dl base attention s8 exp_lut_shift: sweeps n across all block/remainder/tail", "[pie][attention]")
{
    ESP_LOGI(TAG, "attention_exp_lut_i8_shift sweep");
    alignas(16) int32_t scratch[4 + ATTENTION_KEY_TILE_SIZE];

    const int round_mode = probe_exp_shift_round_mode(scratch);
    TEST_ASSERT_TRUE_MESSAGE(round_mode >= 0, "unknown PIE vxrm rounding rule");
    ESP_LOGI(TAG, "PIE vxrm rule = %d (0=trunc, 1=half-up, 2=half-even)", round_mode);

    constexpr int shift = 4;
    constexpr int sat = 127 << shift;
    alignas(16) int32_t table[512];
    for (int i = 0; i < 512; ++i) {
        int q = 127 - i / 4;
        if (q < 0) {
            q = 0;
        }
        table[i] = q;
    }

    alignas(16) int32_t dots[ATTENTION_KEY_TILE_SIZE];
    alignas(16) int8_t e[ATTENTION_KEY_TILE_SIZE];
    const int32_t m_new = 1000;
    for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
        dots[t] = m_new - (t * (sat / 24) - 2 * (shift + 1));
    }

    for (int n = 1; n <= ATTENTION_KEY_TILE_SIZE; ++n) {
        for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
            e[t] = static_cast<int8_t>(-1);
        }
        const int32_t got = dl_esp32p4_s8_attention_exp_lut_shift(e, dots, n, m_new, sat, shift, table, scratch);

        int32_t want_sum = 0;
        for (int t = 0; t < ATTENTION_KEY_TILE_SIZE; ++t) {
            int8_t ref = 0;
            if (t < n) {
                int32_t diff = m_new - dots[t];
                if (diff < 0) {
                    diff = 0;
                }
                if (diff > sat) {
                    diff = sat;
                }
                int32_t idx = exp_shift_idx_ref(diff, shift, round_mode);
                if (idx > 511) {
                    idx = 511;
                }
                ref = static_cast<int8_t>(table[idx]);
            }
            want_sum += ref;
            TEST_ASSERT_EQUAL_INT8_MESSAGE(ref, e[t], "vector exp lane mismatch");
        }
        TEST_ASSERT_EQUAL_INT32_MESSAGE(want_sum, got, "vector exp returned sum mismatch");
    }
}

#endif // CONFIG_PIE_V2_BOOST
