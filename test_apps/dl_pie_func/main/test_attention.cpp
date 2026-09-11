/**
 * @brief Base-layer Attention regression tests.
 *
 * End-to-end operator semantics use the shared .espdl Model::test() harness.
 * These tests cover implementation details that are not visible through an
 * exported model.
 */

#include "dl_base_attention.hpp"
#include "dl_tensor_base.hpp"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include "unity.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

const char *TAG = "TEST DL ATTENTION";

uint32_t float_bits(float x)
{
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

bool is_neg_inf(float x)
{
    return float_bits(x) == 0xff800000u;
}

int selected_head_chunk(
    int Hq, int Hkv, int Lq, int Lk, int D, dl::AttentionLayout layout, dl::runtime_mode_t mode, dl::dtype_t dtype)
{
    uint8_t dummy = 0;
    const std::vector<int> q_shape =
        layout == dl::AttentionLayout::BNHD ? std::vector<int>{1, Lq, Hq, D} : std::vector<int>{1, Hq, Lq, D};
    const std::vector<int> kv_shape =
        layout == dl::AttentionLayout::BNHD ? std::vector<int>{1, Lk, Hkv, D} : std::vector<int>{1, Hkv, Lk, D};
    dl::TensorBase q(q_shape, &dummy, -7, dtype, false);
    dl::TensorBase k(kv_shape, &dummy, -7, dtype, false);
    dl::TensorBase v(kv_shape, &dummy, -7, dtype, false);
    dl::TensorBase o(q_shape, &dummy, -7, dtype, false);
    const dl::base::AttentionMaskArgs mask_args;
    const dl::base::AttentionWorkspace workspace;
    const dl::base::AttentionInt8ExpLut int8_exp_lut;
    dl::base::AttentionTaskBarrier task_barrier;
    const dl::base::AttentionExecutionContext execution_context{&int8_exp_lut, &task_barrier};
    const std::vector<dl::base::AttentionArgs> args = dl::base::get_attention_operation_args(
        &o, &q, &k, &v, mask_args, workspace, execution_context, false, 0.f, false, layout, mode);
    TEST_ASSERT_FALSE(args.empty());
    for (const dl::base::AttentionArgs &arg : args) {
        TEST_ASSERT_EQUAL_INT(args[0].head_chunk_size, arg.head_chunk_size);
    }
    return args[0].head_chunk_size;
}

} // namespace

TEST_CASE("Test dl base attention softmax helpers", "[pie][attention]")
{
    using dl::base::attention_exp_neg_lut_lookup;
    using dl::base::attention_exp_neg_lut_lookup_precise;
    using dl::base::attention_softmax_exp_row_precise;

    double max_rel = 0.0;
    double max_rel_lut = 0.0;
    for (int i = 0; i < 320000; ++i) {
        const float x = -static_cast<float>(i) * 1e-4f;
        const double ref = std::exp(static_cast<double>(x));
        max_rel = std::max(max_rel, std::fabs(attention_exp_neg_lut_lookup_precise(x) - ref) / ref);
        if (x > -16.0f) {
            max_rel_lut = std::max(max_rel_lut, std::fabs(attention_exp_neg_lut_lookup(x) - ref) / ref);
        }
    }
    ESP_LOGI(TAG, "exp rel err: precise=%.3g lut=%.3g", max_rel, max_rel_lut);
    TEST_ASSERT_TRUE(max_rel < 5e-7);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, attention_exp_neg_lut_lookup_precise(0.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, attention_exp_neg_lut_lookup_precise(-32.0f));

    const float neg_inf = -std::numeric_limits<float>::infinity();
    constexpr int n = 4095;
    std::vector<float> scores(n);
    std::vector<float> exp_values(n);
    for (float row_max : {0.0f, 3.25f, -7.5f, 1000.0f}) {
        for (int t = 0; t < n; ++t) {
            scores[t] = (t % 97 == 5) ? neg_inf : row_max - static_cast<float>(t) * (41.0f / n);
        }
        scores[0] = row_max;
        scores[1] = row_max - 16.0f;
        scores[2] = row_max - 32.0f;
        float sum_ref = 0.5f;
        for (int t = 0; t < n; ++t) {
            sum_ref += is_neg_inf(scores[t]) ? 0.0f : attention_exp_neg_lut_lookup_precise(scores[t] - row_max);
        }
        const float sum = attention_softmax_exp_row_precise(exp_values.data(), scores.data(), n, row_max, 0.5f);
        for (int t = 0; t < n; ++t) {
            const float ref = is_neg_inf(scores[t]) ? 0.0f : attention_exp_neg_lut_lookup_precise(scores[t] - row_max);
            TEST_ASSERT_EQUAL_HEX32(float_bits(ref), float_bits(exp_values[t]));
        }
        TEST_ASSERT_FLOAT_WITHIN(1e-6f * sum_ref, sum_ref, sum);
    }
}

TEST_CASE("Test dl base attention validates scale and logit scale", "[pie][attention]")
{
    using dl::base::attention_check_inputs;
    using dl::base::attention_check_scale;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    TEST_ASSERT_TRUE(attention_check_scale(false, nan, "absent"));
    TEST_ASSERT_TRUE(attention_check_scale(true, 0.125f, "valid"));
    for (float bad : {0.0f, -0.0f, -0.125f, nan, inf, -inf, 1e-40f}) {
        TEST_ASSERT_FALSE(attention_check_scale(true, bad, "invalid"));
    }

    const std::vector<int> shape = {1, 2, 8, 16};
    dl::TensorBase q(shape, nullptr, -15, dl::DATA_TYPE_INT8);
    dl::TensorBase k(shape, nullptr, -15, dl::DATA_TYPE_INT8);
    dl::TensorBase v(shape, nullptr, -7, dl::DATA_TYPE_INT8);
    dl::TensorBase o(shape, nullptr, -7, dl::DATA_TYPE_INT8);
    const auto layout = dl::AttentionLayout::BHND;
    const auto dtype = dl::DATA_TYPE_INT8;
    TEST_ASSERT_TRUE(attention_check_inputs(&q, &k, &v, nullptr, &o, false, 0.f, layout, dtype, "default"));
    TEST_ASSERT_TRUE(attention_check_inputs(&q, &k, &v, nullptr, &o, true, 0.125f, layout, dtype, "explicit"));
    TEST_ASSERT_FALSE(attention_check_inputs(&q, &k, &v, nullptr, &o, true, -0.125f, layout, dtype, "negative"));
    TEST_ASSERT_FALSE(attention_check_inputs(&q, &k, &v, nullptr, &o, true, 1e-30f, layout, dtype, "underflow"));
    TEST_ASSERT_FALSE(
        attention_check_inputs(&q, &k, &v, nullptr, &o, false, 0.f, layout, dl::DATA_TYPE_INT16, "dtype"));

    dl::TensorBase q_wide(shape, nullptr, 80, dl::DATA_TYPE_INT8);
    dl::TensorBase k_wide(shape, nullptr, 80, dl::DATA_TYPE_INT8);
    TEST_ASSERT_FALSE(attention_check_inputs(&q_wide, &k_wide, &v, nullptr, &o, true, 1.0f, layout, dtype, "overflow"));
}

TEST_CASE("Test dl base attention constant mask cache fallback", "[pie][attention]")
{
    const std::vector<int> q_shape = {1, 1, 1, 16};
    const std::vector<int> kv_shape = {1, 1, 64, 16};
    dl::TensorBase q(q_shape, nullptr, -6, dl::DATA_TYPE_INT8);
    dl::TensorBase k(kv_shape, nullptr, -6, dl::DATA_TYPE_INT8);

    std::vector<uint8_t> bool_data(64);
    for (size_t i = 0; i < bool_data.size(); ++i) {
        bool_data[i] = (i % 3) != 0;
    }
    dl::TensorBase bool_mask({1, 1, 1, 64}, bool_data.data(), 0, dl::DATA_TYPE_BOOL);
    dl::base::AttentionMaskCache bool_cache;
    dl::base::AttentionMaskArgs bool_args =
        dl::base::get_attention_mask_args(&bool_mask, &q, &k, dl::AttentionLayout::BHND, "bool-mask");
    dl::base::attention_prepare_constant_mask_cache<int8_t>(
        bool_cache, bool_args, &q, &k, false, 0.f, dl::AttentionLayout::BHND, "bool-mask");
    TEST_ASSERT_NOT_NULL(bool_cache.mask_tile_mode);
    TEST_ASSERT_NULL(bool_cache.mask_dots);
    const uint8_t *const bool_modes = bool_cache.mask_tile_mode;

    q.exponent = -5;
    bool_args = dl::base::get_attention_mask_args(&bool_mask, &q, &k, dl::AttentionLayout::BHND, "bool-mask");
    dl::base::attention_prepare_constant_mask_cache<int8_t>(
        bool_cache, bool_args, &q, &k, false, 0.f, dl::AttentionLayout::BHND, "bool-mask");
    TEST_ASSERT_EQUAL_PTR(bool_modes, bool_cache.mask_tile_mode);
    TEST_ASSERT_NULL(bool_args.mask_dots);

    std::vector<float> additive_data(64, 0.f);
    additive_data[1] = -std::numeric_limits<float>::infinity();
    additive_data[3] = -0.25f;
    dl::TensorBase additive_mask({1, 1, 1, 64}, additive_data.data(), 0, dl::DATA_TYPE_FLOAT);
    dl::base::AttentionMaskCache additive_cache;
    dl::base::AttentionMaskArgs additive_args =
        dl::base::get_attention_mask_args(&additive_mask, &q, &k, dl::AttentionLayout::BHND, "additive-mask");
    dl::base::attention_prepare_constant_mask_cache<int8_t>(
        additive_cache, additive_args, &q, &k, false, 1.f, dl::AttentionLayout::BHND, "additive-mask");
    TEST_ASSERT_NOT_NULL(additive_cache.mask_tile_mode);
    TEST_ASSERT_NOT_NULL(additive_cache.mask_dots);
    const uint8_t *const additive_modes = additive_cache.mask_tile_mode;

    heap_caps_free(additive_cache.mask_dots);
    additive_cache.mask_dots = nullptr;
    additive_cache.mask_dots_allocation_failed = true;
    additive_args =
        dl::base::get_attention_mask_args(&additive_mask, &q, &k, dl::AttentionLayout::BHND, "additive-mask");
    dl::base::attention_prepare_constant_mask_cache<int8_t>(
        additive_cache, additive_args, &q, &k, false, 1.f, dl::AttentionLayout::BHND, "additive-mask");
    TEST_ASSERT_EQUAL_PTR(additive_modes, additive_cache.mask_tile_mode);
    TEST_ASSERT_NOT_NULL(additive_args.mask_tile_mode);
    TEST_ASSERT_NULL(additive_args.mask_dots);
}

TEST_CASE("Test dl base attention workspace binds atomically", "[pie][attention]")
{
#if CONFIG_PIE_V2_BOOST
    const std::vector<int> q_shape = {1, 1, 1, 16};
    const std::vector<int> k_shape = {1, 1, 64, 16};
    const dl::base::AttentionWorkspaceSizes sizes =
        dl::base::attention_workspace_sizes(dl::DATA_TYPE_INT8, q_shape, k_shape, dl::AttentionLayout::BHND);
    TEST_ASSERT_EQUAL_size_t(sizes.tile_bytes + sizes.cache_bytes, sizes.bytes);
    TEST_ASSERT_GREATER_THAN_UINT(0u, static_cast<unsigned>(sizes.cache_bytes));

    std::vector<uint8_t> storage(sizes.bytes);
    dl::base::AttentionWorkspace workspace = dl::base::attention_bind_workspace(
        storage.data(), storage.size(), dl::DATA_TYPE_INT8, q_shape, k_shape, dl::AttentionLayout::BHND);
    TEST_ASSERT_NOT_NULL(workspace.tile_buf);
    TEST_ASSERT_EQUAL_PTR(storage.data() + sizes.tile_bytes, workspace.k_cache);
    TEST_ASSERT_EQUAL_PTR(workspace.k_cache + workspace.k_cache_bytes, reinterpret_cast<uint8_t *>(workspace.v_cache));

    workspace = dl::base::attention_bind_workspace(
        storage.data(), storage.size() - 1, dl::DATA_TYPE_INT8, q_shape, k_shape, dl::AttentionLayout::BHND);
    TEST_ASSERT_NULL(workspace.tile_buf);
    TEST_ASSERT_NULL(workspace.k_cache);
    TEST_ASSERT_NULL(workspace.v_cache);
#endif

    const dl::base::AttentionWorkspaceSizes fp32_sizes = dl::base::attention_workspace_sizes(
        dl::DATA_TYPE_FLOAT, {1, 1, 1, 16}, {1, 1, 64, 16}, dl::AttentionLayout::BHND);
    TEST_ASSERT_EQUAL_size_t(0u, fp32_sizes.cache_bytes);
    std::vector<uint8_t> fp32_storage(fp32_sizes.bytes);
    const dl::base::AttentionWorkspace fp32_workspace = dl::base::attention_bind_workspace(fp32_storage.data(),
                                                                                           fp32_storage.size(),
                                                                                           dl::DATA_TYPE_FLOAT,
                                                                                           {1, 1, 1, 16},
                                                                                           {1, 1, 64, 16},
                                                                                           dl::AttentionLayout::BHND);
    TEST_ASSERT_NOT_NULL(fp32_workspace.tile_buf);
    TEST_ASSERT_NULL(fp32_workspace.k_cache);
    TEST_ASSERT_NULL(fp32_workspace.v_cache);
}

#if CONFIG_PIE_V2_BOOST
TEST_CASE("Test dl base attention head chunk policy", "[pie][attention]")
{
    struct ChunkCase {
        dl::dtype_t dtype;
        int Hq;
        int Hkv;
        int Lq;
        int Lk;
        int D;
        dl::AttentionLayout layout;
        dl::runtime_mode_t mode;
        int expected;
    };
    constexpr bool multi_core_build = CONFIG_FREERTOS_NUMBER_OF_CORES > 1;
    const ChunkCase cases[] = {
        {dl::DATA_TYPE_INT8,
         16,
         16,
         64,
         512,
         32,
         dl::AttentionLayout::BNHD,
         dl::RUNTIME_MODE_MULTI_CORE,
         multi_core_build ? 4 : 2},
        {dl::DATA_TYPE_INT8,
         8,
         8,
         64,
         512,
         64,
         dl::AttentionLayout::BNHD,
         dl::RUNTIME_MODE_MULTI_CORE,
         multi_core_build ? 2 : 1},
        {dl::DATA_TYPE_INT8, 8, 8, 64, 1024, 32, dl::AttentionLayout::BHND, dl::RUNTIME_MODE_MULTI_CORE, 1},
        {dl::DATA_TYPE_INT8, 8, 2, 64, 1024, 32, dl::AttentionLayout::BNHD, dl::RUNTIME_MODE_MULTI_CORE, 1},
        {dl::DATA_TYPE_INT16,
         8,
         8,
         64,
         1024,
         8,
         dl::AttentionLayout::BNHD,
         dl::RUNTIME_MODE_MULTI_CORE,
         multi_core_build ? 4 : 1},
        {dl::DATA_TYPE_INT16, 8, 8, 64, 1024, 32, dl::AttentionLayout::BNHD, dl::RUNTIME_MODE_MULTI_CORE, 1},
        {dl::DATA_TYPE_INT16, 8, 8, 64, 1024, 48, dl::AttentionLayout::BNHD, dl::RUNTIME_MODE_SINGLE_CORE, 1},
        {dl::DATA_TYPE_INT16, 8, 2, 64, 1024, 32, dl::AttentionLayout::BNHD, dl::RUNTIME_MODE_MULTI_CORE, 1},
    };
    for (const ChunkCase &test : cases) {
        TEST_ASSERT_EQUAL_INT(
            test.expected,
            selected_head_chunk(test.Hq, test.Hkv, test.Lq, test.Lk, test.D, test.layout, test.mode, test.dtype));
    }
}
#endif

TEST_CASE("Test dl base attention clamps forced GQA head chunk", "[pie][attention]")
{
    constexpr int Hq = 4;
    constexpr int Hkv = 2;
    constexpr int Lq = 8;
    constexpr int Lk = 16;
    constexpr int D = 16;
    constexpr int forced_chunk = 4;
    const std::vector<int> q_shape = {1, Lq, Hq, D};
    const std::vector<int> kv_shape = {1, Lk, Hkv, D};

    std::vector<int16_t> q_data(static_cast<size_t>(Lq * Hq * D));
    std::vector<int16_t> k_data(static_cast<size_t>(Lk * Hkv * D));
    std::vector<int16_t> v_data(static_cast<size_t>(Lk * Hkv * D));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<int16_t>((static_cast<int>(i) * 37 + 11) % 4093 - 2046);
    }
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = static_cast<int16_t>((static_cast<int>(i) * 53 + 7) % 4093 - 2046);
        v_data[i] = static_cast<int16_t>((static_cast<int>(i) * 41 + 19) % 4093 - 2046);
    }

    dl::TensorBase q(q_shape, q_data.data(), -14, dl::DATA_TYPE_INT16, false);
    dl::TensorBase k(kv_shape, k_data.data(), -14, dl::DATA_TYPE_INT16, false);
    dl::TensorBase v(kv_shape, v_data.data(), -14, dl::DATA_TYPE_INT16, false);
    dl::TensorBase expected(q_shape, nullptr, -14, dl::DATA_TYPE_INT16);
    dl::TensorBase actual(q_shape, nullptr, -14, dl::DATA_TYPE_INT16);

    const dl::base::AttentionWorkspaceSizes sizes =
        dl::base::attention_workspace_sizes(dl::DATA_TYPE_INT16, q_shape, kv_shape, dl::AttentionLayout::BNHD);
    const size_t k_head = static_cast<size_t>(((Lk + 7) & ~7) * D * 2);
    const size_t v_head = static_cast<size_t>(((Lk + 63) / 64) * D * 64 * 2);
    std::vector<uint8_t> tile_storage(sizes.tile_bytes + 16);
    std::vector<uint8_t> cache_storage(static_cast<size_t>(forced_chunk) * (k_head + v_head) + 16);
    const auto align16 = [](std::vector<uint8_t> &storage) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(storage.data());
        return storage.data() + ((16u - (address & 15u)) & 15u);
    };

    dl::base::AttentionWorkspace workspace;
    workspace.tile_buf = align16(tile_storage);
    workspace.tile_buf_bytes = sizes.tile_bytes;
    workspace.k_cache = align16(cache_storage);
    workspace.k_cache_bytes = static_cast<size_t>(forced_chunk) * k_head;
    workspace.v_cache = reinterpret_cast<int8_t *>(workspace.k_cache + workspace.k_cache_bytes);
    workspace.v_cache_bytes = static_cast<size_t>(forced_chunk) * v_head;

    const dl::base::AttentionMaskArgs mask_args;
    const dl::base::AttentionInt8ExpLut int8_exp_lut;
    dl::base::AttentionTaskBarrier task_barrier;
    const dl::base::AttentionExecutionContext execution_context{&int8_exp_lut, &task_barrier};
    const auto run = [&](dl::TensorBase &output, int head_chunk) {
        task_barrier.reset();
        std::vector<dl::base::AttentionArgs> args =
            dl::base::get_attention_operation_args(&output,
                                                   &q,
                                                   &k,
                                                   &v,
                                                   mask_args,
                                                   workspace,
                                                   execution_context,
                                                   false,
                                                   0.f,
                                                   false,
                                                   dl::AttentionLayout::BNHD,
                                                   dl::RUNTIME_MODE_SINGLE_CORE);
        TEST_ASSERT_EQUAL_INT(1, static_cast<int>(args.size()));
        args[0].head_chunk_size = head_chunk;
        args[0].workspace = workspace;
        dl::base::attention<int16_t>(&args[0]);
    };
    run(expected, 1);
    run(actual, forced_chunk);
    TEST_ASSERT_EQUAL_INT16_ARRAY(static_cast<const int16_t *>(expected.data),
                                  static_cast<const int16_t *>(actual.data),
                                  static_cast<int>(q_data.size()));
}
