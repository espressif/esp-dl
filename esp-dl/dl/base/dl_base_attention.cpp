#include "dl_base_attention.hpp"

#include "dl_base_attention_common.hpp"
#include "dl_base_attention_kernels.hpp"
#include "dl_base_attention_softmax.hpp"

#include "dl_define.hpp"
#include "dl_tensor_base.hpp"
#include "dl_tool.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace dl {
namespace base {

namespace {
/**
 * @brief Storage and gather buffers impose this absolute BNHD MHA head-chunk limit.
 *
 * It is a capacity bound, not a performance policy. int8 also fits the 256-byte K-gather staging;
 * int16 packs Q into the same q_tile, so the 2-byte occupancy is the tighter cap.
 */
int attention_head_chunk_hard_limit(bool is_bnhd, int Hq, int Hkv, int D, int D_pad, int elem_size)
{
#if CONFIG_PIE_V2_BOOST
    if (!is_bnhd || Hq != Hkv || D <= 0 || D_pad <= 0 || elem_size <= 0) {
        return 1;
    }
    constexpr int q_cap = ATTENTION_QUERY_TILE_SIZE * ATTENTION_MAX_HEAD_DIM * static_cast<int>(sizeof(float));
    int fit = std::min(Hq, q_cap / (ATTENTION_QUERY_TILE_SIZE * D_pad * elem_size));
    if (elem_size == 1) {
        fit = std::min(fit, 256 / D);
    }
    return std::max(1, fit);
#else
    (void)is_bnhd;
    (void)Hq;
    (void)Hkv;
    (void)D;
    (void)D_pad;
    (void)elem_size;
    return 1;
#endif
}

/**
 * @brief Performance policy. int8 and int16 each use their own board sweep.
 *
 * int8 and int16 builders both gather a contiguous BNHD span of several heads per token. The sweeps
 * are separate: element size, t-block length and the QK kernel differ. The real D matters
 * independently of D_pad. Single- and dual-core paths differ because cache-preparation barriers and
 * query-block ownership amortize at different shapes.
 */
int attention_head_chunk_runtime_limit(
    bool is_bnhd, int Hq, int Hkv, int Lq, int Lk, int D, int D_pad, int elem_size, int task_count)
{
    const int hard = attention_head_chunk_hard_limit(is_bnhd, Hq, Hkv, D, D_pad, elem_size);
    if (hard <= 1) {
        return 1;
    }

    // Prefer the smallest chunk within roughly 3% of the measured winner;
    // larger values multiply the mandatory K/V workspace.
    const bool dual_core = task_count > 1;
    const int q_blocks = (Lq + ATTENTION_QUERY_TILE_SIZE - 1) / ATTENTION_QUERY_TILE_SIZE;
    if (elem_size != 1) {
        // P4 int16 sweep after the one-pass K/V gather, non-causal BNHD, PSRAM.
        // At most two query blocks. Dual core: D<=12 and Lk>=1024 takes 4;
        // 12<D<=16 and Lk>=512 takes 2; 16<D<=28 and Lk>=1024 takes 2;
        // 28<D<=48 and Lk>=2048, and D==32 with Lk<=512, take 2.
        // D==64 and Lq=256 stayed at 1 or got slower. Single core takes 2 only for D<=8, Lk>=2048.
        int preferred = 1;
        if (q_blocks <= 2) {
            if (dual_core) {
                if (D <= 12 && Lk >= 1024) {
                    preferred = 4;
                } else if (D > 12 && D <= 16 && Lk >= 512) {
                    preferred = 2;
                } else if (D > 16 && D <= 28 && Lk >= 1024) {
                    preferred = 2;
                } else if (D > 28 && D <= 48 && Lk >= 2048) {
                    preferred = 2;
                } else if (D > 28 && D <= 32 && Lk <= 512) {
                    preferred = 2;
                }
            } else if (D <= 8 && Lk >= 2048) {
                preferred = 2;
            }
        }
        return std::max(1, std::min(hard, preferred));
    }

    int preferred = 1;
    if (Lk <= 512) {
        if (Hq >= 16 && D > 24 && D <= 32) {
            preferred = 4;
        } else if (dual_core && D == 64) {
            preferred = 2;
        }
    } else if (D <= 8) {
        if (Lk >= 2048) {
            preferred = dual_core ? 8 : 2;
        }
    } else if (D <= 15) {
        if (Lk <= 1024) {
            preferred = 1;
        } else {
            preferred = dual_core ? 8 : 4;
        }
    } else if (D <= 24) {
        if (dual_core) {
            // 16 < D <= 24: multi-head gather paid off around Lk*D == 15000 source elements.
            preferred = (Lk * D >= 15000) ? 8 : 1;
        } else if (Lk <= 768) {
            preferred = 1;
        } else {
            preferred = (Lk <= 1024) ? ((D >= 24) ? 2 : 4) : 8;
        }
    } else if (D < 32) {
        if (dual_core) {
            preferred = 8;
        } else {
            preferred = (Lk <= 768) ? 4 : 8;
        }
    } else if (D == 32) {
        if (dual_core) {
            preferred = (q_blocks >= 8) ? ((Lk <= 1024) ? 2 : 1) : 4;
        } else if (q_blocks >= 8) {
            preferred = (Lk <= 1024) ? 2 : 1;
        } else if (Lk <= 1024) {
            preferred = (q_blocks == 1) ? 4 : 2;
        } else if (Hq > 8) {
            preferred = 2;
        } else {
            preferred = std::min(hard, 4);
        }
    } else if (D <= 48) {
        preferred = dual_core ? ((q_blocks >= 8) ? 1 : 4) : 1;
    } else if (D < 64) {
        preferred = (dual_core && Lk > 512) ? 4 : 1;
    } else if (dual_core) {
        if (Lk <= 512) {
            preferred = 2;
        } else if (Lk <= 1024) {
            preferred = (q_blocks == 1) ? 2 : 1;
        } else {
            preferred = (q_blocks == 2) ? 2 : 1;
        }
    }
    return std::max(1, std::min(hard, preferred));
}

/**
 * @brief Planned head-chunk capacity: max of the single- and dual-core policies.
 *
 * @return heads one tiled pass may hold
 */
int attention_head_chunk_capacity(bool is_bnhd, int Hq, int Hkv, int Lq, int Lk, int D, int D_pad, int elem_size)
{
    return std::max(
        attention_head_chunk_runtime_limit(is_bnhd, Hq, Hkv, Lq, Lk, D, D_pad, elem_size, 1),
        attention_head_chunk_runtime_limit(is_bnhd, Hq, Hkv, Lq, Lk, D, D_pad, elem_size, ATTENTION_TASK_COUNT));
}

/**
 * @brief K/V repack cache geometry for one head. k_head_bytes == 0 means the node runs without
 * caches.
 *
 * Load-time sizing, the pre-launch update and the tiled tasks all derive their sizes from here so
 * they cannot drift apart.
 */
struct AttentionCachePlan {
    int D_pad;           /*!< padded head dim used by QACC kernels */
    size_t k_head_bytes; /*!< t-block K: 16 tokens (int8) or 8 tokens (int16) per group */
    size_t v_head_bytes; /*!< d-block V tiles */
    bool chunks_heads;   /*!< int8/int16 may hold several heads per chunk (BNHD MHA) */
};

/**
 * @brief Build the K/V cache geometry for one head of qkv_dtype.
 *
 * @param Lk   key length
 * @param D    head dimension
 *
 * @return cache sizes; k_head_bytes == 0 means the node runs without caches
 */
template <dtype_t qkv_dtype>
AttentionCachePlan attention_cache_plan(int Lk, int D)
{
    AttentionCachePlan plan{D, 0, 0, false};
    if constexpr (qkv_dtype == DATA_TYPE_INT8 || qkv_dtype == DATA_TYPE_INT16) {
        if (!can_use_simd_dot<qkv_dtype>(D)) {
            return plan;
        }
        constexpr int lanes = (qkv_dtype == DATA_TYPE_INT8) ? 16 : 8;
        constexpr size_t elem = static_cast<size_t>(qkv_element_size<qkv_dtype>());
        const int n_k_tiles = (Lk + ATTENTION_KEY_TILE_SIZE - 1) / ATTENTION_KEY_TILE_SIZE;
        plan.D_pad = (D + lanes - 1) & ~(lanes - 1);
        plan.k_head_bytes =
            static_cast<size_t>((Lk + lanes - 1) & ~(lanes - 1)) * static_cast<size_t>(plan.D_pad) * elem;
        plan.v_head_bytes = static_cast<size_t>(n_k_tiles) * attention_v_tile_bytes(plan.D_pad, elem);
        plan.chunks_heads = true;
    }
    return plan;
}

/**
 * @brief Runtime-typed wrapper around attention_cache_plan<qkv_dtype>().
 */
AttentionCachePlan attention_cache_plan(dtype_t qkv_dtype, int Lk, int D)
{
    switch (qkv_dtype) {
    case DATA_TYPE_INT8:
        return attention_cache_plan<DATA_TYPE_INT8>(Lk, D);
    case DATA_TYPE_INT16:
        return attention_cache_plan<DATA_TYPE_INT16>(Lk, D);
    default:
        return attention_cache_plan<DATA_TYPE_FLOAT>(Lk, D);
    }
}

/**
 * @brief Per-node kernel choice, fixed before the head loop.
 *
 * Both tasks derive the same values, so they take the same branches and barriers.
 */
enum class AttentionQkKernel : uint8_t {
    BatchedDotsS8,    /*!< t-block K cache -> int32 dots, softmax stays in dot space */
    BatchedScoresS16, /*!< t-block K cache -> fp32 scores */
    RowSimd,          /*!< fp32 per-row SIMD dot on raw K */
    Scalar,           /*!< dequantized Q, scalar dot */
};

enum class AttentionPvKernel : uint8_t {
    QaccS8,  /*!< int8 exponentials x d-block V on QACC */
    QaccS16, /*!< split int16 exponentials x d-block V on QACC */
    Float,   /*!< dequantized V, fp32 accumulation */
};

/**
 * @brief Layout of one node workspace, each region a multiple of 16 bytes: per-task tile buffers,
 * then the K repack, then the V repack.
 *
 * The caches are sized from the real Lk with no L2 ceiling: each query block streams them once per
 * 32 queries, which costs far less than the raw-K / per-tile-V fallback even when they overflow L2.
 */
struct AttentionWorkspaceLayout {
    size_t tile_bytes; /*!< per-task tile buffers */
    size_t k_bytes;    /*!< K repack cache for one tiled pass */
    size_t v_bytes;    /*!< V repack cache for one tiled pass */
    size_t total() const { return tile_bytes + k_bytes + v_bytes; }
};

/**
 * @brief Size the tile and K/V cache regions for one node.
 */
AttentionWorkspaceLayout attention_workspace_layout(dtype_t qkv_dtype,
                                                    const std::vector<int> &q_shape,
                                                    const std::vector<int> &k_shape,
                                                    AttentionLayout layout)
{
    AttentionWorkspaceLayout ws{tool::get_aligned_size(attention_tile_buffer_bytes()), 0, 0};
    if (q_shape.size() != 4 || k_shape.size() != 4) {
        return ws;
    }
    const AttentionDims dims = attention_dims(layout, q_shape, k_shape);
    if (dims.Lk <= 0 || dims.D <= 0 || dims.D > ATTENTION_MAX_HEAD_DIM || dims.Hq <= 0 || dims.Hkv <= 0) {
        return ws;
    }
    const AttentionCachePlan plan = attention_cache_plan(qkv_dtype, dims.Lk, dims.D);
    if (plan.k_head_bytes == 0) {
        return ws;
    }
    const int elem_size = (qkv_dtype == DATA_TYPE_INT16) ? 2 : 1;
    const int heads = plan.chunks_heads
        ? attention_head_chunk_capacity(
              layout == AttentionLayout::BNHD, dims.Hq, dims.Hkv, dims.Lq, dims.Lk, dims.D, plan.D_pad, elem_size)
        : 1;
    ws.k_bytes = tool::get_aligned_size(static_cast<size_t>(heads) * plan.k_head_bytes);
    ws.v_bytes = tool::get_aligned_size(static_cast<size_t>(heads) * plan.v_head_bytes);
    return ws;
}

/**
 * @brief Map a C++ element type onto dtype_t.
 */
template <typename feature_t>
constexpr dtype_t attention_dtype()
{
    static_assert(std::is_same_v<feature_t, int8_t> || std::is_same_v<feature_t, int16_t> ||
                      std::is_same_v<feature_t, float>,
                  "Attention supports int8_t, int16_t and float");
    return std::is_same_v<feature_t, int8_t> ? DATA_TYPE_INT8
        : std::is_same_v<feature_t, int16_t> ? DATA_TYPE_INT16
                                             : DATA_TYPE_FLOAT;
}

/**
 * @brief Kept out of line: if one variant is inlined into attention<T>, its frame stays live under
 * the other variant's frame on the 2 KB dual-core worker stack.
 */
template <dtype_t qkv_dtype, bool causal>
__attribute__((noinline)) void attention_forward_impl(const TensorBase *q_t,
                                                      const TensorBase *k_t,
                                                      const TensorBase *v_t,
                                                      TensorBase *o_t,
                                                      bool has_scale,
                                                      float scale_attr,
                                                      int task_id,
                                                      int task_count,
                                                      int head_chunk_size,
                                                      AttentionTileBuffer &tile_buf,
                                                      const AttentionWorkspace &ws,
                                                      AttentionLayout layout,
                                                      const AttentionMaskArgs &mask,
                                                      const AttentionExecutionContext &execution_context)
{
    attention_select_qk_rounding<qkv_dtype>();
    const auto &qs = q_t->shape;
    const auto &ks = k_t->shape;
    const AttentionDims dims = attention_dims(layout, qs, ks);
    const bool is_bnhd = (layout == AttentionLayout::BNHD);
    const int B = dims.B;
    const int H = dims.Hq;
    const int Lq = dims.Lq;
    const int D = dims.D;
    const int Lk = dims.Lk;
    assert(D <= ATTENTION_MAX_HEAD_DIM && "Attention: head_dim exceeds ATTENTION_MAX_HEAD_DIM; bump the constant");
    assert(k_t->shape[3] == D && v_t->shape[3] == D);
    const int Hkv = dims.Hkv;
    assert(Hkv > 0 && (H % Hkv) == 0 && "Attention: Hq must be a positive multiple of Hkv");
    const int group = H / Hkv;
    assert((is_bnhd ? v_t->shape[1] : v_t->shape[2]) == Lk);
    assert(task_count >= 1);
    assert(task_id >= 0 && task_id < task_count);
    assert(execution_context.task_barrier != nullptr && "Attention: task barrier must be supplied per forward");

    const float q_scale = dequant_scale_from_exponent(q_t->exponent);
    const float k_scale = dequant_scale_from_exponent(k_t->exponent);
    const float v_scale = dequant_scale_from_exponent(v_t->exponent);
    const float o_inv_scale = (qkv_dtype == DATA_TYPE_FLOAT) ? 1.0f : rescale_to_int_from_exponent(o_t->exponent);

    const float softmax_scale = has_scale ? scale_attr : (1.0f / std::sqrt(static_cast<float>(D)));
    const float logit_scale = attention_logit_scale<qkv_dtype>(q_t, k_t, has_scale, scale_attr);
#if CONFIG_PIE_V2_BOOST
    if constexpr (qkv_dtype == DATA_TYPE_INT8) {
        assert(execution_context.int8_exp_lut != nullptr && execution_context.int8_exp_lut->valid &&
               execution_context.int8_exp_lut->built_for == logit_scale &&
               "Attention: attention_prepare_int8_softmax_exp_lut must prepare the exp LUT before the tiled tasks "
               "launch");
    }
#endif
    const bool has_mask = (mask.data != nullptr);
    // Batched int8 softmax runs on raw int32 dots, so the mask joins in dot units.
    const float mask_to_dots = 1.0f / logit_scale;

    const AttentionStrides q_strides = attention_strides(layout, H, Lq, D);
    const AttentionStrides k_strides = attention_strides(layout, Hkv, Lk, D);
    const AttentionStrides v_strides = attention_strides(layout, Hkv, Lk, D);
    const AttentionStrides o_strides = attention_strides(layout, H, Lq, D);
    const int q_stride_batch = q_strides.batch;
    const int q_stride_head = q_strides.head;
    const int q_stride_pos = q_strides.pos;
    const int k_stride_batch = k_strides.batch;
    const int k_stride_head = k_strides.head;
    const int k_stride_pos = k_strides.pos;
    const int v_stride_batch = v_strides.batch;
    const int v_stride_head = v_strides.head;
    const int v_stride_pos = v_strides.pos;
    const int o_stride_batch = o_strides.batch;
    const int o_stride_head = o_strides.head;
    const int o_stride_pos = o_strides.pos;

    const void *q_data = q_t->data;
    const void *k_data = k_t->data;
    const void *v_data = v_t->data;
    void *o_data = o_t->data;

    // A single decision keeps Q packing, QK, and P·V on the same SIMD/QACC
    // execution plan. For int8/int16 this also implies PIE v2 and valid D.
    const bool use_simd = can_use_simd_dot<qkv_dtype>(D);
    // K/V caches are carved from the node workspace by attention_bind_workspace.
    const AttentionCachePlan cache_plan = attention_cache_plan<qkv_dtype>(Lk, D);
    const int D_pad = cache_plan.D_pad;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const int q_blocks = (Lq + ATTENTION_QUERY_TILE_SIZE - 1) / ATTENTION_QUERY_TILE_SIZE;
    const size_t k_head_bytes = cache_plan.k_head_bytes;
    const size_t v_head_bytes = cache_plan.v_head_bytes;
    // This describes the shape/dtype plan, not allocation success. The public
    // binder rejects partial workspaces, and the assertions below enforce the
    // same contract for direct low-level callers.
    const bool requires_kv_cache = k_head_bytes > 0;
    const AttentionQkKernel qk_kernel = [&]() {
        if (!use_simd) {
            return AttentionQkKernel::Scalar;
        }
        if constexpr (qkv_dtype == DATA_TYPE_INT8) {
            return AttentionQkKernel::BatchedDotsS8;
        } else if constexpr (qkv_dtype == DATA_TYPE_INT16) {
            return AttentionQkKernel::BatchedScoresS16;
        } else {
            return AttentionQkKernel::RowSimd;
        }
    }();
    // Quantized P·V reads the planned V cache; QACC runs on the D_pad grid with
    // a zero-padded tail d-group.
    const AttentionPvKernel pv_kernel = !use_simd ? AttentionPvKernel::Float
        : (qkv_dtype == DATA_TYPE_INT8)           ? AttentionPvKernel::QaccS8
        : (qkv_dtype == DATA_TYPE_INT16)          ? AttentionPvKernel::QaccS16
                                                  : AttentionPvKernel::Float;
    // Batched int8 Q·K keeps softmax in int32 dot space; its P·V accumulates in
    // int32 and dequantizes once at write-out.
    const bool dot_space = (qk_kernel == AttentionQkKernel::BatchedDotsS8);
    const bool batched_qk = dot_space || (qk_kernel == AttentionQkKernel::BatchedScoresS16);
    [[maybe_unused]] const float pv_scale = v_scale / static_cast<float>(1 << PV_S16_BITS);
    static_assert(sizeof(AttentionTileBuffer::exp_scores) >=
                      sizeof(int16_t) * ATTENTION_QUERY_TILE_SIZE * 2 * ATTENTION_KEY_TILE_SIZE,
                  "split int16 exponentials must fit tile_buf.exp_scores");
    static_assert(sizeof(AttentionTileBuffer::dots) >= 4 * 256, "K head-gather staging must fit tile_buf.dots");

    const int n_k_tiles = (Lk + ATTENTION_KEY_TILE_SIZE - 1) / ATTENTION_KEY_TILE_SIZE;
    const int v_tile_stride = static_cast<int>(attention_v_tile_bytes(D_pad, qkv_element_size<qkv_dtype>()));
    head_chunk_size = (group == 1 && cache_plan.chunks_heads && requires_kv_cache) ? std::max(1, head_chunk_size) : 1;
    if (requires_kv_cache) {
        const size_t required_k_bytes = static_cast<size_t>(head_chunk_size) * k_head_bytes;
        const size_t required_v_bytes = static_cast<size_t>(head_chunk_size) * v_head_bytes;
        assert(ws.k_cache != nullptr && ws.k_cache_bytes >= required_k_bytes &&
               "PIE v2 quantized Attention requires its complete K cache");
        assert(ws.v_cache != nullptr && ws.v_cache_bytes >= required_v_bytes &&
               "PIE v2 quantized Attention requires its complete V cache");
    }

    for (int b = 0; b < B; ++b) {
        for (int h0 = 0; h0 < H; h0 += head_chunk_size) {
            const int chunk_heads = std::min(head_chunk_size, H - h0);
            // GQA: head_chunk_size == 1 when group > 1, so h_kv0 uniquely identifies
            // the shared K/V head. Collapses to h_kv0 == h0 for MHA.
            const int h_kv0 = h0 / group;
            const int k_head0_base = b * k_stride_batch + h_kv0 * k_stride_head;
            [[maybe_unused]] const int v_head0_base = b * v_stride_batch + h_kv0 * v_stride_head;
            const int q_head0_base = b * q_stride_batch + h0 * q_stride_head;

            if (requires_kv_cache) {
                // prepare_kv_cache_for_chunk:
                // K and V have independent destination regions, but form one shared
                // preparation phase. Every task reads both complete caches after this
                // block, so one publication barrier follows both repacks.

                // 1. repack_k_cache_slice: each task owns a contiguous t-block range.
                const auto repack_k_group_begin = [&](int n) { return n * task_id / task_count; };
                const auto repack_k_group_end = [&](int n) { return n * (task_id + 1) / task_count; };
                if constexpr (qkv_dtype == DATA_TYPE_INT8) {
                    const int n_group = (Lk + 15) / 16;
                    build_k_heads_tblk_s8(reinterpret_cast<int8_t *>(ws.k_cache),
                                          k_head_bytes,
                                          static_cast<const int8_t *>(k_data) + k_head0_base,
                                          chunk_heads,
                                          Lk,
                                          D,
                                          D_pad,
                                          k_stride_pos,
                                          reinterpret_cast<uint8_t (*)[256]>(tile_buf.dots),
                                          repack_k_group_begin(n_group),
                                          repack_k_group_end(n_group));
                } else if constexpr (qkv_dtype == DATA_TYPE_INT16) {
                    // Only the QACC path caches int16 K: 8-token t-block
                    // groups that the batched QK kernel streams through
                    // esp.vsmulas.s16.qacc.ld.incp. BNHD MHA may pack several
                    // adjacent heads into k_cache[h_local].
                    const int n_group = (Lk + 7) / 8;
                    build_k_heads_tblk_s16(reinterpret_cast<int16_t *>(ws.k_cache),
                                           k_head_bytes / sizeof(int16_t),
                                           static_cast<const int16_t *>(k_data) + k_head0_base,
                                           chunk_heads,
                                           Lk,
                                           D,
                                           D_pad,
                                           k_stride_pos,
                                           repack_k_group_begin(n_group),
                                           repack_k_group_end(n_group),
                                           reinterpret_cast<uint8_t (*)[256]>(tile_buf.dots));
                }
#if CONFIG_PIE_V2_BOOST
                // 2. repack_v_cache_tiles: tile ownership is interleaved, while
                // destination ranges remain disjoint from both K and other tasks.
                int v_tile_index = 0;
                for (int j0 = 0; j0 < Lk; j0 += ATTENTION_KEY_TILE_SIZE, ++v_tile_index) {
                    if ((v_tile_index % task_count) != task_id) {
                        continue;
                    }
                    const int tile_n = std::min(ATTENTION_KEY_TILE_SIZE, Lk - j0);
                    if constexpr (qkv_dtype == DATA_TYPE_INT8) {
                        load_v_tile_dblk_heads_s8(ws.v_cache,
                                                  n_k_tiles,
                                                  v_tile_stride,
                                                  chunk_heads,
                                                  static_cast<const int8_t *>(v_data),
                                                  v_head0_base,
                                                  j0,
                                                  tile_n,
                                                  D,
                                                  v_stride_pos);
                    } else if constexpr (qkv_dtype == DATA_TYPE_INT16) {
                        load_v_tile_dblk_heads_s16(ws.v_cache,
                                                   n_k_tiles,
                                                   v_tile_stride,
                                                   chunk_heads,
                                                   static_cast<const int16_t *>(v_data),
                                                   v_head0_base,
                                                   j0,
                                                   tile_n,
                                                   (tile_n + 7) & ~7,
                                                   D,
                                                   D_pad,
                                                   v_stride_pos);
                    }
                }
#endif
                // 3. publish_kv_cache: both task-owned K and V ranges are ready.
                attention_wait_at_barrier(*execution_context.task_barrier, task_count);
            }

            for (int qb = 0; qb < q_blocks; ++qb) {
                if ((qb % task_count) != task_id) {
                    continue;
                }
                const int i0 = qb * ATTENTION_QUERY_TILE_SIZE;
                const int nq = std::min(ATTENTION_QUERY_TILE_SIZE, Lq - i0);

                if (use_simd) {
                    if constexpr (qkv_dtype == DATA_TYPE_INT8) {
                        copy_q_heads_pad_s8(reinterpret_cast<int8_t *>(tile_buf.q_tile),
                                            ATTENTION_QUERY_TILE_SIZE * D_pad,
                                            static_cast<const int8_t *>(q_data) + q_head0_base + i0 * q_stride_pos,
                                            chunk_heads,
                                            nq,
                                            D,
                                            D_pad,
                                            q_stride_pos);
                    } else if constexpr (qkv_dtype == DATA_TYPE_INT16) {
                        copy_q_heads_pad_s16(reinterpret_cast<int16_t *>(tile_buf.q_tile),
                                             ATTENTION_QUERY_TILE_SIZE * D_pad,
                                             static_cast<const int16_t *>(q_data) + q_head0_base + i0 * q_stride_pos,
                                             chunk_heads,
                                             nq,
                                             D,
                                             D_pad,
                                             q_stride_pos);
                    }
                }

                for (int h_local = 0; h_local < chunk_heads; ++h_local) {
                    const int h = h0 + h_local;
                    const int h_kv = h / group;
                    const int q_head_base = b * q_stride_batch + h * q_stride_head;
                    const int k_head_base = b * k_stride_batch + h_kv * k_stride_head;
                    const int v_head_base = b * v_stride_batch + h_kv * v_stride_head;
                    const int o_head_base = b * o_stride_batch + h * o_stride_head;
                    const int mask_head_base = has_mask ? (b * mask.b_stride + h * mask.h_stride) : 0;
                    const uint8_t *mask_tile_modes = (mask.mask_tile_mode != nullptr)
                        ? mask.mask_tile_mode + b * mask.mask_tile_mode_b_stride + h * mask.mask_tile_mode_h_stride +
                            qb * mask.mask_tile_mode_q_stride
                        : nullptr;
                    const int8_t *q_padded [[maybe_unused]] =
                        reinterpret_cast<const int8_t *>(tile_buf.q_tile) + h_local * ATTENTION_QUERY_TILE_SIZE * D_pad;
                    // Byte view of this head's t-block K; only the batched kernels read it.
                    const int8_t *k_cache_head [[maybe_unused]] = requires_kv_cache
                        ? reinterpret_cast<const int8_t *>(ws.k_cache) + static_cast<size_t>(h_local) * k_head_bytes
                        : nullptr;

                    for (int iq = 0; iq < nq; ++iq) {
                        tile_buf.row_max[iq] = neg_inf;
                        tile_buf.dot_row_max[iq] = INT32_MIN;
                        tile_buf.row_sum[iq] = 0.f;
                    }
                    std::memset(tile_buf.acc, 0, static_cast<size_t>(nq) * ATTENTION_MAX_HEAD_DIM * sizeof(float));

                    if (qk_kernel != AttentionQkKernel::Scalar) {
                        if constexpr (qkv_dtype == DATA_TYPE_FLOAT) {
                            copy_q_block_to_tile_buf<qkv_dtype>(
                                tile_buf.q_tile, q_data, q_head_base, i0, nq, D, D_pad, q_stride_pos);
                        }
                    } else {
                        for (int iq = 0; iq < nq; ++iq) {
                            const int q_row_base = q_head_base + (i0 + iq) * q_stride_pos;
                            float *q_row = tile_buf.q_values + iq * ATTENTION_MAX_HEAD_DIM;
                            for (int d = 0; d < D; ++d) {
                                q_row[d] = load_qkv_element<qkv_dtype>(q_data, q_row_base + d, q_scale);
                            }
                        }
                    }

                    // QK -> softmax -> V tile (V stays L1-resident).
                    // ONNX Attention without past K/V is upper-left aligned:
                    // query i can read keys [0, i]. Keys past the last row's
                    // horizon leave row_max, row_sum and acc unchanged for this
                    // block (the same invariant as MASK_TILE_DROPPED), so skip
                    // their complete tiles. The boundary tile is masked per slot.
                    int causal_visible_key_end = Lk;
                    if constexpr (causal) {
                        causal_visible_key_end = std::min(Lk, i0 + nq);
                    }
                    for (int j0 = 0; j0 < causal_visible_key_end; j0 += ATTENTION_KEY_TILE_SIZE) {
                        const int j1 = std::min(j0 + ATTENTION_KEY_TILE_SIZE, Lk);
                        const int tile_n = j1 - j0;
                        const int n_tok8 = (tile_n + 7) & ~7;
                        // Only a tile whose last key lies past the first row's
                        // horizon needs per-element masking; fully visible tiles
                        // skip it.
                        const bool causal_partial_tile = causal && (j1 - 1 > i0);
                        uint8_t mask_mode = has_mask ? MASK_TILE_MIXED : MASK_TILE_ZERO;
                        if (mask_tile_modes != nullptr) {
                            mask_mode = mask_tile_modes[j0 / ATTENTION_KEY_TILE_SIZE];
                            // All -inf: row_max, row_sum and acc would come out unchanged.
                            if (mask_mode == MASK_TILE_DROPPED) {
                                continue;
                            }
                        }

                        // Row kernels read raw K at its layout stride.
                        const int k_raw_row0 = k_head_base + j0 * k_stride_pos;

#if CONFIG_PIE_V2_BOOST
                        if constexpr (qkv_dtype == DATA_TYPE_INT8) {
                            if (qk_kernel == AttentionQkKernel::BatchedDotsS8) {
                                const int8_t *k_tblock = k_cache_head + j0 * D_pad;
                                if (D_pad == 64) {
                                    dl_esp32p4_s8_attention_qk_qacc_d64(tile_buf.dots,
                                                                        q_padded,
                                                                        k_tblock,
                                                                        nq,
                                                                        tile_n,
                                                                        ATTENTION_KEY_TILE_SIZE,
                                                                        tile_buf.tile_dot_max,
                                                                        tile_buf.qacc_buffer);
                                } else if (D_pad == 32) {
                                    dl_esp32p4_s8_attention_qk_qacc_d32(tile_buf.dots,
                                                                        q_padded,
                                                                        k_tblock,
                                                                        nq,
                                                                        tile_n,
                                                                        ATTENTION_KEY_TILE_SIZE,
                                                                        tile_buf.tile_dot_max,
                                                                        tile_buf.qacc_buffer);
                                } else {
                                    dl_esp32p4_s8_attention_qk_qacc_generic(tile_buf.dots,
                                                                            q_padded,
                                                                            k_tblock,
                                                                            nq,
                                                                            tile_n,
                                                                            ATTENTION_KEY_TILE_SIZE,
                                                                            tile_buf.tile_dot_max,
                                                                            tile_buf.qacc_buffer,
                                                                            D_pad);
                                }
                                if (causal_partial_tile) {
                                    for (int iq = 0; iq < nq; ++iq) {
                                        attention_causal_mask_dots(tile_buf.dots + iq * ATTENTION_KEY_TILE_SIZE,
                                                                   tile_n,
                                                                   j0,
                                                                   i0 + iq,
                                                                   &tile_buf.tile_dot_max[iq]);
                                    }
                                }
                            }
                        } else if constexpr (qkv_dtype == DATA_TYPE_INT16) {
                            // int16 QACC batched Q·K writes rounded fp32 scores.
                            if (qk_kernel == AttentionQkKernel::BatchedScoresS16) {
                                const int16_t *q_block = reinterpret_cast<const int16_t *>(tile_buf.q_tile) +
                                    static_cast<size_t>(h_local) * ATTENTION_QUERY_TILE_SIZE *
                                        static_cast<size_t>(D_pad);
                                const int16_t *k_tblock = reinterpret_cast<const int16_t *>(k_cache_head) + j0 * D_pad;
                                const float dot_scale = logit_scale * static_cast<float>(1 << int16_dot_shift(D));
                                if (D_pad == 32) {
                                    dl_esp32p4_s16_attention_qk_qacc_d32(tile_buf.scores,
                                                                         q_block,
                                                                         k_tblock,
                                                                         nq,
                                                                         tile_n,
                                                                         ATTENTION_KEY_TILE_SIZE,
                                                                         dot_scale,
                                                                         tile_buf.qacc_buffer);
                                } else {
                                    dl_esp32p4_s16_attention_qk_qacc_generic(tile_buf.scores,
                                                                             q_block,
                                                                             k_tblock,
                                                                             nq,
                                                                             tile_n,
                                                                             ATTENTION_KEY_TILE_SIZE,
                                                                             tile_buf.qacc_buffer,
                                                                             D,
                                                                             dot_scale);
                                }
                                if (causal_partial_tile) {
                                    for (int iq = 0; iq < nq; ++iq) {
                                        float *scores = tile_buf.scores + iq * ATTENTION_KEY_TILE_SIZE;
                                        const int causal_max_j = i0 + iq;
                                        for (int t = 0; t < tile_n; ++t) {
                                            if (j0 + t > causal_max_j) {
                                                scores[t] = neg_inf;
                                            }
                                        }
                                    }
                                }
                            }
                        }
#endif

                        // QK-softmax: QK + online softmax -> exp_scores[iq, t] (the P tile).
                        for (int iq = 0; iq < nq; ++iq) {
                            float *acc = tile_buf.acc + iq * ATTENTION_MAX_HEAD_DIM;
                            float *scores = tile_buf.scores + iq * ATTENTION_KEY_TILE_SIZE;
                            float *exp_row = tile_buf.exp_scores + iq * ATTENTION_KEY_TILE_SIZE;
                            float row_max = tile_buf.row_max[iq];
                            float row_sum = tile_buf.row_sum[iq];

                            if (!batched_qk) {
                                switch (qk_kernel) {
                                case AttentionQkKernel::RowSimd: {
#if CONFIG_PIE_V2_BOOST
                                    if constexpr (qkv_dtype == DATA_TYPE_FLOAT) {
                                        const int q_row_base = iq * D_pad;
                                        for (int t = 0; t < tile_n; ++t) {
                                            scores[t] = simd_scaled_dot<qkv_dtype>(q_padded,
                                                                                   q_row_base,
                                                                                   k_data,
                                                                                   k_raw_row0 + t * k_stride_pos,
                                                                                   D_pad,
                                                                                   logit_scale);
                                        }
                                    }
#endif
                                    break;
                                }
                                case AttentionQkKernel::Scalar:
                                default: {
                                    const float *q_row = tile_buf.q_values + iq * ATTENTION_MAX_HEAD_DIM;
                                    for (int t = 0; t < tile_n; ++t) {
                                        const float dot = qk_scalar_dot_row<qkv_dtype>(
                                            q_row, k_data, k_raw_row0 + t * k_stride_pos, D, k_scale);
                                        scores[t] = dot * softmax_scale;
                                    }
                                    break;
                                }
                                }
                                if (causal_partial_tile) {
                                    const int causal_max_j = i0 + iq;
                                    for (int t = 0; t < tile_n; ++t) {
                                        if (j0 + t > causal_max_j) {
                                            scores[t] = neg_inf;
                                        }
                                    }
                                }
                            }

                            [[maybe_unused]] bool mask_dropped = false;
                            if (mask_mode != MASK_TILE_ZERO) {
                                const int mask_row = mask_head_base + (i0 + iq) * mask.q_stride + j0 * mask.k_stride;
                                if (dot_space && mask.mask_dots != nullptr) {
                                    mask_dropped =
                                        attention_mask_dots_cached(tile_buf.dots + iq * ATTENTION_KEY_TILE_SIZE,
                                                                   tile_n,
                                                                   mask.mask_dots + mask_row,
                                                                   mask.k_stride,
                                                                   &tile_buf.tile_dot_max[iq]);
                                } else if (dot_space) {
                                    mask_dropped = attention_mask_dots(tile_buf.dots + iq * ATTENTION_KEY_TILE_SIZE,
                                                                       tile_n,
                                                                       mask,
                                                                       mask_row,
                                                                       mask_to_dots,
                                                                       &tile_buf.tile_dot_max[iq]);
                                } else {
                                    attention_mask_scores(scores, tile_n, mask, mask_row);
                                }
                            }

#if CONFIG_PIE_V2_BOOST
                            if (dot_space) {
                                auto softmax_row = [&](auto masked_slots) {
                                    attention_softmax_quant_row_s8<decltype(masked_slots)::value>(
                                        tile_buf.dots + iq * ATTENTION_KEY_TILE_SIZE,
                                        tile_n,
                                        ATTENTION_KEY_TILE_SIZE,
                                        logit_scale,
                                        &tile_buf.dot_row_max[iq],
                                        acc,
                                        D,
                                        tile_buf.quantized_exp_scores + iq * ATTENTION_KEY_TILE_SIZE,
                                        &tile_buf.row_sum[iq],
                                        &tile_buf.exp_scale[iq],
                                        v_scale,
                                        tile_buf.tile_dot_max[iq],
                                        dot_space,
                                        *execution_context.int8_exp_lut,
                                        tile_buf.exp_scratch);
                                };
                                if (causal_partial_tile || mask_dropped) {
                                    softmax_row(std::true_type{});
                                } else {
                                    softmax_row(std::false_type{});
                                }
                                continue;
                            }
#endif

                            float tile_max = neg_inf;
                            for (int t = 0; t < tile_n; ++t) {
                                if (scores[t] > tile_max)
                                    tile_max = scores[t];
                            }
                            const float new_max = (tile_max > row_max) ? tile_max : row_max;

                            if (!is_neg_inf(row_max)) {
                                const float rescale = softmax_expf<qkv_dtype>(row_max - new_max);
                                row_sum *= rescale;
                                for (int d = 0; d < D; ++d) {
                                    acc[d] *= rescale;
                                }
                            }

                            if constexpr (qkv_dtype == DATA_TYPE_INT16) {
                                if (pv_kernel == AttentionPvKernel::QaccS16) {
                                    row_sum = softmax_exp_row_split_s16(
                                        reinterpret_cast<int16_t *>(tile_buf.exp_scores) + iq * 2 * n_tok8,
                                        scores,
                                        tile_n,
                                        n_tok8,
                                        new_max,
                                        row_sum);
                                } else {
                                    row_sum = softmax_exp_row_precise(exp_row, scores, tile_n, new_max, row_sum);
                                }
                            } else {
                                for (int t = 0; t < tile_n; ++t) {
                                    if (is_neg_inf(scores[t])) {
                                        exp_row[t] = 0.f;
                                        continue;
                                    }
                                    const float exp_value = softmax_expf<qkv_dtype>(scores[t] - new_max);
                                    exp_row[t] = exp_value;
                                    row_sum += exp_value;
                                }
                            }

                            tile_buf.row_max[iq] = new_max;
                            tile_buf.row_sum[iq] = row_sum;
                        }

                        // P-V accumulation: acc += exp_scores x V tile.
                        if (pv_kernel == AttentionPvKernel::QaccS8) {
#if CONFIG_PIE_V2_BOOST
                            if constexpr (qkv_dtype == DATA_TYPE_INT8) {
                                const int8_t *v_tile =
                                    ws.v_cache + (h_local * n_k_tiles + j0 / ATTENTION_KEY_TILE_SIZE) * v_tile_stride;
                                pv_accumulate_int8(tile_buf.acc,
                                                   tile_buf.exp_scores,
                                                   tile_buf.quantized_exp_scores,
                                                   tile_buf.exp_scale,
                                                   v_tile,
                                                   nq,
                                                   tile_n,
                                                   D_pad,
                                                   v_scale,
                                                   dot_space,
                                                   dot_space,
                                                   tile_buf.qacc_buffer);
                            }
#endif
                        } else if (pv_kernel == AttentionPvKernel::QaccS16) {
#if CONFIG_PIE_V2_BOOST
                            const int16_t *v_dblk = reinterpret_cast<const int16_t *>(
                                ws.v_cache + (h_local * n_k_tiles + j0 / ATTENTION_KEY_TILE_SIZE) * v_tile_stride);
                            dl_esp32p4_s16_attention_pv_qacc(tile_buf.acc,
                                                             reinterpret_cast<const int16_t *>(tile_buf.exp_scores),
                                                             v_dblk,
                                                             nq,
                                                             n_tok8,
                                                             D_pad,
                                                             ATTENTION_MAX_HEAD_DIM,
                                                             tile_buf.qacc_buffer,
                                                             pv_scale);
#endif
                        } else {
                            // PIE v2 int8 is always QaccS8 (D is validated to 1..MAX); keeping
                            // the dead Float path out of that instance measurably helps I-cache.
                            if constexpr (qkv_dtype != DATA_TYPE_INT8 || !CONFIG_PIE_V2_BOOST) {
                                load_v_tile_dequant<qkv_dtype>(
                                    tile_buf.v_values, v_data, v_head_base, j0, tile_n, D, v_scale, v_stride_pos);
                                pv_accumulate_fp32(tile_buf.acc, tile_buf.exp_scores, tile_buf.v_values, nq, tile_n, D);
                            }
                        }
                    }

                    // int32 P·V deferred the per-tile dequant; fold into 1/l.
                    const float output_scale = dot_space ? (v_scale * (1.0f / 127.0f)) : 1.0f;
                    for (int iq = 0; iq < nq; ++iq) {
                        const int i = i0 + iq;
                        const float row_sum = tile_buf.row_sum[iq];
                        const float inv_sum = (row_sum > 0.f) ? (output_scale / row_sum) : 0.f;
                        const int o_row_base = o_head_base + i * o_stride_pos;
                        if (dot_space) {
                            const int32_t *acc =
                                reinterpret_cast<const int32_t *>(tile_buf.acc + iq * ATTENTION_MAX_HEAD_DIM);
                            for (int d = 0; d < D; ++d) {
                                store_out_element<qkv_dtype>(
                                    o_data, o_row_base + d, static_cast<float>(acc[d]) * inv_sum, o_inv_scale);
                            }
                        } else {
                            const float *acc = tile_buf.acc + iq * ATTENTION_MAX_HEAD_DIM;
                            for (int d = 0; d < D; ++d) {
                                store_out_element<qkv_dtype>(o_data, o_row_base + d, acc[d] * inv_sum, o_inv_scale);
                            }
                        }
                    }
                } // h_local
            }

            // No trailing barrier after the last chunk: nothing overwrites the
            // caches afterwards, and the framework joins the tasks on return.
            const bool is_last_chunk = (b + 1 >= B) && (h0 + head_chunk_size >= H);
            if (requires_kv_cache && !is_last_chunk) {
                attention_wait_at_barrier(*execution_context.task_barrier, task_count);
            }
        }
    }
}

/**
 * @brief Runs on the calling task before the tiled tasks launch, keeping the LUT build off the
 * dual-core task stacks.
 */
bool prepare_int8_softmax_exp_lut_impl(AttentionInt8ExpLut &lut,
                                       const TensorBase *q_t,
                                       const TensorBase *k_t,
                                       bool has_scale,
                                       float scale_attr,
                                       const char *who)
{
#if CONFIG_PIE_V2_BOOST
    // The batched int8 softmax has no other exp table, so a failed build
    // fails the node. build_int8_exp_lut is idempotent for an unchanged scale.
    const float logit_scale = attention_logit_scale<DATA_TYPE_INT8>(q_t, k_t, has_scale, scale_attr);
    if (!build_int8_exp_lut(lut, logit_scale)) {
        ESP_LOGE("Attention", "%s: cannot build the int8 softmax exp LUT (logit_scale=%.6g).", who, logit_scale);
        return false;
    }
#else
    (void)lut;
    (void)q_t;
    (void)k_t;
    (void)has_scale;
    (void)scale_attr;
    (void)who;
#endif
    return true;
}

/**
 * @brief Admits attn_mask iff it right-aligns to <=4 dims broadcastable onto [B, Hq, Lq, Lk].
 *
 * Rejection fails the node (silently dropping a mask is the worst failure mode); ai.onnx::Attention
 * spec disallows rank > 4.
 */
static bool attention_check_mask_dims(
    const TensorBase *mask_t, int B, int Hq, int Lq, int Lk, int dims_out[4], const char *who)
{
    dims_out[0] = dims_out[1] = dims_out[2] = dims_out[3] = 1;
    if (mask_t == nullptr) {
        ESP_LOGE("Attention", "%s: attn_mask tensor is null", who);
        return false;
    }
    if (mask_t->data == nullptr) {
        ESP_LOGE("Attention", "%s: attn_mask is connected but has no data", who);
        return false;
    }

    // ONNX Attention / SDPA semantics: bool (keep/drop) or float (additive).
    // A quantized mask cannot hold -inf, so int8/int16 are rejected, not dequantized.
    if (mask_t->dtype != DATA_TYPE_FLOAT && mask_t->dtype != DATA_TYPE_BOOL) {
        ESP_LOGE("Attention",
                 "%s: attn_mask dtype=%d unsupported (need FLOAT or BOOL)",
                 who,
                 static_cast<int>(mask_t->dtype));
        return false;
    }

    const auto &mask_shape = mask_t->shape;
    const int mask_rank = static_cast<int>(mask_shape.size());
    if (mask_rank > 4) {
        ESP_LOGE(
            "Attention", "%s: rank-%d attn_mask cannot broadcast onto 4-D scores; squeeze to <=4 dims", who, mask_rank);
        return false;
    }

    // Right-align onto [B, Hq, Lq, Lk] (ONNX/NumPy broadcast rule).
    for (int i = 0; i < mask_rank; ++i) {
        dims_out[4 - mask_rank + i] = mask_shape[i];
    }
    for (int i = 0; i < 4; ++i) {
        if (dims_out[i] <= 0) {
            ESP_LOGE("Attention", "%s: attn_mask has non-positive dim[%d]=%d", who, i, dims_out[i]);
            return false;
        }
    }

    // Each dim must be 1 (broadcast) or the full extent.
    const int Bm = dims_out[0];
    const int Hm = dims_out[1];
    const int Lqm = dims_out[2];
    const int Lkm = dims_out[3];
    if (!((Bm == 1 || Bm == B) && (Hm == 1 || Hm == Hq) && (Lqm == 1 || Lqm == Lq) && (Lkm == 1 || Lkm == Lk))) {
        ESP_LOGE("Attention",
                 "%s: attn_mask [%d,%d,%d,%d] not broadcastable to [B=%d,H=%d,Lq=%d,Lk=%d]",
                 who,
                 Bm,
                 Hm,
                 Lqm,
                 Lkm,
                 B,
                 Hq,
                 Lq,
                 Lk);
        return false;
    }
    return true;
}

/**
 * @brief Build broadcast strides for an optional attn_mask tensor.
 */
inline AttentionMaskArgs make_mask_args(const TensorBase *mask_t, int B, int Hq, int Lq, int Lk, const char *who)
{
    AttentionMaskArgs mask_args{nullptr, DATA_TYPE_UNDEFINED, 0, 0, 0, 0};
    if (mask_t == nullptr)
        return mask_args;

    int mask_dims[4];
    if (!attention_check_mask_dims(mask_t, B, Hq, Lq, Lk, mask_dims, who)) {
        return mask_args;
    }
    const int Bm = mask_dims[0];
    const int Hm = mask_dims[1];
    const int Lqm = mask_dims[2];
    const int Lkm = mask_dims[3];

    // Row-major strides; broadcast axes get stride 0.
    const int s_k = 1;
    const int s_q = Lkm;
    const int s_h = Lqm * Lkm;
    const int s_b = Hm * Lqm * Lkm;
    mask_args.data = mask_t->data;
    mask_args.dtype = mask_t->dtype;
    mask_args.b_stride = (Bm == 1) ? 0 : s_b;
    mask_args.h_stride = (Hm == 1) ? 0 : s_h;
    mask_args.q_stride = (Lqm == 1) ? 0 : s_q;
    mask_args.k_stride = (Lkm == 1) ? 0 : s_k;
    return mask_args;
}

/**
 * @brief Builds (or reuses) the constant-mask cache and attaches it to mask_args.
 *
 * Only float int8 masks need mask_dots: bool masks are already exact keep/drop
 * values and continue to use their compact byte representation. If the optional
 * dot cache cannot be allocated, mask_tile_mode remains useful and the kernel
 * reads the original mask for mixed tiles.
 */
__attribute__((noinline)) static void prepare_constant_mask_cache_impl(AttentionMaskCache &mask_cache,
                                                                       AttentionMaskArgs &mask_args,
                                                                       int B,
                                                                       int H,
                                                                       int Lq,
                                                                       int Lk,
                                                                       float to_dots,
                                                                       const char *who)
{
    const bool source_hit = mask_cache.mask_tile_mode != nullptr && mask_cache.source_data == mask_args.data &&
        mask_cache.source_dtype == mask_args.dtype && mask_cache.B == B && mask_cache.H == H && mask_cache.Lq == Lq &&
        mask_cache.Lk == Lk;
    bool rebuilt_tile_mode = false;
    int counts[4] = {0, 0, 0, 0};
    if (!source_hit) {
        mask_cache.release();
        const int Bm = mask_args.b_stride ? B : 1;
        const int Hm = mask_args.h_stride ? H : 1;
        const int Lqm = mask_args.q_stride ? Lq : 1;
        const int Lkm = mask_args.k_stride ? Lk : 1;
        const int n_q_blocks =
            mask_args.q_stride ? (Lq + ATTENTION_QUERY_TILE_SIZE - 1) / ATTENTION_QUERY_TILE_SIZE : 1;
        const int n_k_tiles = (Lk + ATTENTION_KEY_TILE_SIZE - 1) / ATTENTION_KEY_TILE_SIZE;
        const size_t n_tiles =
            static_cast<size_t>(Bm) * static_cast<size_t>(Hm) * static_cast<size_t>(n_q_blocks) * n_k_tiles;
        // Fail fast on OOM before the O(numel) mask_dots fill; mask_tile_mode must
        // exist for the cache to be usable at all.
        mask_cache.mask_tile_mode = static_cast<uint8_t *>(tool::malloc_aligned(n_tiles, MALLOC_CAP_8BIT));
        if (mask_cache.mask_tile_mode == nullptr) {
            mask_cache.release();
            return;
        }
        mask_cache.mask_tile_mode_q_stride = mask_args.q_stride ? n_k_tiles : 0;
        mask_cache.mask_tile_mode_h_stride = mask_args.h_stride ? n_q_blocks * n_k_tiles : 0;
        mask_cache.mask_tile_mode_b_stride = mask_args.b_stride ? Hm * n_q_blocks * n_k_tiles : 0;
        uint8_t *mode_out = mask_cache.mask_tile_mode;
        for (int bm = 0; bm < Bm; ++bm) {
            for (int hm = 0; hm < Hm; ++hm) {
                for (int qb = 0; qb < n_q_blocks; ++qb) {
                    const int i0 = qb * ATTENTION_QUERY_TILE_SIZE;
                    const int i1 = std::min(i0 + ATTENTION_QUERY_TILE_SIZE, Lqm);
                    for (int kt = 0; kt < n_k_tiles; ++kt) {
                        const int j0 = kt * ATTENTION_KEY_TILE_SIZE;
                        const int j1 = mask_args.k_stride ? std::min(j0 + ATTENTION_KEY_TILE_SIZE, Lkm) : 1;
                        const int jb = mask_args.k_stride ? j0 : 0;
                        int n = 0;
                        int n_zero = 0;
                        int n_inf = 0;
                        for (int i = i0; i < i1; ++i) {
                            const int row = bm * mask_args.b_stride + hm * mask_args.h_stride + i * mask_args.q_stride;
                            for (int j = jb; j < j1; ++j) {
                                const float m = load_mask_element(mask_args, row + j * mask_args.k_stride);
                                ++n;
                                n_zero += (m == 0.f);
                                n_inf += is_neg_inf(m);
                            }
                        }
                        const uint8_t c = (n_inf == n) ? MASK_TILE_DROPPED
                            : (n_inf > 0)              ? MASK_TILE_MIXED
                            : (n_zero == n)            ? MASK_TILE_ZERO
                                                       : MASK_TILE_FINITE;
                        *mode_out++ = c;
                        ++counts[c];
                    }
                }
            }
        }
        mask_cache.source_data = mask_args.data;
        mask_cache.source_dtype = mask_args.dtype;
        mask_cache.B = B;
        mask_cache.H = H;
        mask_cache.Lq = Lq;
        mask_cache.Lk = Lk;
        rebuilt_tile_mode = true;
    }

    const bool needs_mask_dots = (mask_args.dtype == DATA_TYPE_FLOAT) && (to_dots != 0.f);
    if (needs_mask_dots) {
        if (mask_cache.to_dots != to_dots) {
            heap_caps_free(mask_cache.mask_dots);
            mask_cache.mask_dots = nullptr;
            mask_cache.to_dots = to_dots;
            mask_cache.mask_dots_allocation_failed = false;
        }
        if (mask_cache.mask_dots == nullptr && !mask_cache.mask_dots_allocation_failed) {
            const int Bm = mask_args.b_stride ? B : 1;
            const int Hm = mask_args.h_stride ? H : 1;
            const int Lqm = mask_args.q_stride ? Lq : 1;
            const int Lkm = mask_args.k_stride ? Lk : 1;
            const size_t numel =
                static_cast<size_t>(Bm) * static_cast<size_t>(Hm) * static_cast<size_t>(Lqm) * static_cast<size_t>(Lkm);
            const size_t bytes = numel * sizeof(int32_t);
            mask_cache.mask_dots =
                static_cast<int32_t *>(tool::malloc_aligned(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT));
            if (mask_cache.mask_dots == nullptr) {
                mask_cache.mask_dots = static_cast<int32_t *>(tool::malloc_aligned(bytes, MALLOC_CAP_32BIT));
            }
            if (mask_cache.mask_dots == nullptr) {
                mask_cache.mask_dots_allocation_failed = true;
                ESP_LOGW("Attention", "%s: cannot allocate constant-mask dot cache; using direct mask reads.", who);
            } else {
                for (size_t i = 0; i < numel; ++i) {
                    const float m = load_mask_element(mask_args, static_cast<int>(i));
                    mask_cache.mask_dots[i] = is_neg_inf(m) ? QK_DOTS_MASKED : mask_dot_units(m, to_dots);
                }
            }
        }
    }

    if (rebuilt_tile_mode) {
        ESP_LOGI("Attention",
                 "%s: constant attn_mask precomputed (mask_dots %s); tiles zero=%d finite=%d mixed=%d skipped=%d",
                 who,
                 mask_cache.mask_dots ? "int32" : "none",
                 counts[MASK_TILE_ZERO],
                 counts[MASK_TILE_FINITE],
                 counts[MASK_TILE_MIXED],
                 counts[MASK_TILE_DROPPED]);
    }
    mask_args.mask_dots = mask_cache.mask_dots;
    mask_args.mask_tile_mode = mask_cache.mask_tile_mode;
    mask_args.mask_tile_mode_b_stride = mask_cache.mask_tile_mode_b_stride;
    mask_args.mask_tile_mode_h_stride = mask_cache.mask_tile_mode_h_stride;
    mask_args.mask_tile_mode_q_stride = mask_cache.mask_tile_mode_q_stride;
}
} // namespace

/**
 * @brief Runtime checks (assert() dies in NDEBUG).
 */
bool attention_check_shapes(const std::vector<int> &q,
                            const std::vector<int> &k,
                            const std::vector<int> &v,
                            const std::vector<int> *o,
                            AttentionLayout layout,
                            const char *who)
{
    // "-1" in the logs below = "O not available" (load-time call).
    if (q.size() != 4 || k.size() != 4 || v.size() != 4 || (o != nullptr && o->size() != 4)) {
        ESP_LOGE("Attention",
                 "%s: Q/K/V/O must be 4-D, got Q.dim=%zu K.dim=%zu V.dim=%zu O.dim=%d",
                 who,
                 q.size(),
                 k.size(),
                 v.size(),
                 o != nullptr ? static_cast<int>(o->size()) : -1);
        return false;
    }
    const int D = q[3];
    if (D <= 0 || D > ATTENTION_MAX_HEAD_DIM) {
        ESP_LOGE("Attention",
                 "%s: head_dim=%d out of [1, %d]; raise ESP_DL_ATTENTION_MAX_HEAD_DIM "
                 "or export without fused Attention.",
                 who,
                 D,
                 ATTENTION_MAX_HEAD_DIM);
        return false;
    }
    if (k[3] != D || v[3] != D || (o != nullptr && (*o)[3] != D)) {
        ESP_LOGE("Attention",
                 "%s: Q/K/V/O head_dim mismatch (Q=%d K=%d V=%d O=%d); v_head_size != head_size is not supported",
                 who,
                 D,
                 k[3],
                 v[3],
                 o != nullptr ? (*o)[3] : -1);
        return false;
    }
    if (q[0] != k[0] || q[0] != v[0] || (o != nullptr && q[0] != (*o)[0])) {
        ESP_LOGE(
            "Attention", "%s: batch mismatch Q=%d K=%d V=%d O=%d", who, q[0], k[0], v[0], o != nullptr ? (*o)[0] : -1);
        return false;
    }
    if (k[1] != v[1] || k[2] != v[2]) {
        ESP_LOGE(
            "Attention", "%s: K and V must agree on (Hkv, Lk); got K=(%d,%d) V=(%d,%d)", who, k[1], k[2], v[1], v[2]);
        return false;
    }
    const AttentionDims dims = attention_dims(layout, q, k);
    const int h_q = dims.Hq;
    const int h_kv = dims.Hkv;
    if (h_q <= 0 || h_kv <= 0 || (h_q % h_kv) != 0) {
        ESP_LOGE("Attention", "%s: Hq must be a positive multiple of Hkv; got Hq=%d Hkv=%d", who, h_q, h_kv);
        return false;
    }
    if (o != nullptr && ((*o)[1] != q[1] || (*o)[2] != q[2])) {
        ESP_LOGE(
            "Attention", "%s: O shape (H,L)=(%d,%d) must match Q (H,L)=(%d,%d)", who, (*o)[1], (*o)[2], q[1], q[2]);
        return false;
    }
    const int lk = dims.Lk;
    if (lk <= 0 || lk > ATTENTION_MAX_KV_TOKENS) {
        ESP_LOGE("Attention",
                 "%s: Lk=%d > ATTENTION_MAX_KV_TOKENS=%d; raise ESP_DL_ATTENTION_MAX_KV_TOKENS.",
                 who,
                 lk,
                 ATTENTION_MAX_KV_TOKENS);
        return false;
    }
    return true;
}

/**
 * @brief The int8 kernel runs softmax on raw integer dots and takes the row maximum in dot units,
 * which is the score maximum only for a positive logit_scale.
 *
 * The rule covers every dtype so admission does not depend on the quantization; esp-ppq applies the
 * same rule and leaves other scales as primitives.
 */
bool attention_check_scale(bool has_scale, float scale, const char *who)
{
    if (has_scale && !is_positive_normal(scale)) {
        ESP_LOGE("Attention", "%s: scale=%.6g must be a finite, positive, normal float32.", who, scale);
        return false;
    }
    return true;
}

/**
 * @brief Per-forward checks: shapes plus conditions that need live tensors.
 */
bool attention_check_inputs(const TensorBase *q,
                            const TensorBase *k,
                            const TensorBase *v,
                            const TensorBase *mask,
                            const TensorBase *o,
                            bool has_scale,
                            float scale,
                            AttentionLayout layout,
                            dtype_t expected_qkv_dtype,
                            const char *who)
{
    if (layout != AttentionLayout::BHND && layout != AttentionLayout::BNHD) {
        ESP_LOGE("Attention", "%s: layout must be BHND or BNHD; got %d.", who, static_cast<int>(layout));
        return false;
    }
    if (q == nullptr || k == nullptr || v == nullptr || o == nullptr) {
        ESP_LOGE("Attention", "%s: Q/K/V/O tensor is null", who);
        return false;
    }
    if (!attention_check_shapes(q->shape, k->shape, v->shape, &o->shape, layout, who)) {
        return false;
    }
    if (!attention_check_scale(has_scale, scale, who)) {
        return false;
    }
    if (q->dtype != expected_qkv_dtype || k->dtype != expected_qkv_dtype || v->dtype != expected_qkv_dtype ||
        o->dtype != expected_qkv_dtype) {
        ESP_LOGE("Attention",
                 "%s: Q/K/V/O dtype must match quant_type (expected %s, got Q=%s K=%s V=%s O=%s)",
                 who,
                 dtype_to_string(expected_qkv_dtype),
                 dtype_to_string(q->dtype),
                 dtype_to_string(k->dtype),
                 dtype_to_string(v->dtype),
                 dtype_to_string(o->dtype));
        return false;
    }
    // Q/K exponents can still push the product out of the normal range.
    const float logit_scale = (expected_qkv_dtype == DATA_TYPE_FLOAT)
        ? attention_logit_scale<DATA_TYPE_FLOAT>(q, k, has_scale, scale)
        : attention_logit_scale<DATA_TYPE_INT8>(q, k, has_scale, scale);
    if (!is_positive_normal(logit_scale)) {
        ESP_LOGE("Attention",
                 "%s: logit_scale=%.6g (scale x Q/K dequant scales, exponents %d/%d) must be a finite, positive, "
                 "normal float32.",
                 who,
                 logit_scale,
                 static_cast<int>(q->exponent),
                 static_cast<int>(k->exponent));
        return false;
    }
    if (mask != nullptr) {
        const AttentionDims dims = attention_dims(layout, q->shape, k->shape);
        int mask_dims[4];
        if (!attention_check_mask_dims(mask, dims.B, dims.Hq, dims.Lq, dims.Lk, mask_dims, who)) {
            return false;
        }
    }
    return true;
}

void AttentionMaskCache::release()
{
    heap_caps_free(mask_dots);
    heap_caps_free(mask_tile_mode);
    mask_dots = nullptr;
    mask_tile_mode = nullptr;
    source_data = nullptr;
    source_dtype = DATA_TYPE_UNDEFINED;
    B = H = Lq = Lk = 0;
    to_dots = 0.f;
    mask_dots_allocation_failed = false;
    mask_tile_mode_b_stride = 0;
    mask_tile_mode_h_stride = 0;
    mask_tile_mode_q_stride = 0;
}

float attention_exp_neg_lut_lookup(float x)
{
    return attention_float_exp_lut().lookup(x);
}

float attention_exp_neg_lut_lookup_precise(float x)
{
    return attention_float_exp_lut().lookup_precise(x);
}

float attention_softmax_exp_row_precise(float *e, const float *s, int n, float m, float l)
{
    return softmax_exp_row_precise(e, s, n, m, l);
}

AttentionWorkspaceSizes attention_workspace_sizes(dtype_t qkv_dtype,
                                                  const std::vector<int> &q_shape,
                                                  const std::vector<int> &k_shape,
                                                  AttentionLayout layout)
{
    const AttentionWorkspaceLayout regions = attention_workspace_layout(qkv_dtype, q_shape, k_shape, layout);
    return {regions.total(), regions.tile_bytes, regions.k_bytes + regions.v_bytes};
}

AttentionWorkspace attention_bind_workspace(void *storage,
                                            size_t storage_bytes,
                                            dtype_t qkv_dtype,
                                            const std::vector<int> &q_shape,
                                            const std::vector<int> &k_shape,
                                            AttentionLayout layout)
{
    AttentionWorkspace ws;
    const AttentionWorkspaceLayout regions = attention_workspace_layout(qkv_dtype, q_shape, k_shape, layout);
    if (storage == nullptr || storage_bytes < regions.total()) {
        return ws;
    }
    uint8_t *base = static_cast<uint8_t *>(storage);
    ws.tile_buf = base;
    ws.tile_buf_bytes = regions.tile_bytes;
    if (regions.k_bytes > 0) {
        ws.k_cache = base + regions.tile_bytes;
        ws.k_cache_bytes = regions.k_bytes;
    }
    if (regions.v_bytes > 0) {
        ws.v_cache = reinterpret_cast<int8_t *>(base + regions.tile_bytes + regions.k_bytes);
        ws.v_cache_bytes = regions.v_bytes;
    }
    return ws;
}

AttentionMaskArgs get_attention_mask_args(
    const TensorBase *mask, const TensorBase *q, const TensorBase *k, AttentionLayout layout, const char *who)
{
    const AttentionDims dims = attention_dims(layout, q->shape, k->shape);
    return make_mask_args(mask, dims.B, dims.Hq, dims.Lq, dims.Lk, who);
}

template <typename feature_t>
void attention_prepare_constant_mask_cache(AttentionMaskCache &mask_cache,
                                           AttentionMaskArgs &mask_args,
                                           const TensorBase *q,
                                           const TensorBase *k,
                                           bool has_scale,
                                           float scale,
                                           AttentionLayout layout,
                                           const char *who)
{
    constexpr dtype_t qkv_dtype = attention_dtype<feature_t>();
    const AttentionDims dims = attention_dims(layout, q->shape, k->shape);
    // Must equal attention_forward_impl's mask_to_dots bit for bit.
    const float to_dots =
        (qkv_dtype == DATA_TYPE_INT8) ? 1.0f / attention_logit_scale<DATA_TYPE_INT8>(q, k, has_scale, scale) : 0.f;
    prepare_constant_mask_cache_impl(mask_cache, mask_args, dims.B, dims.Hq, dims.Lq, dims.Lk, to_dots, who);
}

bool attention_prepare_int8_softmax_exp_lut(
    AttentionInt8ExpLut &lut, const TensorBase *q, const TensorBase *k, bool has_scale, float scale, const char *who)
{
    return prepare_int8_softmax_exp_lut_impl(lut, q, k, has_scale, scale, who);
}

std::vector<AttentionArgs> get_attention_operation_args(TensorBase *output,
                                                        TensorBase *q,
                                                        TensorBase *k,
                                                        TensorBase *v,
                                                        const AttentionMaskArgs &mask_args,
                                                        const AttentionWorkspace &workspace,
                                                        const AttentionExecutionContext &execution_context,
                                                        bool has_scale,
                                                        float scale,
                                                        bool is_causal,
                                                        AttentionLayout layout,
                                                        const runtime_mode_t runtime_mode)
{
    AttentionArgs args;
    args.q = q;
    args.k = k;
    args.v = v;
    args.o = output;
    args.has_scale = has_scale;
    args.scale = scale;
    args.is_causal = is_causal;
    args.layout = layout;
    args.mask = mask_args;
    args.workspace = workspace;
    args.execution_context = &execution_context;
    args.task_id = 0;
    args.task_count = 1;
    const auto select_head_chunk = [&](AttentionArgs &selected) {
        if (q == nullptr || k == nullptr || q->shape.size() != 4 || k->shape.size() != 4) {
            selected.head_chunk_size = 1;
            return;
        }
        const AttentionDims dims = attention_dims(layout, q->shape, k->shape);
        const AttentionCachePlan plan = attention_cache_plan(q->dtype, dims.Lk, dims.D);
        const int elem_size = (q->dtype == DATA_TYPE_INT16) ? 2 : 1;
        selected.head_chunk_size = plan.chunks_heads
            ? attention_head_chunk_runtime_limit(layout == AttentionLayout::BNHD,
                                                 dims.Hq,
                                                 dims.Hkv,
                                                 dims.Lq,
                                                 dims.Lk,
                                                 dims.D,
                                                 plan.D_pad,
                                                 elem_size,
                                                 selected.task_count)
            : 1;
    };

#if CONFIG_FREERTOS_NUMBER_OF_CORES > 1
    // Split across both cores unless single-core is requested; AUTO keeps that.
    if (runtime_mode != RUNTIME_MODE_SINGLE_CORE) {
        static_assert(ATTENTION_TASK_COUNT == 2, "module_forward_dual_core runs exactly two args");
        std::vector<AttentionArgs> m_args(ATTENTION_TASK_COUNT, args);
        for (int i = 0; i < ATTENTION_TASK_COUNT; ++i) {
            m_args[i].task_id = i;
            m_args[i].task_count = ATTENTION_TASK_COUNT;
            select_head_chunk(m_args[i]);
        }
        return m_args;
    }
#endif
    select_head_chunk(args);
    return {args};
}

template <typename feature_t>
void attention(void *args_ptr)
{
    constexpr dtype_t qkv_dtype = attention_dtype<feature_t>();
    const AttentionArgs &a = *static_cast<const AttentionArgs *>(args_ptr);
    assert(a.workspace.tile_buf != nullptr && a.workspace.tile_buf_bytes >= attention_tile_buffer_bytes() &&
           "Attention requires its complete per-task tile buffer");
    assert(a.task_id >= 0 && a.task_id < ATTENTION_TILE_BUFFER_COUNT &&
           "Attention task_id exceeds the planned tile-buffer count");
    AttentionTileBuffer *tile_buf = static_cast<AttentionTileBuffer *>(a.workspace.tile_buf);
    assert(a.execution_context != nullptr && "Attention: execution context must be supplied");
    if (a.is_causal) {
        attention_forward_impl<qkv_dtype, true>(a.q,
                                                a.k,
                                                a.v,
                                                a.o,
                                                a.has_scale,
                                                a.scale,
                                                a.task_id,
                                                a.task_count,
                                                a.head_chunk_size,
                                                tile_buf[a.task_id],
                                                a.workspace,
                                                a.layout,
                                                a.mask,
                                                *a.execution_context);
    } else {
        attention_forward_impl<qkv_dtype, false>(a.q,
                                                 a.k,
                                                 a.v,
                                                 a.o,
                                                 a.has_scale,
                                                 a.scale,
                                                 a.task_id,
                                                 a.task_count,
                                                 a.head_chunk_size,
                                                 tile_buf[a.task_id],
                                                 a.workspace,
                                                 a.layout,
                                                 a.mask,
                                                 *a.execution_context);
    }
}

#define DL_BASE_ATTENTION_INSTANTIATE(T)                                         \
    template void attention_prepare_constant_mask_cache<T>(AttentionMaskCache &, \
                                                           AttentionMaskArgs &,  \
                                                           const TensorBase *,   \
                                                           const TensorBase *,   \
                                                           bool,                 \
                                                           float,                \
                                                           AttentionLayout,      \
                                                           const char *);        \
    template void attention<T>(void *);

DL_BASE_ATTENTION_INSTANTIATE(int8_t)
DL_BASE_ATTENTION_INSTANTIATE(int16_t)
DL_BASE_ATTENTION_INSTANTIATE(float)
#undef DL_BASE_ATTENTION_INSTANTIATE

} // namespace base
} // namespace dl
