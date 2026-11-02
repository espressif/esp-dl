#pragma once

#include "dl_base.hpp"
#include "sdkconfig.h"

#include <cstdint>
#include <vector>

namespace dl {

enum class AttentionLayout : uint8_t {
    BHND = 0, /*!< [B, H, N, D] */
    BNHD = 1, /*!< [B, N, H, D] */
};

namespace base {

#ifndef ESP_DL_ATTENTION_MAX_HEAD_DIM
#define ESP_DL_ATTENTION_MAX_HEAD_DIM 64 /*!< override with -DESP_DL_ATTENTION_MAX_HEAD_DIM=128 */
#endif
constexpr int ATTENTION_MAX_HEAD_DIM =
    ESP_DL_ATTENTION_MAX_HEAD_DIM; /*!< head_dim ceiling; keep in sync with esp-ppq */
static_assert(ATTENTION_MAX_HEAD_DIM >= 16, "head_dim must be at least 16 (one QACC lane)");
static_assert(ATTENTION_MAX_HEAD_DIM <= 128, "head_dim ceiling has only been validated up to 128");
static_assert((ATTENTION_MAX_HEAD_DIM % 16) == 0, "head_dim ceiling must be a multiple of 16 (QACC lane count)");

/**
 * @brief attn_mask broadcast onto [B, Hq, Lq, Lk]; data == nullptr means none.
 *
 * FLOAT is an additive bias; BOOL follows ONNX / SDPA (true = attend, false = -inf).
 */
struct AttentionMaskArgs {
    const void *data = nullptr;              /*!< mask payload; nullptr means none */
    dtype_t dtype = DATA_TYPE_UNDEFINED;     /*!< FLOAT / BOOL */
    int b_stride = 0;                        /*!< batch stride in elements; broadcast dims get 0 */
    int h_stride = 0;                        /*!< head stride in elements; broadcast dims get 0 */
    int q_stride = 0;                        /*!< query stride in elements; broadcast dims get 0 */
    int k_stride = 0;                        /*!< key stride in elements; broadcast dims get 0 */
    const int32_t *mask_dots = nullptr;      /*!< constant-mask precompute; -inf is QK_DOTS_MASKED */
    const uint8_t *mask_tile_mode = nullptr; /*!< [b][h][q block][k tile]; k tile stride is 1 */
    int mask_tile_mode_b_stride = 0;         /*!< broadcast dims get stride 0 */
    int mask_tile_mode_h_stride = 0;         /*!< broadcast dims get stride 0 */
    int mask_tile_mode_q_stride = 0;         /*!< broadcast dims get stride 0 */
};

/**
 * @brief Load-once precompute for an attn_mask that is a model parameter.
 */
struct AttentionMaskCache {
    const void *source_data = nullptr;          /*!< original mask pointer used as a cache key */
    dtype_t source_dtype = DATA_TYPE_UNDEFINED; /*!< original mask dtype */
    int B = 0;                                  /*!< cached batch */
    int H = 0;                                  /*!< cached query heads */
    int Lq = 0;                                 /*!< cached query length */
    int Lk = 0;                                 /*!< cached key length */
    float to_dots = 0.f;                        /*!< dot-space scale used for mask_dots; 0 means none */
    int32_t *mask_dots = nullptr;               /*!< converted-dot cache */
    bool mask_dots_allocation_failed = false;   /*!< remember OOM so later forwards skip retry */
    uint8_t *mask_tile_mode = nullptr;          /*!< tile classification cache */
    int mask_tile_mode_b_stride = 0;            /*!< broadcast dims get stride 0 */
    int mask_tile_mode_h_stride = 0;            /*!< broadcast dims get stride 0 */
    int mask_tile_mode_q_stride = 0;            /*!< broadcast dims get stride 0 */

    ~AttentionMaskCache() { release(); }
    void release();
};

/**
 * @brief Per-Attention int8 softmax lookup table.
 */
struct AttentionInt8ExpLut {
    alignas(16) int32_t table[512]{}; /*!< 512-entry exp table */
    int32_t shift = 0;                /*!< index shift applied before gather */
    int32_t sat = 0;                  /*!< 511 << shift; clamp applied before the shift */
    float built_for = 0.f;            /*!< logit scale this table was built for */
    bool valid = false;               /*!< whether the table is ready */
};

/**
 * @brief Synchronization state shared only by the tasks of one forward call.
 */
struct AttentionTaskBarrier {
    int bar_count = 0; /*!< arriving task count */
    int bar_phase = 0; /*!< generation used to distinguish consecutive waits */

    void reset()
    {
        bar_count = 0;
        bar_phase = 0;
    }
};

/**
 * @brief Non-owning per-forward state passed to each Attention task.
 */
struct AttentionExecutionContext {
    const AttentionInt8ExpLut *int8_exp_lut = nullptr; /*!< module-owned LUT, stable for this call */
    AttentionTaskBarrier *task_barrier = nullptr;      /*!< invocation-owned barrier, alive until join */
};

/**
 * @brief Byte size of a node's one workspace buffer, and the cut inside it.
 */
struct AttentionWorkspaceSizes {
    size_t bytes = 0;       /*!< single request: tile, then K, then V */
    size_t tile_bytes = 0;  /*!< leading bytes for the per-task tile buffers */
    size_t cache_bytes = 0; /*!< following K then V bytes; included in bytes */
};

/**
 * @brief One node's workspace, bound by attention_bind_workspace() from the buffer
 * attention_workspace_sizes() sizes.
 */
struct AttentionWorkspace {
    void *tile_buf = nullptr;   /*!< one tile buffer per task; nullptr means the node cannot run */
    size_t tile_buf_bytes = 0;  /*!< bytes available at tile_buf */
    uint8_t *k_cache = nullptr; /*!< t-block K repack; mandatory for PIE v2 int8/int16 */
    size_t k_cache_bytes = 0;   /*!< bytes available at k_cache */
    int8_t *v_cache = nullptr;  /*!< d-block V tiles; mandatory for PIE v2 int8/int16 */
    size_t v_cache_bytes = 0;   /*!< bytes available at v_cache */
};

/**
 * @brief Attention operation args.
 */
struct AttentionArgs {
    const TensorBase *q;                                          /*!< query */
    const TensorBase *k;                                          /*!< key */
    const TensorBase *v;                                          /*!< value */
    TensorBase *o;                                                /*!< output */
    bool has_scale;                                               /*!< whether an explicit scale is provided */
    float scale;                                                  /*!< explicit softmax scale */
    bool is_causal;                                               /*!< whether to apply a causal mask */
    AttentionLayout layout;                                       /*!< Q/K/V layout */
    AttentionMaskArgs mask;                                       /*!< optional attn_mask */
    AttentionWorkspace workspace;                                 /*!< bound tile and K/V cache buffers */
    const AttentionExecutionContext *execution_context = nullptr; /*!< module LUT and invocation barrier */
    int task_id;             /*!< Q blocks / K tiles with index % task_count == task_id */
    int task_count;          /*!< 2 only when both args run concurrently */
    int head_chunk_size = 1; /*!< runtime-selected BNHD MHA chunk */
};

/**
 * @brief Check shape admission at load time (o == nullptr) or during forward.
 *
 * @param q       Query shape
 * @param k       Key shape
 * @param v       Value shape
 * @param o       Output shape; nullptr at load time
 * @param layout  Q/K/V layout
 * @param who     Node name used in error logs
 *
 * @return true if the shapes are admitted
 */
bool attention_check_shapes(const std::vector<int> &q,
                            const std::vector<int> &k,
                            const std::vector<int> &v,
                            const std::vector<int> *o,
                            AttentionLayout layout,
                            const char *who);

/**
 * @brief Check the explicit scale attribute at load time.
 *
 * @param has_scale  whether an explicit scale is provided
 * @param scale      explicit softmax scale
 * @param who        Node name used in error logs
 *
 * @return true if the scale is absent or a finite, positive, normal float32
 */
bool attention_check_scale(bool has_scale, float scale, const char *who);

/**
 * @brief Check live tensors before constructing operation args.
 *
 * @param q                   Query tensor
 * @param k                   Key tensor
 * @param v                   Value tensor
 * @param mask                Optional attn_mask
 * @param o                   Output tensor
 * @param has_scale           whether an explicit scale is provided
 * @param scale               explicit softmax scale
 * @param layout              Q/K/V layout
 * @param expected_qkv_dtype  dtype implied by the node's quant_type
 * @param who                 Node name used in error logs
 *
 * @return true if the tensors, dtypes, shapes and scales are valid
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
                            const char *who);

/**
 * @brief Workspace one node needs.
 *
 * @param qkv_dtype  Q/K/V data type
 * @param q_shape    Query shape, following layout
 * @param k_shape    Key shape, following layout
 * @param layout     Q/K/V layout
 *
 * @return one buffer holding the per-task tile buffers, then the K/V caches for the heads a tiled pass processes
 * together
 */
AttentionWorkspaceSizes attention_workspace_sizes(dtype_t qkv_dtype,
                                                  const std::vector<int> &q_shape,
                                                  const std::vector<int> &k_shape,
                                                  AttentionLayout layout);

/**
 * @brief Slice the buffer attention_workspace_sizes() sized into tile, K and V.
 *
 * @param storage          Node workspace
 * @param storage_bytes    Bytes available at storage
 * @param qkv_dtype        Q/K/V data type
 * @param q_shape          Query shape, following layout
 * @param k_shape          Key shape, following layout
 * @param layout           Q/K/V layout
 *
 * @return a workspace whose tile_buf is nullptr when storage is missing or shorter than the planned size
 */
AttentionWorkspace attention_bind_workspace(void *storage,
                                            size_t storage_bytes,
                                            dtype_t qkv_dtype,
                                            const std::vector<int> &q_shape,
                                            const std::vector<int> &k_shape,
                                            AttentionLayout layout);

/**
 * @brief Build the broadcast strides for an optional mask.
 *
 * Does not allocate memory.
 *
 * @param mask    Optional attn_mask tensor
 * @param q       Query tensor
 * @param k       Key tensor
 * @param layout  Q/K/V layout
 * @param who     Node name used in error logs
 *
 * @return mask arguments with broadcast strides
 */
AttentionMaskArgs get_attention_mask_args(
    const TensorBase *mask, const TensorBase *q, const TensorBase *k, AttentionLayout layout, const char *who);

/**
 * @brief Prepare or reuse the constant-mask cache and attach it to mask_args.
 *
 * @param mask_cache  Module-owned cache
 * @param mask_args   Mask arguments updated in place
 * @param q           Query tensor
 * @param k           Key tensor
 * @param has_scale   whether an explicit scale is provided
 * @param scale       explicit softmax scale
 * @param layout      Q/K/V layout
 * @param who         Node name used in error logs
 */
template <typename feature_t>
void attention_prepare_constant_mask_cache(AttentionMaskCache &mask_cache,
                                           AttentionMaskArgs &mask_args,
                                           const TensorBase *q,
                                           const TensorBase *k,
                                           bool has_scale,
                                           float scale,
                                           AttentionLayout layout,
                                           const char *who);

/**
 * @brief Prepare the int8 softmax exp LUT for this node's logit scale.
 *
 * @param lut        Module-owned LUT
 * @param q          Query tensor
 * @param k          Key tensor
 * @param has_scale  whether an explicit scale is provided
 * @param scale      explicit softmax scale
 * @param who        Node name used in error logs
 *
 * @return false when a valid LUT cannot be prepared
 */
bool attention_prepare_int8_softmax_exp_lut(
    AttentionInt8ExpLut &lut, const TensorBase *q, const TensorBase *k, bool has_scale, float scale, const char *who);

/**
 * @brief exp(x) for x <= 0 as computed by the softmax.
 *
 * Exposed for unit tests.
 *
 * @param x  Non-positive input
 *
 * @return lookup: linear interpolation over the 1025-entry LUT
 */
float attention_exp_neg_lut_lookup(float x);

/**
 * @brief exp(x) for x <= 0 as computed by the softmax.
 *
 * Exposed for unit tests.
 *
 * @param x  Non-positive input
 *
 * @return lookup_precise: LUT plus cubic correction (int16 path)
 */
float attention_exp_neg_lut_lookup_precise(float x);

/**
 * @brief e[t] = exp(s[t] - m) via attention_exp_neg_lut_lookup_precise.
 *
 * @param e  Output exponentials
 * @param s  Input scores
 * @param n  Valid length
 * @param m  Row max
 * @param l  Previous row sum
 *
 * @return l + sum(e)
 */
float attention_softmax_exp_row_precise(float *e, const float *s, int n, float m, float l);

/**
 * @brief Get the attention operation args object
 *
 * @param output             Output tensor
 * @param q                  Query tensor
 * @param k                  Key tensor
 * @param v                  Value tensor
 * @param mask_args          Optional attn_mask arguments
 * @param workspace          From attention_bind_workspace()
 * @param execution_context  Module-owned LUT and invocation-owned barrier
 * @param has_scale          whether an explicit scale is provided
 * @param scale              explicit softmax scale
 * @param is_causal          whether to apply a causal mask
 * @param layout             Q/K/V layout
 * @param runtime_mode       Runtime mode
 *
 * @return std::vector<AttentionArgs>
 */
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
                                                        const runtime_mode_t runtime_mode = RUNTIME_MODE_AUTO);

/**
 * @brief Attention(Q, K, V[, attn_mask]).
 *
 * @tparam feature_t
 * @param args_ptr
 */
template <typename feature_t>
void attention(void *args_ptr);

} // namespace base
} // namespace dl
