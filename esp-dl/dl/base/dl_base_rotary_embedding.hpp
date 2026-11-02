#pragma once

#include "dl_base.hpp"
#include "dl_tensor_base.hpp"

#include <climits>
#include <cstdint>
#include <vector>

namespace dl {
namespace base {

/**
 * @brief Logical dimensions and strides decoded from X and cos/sin cache shapes.
 */
struct RopeDims {
    int batch = 0;              /*!< batch size B */
    int head_count = 0;         /*!< number of attention heads H */
    int seq_len = 0;            /*!< sequence length L */
    int head_dim = 0;           /*!< head dimension D */
    int half_dim = 0;           /*!< D / 2, number of rotated pairs */
    int x_elements = 0;         /*!< total elements in X/Y */
    int cache_elements = 0;     /*!< total elements in cos/sin cache */
    int x_stride_head = 0;      /*!< X head stride in elements (L * D) */
    int x_stride_batch = 0;     /*!< X batch stride in elements (H * L * D) */
    int cache_stride_batch = 0; /*!< cache batch stride in elements (0 when broadcast) */
    int cache_stride_pos = 0;   /*!< cache position stride in elements (D / 2) */
    int token_count = 0;        /*!< total tokens (B * L) */
};

/**
 * @brief RotaryEmbedding operation arguments for one single- or dual-core task.
 */
struct RopeArgs {
    const void *x = nullptr;   /*!< input tensor X buffer */
    const void *cos = nullptr; /*!< cos_cache buffer */
    const void *sin = nullptr; /*!< sin_cache buffer */
    void *y = nullptr;         /*!< output tensor Y buffer */

    dtype_t x_dtype = DATA_TYPE_INT8;    /*!< X data type */
    dtype_t cos_dtype = DATA_TYPE_INT16; /*!< cos_cache data type */
    dtype_t sin_dtype = DATA_TYPE_INT16; /*!< sin_cache data type */
    dtype_t y_dtype = DATA_TYPE_INT8;    /*!< Y data type */

    float x_scale = 1.f;     /*!< X dequantization scale */
    float cos_scale = 1.f;   /*!< cos_cache dequantization scale */
    float sin_scale = 1.f;   /*!< sin_cache dequantization scale */
    float y_inv_scale = 1.f; /*!< Y requantization inverse scale */

    int head_dim = 0;   /*!< head dimension D */
    int half_dim = 0;   /*!< D / 2, number of rotated pairs */
    int head_count = 0; /*!< number of attention heads H */
    int seq_len = 0;    /*!< sequence length L */

    int x_stride_batch = 0; /*!< X batch stride in elements (H * L * D) */
    int x_stride_head = 0;  /*!< X head stride in elements (L * D) */
    int x_stride_pos = 0;   /*!< X position stride in elements (D) */

    int cache_stride_batch = 0; /*!< cache batch stride in elements (0 when broadcast) */
    int cache_stride_pos = 0;   /*!< cache position stride in elements (D / 2) */

    int token_start = 0;      /*!< start of flat token interval [token_start, token_end) */
    int token_end = 0;        /*!< end of flat token interval [token_start, token_end) */
    int token_block_size = 8; /*!< token block size per head traversal */

    int simd_mac_shift = -1;         /*!< QACC shift for PIE v2 SIMD (-1 means ineligible) */
    bool simd_same_k = false;        /*!< whether cos and sin share the same exponent */
    int integer_mac_shift = INT_MIN; /*!< integer rescale shift for typed scalar path (INT_MIN uses fp32 fallback) */
};

/**
 * @brief Check static RotaryEmbedding attributes at load time or runtime.
 */
bool rope_check_attributes(int interleaved, int rotary_embedding_dim, int num_heads, const char *who);

/**
 * @brief Check shape admission at load time, shape inference, or runtime.
 */
bool rope_check_shapes(const std::vector<int> &x_shape,
                       const std::vector<int> &cos_shape,
                       const std::vector<int> &sin_shape,
                       int interleaved,
                       int rotary_embedding_dim,
                       int num_heads,
                       RopeDims &dims,
                       const char *who);

/**
 * @brief Zero-fill the Y output tensor on any error path.
 */
void rope_zero_output(TensorBase *y);

/**
 * @brief Validate live tensors and build 1 or 2 task arguments; returns empty and zeroes Y on failure.
 */
std::vector<RopeArgs> get_rotary_embedding_operation_args(TensorBase *y,
                                                          TensorBase *x,
                                                          TensorBase *cos_cache,
                                                          TensorBase *sin_cache,
                                                          int interleaved,
                                                          int rotary_embedding_dim,
                                                          int num_heads,
                                                          quant_type_t quant_type,
                                                          runtime_mode_t mode,
                                                          const char *who);

/**
 * @brief Execute one RotaryEmbedding task (RopeArgs).
 *
 * @param args_ptr    Pointer to RopeArgs
 */
void rotary_embedding(void *args_ptr);

} // namespace base
} // namespace dl
