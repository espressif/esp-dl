#pragma once

/**
 * @brief Private Attention kernels for dl_base_attention.cpp and white-box tests.
 *
 * Internal linkage; see dl_base_attention_common.hpp.
 */

#include "dl_base_attention_common.hpp"
#include "dl_base_attention_softmax.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace dl {
namespace base {
namespace {

constexpr int ATTENTION_TBLK_S8_GROUP = 16;     /*!< int8 t-block: tokens per QACC pass */
constexpr int ATTENTION_TBLK_S16_GROUP = 8;     /*!< int16 t-block: tokens per QACC pass */
constexpr float MASK_DOTS_CLAMP = 268435456.0f; /*!< 2^28; |bias| bound in int8 dot units */

/**
 * @brief Single-query P·V, 8-wide acc-resident.
 */
inline void pv_row_8lane(float *acc, const float *exp_row, const float *v_values, int tile_n, int D)
{
    int d = 0;
    for (; d + 7 < D; d += 8) {
        float a0 = acc[d];
        float a1 = acc[d + 1];
        float a2 = acc[d + 2];
        float a3 = acc[d + 3];
        float a4 = acc[d + 4];
        float a5 = acc[d + 5];
        float a6 = acc[d + 6];
        float a7 = acc[d + 7];
        for (int t = 0; t < tile_n; ++t) {
            const float e = exp_row[t];
            const float *v = v_values + t * D + d;
            a0 += e * v[0];
            a1 += e * v[1];
            a2 += e * v[2];
            a3 += e * v[3];
            a4 += e * v[4];
            a5 += e * v[5];
            a6 += e * v[6];
            a7 += e * v[7];
        }
        acc[d] = a0;
        acc[d + 1] = a1;
        acc[d + 2] = a2;
        acc[d + 3] = a3;
        acc[d + 4] = a4;
        acc[d + 5] = a5;
        acc[d + 6] = a6;
        acc[d + 7] = a7;
    }
    for (; d < D; ++d) {
        float a = acc[d];
        for (int t = 0; t < tile_n; ++t) {
            a += exp_row[t] * v_values[t * D + d];
        }
        acc[d] = a;
    }
}

/**
 * @brief P·V, 2 queries per step (V row reuse); MQ=4 spills P4's 32 fp regs.
 */
inline void pv_accumulate_fp32(
    float *acc_base, const float *exp_scores_base, const float *v_values, int nq, int tile_n, int D)
{
    int iq = 0;
    for (; iq + 1 < nq; iq += 2) {
        float *acc0 = acc_base + iq * ATTENTION_MAX_HEAD_DIM;
        float *acc1 = acc_base + (iq + 1) * ATTENTION_MAX_HEAD_DIM;
        const float *e0 = exp_scores_base + iq * ATTENTION_KEY_TILE_SIZE;
        const float *e1 = exp_scores_base + (iq + 1) * ATTENTION_KEY_TILE_SIZE;

        int d = 0;
        for (; d + 7 < D; d += 8) {
            float a00 = acc0[d];
            float a01 = acc0[d + 1];
            float a02 = acc0[d + 2];
            float a03 = acc0[d + 3];
            float a04 = acc0[d + 4];
            float a05 = acc0[d + 5];
            float a06 = acc0[d + 6];
            float a07 = acc0[d + 7];
            float a10 = acc1[d];
            float a11 = acc1[d + 1];
            float a12 = acc1[d + 2];
            float a13 = acc1[d + 3];
            float a14 = acc1[d + 4];
            float a15 = acc1[d + 5];
            float a16 = acc1[d + 6];
            float a17 = acc1[d + 7];

            for (int t = 0; t < tile_n; ++t) {
                const float ee0 = e0[t];
                const float ee1 = e1[t];
                const float *v = v_values + t * D + d;
                const float v0 = v[0];
                const float v1 = v[1];
                const float v2 = v[2];
                const float v3 = v[3];
                const float v4 = v[4];
                const float v5 = v[5];
                const float v6 = v[6];
                const float v7 = v[7];
                a00 += ee0 * v0;
                a01 += ee0 * v1;
                a02 += ee0 * v2;
                a03 += ee0 * v3;
                a04 += ee0 * v4;
                a05 += ee0 * v5;
                a06 += ee0 * v6;
                a07 += ee0 * v7;
                a10 += ee1 * v0;
                a11 += ee1 * v1;
                a12 += ee1 * v2;
                a13 += ee1 * v3;
                a14 += ee1 * v4;
                a15 += ee1 * v5;
                a16 += ee1 * v6;
                a17 += ee1 * v7;
            }

            acc0[d] = a00;
            acc0[d + 1] = a01;
            acc0[d + 2] = a02;
            acc0[d + 3] = a03;
            acc0[d + 4] = a04;
            acc0[d + 5] = a05;
            acc0[d + 6] = a06;
            acc0[d + 7] = a07;
            acc1[d] = a10;
            acc1[d + 1] = a11;
            acc1[d + 2] = a12;
            acc1[d + 3] = a13;
            acc1[d + 4] = a14;
            acc1[d + 5] = a15;
            acc1[d + 6] = a16;
            acc1[d + 7] = a17;
        }
        for (; d < D; ++d) {
            float aa0 = acc0[d];
            float aa1 = acc1[d];
            for (int t = 0; t < tile_n; ++t) {
                const float vt = v_values[t * D + d];
                aa0 += e0[t] * vt;
                aa1 += e1[t] * vt;
            }
            acc0[d] = aa0;
            acc1[d] = aa1;
        }
    }
    for (; iq < nq; ++iq) {
        pv_row_8lane(acc_base + iq * ATTENTION_MAX_HEAD_DIM,
                     exp_scores_base + iq * ATTENTION_KEY_TILE_SIZE,
                     v_values,
                     tile_n,
                     D);
    }
}

#if CONFIG_PIE_V2_BOOST
/**
 * @brief Int8 P·V: QACC outer product over a d-block V tile of ATTENTION_KEY_TILE_SIZE=64 tokens.
 *
 * ``exp_scores_ready`` skips requantization when batched Q·K wrote the
 * quantized exponentials directly.
 */
inline void pv_accumulate_int8(float *acc_base,
                               float *exp_scores_base,
                               int8_t *quantized_exp_base,
                               float *exp_scale,
                               const int8_t *v_block,
                               int nq,
                               int tile_n,
                               int D_pad,
                               float v_scale,
                               bool exp_scores_ready,
                               bool defer_pv_scale,
                               int32_t *qacc_buffer)
{
    const int n_dot = ATTENTION_KEY_TILE_SIZE;

    if (!exp_scores_ready) {
        for (int iq = 0; iq < nq; ++iq) {
            float *exp_row = exp_scores_base + iq * ATTENTION_KEY_TILE_SIZE;
            int8_t *quantized_exp_row = quantized_exp_base + iq * ATTENTION_KEY_TILE_SIZE;

            float e_max = 0.f;
            for (int t = 0; t < tile_n; ++t) {
                const float ev = exp_row[t];
                if (ev > e_max)
                    e_max = ev;
            }

            if (e_max <= 0.f) {
                std::memset(quantized_exp_row, 0, static_cast<size_t>(n_dot));
                exp_scale[iq] = 0.f;
                continue;
            }

            const float qscale = 127.0f / e_max;
            for (int t = 0; t < tile_n; ++t) {
                int qv = static_cast<int>(exp_row[t] * qscale + 0.5f);
                if (qv > 127)
                    qv = 127;
                quantized_exp_row[t] = static_cast<int8_t>(qv);
            }
            for (int t = tile_n; t < n_dot; ++t) {
                quantized_exp_row[t] = 0;
            }
            exp_scale[iq] = (e_max / 127.0f) * v_scale;
        }
    }

    // The kernels walk ceil(D/16) d-groups of D_pad.
    if (defer_pv_scale) {
        dl_esp32p4_s8_attention_pv_qacc_i32(reinterpret_cast<int32_t *>(acc_base),
                                            quantized_exp_base,
                                            v_block,
                                            nq,
                                            D_pad,
                                            ATTENTION_MAX_HEAD_DIM,
                                            qacc_buffer);
    } else {
        dl_esp32p4_s8_attention_pv_qacc(
            acc_base, quantized_exp_base, v_block, exp_scale, nq, D_pad, ATTENTION_MAX_HEAD_DIM, qacc_buffer);
    }
}
#endif // CONFIG_PIE_V2_BOOST

/**
 * @brief src_stride_pos: BHND -> D (contiguous), BNHD -> H*D (memcpy-per-row).
 */
template <dtype_t qkv_dtype>
inline void copy_q_block_to_tile_buf(
    uint8_t *dst, const void *q_data, int q_head_base, int i0, int nq, int D, int D_pad, int src_stride_pos)
{
    const int elem = qkv_element_size<qkv_dtype>();
    const uint8_t *src = static_cast<const uint8_t *>(q_data) +
        static_cast<size_t>(q_head_base + i0 * src_stride_pos) * static_cast<size_t>(elem);
    if (D_pad == D && src_stride_pos == D) {
        std::memcpy(dst, src, static_cast<size_t>(nq * D) * static_cast<size_t>(elem));
        return;
    }
    const size_t row_bytes = static_cast<size_t>(D) * static_cast<size_t>(elem);
    const size_t padded_row_bytes = static_cast<size_t>(D_pad) * static_cast<size_t>(elem);
    const size_t src_step = static_cast<size_t>(src_stride_pos) * static_cast<size_t>(elem);
    for (int r = 0; r < nq; ++r) {
        if (r + 1 < nq) {
            __builtin_prefetch(src + static_cast<size_t>(r + 1) * src_step, 0, 1);
        }
        uint8_t *out = dst + static_cast<size_t>(r) * padded_row_bytes;
        std::memcpy(out, src + static_cast<size_t>(r) * src_step, row_bytes);
        if (D_pad > D) {
            std::memset(out + row_bytes, 0, padded_row_bytes - row_bytes);
        }
    }
}

/**
 * @brief Re-chunk K[t][d] -> t-block-major (dst[g*16*D_pad + d*16 + tl]) so QACC Q·K covers 16
 * tokens per pass; tail zero-padded.
 *
 * Only groups in [g_begin, g_end) are written, so tasks can build disjoint ranges.
 */
inline void build_k_head_tblk_s8(int8_t *dst,
                                 const int8_t *src,
                                 int Lk,
                                 int D,
                                 int D_pad,
                                 int src_stride_pos,
                                 int g_begin = 0,
                                 int g_end = std::numeric_limits<int>::max())
{
    constexpr int group_size = ATTENTION_TBLK_S8_GROUP;
    const int n_group = std::min(g_end, (Lk + group_size - 1) / group_size);
    // Fast 4x4 byte transpose when rows are D bytes and 4-B aligned.
    const bool fast = (D == D_pad) && ((D & 3) == 0);

    for (int g = g_begin; g < n_group; ++g) {
        const int t0 = g * group_size;
        const int nt = std::min(group_size, Lk - t0);
        int8_t *dg = dst + static_cast<size_t>(g) * group_size * static_cast<size_t>(D_pad);
        for (int tl = 0; tl < nt; ++tl) {
            __builtin_prefetch(src + static_cast<size_t>(t0 + tl) * static_cast<size_t>(src_stride_pos), 0, 1);
        }

        if (nt < group_size || !fast) {
            std::memset(dg, 0, static_cast<size_t>(group_size) * static_cast<size_t>(D_pad));
        }

        if (!fast) {
            for (int tl = 0; tl < nt; ++tl) {
                const int8_t *srow = src + static_cast<size_t>(t0 + tl) * static_cast<size_t>(src_stride_pos);
                for (int d = 0; d < D; ++d) {
                    dg[d * group_size + tl] = srow[d];
                }
            }
            continue;
        }

        for (int tl = 0; tl < nt; tl += 4) {
            const int nrow = std::min(4, nt - tl);
            for (int d = 0; d < D; d += 4) {
                uint32_t r[4] = {0, 0, 0, 0};
                for (int i = 0; i < nrow; ++i) {
                    std::memcpy(
                        &r[i], src + static_cast<size_t>(t0 + tl + i) * static_cast<size_t>(src_stride_pos) + d, 4);
                }
                for (int k = 0; k < 4; ++k) {
                    const unsigned sh = 8u * static_cast<unsigned>(k);
                    const uint32_t packed = ((r[0] >> sh) & 0xffu) | (((r[1] >> sh) & 0xffu) << 8) |
                        (((r[2] >> sh) & 0xffu) << 16) | (((r[3] >> sh) & 0xffu) << 24);
                    std::memcpy(dg + (d + k) * group_size + tl, &packed, 4);
                }
            }
        }
    }
}

/**
 * @brief Gather chunk_heads adjacent BNHD heads into per-head t-block caches in one pass. staging
 * holds 4 tokens of chunk_heads * D (<= 256) bytes; kept off the stack, which is too small on the
 * shared dual-core tasks.
 */
inline void build_k_heads_tblk_s8(int8_t *dst,
                                  size_t dst_head_bytes,
                                  const int8_t *src_h0,
                                  int chunk_heads,
                                  int Lk,
                                  int D,
                                  int D_pad,
                                  int src_stride_pos,
                                  uint8_t (*staging)[256],
                                  int g_begin = 0,
                                  int g_end = std::numeric_limits<int>::max())
{
    if (chunk_heads <= 1) {
        build_k_head_tblk_s8(dst, src_h0, Lk, D, D_pad, src_stride_pos, g_begin, g_end);
        return;
    }
    constexpr int group_size = ATTENTION_TBLK_S8_GROUP;
    const int n_group = std::min(g_end, (Lk + group_size - 1) / group_size);
    const bool fast = (D == D_pad) && ((D & 3) == 0);
    const int chunk_bytes = chunk_heads * D;

    for (int g = g_begin; g < n_group; ++g) {
        const int t0 = g * group_size;
        const int nt = std::min(group_size, Lk - t0);
        for (int tl = 0; tl < nt; ++tl) {
            __builtin_prefetch(src_h0 + static_cast<size_t>(t0 + tl) * static_cast<size_t>(src_stride_pos), 0, 1);
        }
        for (int h = 0; h < chunk_heads; ++h) {
            int8_t *dg = dst + static_cast<size_t>(h) * dst_head_bytes +
                static_cast<size_t>(g) * group_size * static_cast<size_t>(D_pad);
            if (nt < group_size || !fast) {
                std::memset(dg, 0, static_cast<size_t>(group_size) * static_cast<size_t>(D_pad));
            }
            if (!fast) {
                for (int tl = 0; tl < nt; ++tl) {
                    const int8_t *srow =
                        src_h0 + static_cast<size_t>(t0 + tl) * static_cast<size_t>(src_stride_pos) + h * D;
                    for (int d = 0; d < D; ++d) {
                        dg[d * group_size + tl] = srow[d];
                    }
                }
            }
        }
        if (!fast) {
            continue;
        }
        for (int tl = 0; tl < nt; tl += 4) {
            const int nrow = std::min(4, nt - tl);
            uint8_t (*tok)[256] = staging;
            for (int i = 0; i < nrow; ++i) {
                std::memcpy(tok[i],
                            src_h0 + static_cast<size_t>(t0 + tl + i) * static_cast<size_t>(src_stride_pos),
                            static_cast<size_t>(chunk_bytes));
            }
            for (int h = 0; h < chunk_heads; ++h) {
                int8_t *dg = dst + static_cast<size_t>(h) * dst_head_bytes +
                    static_cast<size_t>(g) * group_size * static_cast<size_t>(D_pad);
                for (int d = 0; d < D; d += 4) {
                    uint32_t r[4] = {0, 0, 0, 0};
                    for (int i = 0; i < nrow; ++i) {
                        std::memcpy(&r[i], tok[i] + h * D + d, 4);
                    }
                    for (int k = 0; k < 4; ++k) {
                        const unsigned sh = 8u * static_cast<unsigned>(k);
                        const uint32_t packed = ((r[0] >> sh) & 0xffu) | (((r[1] >> sh) & 0xffu) << 8) |
                            (((r[2] >> sh) & 0xffu) << 16) | (((r[3] >> sh) & 0xffu) << 24);
                        std::memcpy(dg + (d + k) * group_size + tl, &packed, 4);
                    }
                }
            }
        }
    }
}

/**
 * @brief int16 t-block builder (mirror of build_k_head_tblk_s8, group size 8).
 *
 * Layout: dst[g * 8 * D_pad + d * 8 + tl], one 8-token group == 16*D_pad
 * bytes so QACC MAC can walk 16 B per s16 broadcast lane. Tail zero-padded.
 */
inline void build_k_head_tblk_s16(int16_t *dst,
                                  const int16_t *src,
                                  int Lk,
                                  int D,
                                  int D_pad,
                                  int src_stride_pos,
                                  int g_begin = 0,
                                  int g_end = std::numeric_limits<int>::max())
{
    constexpr int group_size = ATTENTION_TBLK_S16_GROUP;
    const int n_group = std::min(g_end, (Lk + group_size - 1) / group_size);
    for (int g = g_begin; g < n_group; ++g) {
        const int t0 = g * group_size;
        const int nt = std::min(group_size, Lk - t0);
        int16_t *dg = dst + static_cast<size_t>(g) * static_cast<size_t>(group_size) * static_cast<size_t>(D_pad);
        for (int tl = 0; tl < nt; ++tl) {
            __builtin_prefetch(src + static_cast<size_t>(t0 + tl) * static_cast<size_t>(src_stride_pos), 0, 1);
        }
        if (nt == group_size && D == D_pad) {
            const int16_t *s[group_size];
            for (int tl = 0; tl < group_size; ++tl) {
                s[tl] = src + static_cast<size_t>(t0 + tl) * static_cast<size_t>(src_stride_pos);
            }
            for (int d = 0; d < D; ++d) {
                int16_t *o = dg + d * group_size;
                o[0] = s[0][d];
                o[1] = s[1][d];
                o[2] = s[2][d];
                o[3] = s[3][d];
                o[4] = s[4][d];
                o[5] = s[5][d];
                o[6] = s[6][d];
                o[7] = s[7][d];
            }
            continue;
        }
        std::memset(dg, 0, static_cast<size_t>(group_size) * static_cast<size_t>(D_pad) * sizeof(int16_t));
        for (int tl = 0; tl < nt; ++tl) {
            const int16_t *srow = src + static_cast<size_t>(t0 + tl) * static_cast<size_t>(src_stride_pos);
            for (int d = 0; d < D; ++d) {
                dg[d * group_size + tl] = srow[d];
            }
        }
    }
}

/**
 * @brief V tile -> dl_esp32p4_s16_attention_pv_qacc layout: dst[dc * n_tok * 8 + t * 8 + i] = V[j0
 * + t][dc * 8 + i]; tokens in [tile_n, n_tok) and dimensions in [D, D_pad) are zero.
 */
inline void load_v_tile_dblk_s16(int16_t *dst,
                                 const int16_t *v_data,
                                 int head_base,
                                 int j0,
                                 int tile_n,
                                 int n_tok,
                                 int D,
                                 int D_pad,
                                 int src_stride_pos)
{
    const int16_t *src0 = v_data + head_base + static_cast<size_t>(j0) * static_cast<size_t>(src_stride_pos);
    const int n_dc = D_pad / 8;
    const int n_full = D / 8;
    const size_t dc_step = static_cast<size_t>(n_tok) * 8;
    for (int t = 0; t < tile_n; ++t) {
        const int16_t *row = src0 + static_cast<size_t>(t) * static_cast<size_t>(src_stride_pos);
        int16_t *o = dst + static_cast<size_t>(t) * 8;
        // Constant-size copies compile to plain loads/stores, not a memcpy call.
        for (int dc = 0; dc < n_full; ++dc) {
            std::memcpy(o + dc * dc_step, row + dc * 8, 8 * sizeof(int16_t));
        }
        for (int dc = n_full; dc < n_dc; ++dc) {
            const int d0 = dc * 8;
            const int valid = std::max(0, std::min(8, D - d0));
            for (int i = 0; i < valid; ++i) {
                o[dc * dc_step + i] = row[d0 + i];
            }
            for (int i = valid; i < 8; ++i) {
                o[dc * dc_step + i] = 0;
            }
        }
    }
    if (tile_n < n_tok) {
        for (int dc = 0; dc < n_dc; ++dc) {
            std::memset(dst + dc * dc_step + static_cast<size_t>(tile_n) * 8,
                        0,
                        static_cast<size_t>(n_tok - tile_n) * 8 * sizeof(int16_t));
        }
    }
}

/**
 * @brief C reference for dl_esp32p4_s16_attention_pv_qacc (same integer total).
 */
inline void attention_pv_s16_qacc_c(
    float *acc, const int16_t *e, const int16_t *v_dblk, int nq, int n_tok, int D, int acc_stride, float scale)
{
    for (int iq = 0; iq < nq; ++iq) {
        const int16_t *e_hi = e + static_cast<size_t>(iq) * 2 * static_cast<size_t>(n_tok);
        const int16_t *e_lo = e_hi + n_tok;
        float *a = acc + static_cast<size_t>(iq) * static_cast<size_t>(acc_stride);
        for (int d = 0; d < D; ++d) {
            const int16_t *v = v_dblk + static_cast<size_t>(d / 8) * static_cast<size_t>(n_tok) * 8 + (d % 8);
            int64_t s_hi = 0;
            int64_t s_lo = 0;
            for (int t = 0; t < n_tok; ++t) {
                s_hi += static_cast<int64_t>(e_hi[t]) * v[t * 8];
                s_lo += static_cast<int64_t>(e_lo[t]) * v[t * 8];
            }
            a[d] += static_cast<float>(s_hi * (int64_t(1) << PV_S16_LO_BITS) + s_lo) * scale;
        }
    }
}

/**
 * @brief BNHD variant: one visit per token copies the contiguous chunk_heads * D span.
 *
 * staging holds 4 tokens of that span and must be 4 * 256 bytes (tile_buf.dots during K prep).
 * A null staging pointer, or a span wider than 256 bytes, falls back to one head at a time.
 * Only groups in [g_begin, g_end) are written, matching the dual-core K split.
 */
inline void build_k_heads_tblk_s16(int16_t *dst,
                                   size_t dst_head_elems,
                                   const int16_t *src_h0,
                                   int chunk_heads,
                                   int Lk,
                                   int D,
                                   int D_pad,
                                   int src_stride_pos,
                                   int g_begin = 0,
                                   int g_end = std::numeric_limits<int>::max(),
                                   uint8_t (*staging)[256] = nullptr)
{
    const size_t row_bytes = static_cast<size_t>(chunk_heads) * static_cast<size_t>(D) * sizeof(int16_t);
    if (chunk_heads <= 1 || staging == nullptr || row_bytes == 0 || row_bytes > 256) {
        if (chunk_heads <= 1) {
            build_k_head_tblk_s16(dst, src_h0, Lk, D, D_pad, src_stride_pos, g_begin, g_end);
            return;
        }
        for (int h = 0; h < chunk_heads; ++h) {
            build_k_head_tblk_s16(dst + static_cast<size_t>(h) * dst_head_elems,
                                  src_h0 + static_cast<size_t>(h) * static_cast<size_t>(D),
                                  Lk,
                                  D,
                                  D_pad,
                                  src_stride_pos,
                                  g_begin,
                                  g_end);
        }
        return;
    }

    constexpr int group_size = ATTENTION_TBLK_S16_GROUP;
    const int n_group = std::min(g_end, (Lk + group_size - 1) / group_size);
    for (int g = g_begin; g < n_group; ++g) {
        const int t0 = g * group_size;
        const int nt = std::min(group_size, Lk - t0);
        if (nt < group_size || D != D_pad) {
            for (int h = 0; h < chunk_heads; ++h) {
                int16_t *dg = dst + static_cast<size_t>(h) * dst_head_elems +
                    static_cast<size_t>(g) * static_cast<size_t>(group_size) * static_cast<size_t>(D_pad);
                std::memset(dg, 0, static_cast<size_t>(group_size) * static_cast<size_t>(D_pad) * sizeof(int16_t));
            }
        }
        for (int tl = 0; tl < nt;) {
            const int nrow = std::min(4, nt - tl);
            for (int i = 0; i < nrow; ++i) {
                const int16_t *srow = src_h0 + static_cast<size_t>(t0 + tl + i) * static_cast<size_t>(src_stride_pos);
                if (i + 1 < nrow) {
                    __builtin_prefetch(srow + src_stride_pos, 0, 1);
                }
                std::memcpy(staging[i], srow, row_bytes);
            }
            for (int h = 0; h < chunk_heads; ++h) {
                int16_t *dg = dst + static_cast<size_t>(h) * dst_head_elems +
                    static_cast<size_t>(g) * static_cast<size_t>(group_size) * static_cast<size_t>(D_pad);
                for (int d = 0; d < D; ++d) {
                    int16_t *o = dg + d * group_size + tl;
                    for (int i = 0; i < nrow; ++i) {
                        const int16_t *tok = reinterpret_cast<const int16_t *>(staging[i]);
                        o[i] = tok[h * D + d];
                    }
                }
            }
            tl += nrow;
        }
    }
}

/**
 * @brief C reference for the QACC batched QK, exercised bit-exact by PIE tests before the hardware
 * kernel is trusted.
 *
 * Same 8-token group semantics.
 */
inline void attention_qk_s16_qacc_c(float *scores,
                                    const int16_t *q,
                                    const int16_t *k_blk,
                                    int nq,
                                    int tile_n,
                                    int score_stride,
                                    int D_real,
                                    float dot_scale)
{
    constexpr int group_size = ATTENTION_TBLK_S16_GROUP;
    const int D_pad = (D_real + group_size - 1) & -group_size;
    const int shift = int16_dot_shift(D_real);
    for (int iq = 0; iq < nq; ++iq) {
        const int16_t *qrow = q + static_cast<size_t>(iq) * static_cast<size_t>(D_pad);
        float *srow = scores + static_cast<size_t>(iq) * static_cast<size_t>(score_stride);
        for (int t = 0; t < tile_n; ++t) {
            const int tg = t / group_size;
            const int tl = t % group_size;
            int64_t acc = 0;
            for (int d = 0; d < D_pad; ++d) {
                const int16_t kv =
                    k_blk[static_cast<size_t>(tg) * static_cast<size_t>(group_size) * static_cast<size_t>(D_pad) +
                          static_cast<size_t>(d) * static_cast<size_t>(group_size) + static_cast<size_t>(tl)];
                acc += static_cast<int64_t>(qrow[d]) * static_cast<int64_t>(kv);
            }
            const int32_t dot = static_cast<int32_t>(shift_round_half_even(acc, shift));
            srow[t] = static_cast<float>(dot) * dot_scale;
        }
    }
}

/**
 * @brief Zero-pad int8 rows from D to D_pad (multiple of 16).
 */
inline void copy_rows_pad_s8(int8_t *dst, const int8_t *src, int rows, int D, int D_pad, int src_stride_pos)
{
    if (D_pad == D && src_stride_pos == D) {
        std::memcpy(dst, src, static_cast<size_t>(rows) * static_cast<size_t>(D));
        return;
    }
    for (int r = 0; r < rows; ++r) {
        const int8_t *row = src + static_cast<size_t>(r) * static_cast<size_t>(src_stride_pos);
        if (r + 1 < rows) {
            __builtin_prefetch(src + static_cast<size_t>(r + 1) * static_cast<size_t>(src_stride_pos), 0, 1);
        }
        std::memcpy(dst + r * D_pad, row, static_cast<size_t>(D));
        if (D_pad > D) {
            std::memset(dst + r * D_pad + D, 0, static_cast<size_t>(D_pad - D));
        }
    }
}

/**
 * @brief BNHD: chunk_heads adjacent heads of a Q-block into
 * dst[h][ATTENTION_QUERY_TILE_SIZE][D_pad].
 */
inline void copy_q_heads_pad_s8(
    int8_t *dst, int q_head_stride, const int8_t *src_h0, int chunk_heads, int nq, int D, int D_pad, int src_stride_pos)
{
    if (chunk_heads <= 1) {
        copy_rows_pad_s8(dst, src_h0, nq, D, D_pad, src_stride_pos);
        return;
    }
    for (int r = 0; r < nq; ++r) {
        const int8_t *row = src_h0 + static_cast<size_t>(r) * static_cast<size_t>(src_stride_pos);
        if (r + 1 < nq) {
            __builtin_prefetch(src_h0 + static_cast<size_t>(r + 1) * static_cast<size_t>(src_stride_pos), 0, 1);
        }
        for (int h = 0; h < chunk_heads; ++h) {
            int8_t *out = dst + h * q_head_stride + r * D_pad;
            std::memcpy(out, row + h * D, static_cast<size_t>(D));
            if (D_pad > D) {
                std::memset(out + D, 0, static_cast<size_t>(D_pad - D));
            }
        }
    }
}

/**
 * @brief BNHD: chunk_heads adjacent heads of a Q-block into
 * dst[h][ATTENTION_QUERY_TILE_SIZE][D_pad].
 */
inline void copy_q_heads_pad_s16(int16_t *dst,
                                 int q_head_stride,
                                 const int16_t *src_h0,
                                 int chunk_heads,
                                 int nq,
                                 int D,
                                 int D_pad,
                                 int src_stride_pos)
{
    if (chunk_heads <= 1) {
        if (D_pad == D && src_stride_pos == D) {
            std::memcpy(dst, src_h0, static_cast<size_t>(nq) * static_cast<size_t>(D) * sizeof(int16_t));
            return;
        }
        for (int r = 0; r < nq; ++r) {
            const int16_t *row = src_h0 + static_cast<size_t>(r) * static_cast<size_t>(src_stride_pos);
            int16_t *out = dst + r * D_pad;
            std::memcpy(out, row, static_cast<size_t>(D) * sizeof(int16_t));
            if (D_pad > D) {
                std::memset(out + D, 0, static_cast<size_t>(D_pad - D) * sizeof(int16_t));
            }
        }
        return;
    }
    for (int r = 0; r < nq; ++r) {
        const int16_t *row = src_h0 + static_cast<size_t>(r) * static_cast<size_t>(src_stride_pos);
        if (r + 1 < nq) {
            __builtin_prefetch(src_h0 + static_cast<size_t>(r + 1) * static_cast<size_t>(src_stride_pos), 0, 1);
        }
        for (int h = 0; h < chunk_heads; ++h) {
            int16_t *out = dst + h * q_head_stride + r * D_pad;
            std::memcpy(out, row + h * D, static_cast<size_t>(D) * sizeof(int16_t));
            if (D_pad > D) {
                std::memset(out + D, 0, static_cast<size_t>(D_pad - D) * sizeof(int16_t));
            }
        }
    }
}

/**
 * @brief Q·K^T scalar fallback (K from cache or source).
 */
template <dtype_t qkv_dtype>
inline float qk_scalar_dot_row(const float *q_row, const void *k_data, int k_row_base, int D, float k_scale)
{
    float dot = 0.f;
    if constexpr (qkv_dtype == DATA_TYPE_FLOAT) {
        (void)k_scale;
        const float *krow = static_cast<const float *>(k_data) + k_row_base;
        for (int d = 0; d < D; ++d) {
            dot += q_row[d] * krow[d];
        }
    } else if constexpr (qkv_dtype == DATA_TYPE_INT8) {
        const int8_t *krow = static_cast<const int8_t *>(k_data) + k_row_base;
        for (int d = 0; d < D; ++d) {
            dot += q_row[d] * static_cast<float>(krow[d]);
        }
        dot *= k_scale;
    } else { // DATA_TYPE_INT16
        const int16_t *krow = static_cast<const int16_t *>(k_data) + k_row_base;
        for (int d = 0; d < D; ++d) {
            dot += q_row[d] * static_cast<float>(krow[d]);
        }
        dot *= k_scale;
    }
    return dot;
}

/**
 * @brief V[t, d] -> d-block-major dst[dg][t][dl] for QACC P·V; tail zero-padded.
 */
inline void load_v_tile_dblk_s8(
    int8_t *dst, const int8_t *v_data, int head_base, int j0, int tile_n, int D, int src_stride_pos)
{
    const int8_t *src = v_data + head_base + j0 * src_stride_pos;
    const int n_dgroup = (D + 15) >> 4;
    for (int dg = 0; dg < n_dgroup; ++dg) {
        int8_t *out = dst + dg * (ATTENTION_KEY_TILE_SIZE * 16);
        const int8_t *in = src + dg * 16;
        const int d_valid = std::min(16, D - dg * 16);
        if (d_valid == 16) {
            for (int t = 0; t < tile_n; ++t) {
                if (t + 1 < tile_n) {
                    __builtin_prefetch(in + static_cast<size_t>(t + 1) * static_cast<size_t>(src_stride_pos), 0, 1);
                }
                const uint32_t *s = reinterpret_cast<const uint32_t *>(
                    in + static_cast<size_t>(t) * static_cast<size_t>(src_stride_pos));
                uint32_t *d = reinterpret_cast<uint32_t *>(out + t * 16);
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
                d[3] = s[3];
            }
        } else {
            for (int t = 0; t < tile_n; ++t) {
                if (t + 1 < tile_n) {
                    __builtin_prefetch(in + static_cast<size_t>(t + 1) * static_cast<size_t>(src_stride_pos), 0, 1);
                }
                int8_t *d = out + t * 16;
                std::memcpy(
                    d, in + static_cast<size_t>(t) * static_cast<size_t>(src_stride_pos), static_cast<size_t>(d_valid));
                std::memset(d + d_valid, 0, static_cast<size_t>(16 - d_valid));
            }
        }
        if (tile_n < ATTENTION_KEY_TILE_SIZE) {
            std::memset(out + tile_n * 16, 0, static_cast<size_t>(ATTENTION_KEY_TILE_SIZE - tile_n) * 16);
        }
    }
}

/**
 * @brief BNHD: token-major pass of chunk_heads adjacent heads into d-block V tiles.
 */
inline void load_v_tile_dblk_heads_s8(int8_t *dst_head0,
                                      int n_tiles,
                                      int v_tile_stride,
                                      int chunk_heads,
                                      const int8_t *v_data,
                                      int head0_base,
                                      int j0,
                                      int tile_n,
                                      int D,
                                      int src_stride_pos)
{
    if (chunk_heads <= 1) {
        load_v_tile_dblk_s8(dst_head0 + (j0 / ATTENTION_KEY_TILE_SIZE) * v_tile_stride,
                            v_data,
                            head0_base,
                            j0,
                            tile_n,
                            D,
                            src_stride_pos);
        return;
    }
    const int ti = j0 / ATTENTION_KEY_TILE_SIZE;
    const int n_dgroup = (D + 15) >> 4;
    const int8_t *src = v_data + head0_base + j0 * src_stride_pos;
    for (int h = 0; h < chunk_heads; ++h) {
        int8_t *dst = dst_head0 + (h * n_tiles + ti) * v_tile_stride;
        for (int dg = 0; dg < n_dgroup; ++dg) {
            if (tile_n < ATTENTION_KEY_TILE_SIZE) {
                std::memset(dst + dg * (ATTENTION_KEY_TILE_SIZE * 16) + tile_n * 16,
                            0,
                            static_cast<size_t>(ATTENTION_KEY_TILE_SIZE - tile_n) * 16);
            }
        }
    }
    for (int t = 0; t < tile_n; ++t) {
        if (t + 1 < tile_n) {
            __builtin_prefetch(src + static_cast<size_t>(t + 1) * static_cast<size_t>(src_stride_pos), 0, 1);
        }
        const int8_t *tok = src + static_cast<size_t>(t) * static_cast<size_t>(src_stride_pos);
        for (int h = 0; h < chunk_heads; ++h) {
            int8_t *dst = dst_head0 + (h * n_tiles + ti) * v_tile_stride;
            const int8_t *row = tok + h * D;
            for (int dg = 0; dg < n_dgroup; ++dg) {
                const int d_valid = std::min(16, D - dg * 16);
                int8_t *out = dst + dg * (ATTENTION_KEY_TILE_SIZE * 16) + t * 16;
                if (d_valid == 16) {
                    std::memcpy(out, row + dg * 16, 16);
                } else {
                    std::memcpy(out, row + dg * 16, static_cast<size_t>(d_valid));
                    std::memset(out + d_valid, 0, static_cast<size_t>(16 - d_valid));
                }
            }
        }
    }
}

/**
 * @brief BNHD: token-major pass of chunk_heads adjacent heads into int16 d-block V tiles.
 * v_tile_stride is in bytes; destination index is (h * n_tiles + j0/TILE) like the int8 helper.
 */
inline void load_v_tile_dblk_heads_s16(int8_t *dst_head0,
                                       int n_tiles,
                                       int v_tile_stride,
                                       int chunk_heads,
                                       const int16_t *v_data,
                                       int head0_base,
                                       int j0,
                                       int tile_n,
                                       int n_tok,
                                       int D,
                                       int D_pad,
                                       int src_stride_pos)
{
    const int ti = j0 / ATTENTION_KEY_TILE_SIZE;
    if (chunk_heads <= 1) {
        load_v_tile_dblk_s16(reinterpret_cast<int16_t *>(dst_head0 + static_cast<size_t>(ti) * v_tile_stride),
                             v_data,
                             head0_base,
                             j0,
                             tile_n,
                             n_tok,
                             D,
                             D_pad,
                             src_stride_pos);
        return;
    }
    const int n_dc = D_pad / 8;
    const int n_full = D / 8;
    const size_t dc_step = static_cast<size_t>(n_tok) * 8;
    const int16_t *src0 = v_data + head0_base + static_cast<size_t>(j0) * static_cast<size_t>(src_stride_pos);
    for (int h = 0; h < chunk_heads; ++h) {
        if (tile_n >= n_tok) {
            continue;
        }
        int16_t *dst = reinterpret_cast<int16_t *>(dst_head0 + static_cast<size_t>(h * n_tiles + ti) * v_tile_stride);
        for (int dc = 0; dc < n_dc; ++dc) {
            std::memset(dst + dc * dc_step + static_cast<size_t>(tile_n) * 8,
                        0,
                        static_cast<size_t>(n_tok - tile_n) * 8 * sizeof(int16_t));
        }
    }
    for (int t = 0; t < tile_n; ++t) {
        const int16_t *tok = src0 + static_cast<size_t>(t) * static_cast<size_t>(src_stride_pos);
        if (t + 1 < tile_n) {
            __builtin_prefetch(tok + src_stride_pos, 0, 1);
        }
        for (int h = 0; h < chunk_heads; ++h) {
            const int16_t *row = tok + h * D;
            int16_t *o =
                reinterpret_cast<int16_t *>(dst_head0 + static_cast<size_t>(h * n_tiles + ti) * v_tile_stride) +
                static_cast<size_t>(t) * 8;
            for (int dc = 0; dc < n_full; ++dc) {
                std::memcpy(o + dc * dc_step, row + dc * 8, 8 * sizeof(int16_t));
            }
            for (int dc = n_full; dc < n_dc; ++dc) {
                const int d0 = dc * 8;
                const int valid = std::max(0, std::min(8, D - d0));
                for (int i = 0; i < valid; ++i) {
                    o[dc * dc_step + i] = row[d0 + i];
                }
                for (int i = valid; i < 8; ++i) {
                    o[dc * dc_step + i] = 0;
                }
            }
        }
    }
}

/**
 * @brief Source V -> fp32 tile-buffer tile with v_scale baked in; dst is dense [tile_n, D].
 */
template <dtype_t qkv_dtype>
inline void load_v_tile_dequant(
    float *dst, const void *v_data, int head_base, int j0, int tile_n, int D, float v_scale, int src_stride_pos)
{
    const int src_base = head_base + j0 * src_stride_pos;
    if constexpr (qkv_dtype == DATA_TYPE_FLOAT) {
        (void)v_scale;
        const float *src = static_cast<const float *>(v_data) + src_base;
        if (src_stride_pos == D) {
            std::memcpy(dst, src, static_cast<size_t>(tile_n * D) * sizeof(float));
        } else {
            for (int t = 0; t < tile_n; ++t) {
                std::memcpy(dst + static_cast<size_t>(t) * static_cast<size_t>(D),
                            src + static_cast<size_t>(t) * static_cast<size_t>(src_stride_pos),
                            static_cast<size_t>(D) * sizeof(float));
            }
        }
    } else if constexpr (qkv_dtype == DATA_TYPE_INT8) {
        const int8_t *src0 = static_cast<const int8_t *>(v_data) + src_base;
        for (int t = 0; t < tile_n; ++t) {
            const int8_t *src = src0 + static_cast<size_t>(t) * static_cast<size_t>(src_stride_pos);
            float *out = dst + static_cast<size_t>(t) * static_cast<size_t>(D);
            int i = 0;
            for (; i + 3 < D; i += 4) {
                out[i] = static_cast<float>(src[i]) * v_scale;
                out[i + 1] = static_cast<float>(src[i + 1]) * v_scale;
                out[i + 2] = static_cast<float>(src[i + 2]) * v_scale;
                out[i + 3] = static_cast<float>(src[i + 3]) * v_scale;
            }
            for (; i < D; ++i) {
                out[i] = static_cast<float>(src[i]) * v_scale;
            }
        }
    } else { // DATA_TYPE_INT16
        const int16_t *src0 = static_cast<const int16_t *>(v_data) + src_base;
        for (int t = 0; t < tile_n; ++t) {
            const int16_t *src = src0 + static_cast<size_t>(t) * static_cast<size_t>(src_stride_pos);
            float *out = dst + static_cast<size_t>(t) * static_cast<size_t>(D);
            for (int i = 0; i < D; ++i) {
                out[i] = static_cast<float>(src[i]) * v_scale;
            }
        }
    }
}

/**
 * @brief Integer QK dot -> softmax logit: softmax_scale * q_scale * k_scale.
 */
template <dtype_t qkv_dtype>
inline float attention_logit_scale(const TensorBase *q_t, const TensorBase *k_t, bool has_scale, float scale_attr)
{
    const float softmax_scale = has_scale ? scale_attr : (1.0f / std::sqrt(static_cast<float>(q_t->shape[3])));
    if constexpr (qkv_dtype == DATA_TYPE_FLOAT) {
        return softmax_scale;
    } else {
        return softmax_scale * dequant_scale_from_exponent(q_t->exponent) * dequant_scale_from_exponent(k_t->exponent);
    }
}

/**
 * @brief Apply the causal frontier to one QK tile and recompute its max.
 */
inline void attention_causal_mask_dots(int32_t *dots, int tile_n, int j0, int qi, int32_t *tile_max_out)
{
    int32_t tmax = INT32_MIN;
    for (int t = 0; t < tile_n; ++t) {
        // ONNX Attention without past K/V is upper-left aligned: query qi
        // can read exactly keys [0, qi]. Do not infer a decode offset from
        // Lk-Lq; this operator does not accept a past-K/V input.
        if (j0 + t > qi) {
            dots[t] = QK_DOTS_MASKED;
        } else if (dots[t] > tmax) {
            tmax = dots[t];
        }
    }
    *tile_max_out = tmax;
}

/**
 * @brief Convert one additive mask value into rounded int32 dot units.
 *
 * |bias| bound in dot units: int8 dots stay under 2^22 (D <= 256), so dots + bias
 * and m_new - dots cannot overflow int32. Anything this far below the row max
 * already lands on the zero end of the exp LUT.
 */
inline int32_t mask_dot_units(float m, float to_dots)
{
    const float x = std::fmin(std::fmax(m * to_dots, -MASK_DOTS_CLAMP), MASK_DOTS_CLAMP);
#if CONFIG_PIE_V2_BOOST
    int32_t r;
    __asm__("fcvt.w.s %0, %1, rne" : "=r"(r) : "f"(x));
    return r;
#else
    return tool::round_half_even(x);
#endif
}

/**
 * @brief Adds one row of attn_mask to int32 QK dots in dot units (mask / logit_scale, rounded
 * half-to-even) and recomputes the tile max. -inf entries become QK_DOTS_MASKED; slots already
 * masked (causal frontier) are left alone.
 *
 * Returns true iff this call dropped a slot.
 */
template <typename LoadMask>
inline bool attention_mask_dots_impl(int32_t *dots, int tile_n, LoadMask load, float to_dots, int32_t *tile_max_out)
{
    int32_t tmax = INT32_MIN;
    bool dropped = false;
    for (int t = 0; t < tile_n; ++t) {
        if (dots[t] == QK_DOTS_MASKED) {
            continue;
        }
        const float m = load(t);
        if (is_neg_inf(m)) {
            dots[t] = QK_DOTS_MASKED;
            dropped = true;
            continue;
        }
        const int32_t d = dots[t] + mask_dot_units(m, to_dots);
        dots[t] = d;
        if (d > tmax) {
            tmax = d;
        }
    }
    *tile_max_out = tmax;
    return dropped;
}

/**
 * @brief Apply a live attn_mask row to int32 QK dots.
 */
inline bool attention_mask_dots(
    int32_t *dots, int tile_n, const AttentionMaskArgs &mask_args, int mask_base, float to_dots, int32_t *tile_max_out)
{
    const int ks = mask_args.k_stride;
    if (mask_args.dtype == DATA_TYPE_FLOAT) {
        const float *mp = static_cast<const float *>(mask_args.data) + mask_base;
        if (ks == 1) {
            return attention_mask_dots_impl(dots, tile_n, [mp](int t) { return mp[t]; }, to_dots, tile_max_out);
        }
        return attention_mask_dots_impl(dots, tile_n, [mp, ks](int t) { return mp[t * ks]; }, to_dots, tile_max_out);
    }
    return attention_mask_dots_impl(
        dots,
        tile_n,
        [&mask_args, mask_base, ks](int t) { return load_mask_element(mask_args, mask_base + t * ks); },
        to_dots,
        tile_max_out);
}

/**
 * @brief Same result as attention_mask_dots, from an AttentionMaskCache's precomputed mask_dots.
 */
inline bool attention_mask_dots_cached(
    int32_t *dots, int tile_n, const int32_t *mask_dots, int ks, int32_t *tile_max_out)
{
    int32_t tmax = INT32_MIN;
    bool dropped = false;
    for (int t = 0; t < tile_n; ++t) {
        const int32_t d = dots[t];
        if (d == QK_DOTS_MASKED) {
            continue;
        }
        const int32_t b = mask_dots[t * ks];
        if (b == QK_DOTS_MASKED) {
            dots[t] = QK_DOTS_MASKED;
            dropped = true;
            continue;
        }
        const int32_t s = d + b;
        dots[t] = s;
        if (s > tmax) {
            tmax = s;
        }
    }
    *tile_max_out = tmax;
    return dropped;
}

/**
 * @brief Add one attn_mask row onto fp32 QK scores.
 */
inline void attention_mask_scores(float *scores, int tile_n, const AttentionMaskArgs &mask_args, int mask_base)
{
    for (int t = 0; t < tile_n; ++t) {
        scores[t] += load_mask_element(mask_args, mask_base + t * mask_args.k_stride);
    }
}

} // namespace
} // namespace base
} // namespace dl
