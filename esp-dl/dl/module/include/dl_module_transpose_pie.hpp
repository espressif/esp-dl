#pragma once

// Uncomment the line below to enable verbose debug prints for every forward call
// #define TRANSPOSE_PIE_DEBUG

#include "dl_module_base.hpp"
#include <vector>
#include <cstring>

// Assembly kernels (linked from .S files)
extern "C" {
void dl_esp32p4_s8_transpose(int8_t* out, int8_t* in, int32_t N, int32_t M);
void dl_esp32p4_block_transpose(int8_t* out, int8_t* in, int32_t N, int32_t M, int32_t K);
}

namespace dl {
namespace module {

/**
 * TransposePIE — SIMD-accelerated transpose for ESP32-P4
 *
 * Supports any permutation on INT8 data with automatic dispatch:
 *   - Kernel 1 (byte zip):   14.6x faster, for swaps involving innermost dim
 *   - Kernel 2 (block copy): 30x faster,   for swaps with fixed innermost dim (K=16)
 *   - Scalar fallback:       exact esp-dl logic, for unsupported cases
 *
 * The 6-step dispatch algorithm runs once per forward() call (~20 integer ops).
 * All complexity is in the analysis; the kernels are pure PIE SIMD.
 *
 * Validated 48/48 on ESP32-P4 silicon including all 16 YOLO26n attention transposes.
 */
class TransposePIE : public Module {
private:
    std::vector<int> m_perm;

    // Cached dispatch result (computed on first forward)
    bool m_dispatch_cached = false;
    int m_kernel_id = 0;   // 0=scalar, 1=byte_zip, 2=block_copy
    int m_batch = 1;
    int m_N = 0;
    int m_M = 0;
    int m_K = 1;

    /**
     * 6-step dispatch algorithm: try_simd_transpose
     *
     * Step 1: Normalize perm (handle negatives)
     * Step 2: Peel leading batch dims (perm[i] == i)
     * Step 3: Peel trailing fixed dims (perm[i] == i) → accumulate K
     * Step 4: Merge consecutive ascending groups in active region
     * Step 5: If exactly 2 groups → two-block swap → select kernel
     * Step 6: Otherwise → scalar fallback
     *
     * Returns kernel_id (0, 1, or 2). Sets m_batch, m_N, m_M, m_K.
     */
    int try_simd_transpose(const std::vector<int>& shape, std::vector<int>& perm)
    {
        int ndims = perm.size();
        if (ndims < 2 || ndims > 8) return 0;

        // Step 1: normalize negative indices
        for (int i = 0; i < ndims; i++)
            if (perm[i] < 0) perm[i] += ndims;

        // Step 2: peel leading batch dims (identity prefix)
        m_batch = 1;
        int lo = 0;
        while (lo < ndims && perm[lo] == lo) {
            m_batch *= shape[lo];
            lo++;
        }

        // Step 3: peel trailing fixed dims (identity suffix)
        m_K = 1;
        int hi = ndims - 1;
        while (hi >= lo && perm[hi] == hi) {
            m_K *= shape[hi];
            hi--;
        }

        // Identity permutation → nothing to do
        if (lo > hi) return 0;

        // Step 4: group consecutive ascending values in active region [lo..hi]
        int act_len = hi - lo + 1;
        int g_first[8], g_len[8];
        int ng = 1;
        g_first[0] = perm[lo];
        g_len[0] = 1;
        for (int k = 1; k < act_len; k++) {
            if (perm[lo + k] == perm[lo + k - 1] + 1) {
                g_len[ng - 1]++;
            } else {
                g_first[ng] = perm[lo + k];
                g_len[ng] = 1;
                ng++;
            }
        }

        // Step 5: must be exactly 2 groups (a two-block swap)
        if (ng != 2) return 0;

        // Compute merged sizes for each group
        int s1 = 1, s2 = 1;
        for (int k = 0; k < g_len[0]; k++)
            s1 *= shape[g_first[0] + k];
        for (int k = 0; k < g_len[1]; k++)
            s2 *= shape[g_first[1] + k];

        // Determine N (input-first block) and M (input-second block)
        if (g_first[0] < g_first[1]) {
            m_N = s1; m_M = s2;
        } else {
            m_N = s2; m_M = s1;
        }

        // Step 6: select kernel based on alignment
        if (m_K == 1) {
            // Kernel 1: byte zip — needs N%8==0 and M%16==0
            if ((m_N % 8 == 0) && (m_M % 16 == 0)) return 1;
            return 0;
        } else if (m_K >= 16 && (m_K % 16 == 0)) {
            // Kernel 2: block copy — any M, K must be 16-aligned
            return 2;
        }
        return 0;
    }

public:
    TransposePIE(const char *name = NULL,
                 module_inplace_t inplace = MODULE_NON_INPLACE,
                 quant_type_t quant_type = QUANT_TYPE_NONE,
                 std::vector<int> perm = {}) :
        Module(name, inplace, quant_type), m_perm(perm)
    {
    }

    ~TransposePIE() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        assert(input_shapes[0].size() == m_perm.size() || m_perm.size() == 0);

        std::vector<int> output_shape;
        std::vector<int> perm_copy = m_perm;
        int dims = perm_copy.size();

        for (int i = 0; i < dims; i++) {
            if (perm_copy[i] < 0)
                perm_copy[i] = dims + perm_copy[i];
            output_shape.push_back(input_shapes[0][perm_copy[i]]);
        }

        return std::vector<std::vector<int>>(1, output_shape);
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        int8_t *in_ptr = (int8_t *)input->get_element_ptr();
        int8_t *out_ptr = (int8_t *)output->get_element_ptr();

        // Cache dispatch on first call (shapes are fixed between inferences)
        if (!m_dispatch_cached) {
            std::vector<int> perm_copy = m_perm;
            m_kernel_id = try_simd_transpose(input->shape, perm_copy);
            m_perm = perm_copy; // store normalized perm
            m_dispatch_cached = true;

            const char *kname = (m_kernel_id == 1) ? "K1_zip" :
                                (m_kernel_id == 2) ? "K2_blk" : "scalar";
#ifdef TRANSPOSE_PIE_DEBUG
            ESP_LOGI("TransposePIE",
                     "kernel=%s, batch=%d, N=%d, M=%d, K=%d",
                     kname, m_batch, m_N, m_M, m_K);
#endif
        }

#ifdef TRANSPOSE_PIE_DEBUG
        // --- Debug: print full context for every forward ---
        {
            int ndims = (int)input->shape.size();
            ESP_LOGW("TransposePIE_DBG", "--- forward ---");

            // Print input shape
            char buf[128];
            int pos = 0;
            pos += snprintf(buf + pos, sizeof(buf) - pos, "in_shape=[");
            for (int i = 0; i < ndims; i++)
                pos += snprintf(buf + pos, sizeof(buf) - pos, "%d%s",
                                input->shape[i], i < ndims - 1 ? "," : "");
            pos += snprintf(buf + pos, sizeof(buf) - pos, "]");
            ESP_LOGW("TransposePIE_DBG", "  %s", buf);

            // Print perm
            pos = 0;
            pos += snprintf(buf + pos, sizeof(buf) - pos, "perm=[");
            for (int i = 0; i < (int)m_perm.size(); i++)
                pos += snprintf(buf + pos, sizeof(buf) - pos, "%d%s",
                                m_perm[i], i < (int)m_perm.size() - 1 ? "," : "");
            pos += snprintf(buf + pos, sizeof(buf) - pos, "]");
            ESP_LOGW("TransposePIE_DBG", "  %s", buf);

            // Print dispatch
            ESP_LOGW("TransposePIE_DBG",
                     "  kernel=%d batch=%d N=%d M=%d K=%d",
                     m_kernel_id, m_batch, m_N, m_M, m_K);

            // Total elements and pointers
            int total = 1;
            for (int i = 0; i < ndims; i++) total *= input->shape[i];
            ESP_LOGW("TransposePIE_DBG",
                     "  total_bytes=%d in_ptr=%p out_ptr=%p",
                     total, in_ptr, out_ptr);

            // Hex dump first 32 bytes of input
            int dump_len = total < 32 ? total : 32;
            pos = 0;
            for (int i = 0; i < dump_len; i++)
                pos += snprintf(buf + pos, sizeof(buf) - pos, "%02x ",
                                (uint8_t)in_ptr[i]);
            ESP_LOGW("TransposePIE_DBG", "  in[0..%d]: %s", dump_len - 1, buf);
        }
#endif // TRANSPOSE_PIE_DEBUG

        if (m_kernel_id == 1) {
            // Kernel 1: byte zip transpose (K==1)
            int chunk = m_N * m_M;
            for (int b = 0; b < m_batch; b++)
                dl_esp32p4_s8_transpose(out_ptr + b * chunk,
                                        in_ptr + b * chunk, m_N, m_M);
        } else if (m_kernel_id == 2) {
            // Kernel 2: block copy transpose (K>=16, K%16==0)
            int chunk = m_N * m_M * m_K;
            for (int b = 0; b < m_batch; b++)
                dl_esp32p4_block_transpose(out_ptr + b * chunk,
                                           in_ptr + b * chunk, m_N, m_M, m_K);
        } else {
            // Scalar fallback: use existing TensorBase::transpose
            output->transpose(input, m_perm);
            return; // transpose() handles shape/stride internally
        }

        // Update output shape and strides for SIMD paths
        int ndims = m_perm.size();
        for (int i = 0; i < ndims; i++)
            output->shape[i] = input->shape[m_perm[i]];
        output->axis_offset[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            output->axis_offset[i] = output->axis_offset[i + 1] * output->shape[i + 1];

#ifdef TRANSPOSE_PIE_DEBUG
        // --- Debug: print output after kernel ---
        {
            int total = 1;
            for (int i = 0; i < ndims; i++) total *= output->shape[i];
            int dump_len = total < 32 ? total : 32;
            char buf[128];
            int pos = 0;
            for (int i = 0; i < dump_len; i++)
                pos += snprintf(buf + pos, sizeof(buf) - pos, "%02x ",
                                (uint8_t)out_ptr[i]);
            ESP_LOGW("TransposePIE_DBG", "  out[0..%d]: %s", dump_len - 1, buf);
            ESP_LOGW("TransposePIE_DBG", "--- done ---");
        }
#endif // TRANSPOSE_PIE_DEBUG
    }

    void forward_args(void *args) {}

    /**
     * @brief Deserialize TransposePIE module from flatbuffers model
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        std::vector<int> perm;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "perm", perm);

        op = new TransposePIE(node_name.c_str(), MODULE_NON_INPLACE, quant_type, perm);
        return op;
    }

    void print()
    {
        const char *kname = (m_kernel_id == 1) ? "K1_zip" :
                            (m_kernel_id == 2) ? "K2_blk" : "scalar";
        ESP_LOGI("TransposePIE",
                 "quant_type: %s, perm: %s, kernel: %s, batch: %d, N: %d, M: %d, K: %d",
                 quant_type_to_string(quant_type),
                 vector_to_string(m_perm).c_str(),
                 kname, m_batch, m_N, m_M, m_K);
    }
};

} // namespace module
} // namespace dl
