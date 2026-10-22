#pragma once

#include "dl_base_attention.hpp"
#include "dl_module_base.hpp"

#include <memory>
#include <string>
#include <vector>

namespace dl {
namespace module {

/**
 * @brief Attention(Q, K, V[, attn_mask]).
 */
class Attention : public Module {
private:
    bool m_has_scale = false;                         /*!< whether an explicit scale is provided */
    float m_scale = 0.f;                              /*!< explicit softmax scale; unused when m_has_scale is false */
    bool m_is_causal = false;                         /*!< whether to apply a causal mask */
    AttentionLayout m_layout = AttentionLayout::BHND; /*!< Q/K/V layout */

    std::unique_ptr<base::AttentionMaskCache>
        m_mask_cache; /*!< precompute of a constant attn_mask, built on first forward */
    std::unique_ptr<base::AttentionInt8ExpLut>
        m_int8_exp_lut; /*!< per-module int8 exp LUT; lazy so fp32/int16 nodes do not reserve it */

    struct FallbackWorkspace {
        void *data = nullptr; /*!< fallback buffer */
        size_t bytes = 0;     /*!< allocated bytes */
    };

    FallbackWorkspace m_fallback_workspace; /*!< fallback when the planner did not bind a workspace */
    bool m_workspace_logged = false;        /*!< whether a workspace alloc failure has been logged */

    dtype_t qkv_dtype() const;
    void *get_workspace_or_fallback(ModelContext *context, size_t required_bytes, size_t &bytes);

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode);

public:
    /**
     * @brief Construct a new Attention object.
     *
     * @param name            name of module
     * @param has_scale       whether an explicit scale is provided
     * @param scale           explicit softmax scale; ignored when has_scale is false
     * @param is_causal       whether to apply a causal mask
     * @param layout          Q/K/V layout, BHND or BNHD
     * @param inplace         inplace operation mode
     * @param quant_type      quantization type
     */
    Attention(const char *name = NULL,
              bool has_scale = false,
              float scale = 0.f,
              bool is_causal = false,
              AttentionLayout layout = AttentionLayout::BHND,
              module_inplace_t inplace = MODULE_NON_INPLACE,
              quant_type_t quant_type = QUANT_TYPE_NONE);

    /**
     * @brief Destroy the Attention object.
     */
    ~Attention() override;

    Attention(const Attention &) = delete;
    Attention &operator=(const Attention &) = delete;
    Attention(Attention &&) = delete;
    Attention &operator=(Attention &&) = delete;

    /**
     * @brief Calculate the output shape
     *
     * @param input_shapes The shape of inputs
     *
     * @return output shape
     */
    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) override;

    /**
     * @brief Scratch memory needed while this module's forward runs
     *
     * @param input_shapes   Input shapes
     *
     * @return One request per workspace buffer; empty when none is needed
     */
    std::vector<WorkspaceRequest> get_workspace_requests(std::vector<std::vector<int>> &input_shapes) override;

    /**
     * @brief Build the module, high-level interface for Module layer
     *
     * @param context   Model context including all inputs and outputs and other runtime information
     * @param mode      Runtime mode, default is RUNTIME_MODE_AUTO
     */
    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO) override;

    /**
     * @brief Run the module, Low-level interface for base layer and multi-core processing
     *
     * @param args      AttentionArgs
     */
    void forward_args(void *args) override;

    /**
     * @brief print module information
     */
    void print() override;

    /**
     * @brief deserialize Attention module instance by node serialization information
     *
     * @param fbs_model  Flatbuffer's model
     * @param node_name  The node name in model's graph
     *
     * @return The pointer of module instance
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name);
};

} // namespace module
} // namespace dl
