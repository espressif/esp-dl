#include "dl_module_attention.hpp"

#include "dl_model_context.hpp"
#include "esp_log.h"
#include "fbs_model.hpp"

#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace dl {
namespace module {

Attention::Attention(const char *name,
                     bool has_scale,
                     float scale,
                     bool is_causal,
                     AttentionLayout layout,
                     module_inplace_t inplace,
                     quant_type_t quant_type) :
    Module(name, inplace, quant_type), m_has_scale(has_scale), m_scale(scale), m_is_causal(is_causal), m_layout(layout)
{
}

Attention::~Attention()
{
    heap_caps_free(m_fallback_workspace.data);
}

std::vector<std::vector<int>> Attention::get_output_shape(std::vector<std::vector<int>> &input_shapes)
{
    // Output shape == Q's shape. Mismatches are logged; runtime checks run later.
    if (input_shapes.size() < 3) {
        ESP_LOGE("Attention",
                 "%s: needs at least 3 inputs (Q,K,V), got %zu",
                 name ? name : "<unnamed>",
                 input_shapes.size());
        return {{}};
    }
    const auto &q = input_shapes[0];
    const auto &k = input_shapes[1];
    const auto &v = input_shapes[2];
    if (q.size() != 4 || k.size() != 4 || v.size() != 4) {
        ESP_LOGE("Attention",
                 "%s: Q/K/V must be 4-D (got %zu/%zu/%zu)",
                 name ? name : "<unnamed>",
                 q.size(),
                 k.size(),
                 v.size());
        return {q};
    }
    if (q[0] != k[0] || q[0] != v[0]) {
        ESP_LOGE("Attention", "%s: batch mismatch Q=%d K=%d V=%d", name ? name : "<unnamed>", q[0], k[0], v[0]);
    }
    if (k[1] != v[1] || k[2] != v[2]) {
        ESP_LOGE("Attention",
                 "%s: K and V must agree on (Hkv,Lk); got K=(%d,%d) V=(%d,%d)",
                 name ? name : "<unnamed>",
                 k[1],
                 k[2],
                 v[1],
                 v[2]);
    }
    if (q[3] != k[3] || q[3] != v[3]) {
        ESP_LOGE("Attention", "%s: head_dim mismatch Q=%d K=%d V=%d", name ? name : "<unnamed>", q[3], k[3], v[3]);
    }
    if (q[3] <= 0 || q[3] > base::ATTENTION_MAX_HEAD_DIM) {
        ESP_LOGE("Attention",
                 "%s: head_dim=%d out of [1, %d]; forward will zero the output.",
                 name ? name : "<unnamed>",
                 q[3],
                 base::ATTENTION_MAX_HEAD_DIM);
    }
    return {{q[0], q[1], q[2], q[3]}};
}

std::vector<WorkspaceRequest> Attention::get_workspace_requests(std::vector<std::vector<int>> &input_shapes)
{
    if (input_shapes.size() < 2) {
        return {};
    }
    const base::AttentionWorkspaceSizes sizes =
        base::attention_workspace_sizes(qkv_dtype(), input_shapes[0], input_shapes[1], m_layout);
    if (sizes.bytes == 0) {
        return {};
    }
    // Internal RAM is left to the rest of the application. One buffer holds the tile buffers, then K, then V.
    return {{sizes.bytes, true}};
}

/**
 * @brief Zero the output tensor when a forward cannot run.
 */
static void attention_zero_output(dl::TensorBase *output)
{
    if (output != nullptr && output->data != nullptr) {
        std::memset(output->data, 0, output->get_bytes());
    }
}

void Attention::forward(ModelContext *context, runtime_mode_t mode)
{
    if (quant_type == QUANT_TYPE_SYMM_8BIT) {
        forward_template<int8_t>(context, mode);
    } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
        forward_template<int16_t>(context, mode);
    } else if (quant_type == QUANT_TYPE_FLOAT32) {
        forward_template<float>(context, mode);
    } else {
        ESP_LOGE("Attention",
                 "%s: quant_type=%s unsupported (need S8, S16 or F32)",
                 name ? name : "<unnamed>",
                 quant_type_to_string(quant_type));
        if (!m_outputs_index.empty()) {
            attention_zero_output(context->get_tensor(m_outputs_index[0]));
        }
    }
}

void Attention::forward_args(void *args)
{
    if (quant_type == QUANT_TYPE_SYMM_8BIT) {
        base::attention<int8_t>(args);
    } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
        base::attention<int16_t>(args);
    } else if (quant_type == QUANT_TYPE_FLOAT32) {
        base::attention<float>(args);
    } else if (args != nullptr) {
        attention_zero_output(static_cast<base::AttentionArgs *>(args)->o);
    }
}

void Attention::print()
{
    const char *qt = "float32";
    switch (quant_type) {
    case QUANT_TYPE_SYMM_8BIT:
        qt = "int8";
        break;
    case QUANT_TYPE_SYMM_16BIT:
        qt = "int16";
        break;
    case QUANT_TYPE_FLOAT32:
        qt = "float32";
        break;
    default:
        qt = "none";
        break;
    }
    const char *lo = (m_layout == AttentionLayout::BNHD) ? "BNHD" : "BHND";
    if (m_has_scale) {
        ESP_LOGI("Attention",
                 "quant_type=%s, scale=%.6f (explicit), is_causal=%d, layout=%s",
                 qt,
                 m_scale,
                 static_cast<int>(m_is_causal),
                 lo);
    } else {
        ESP_LOGI("Attention",
                 "quant_type=%s, scale=1/sqrt(D) (auto), is_causal=%d, layout=%s",
                 qt,
                 static_cast<int>(m_is_causal),
                 lo);
    }
}

/**
 * @brief Fails the load, like ORT session creation / TensorRT engine build, instead of leaving the
 * forward-time check to zero the output of an otherwise running model.
 *
 * UNDEFINED (no dtype recorded) is left to the forward-time check.
 */
static bool attention_check_mask_dtype(fbs::FbsModel *fbs_model, const std::string &node_name)
{
    std::vector<std::string> inputs, outputs;
    if (fbs_model->get_operation_inputs_and_outputs(node_name, inputs, outputs) != ESP_OK || inputs.size() < 4 ||
        inputs[3].empty()) {
        return true;
    }
    const std::string &mask_name = inputs[3];
    const dtype_t dtype = fbs_model->is_parameter(mask_name) ? fbs_model->get_tensor_dtype(mask_name)
                                                             : fbs_model->get_value_info_dtype(mask_name);
    if (dtype == DATA_TYPE_UNDEFINED || dtype == DATA_TYPE_FLOAT || dtype == DATA_TYPE_BOOL) {
        return true;
    }
    ESP_LOGE("Attention",
             "%s: attn_mask \"%s\" dtype=%s unsupported (need FLOAT or BOOL); re-export the mask as bool "
             "(true = attend) or fp32 additive",
             node_name.c_str(),
             mask_name.c_str(),
             dtype_to_string(dtype));
    return false;
}

/**
 * @brief Whether this quant_type can run through Attention::forward.
 */
static bool attention_quant_type_supported(quant_type_t quant_type)
{
    return quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT ||
        quant_type == QUANT_TYPE_FLOAT32;
}

/**
 * @brief Map a module quant_type onto the Q/K/V/O dtype_t.
 */
static dtype_t attention_dtype_from_quant_type(quant_type_t quant_type)
{
    switch (quant_type) {
    case QUANT_TYPE_SYMM_8BIT:
        return DATA_TYPE_INT8;
    case QUANT_TYPE_SYMM_16BIT:
        return DATA_TYPE_INT16;
    case QUANT_TYPE_FLOAT32:
        return DATA_TYPE_FLOAT;
    default:
        return DATA_TYPE_UNDEFINED;
    }
}

/**
 * @brief Recorded dtype of a named tensor, from a parameter or a value_info.
 */
static dtype_t attention_recorded_tensor_dtype(fbs::FbsModel *fbs_model, const std::string &name)
{
    if (name.empty()) {
        return DATA_TYPE_UNDEFINED;
    }
    return fbs_model->is_parameter(name) ? fbs_model->get_tensor_dtype(name) : fbs_model->get_value_info_dtype(name);
}

/**
 * @brief Fail the load when a recorded Q/K/V/O dtype disagrees with quant_type.
 */
static bool attention_check_recorded_qkv_dtype(fbs::FbsModel *fbs_model, const std::string &node_name, dtype_t expected)
{
    std::vector<std::string> inputs, outputs;
    if (fbs_model->get_operation_inputs_and_outputs(node_name, inputs, outputs) != ESP_OK || inputs.size() < 3) {
        return true;
    }
    const char *roles[4] = {"Q", "K", "V", "O"};
    const std::string *names[4] = {&inputs[0], &inputs[1], &inputs[2], outputs.empty() ? nullptr : &outputs[0]};
    for (int i = 0; i < 4; ++i) {
        if (names[i] == nullptr || names[i]->empty()) {
            continue;
        }
        const dtype_t dtype = attention_recorded_tensor_dtype(fbs_model, *names[i]);
        if (dtype != DATA_TYPE_UNDEFINED && dtype != expected) {
            ESP_LOGE("Attention",
                     "%s: %s \"%s\" dtype=%s does not match quant_type (expected %s)",
                     node_name.c_str(),
                     roles[i],
                     names[i]->c_str(),
                     dtype_to_string(dtype),
                     dtype_to_string(expected));
            return false;
        }
    }
    return true;
}

Module *Attention::deserialize(fbs::FbsModel *fbs_model, std::string node_name)
{
    quant_type_t quant_type = QUANT_TYPE_NONE;
    fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
    if (!attention_quant_type_supported(quant_type)) {
        ESP_LOGE("Attention",
                 "%s: quant_type=%s unsupported (need S8, S16 or F32); W8A16 Attention is exported as S16",
                 node_name.c_str(),
                 quant_type_to_string(quant_type));
        return nullptr;
    }

    // "scale" is optional; missing -> 1/sqrt(D) at runtime.
    float scale = 0.f;
    bool has_scale = (fbs_model->get_operation_attribute(node_name, "scale", scale) == ESP_OK);
    if (!base::attention_check_scale(has_scale, scale, node_name.c_str())) {
        return nullptr;
    }

    // "is_causal" optional (default 0).
    int is_causal_i = 0;
    fbs_model->get_operation_attribute(node_name, "is_causal", is_causal_i);
    const bool is_causal = (is_causal_i != 0);

    std::string layout;
    (void)fbs_model->get_operation_attribute(node_name, "layout", layout);
    for (char &c : layout) {
        if (c >= 'a' && c <= 'z')
            c = static_cast<char>(c - 32);
    }
    AttentionLayout parsed = AttentionLayout::BHND;
    if (layout.empty() || layout == "BHND") {
        parsed = AttentionLayout::BHND;
    } else if (layout == "BNHD") {
        parsed = AttentionLayout::BNHD;
    } else {
        ESP_LOGE(
            "Attention", "%s: layout=\"%s\" is not supported; use BHND or BNHD.", node_name.c_str(), layout.c_str());
        return nullptr;
    }

    if (!attention_check_mask_dtype(fbs_model, node_name)) {
        return nullptr;
    }
    if (!attention_check_recorded_qkv_dtype(fbs_model, node_name, attention_dtype_from_quant_type(quant_type))) {
        return nullptr;
    }

    // Reject unsupported shapes at load time when the .espdl carries shapes.
    std::vector<int> q_shape, k_shape, v_shape;
    if (fbs_model->get_operation_input_shape(node_name, 0, q_shape) == ESP_OK &&
        fbs_model->get_operation_input_shape(node_name, 1, k_shape) == ESP_OK &&
        fbs_model->get_operation_input_shape(node_name, 2, v_shape) == ESP_OK) {
        if (!base::attention_check_shapes(q_shape, k_shape, v_shape, nullptr, parsed, node_name.c_str())) {
            return nullptr;
        }
    }

    return new Attention(node_name.c_str(), has_scale, scale, is_causal, parsed, MODULE_NON_INPLACE, quant_type);
}

dtype_t Attention::qkv_dtype() const
{
    return (quant_type == QUANT_TYPE_SYMM_8BIT) ? DATA_TYPE_INT8
        : (quant_type == QUANT_TYPE_SYMM_16BIT) ? DATA_TYPE_INT16
                                                : DATA_TYPE_FLOAT;
}

void *Attention::get_workspace_or_fallback(ModelContext *context, size_t required_bytes, size_t &bytes)
{
    bytes = 0;
    if (required_bytes == 0) {
        return nullptr;
    }
    // run() builds its own context, in which the planner's workspace binding does not exist.
    if (!m_workspace_binding_ids.empty()) {
        size_t bound_bytes = 0;
        void *bound_workspace = context->get_bound_workspace(m_workspace_binding_ids[0], bound_bytes);
        if (bound_workspace != nullptr && bound_bytes >= required_bytes) {
            bytes = bound_bytes;
            return bound_workspace;
        }
    }
    FallbackWorkspace &fallback = m_fallback_workspace;
    if (fallback.bytes < required_bytes) {
        heap_caps_free(fallback.data);
#if CONFIG_SPIRAM
        fallback.data = tool::malloc_aligned(required_bytes, MALLOC_CAP_SPIRAM);
#else
        fallback.data = tool::malloc_aligned(required_bytes, MALLOC_CAP_DEFAULT);
#endif
        fallback.bytes = (fallback.data != nullptr) ? required_bytes : 0;
    }
    bytes = fallback.bytes;
    return fallback.data;
}

template <typename T>
void Attention::forward_template(ModelContext *context, runtime_mode_t mode)
{
    TensorBase *q = context->get_tensor(m_inputs_index[0]);
    TensorBase *k = context->get_tensor(m_inputs_index[1]);
    TensorBase *v = context->get_tensor(m_inputs_index[2]);
    TensorBase *o = context->get_tensor(m_outputs_index[0]);

    // Optional 4th input: attn_mask (fp32 additive or bool keep/drop).
    TensorBase *mask = nullptr;
    base::AttentionMaskCache *mask_cache = nullptr;
    if (m_inputs_index.size() > 3) {
        mask = context->get_tensor(m_inputs_index[3]);
        // Parameters are immutable, so their precompute can be cached.
        if (mask != nullptr && m_inputs_index[3] >= CONTEXT_PARAMETER_OFFSET) {
            if (!m_mask_cache) {
                m_mask_cache.reset(new (std::nothrow) base::AttentionMaskCache());
            }
            mask_cache = m_mask_cache.get();
        }
    }

    const char *node_name = name ? name : "<unnamed>";
    bool ok = base::attention_check_inputs(q, k, v, mask, o, m_has_scale, m_scale, m_layout, qkv_dtype(), node_name);
    base::AttentionWorkspace workspace;
    if (ok) {
        const dtype_t dtype = qkv_dtype();
        const base::AttentionWorkspaceSizes required =
            base::attention_workspace_sizes(dtype, q->shape, k->shape, m_layout);
        size_t workspace_bytes = 0;
        void *storage = get_workspace_or_fallback(context, required.bytes, workspace_bytes);
        workspace = base::attention_bind_workspace(storage, workspace_bytes, dtype, q->shape, k->shape, m_layout);
        if (workspace.tile_buf == nullptr) {
            if (!m_workspace_logged) {
                m_workspace_logged = true;
                ESP_LOGE("Attention",
                         "%s: workspace alloc failed (%u KB) — Attention cannot run",
                         node_name,
                         static_cast<unsigned>(required.bytes / 1024));
            }
            ok = false;
        }
    }
    if constexpr (std::is_same<T, int8_t>::value) {
        if (ok && !m_int8_exp_lut) {
            m_int8_exp_lut.reset(new (std::nothrow) base::AttentionInt8ExpLut());
            if (!m_int8_exp_lut) {
                ESP_LOGE("Attention", "%s: cannot allocate the int8 softmax exp LUT.", node_name);
                ok = false;
            }
        }
        if (ok) {
            ok = base::attention_prepare_int8_softmax_exp_lut(*m_int8_exp_lut, q, k, m_has_scale, m_scale, node_name);
        }
    }
    if (!ok) {
        if (o != nullptr && o->data != nullptr) {
            std::memset(o->data, 0, o->get_bytes());
        }
        return;
    }

    base::AttentionMaskArgs mask_args = base::get_attention_mask_args(mask, q, k, m_layout, node_name);
    if (mask_cache != nullptr && mask_args.data != nullptr) {
        base::attention_prepare_constant_mask_cache<T>(
            *mask_cache, mask_args, q, k, m_has_scale, m_scale, m_layout, node_name);
    }
    // The barrier is invocation-local, so separate model instances cannot
    // reset a task pair that is still consuming another Attention cache.
    base::AttentionTaskBarrier task_barrier{};
    const base::AttentionExecutionContext execution_context{m_int8_exp_lut.get(), &task_barrier};

    std::vector<base::AttentionArgs> m_args = base::get_attention_operation_args(
        o, q, k, v, mask_args, workspace, execution_context, m_has_scale, m_scale, m_is_causal, m_layout, mode);
    if (m_args.size() == 1) {
        forward_args(&m_args[0]);
    } else if (m_args.size() == 2) {
        module_forward_dual_core(this, &m_args[0], &m_args[1]);
    }
}

} // namespace module
} // namespace dl
