#pragma once

#include "dl_base_rotary_embedding.hpp"
#include "dl_model_context.hpp"
#include "dl_module_base.hpp"
#include "esp_log.h"
#include "fbs_model.hpp"

#include <string>
#include <vector>

namespace dl {
namespace module {

/**
 * @brief RotaryEmbedding (vision subset of ONNX opset 23 RotaryEmbedding).
 *
 * Applies non-interleaved (half-split) rotary position embedding to 4D input X [B, H, L, D]
 * using precomputed cos_cache and sin_cache.
 *
 * Reference: https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html
 */
class RotaryEmbedding : public Module {
private:
    int m_interleaved = 0;          /*!< whether rotation pairs are interleaved; must be 0 (half-split layout) */
    int m_rotary_embedding_dim = 0; /*!< rotary embedding dimension; must be 0 or head_dim D */
    int m_num_heads = 0;            /*!< number of attention heads; 0 or X.shape[1] */

    static bool is_supported_quant_type(quant_type_t type)
    {
        return type == QUANT_TYPE_FLOAT32 || type == QUANT_TYPE_SYMM_8BIT || type == QUANT_TYPE_SYMM_16BIT;
    }

    static dtype_t dtype_from_quant_type(quant_type_t type)
    {
        switch (type) {
        case QUANT_TYPE_FLOAT32:
            return DATA_TYPE_FLOAT;
        case QUANT_TYPE_SYMM_8BIT:
            return DATA_TYPE_INT8;
        case QUANT_TYPE_SYMM_16BIT:
            return DATA_TYPE_INT16;
        default:
            return DATA_TYPE_UNDEFINED;
        }
    }

    static bool is_supported_tensor_dtype(dtype_t dtype)
    {
        return dtype == DATA_TYPE_FLOAT || dtype == DATA_TYPE_INT8 || dtype == DATA_TYPE_INT16;
    }

    static dtype_t recorded_tensor_dtype(fbs::FbsModel *fbs_model, const std::string &name)
    {
        if (name.empty()) {
            return DATA_TYPE_UNDEFINED;
        }
        return fbs_model->is_parameter(name) ? fbs_model->get_tensor_dtype(name)
                                             : fbs_model->get_value_info_dtype(name);
    }

    static bool check_recorded_quant_exponent(fbs::FbsModel *fbs_model, const std::string &tensor_name, dtype_t dtype)
    {
        if (dtype != DATA_TYPE_INT8 && dtype != DATA_TYPE_INT16) {
            return true;
        }
        int exponent = 0;
        if (fbs_model->is_parameter(tensor_name)) {
            const std::vector<int> exponents = fbs_model->get_tensor_exponents(tensor_name);
            if (exponents.size() != 1) {
                return false;
            }
            exponent = exponents[0];
        } else {
            exponent = fbs_model->get_value_info_exponent(tensor_name);
        }
        return exponent > -31 && exponent < 31;
    }

public:
    /**
     * @brief Construct a new RotaryEmbedding object.
     *
     * @param name                  name of module
     * @param interleaved           whether rotation pairs are interleaved (must be 0)
     * @param rotary_embedding_dim  rotary embedding dimension (0 or head_dim D)
     * @param num_heads             number of attention heads (0 or X.shape[1])
     * @param inplace               inplace type
     * @param quant_type            quantization type
     */
    RotaryEmbedding(const char *name = NULL,
                    int interleaved = 0,
                    int rotary_embedding_dim = 0,
                    int num_heads = 0,
                    module_inplace_t inplace = MODULE_NON_INPLACE,
                    quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type),
        m_interleaved(interleaved),
        m_rotary_embedding_dim(rotary_embedding_dim),
        m_num_heads(num_heads)
    {
    }

    /**
     * @brief Destroy the RotaryEmbedding object.
     */
    ~RotaryEmbedding() override = default;

    /**
     * @brief Calculate the output shape.
     *
     * @param input_shapes The shapes of inputs [X, cos_cache, sin_cache]
     * @return output shape
     */
    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) override
    {
        if (input_shapes.size() != 3) {
            ESP_LOGE("RotaryEmbedding",
                     "shape inference expected exactly 3 inputs, got %d",
                     static_cast<int>(input_shapes.size()));
            return {};
        }
        base::RopeDims dims;
        if (!base::rope_check_shapes(input_shapes[0],
                                     input_shapes[1],
                                     input_shapes[2],
                                     m_interleaved,
                                     m_rotary_embedding_dim,
                                     m_num_heads,
                                     dims,
                                     name)) {
            return {};
        }
        return {input_shapes[0]};
    }

    /**
     * @brief Forward pass for RotaryEmbedding.
     *
     * @param context Model context
     * @param mode    Runtime mode
     */
    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO) override
    {
        if (context == nullptr) {
            ESP_LOGE("RotaryEmbedding", "model context is null");
            return;
        }

        TensorBase *y = m_outputs_index.empty() ? nullptr : context->get_tensor(m_outputs_index[0]);
        if (m_inputs_index.size() != 3 || m_outputs_index.size() != 1) {
            ESP_LOGE("RotaryEmbedding",
                     "expected exactly 3 inputs and 1 output, got %d/%d",
                     static_cast<int>(m_inputs_index.size()),
                     static_cast<int>(m_outputs_index.size()));
            base::rope_zero_output(y);
            return;
        }
        TensorBase *x = context->get_tensor(m_inputs_index[0]);
        TensorBase *cos_cache = context->get_tensor(m_inputs_index[1]);
        TensorBase *sin_cache = context->get_tensor(m_inputs_index[2]);

        std::vector<base::RopeArgs> m_args = base::get_rotary_embedding_operation_args(
            y, x, cos_cache, sin_cache, m_interleaved, m_rotary_embedding_dim, m_num_heads, quant_type, mode, name);
        if (m_args.size() == 1) {
            forward_args(&m_args[0]);
        } else if (m_args.size() == 2) {
            module_forward_dual_core(this, &m_args[0], &m_args[1]);
        }
    }

    /**
     * @brief Low-level forward interface for base layer and multi-core processing.
     *
     * @param args Pointer to RopeArgs
     */
    void forward_args(void *args) override { base::rotary_embedding(args); }

    /**
     * @brief Print module information.
     */
    void print() override
    {
        ESP_LOGI("RotaryEmbedding",
                 "quant_type: %s, interleaved: %d, rotary_embedding_dim: %d, num_heads: %d.",
                 quant_type_to_string(quant_type),
                 m_interleaved,
                 m_rotary_embedding_dim,
                 m_num_heads);
    }

    /**
     * @brief Deserialize RotaryEmbedding module instance by node serialization information.
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        if (!fbs_model) {
            ESP_LOGE("RotaryEmbedding", "%s: null FbsModel", node_name.c_str());
            return nullptr;
        }

        quant_type_t quant_type = QUANT_TYPE_NONE;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        if (!is_supported_quant_type(quant_type)) {
            ESP_LOGE("RotaryEmbedding",
                     "%s: unsupported quant_type %d; expected F32, S8, or S16",
                     node_name.c_str(),
                     static_cast<int>(quant_type));
            return nullptr;
        }

        int interleaved = 0;
        fbs_model->get_operation_attribute(node_name, "interleaved", interleaved);

        int rotary_embedding_dim = 0;
        fbs_model->get_operation_attribute(node_name, "rotary_embedding_dim", rotary_embedding_dim);

        int num_heads = 0;
        fbs_model->get_operation_attribute(node_name, "num_heads", num_heads);

        if (!base::rope_check_attributes(interleaved, rotary_embedding_dim, num_heads, node_name.c_str())) {
            return nullptr;
        }

        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        if (fbs_model->get_operation_inputs_and_outputs(node_name, inputs, outputs) != ESP_OK || inputs.size() != 3 ||
            outputs.size() != 1) {
            ESP_LOGE("RotaryEmbedding",
                     "%s: expected exactly 3 inputs and 1 output, got %d/%d",
                     node_name.c_str(),
                     static_cast<int>(inputs.size()),
                     static_cast<int>(outputs.size()));
            return nullptr;
        }

        const dtype_t expected_xy_dtype = dtype_from_quant_type(quant_type);
        const dtype_t x_dtype = recorded_tensor_dtype(fbs_model, inputs[0]);
        const dtype_t cos_dtype = recorded_tensor_dtype(fbs_model, inputs[1]);
        const dtype_t sin_dtype = recorded_tensor_dtype(fbs_model, inputs[2]);
        const dtype_t y_dtype = recorded_tensor_dtype(fbs_model, outputs[0]);
        if ((x_dtype != DATA_TYPE_UNDEFINED && x_dtype != expected_xy_dtype) ||
            (y_dtype != DATA_TYPE_UNDEFINED && y_dtype != expected_xy_dtype)) {
            ESP_LOGE("RotaryEmbedding",
                     "%s: X/Y dtype %s/%s does not match quant_type %s",
                     node_name.c_str(),
                     dtype_to_string(x_dtype),
                     dtype_to_string(y_dtype),
                     quant_type_to_string(quant_type));
            return nullptr;
        }
        if ((cos_dtype != DATA_TYPE_UNDEFINED && !is_supported_tensor_dtype(cos_dtype)) ||
            (sin_dtype != DATA_TYPE_UNDEFINED && !is_supported_tensor_dtype(sin_dtype)) ||
            (cos_dtype != DATA_TYPE_UNDEFINED && sin_dtype != DATA_TYPE_UNDEFINED && cos_dtype != sin_dtype)) {
            ESP_LOGE("RotaryEmbedding",
                     "%s: invalid cos/sin cache dtype %s/%s",
                     node_name.c_str(),
                     dtype_to_string(cos_dtype),
                     dtype_to_string(sin_dtype));
            return nullptr;
        }
        if (!check_recorded_quant_exponent(fbs_model, inputs[0], x_dtype) ||
            !check_recorded_quant_exponent(fbs_model, inputs[1], cos_dtype) ||
            !check_recorded_quant_exponent(fbs_model, inputs[2], sin_dtype) ||
            !check_recorded_quant_exponent(fbs_model, outputs[0], y_dtype)) {
            ESP_LOGE("RotaryEmbedding",
                     "%s: quantized tensors require valid per-tensor exponents in (-31, 31)",
                     node_name.c_str());
            return nullptr;
        }

        std::vector<int> x_shape;
        std::vector<int> cos_shape;
        std::vector<int> sin_shape;
        std::vector<int> y_shape;
        if (fbs_model->get_operation_input_shape(node_name, 0, x_shape) == ESP_OK &&
            fbs_model->get_operation_input_shape(node_name, 1, cos_shape) == ESP_OK &&
            fbs_model->get_operation_input_shape(node_name, 2, sin_shape) == ESP_OK) {
            base::RopeDims dims;
            if (!base::rope_check_shapes(x_shape,
                                         cos_shape,
                                         sin_shape,
                                         interleaved,
                                         rotary_embedding_dim,
                                         num_heads,
                                         dims,
                                         node_name.c_str())) {
                return nullptr;
            }
            if (fbs_model->get_operation_output_shape(node_name, 0, y_shape) == ESP_OK && y_shape != x_shape) {
                ESP_LOGE("RotaryEmbedding",
                         "%s: output shape (rank=%d) must match X shape (rank=%d)",
                         node_name.c_str(),
                         static_cast<int>(y_shape.size()),
                         static_cast<int>(x_shape.size()));
                return nullptr;
            }
        }

        return new RotaryEmbedding(
            node_name.c_str(), interleaved, rotary_embedding_dim, num_heads, MODULE_NON_INPLACE, quant_type);
    }
};

} // namespace module
} // namespace dl
