#pragma once

#include "dl_define.hpp"
#include "dl_model_context.hpp"
#include "dl_module_base.hpp"
#include "dl_tensor_base.hpp"
#include "dl_tool.hpp"
#include "esp_log.h"
#include "fbs_model.hpp"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace dl {
namespace module {

/**
 * @brief InverseSigmoid operator.
 *
 * Computes element-wise inverse-sigmoid with clip guard eps in (0, 1):
 *   xc  = clip(X, 0, 1)
 *   num = max(xc, eps)
 *   den = max(1 - xc, eps)
 *   Y   = log(num / den)
 */
class InverseSigmoid : public Module {
private:
    float m_eps = 1e-5f; /*!< Clip guard that keeps the argument of Log strictly positive, in (0, 1) */

    static bool is_valid_eps(float eps) { return std::isfinite(eps) && eps > 0.0f && eps < 1.0f; }

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

    static bool checked_positive_shape_size(const std::vector<int> &shape, int *total_out = nullptr)
    {
        if (shape.empty()) {
            return false;
        }
        int total = 1;
        for (int dim : shape) {
            if (dim <= 0 || total > INT_MAX / dim) {
                return false;
            }
            total *= dim;
        }
        if (total_out != nullptr) {
            *total_out = total;
        }
        return true;
    }

    static float compute_inverse_sigmoid(float x_val, float eps)
    {
        if (x_val < 0.0f) {
            x_val = 0.0f;
        } else if (x_val > 1.0f) {
            x_val = 1.0f;
        }

        const float num = (x_val > eps) ? x_val : eps;
        const float one_sub_x = 1.0f - x_val;
        const float denom = (one_sub_x > eps) ? one_sub_x : eps;

        // Must stay logf(num / denom), not logf(num) - logf(denom): the unfused
        // ONNX graph is Div then Log, and Model::test() is exact on int8.
        return logf(num / denom);
    }

    static void inverse_sigmoid_zero_output(TensorBase *y)
    {
        if (y != nullptr && y->data != nullptr && y->get_size() > 0 && y->get_bytes() > 0) {
            std::memset(y->data, 0, y->get_bytes());
        }
    }

    static bool check_per_tensor_exponent(const TensorBase *tensor, const char *role)
    {
        if (!tensor->exponent.is_valid() || tensor->exponent.is_per_channel()) {
            ESP_LOGE("InverseSigmoid", "%s requires a valid per-tensor exponent", role);
            return false;
        }
        const int exp = tensor->exponent.get();
        if (exp <= -31 || exp >= 31) {
            ESP_LOGE("InverseSigmoid", "%s exponent %d is out of range (-31, 31)", role, exp);
            return false;
        }
        return true;
    }

    bool check_forward_tensors(ModelContext *context, dtype_t expected_dtype, TensorBase *&x, TensorBase *&y) const
    {
        if (!context) {
            ESP_LOGE("InverseSigmoid", "null ModelContext");
            return false;
        }
        y = m_outputs_index.empty() ? nullptr : context->get_tensor(m_outputs_index[0]);
        if (m_inputs_index.size() != 1 || m_outputs_index.size() != 1) {
            ESP_LOGE("InverseSigmoid",
                     "expected exactly 1 input and 1 output, got %d/%d",
                     static_cast<int>(m_inputs_index.size()),
                     static_cast<int>(m_outputs_index.size()));
            inverse_sigmoid_zero_output(y);
            return false;
        }

        x = context->get_tensor(m_inputs_index[0]);
        // Release-safe: an assert would compile out and leave a null deref.
        if (!x || !y) {
            ESP_LOGE("InverseSigmoid",
                     "missing input/output tensor (x=%p y=%p)",
                     static_cast<void *>(x),
                     static_cast<void *>(y));
            inverse_sigmoid_zero_output(y);
            return false;
        }
        if (x->dtype != y->dtype) {
            ESP_LOGE(
                "InverseSigmoid", "X dtype %d != Y dtype %d", static_cast<int>(x->dtype), static_cast<int>(y->dtype));
            inverse_sigmoid_zero_output(y);
            return false;
        }
        if (x->dtype != expected_dtype) {
            ESP_LOGE("InverseSigmoid",
                     "tensor dtype %s does not match quant_type %s",
                     dtype_to_string(x->dtype),
                     quant_type_to_string(quant_type));
            inverse_sigmoid_zero_output(y);
            return false;
        }
        if (x->shape != y->shape) {
            ESP_LOGE("InverseSigmoid",
                     "Y rank %d != X rank %d or shape mismatch",
                     static_cast<int>(y->shape.size()),
                     static_cast<int>(x->shape.size()));
            inverse_sigmoid_zero_output(y);
            return false;
        }
        int total = 0;
        if (!checked_positive_shape_size(x->shape, &total) || x->get_size() != total || y->get_size() != total) {
            ESP_LOGE("InverseSigmoid",
                     "invalid or mismatched X/Y size (%d/%d vs shape product %d)",
                     x->get_size(),
                     y->get_size(),
                     total);
            inverse_sigmoid_zero_output(y);
            return false;
        }
        if (!is_valid_eps(m_eps)) {
            ESP_LOGE("InverseSigmoid", "eps=%.6g must be finite and in (0, 1)", static_cast<double>(m_eps));
            inverse_sigmoid_zero_output(y);
            return false;
        }
        if (total > 0 && (!x->get_element_ptr() || !y->get_element_ptr())) {
            ESP_LOGE("InverseSigmoid", "null element pointer");
            inverse_sigmoid_zero_output(y);
            return false;
        }
        if (expected_dtype == DATA_TYPE_INT8 || expected_dtype == DATA_TYPE_INT16) {
            if (!check_per_tensor_exponent(x, "X") || !check_per_tensor_exponent(y, "Y")) {
                inverse_sigmoid_zero_output(y);
                return false;
            }
        }
        return true;
    }

public:
    /**
     * @brief Construct a new InverseSigmoid object.
     *
     * @param name        name of module
     * @param eps         clip guard in (0, 1), default 1e-5f
     * @param inplace     inplace type
     * @param quant_type  quantization type
     */
    InverseSigmoid(const char *name = NULL,
                   float eps = 1e-5f,
                   module_inplace_t inplace = MODULE_NON_INPLACE,
                   quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_eps(eps)
    {
    }

    /**
     * @brief Destroy the InverseSigmoid object.
     */
    ~InverseSigmoid() override = default;

    /**
     * @brief Calculate the output shape.
     *
     * @param input_shapes The shape of inputs
     * @return output shape
     */
    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) override
    {
        if (!is_valid_eps(m_eps)) {
            ESP_LOGE("InverseSigmoid", "eps=%.6g must be finite and in (0, 1)", static_cast<double>(m_eps));
            return {};
        }
        if (input_shapes.size() != 1 || !checked_positive_shape_size(input_shapes[0])) {
            ESP_LOGE("InverseSigmoid",
                     "expected 1 positive non-empty input shape, got %d",
                     static_cast<int>(input_shapes.size()));
            return {};
        }
        return {input_shapes[0]};
    }

    void forward_float(ModelContext *context)
    {
        TensorBase *input = nullptr;
        TensorBase *output = nullptr;
        if (!check_forward_tensors(context, DATA_TYPE_FLOAT, input, output)) {
            return;
        }
        const int total = input->get_size();
        if (total <= 0) {
            return;
        }
        const float *input_ptr = input->get_element_ptr<float>();
        float *output_ptr = output->get_element_ptr<float>();
        const float eps = m_eps;
        for (int i = 0; i < total; ++i) {
            output_ptr[i] = compute_inverse_sigmoid(input_ptr[i], eps);
        }
    }

    /**
     * @brief Forward pass for InverseSigmoid.
     *
     * @param context Model context
     * @param mode    Runtime mode
     */
    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO) override
    {
        quant_type_t effective_quant = quant_type;
        if (effective_quant == QUANT_TYPE_NONE && context != nullptr && m_inputs_index.size() == 1 &&
            m_outputs_index.size() == 1) {
            TensorBase *x = context->get_tensor(m_inputs_index[0]);
            if (x != nullptr) {
                if (x->dtype == DATA_TYPE_INT8) {
                    effective_quant = QUANT_TYPE_SYMM_8BIT;
                } else if (x->dtype == DATA_TYPE_INT16) {
                    effective_quant = QUANT_TYPE_SYMM_16BIT;
                } else if (x->dtype == DATA_TYPE_FLOAT) {
                    effective_quant = QUANT_TYPE_FLOAT32;
                }
            }
        }

        if (effective_quant == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (effective_quant == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        } else if (effective_quant == QUANT_TYPE_FLOAT32) {
            forward_float(context);
        } else {
            ESP_LOGE("InverseSigmoid", "unsupported quant_type %s", quant_type_to_string(quant_type));
            if (context != nullptr && !m_outputs_index.empty()) {
                inverse_sigmoid_zero_output(context->get_tensor(m_outputs_index[0]));
            }
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t /*mode*/)
    {
        constexpr dtype_t expected_dtype = std::is_same_v<T, int8_t> ? DATA_TYPE_INT8 : DATA_TYPE_INT16;
        TensorBase *input = nullptr;
        TensorBase *output = nullptr;
        if (!check_forward_tensors(context, expected_dtype, input, output)) {
            return;
        }
        const int total = input->get_size();
        if (total <= 0) {
            return;
        }
        const T *input_ptr = input->get_element_ptr<T>();
        T *output_ptr = output->get_element_ptr<T>();
        const float input_scale = DL_SCALE(input->exponent);
        const float output_scale = DL_RESCALE(output->exponent);
        const float eps = m_eps;
        for (int i = 0; i < total; ++i) {
            const float temp = compute_inverse_sigmoid(static_cast<float>(input_ptr[i]) * input_scale, eps);
            tool::truncate(output_ptr[i], tool::round(temp * output_scale));
        }
    }

    void forward_args(void * /*args*/) override {}

    /**
     * @brief Print module information.
     */
    void print() override
    {
        ESP_LOGI("InverseSigmoid",
                 "quant_type: %s, eps: %.6g.",
                 quant_type_to_string(quant_type),
                 static_cast<double>(m_eps));
    }

    /**
     * @brief Deserialize InverseSigmoid module instance by node serialization information.
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        if (!fbs_model) {
            ESP_LOGE("InverseSigmoid", "%s: null FbsModel", node_name.c_str());
            return nullptr;
        }

        quant_type_t quant_type = QUANT_TYPE_NONE;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        if (!is_supported_quant_type(quant_type)) {
            ESP_LOGE("InverseSigmoid",
                     "%s: unsupported quant_type %d; expected F32, S8, or S16",
                     node_name.c_str(),
                     static_cast<int>(quant_type));
            return nullptr;
        }

        float eps = 1e-5f;
        fbs_model->get_operation_attribute(node_name, "eps", eps);
        if (!is_valid_eps(eps)) {
            ESP_LOGE("InverseSigmoid",
                     "%s: eps=%.6g must be finite and in (0, 1)",
                     node_name.c_str(),
                     static_cast<double>(eps));
            return nullptr;
        }

        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        if (fbs_model->get_operation_inputs_and_outputs(node_name, inputs, outputs) != ESP_OK || inputs.size() != 1 ||
            outputs.size() != 1) {
            ESP_LOGE("InverseSigmoid",
                     "%s: expected exactly 1 input and 1 output, got %d/%d",
                     node_name.c_str(),
                     static_cast<int>(inputs.size()),
                     static_cast<int>(outputs.size()));
            return nullptr;
        }

        const dtype_t expected_dtype = dtype_from_quant_type(quant_type);
        const dtype_t input_dtype = recorded_tensor_dtype(fbs_model, inputs[0]);
        const dtype_t output_dtype = recorded_tensor_dtype(fbs_model, outputs[0]);
        if ((input_dtype != DATA_TYPE_UNDEFINED && input_dtype != expected_dtype) ||
            (output_dtype != DATA_TYPE_UNDEFINED && output_dtype != expected_dtype)) {
            ESP_LOGE("InverseSigmoid",
                     "%s: input/output dtype %s/%s does not match quant_type %s",
                     node_name.c_str(),
                     dtype_to_string(input_dtype),
                     dtype_to_string(output_dtype),
                     quant_type_to_string(quant_type));
            return nullptr;
        }

        if (!check_recorded_quant_exponent(fbs_model, inputs[0], input_dtype) ||
            !check_recorded_quant_exponent(fbs_model, outputs[0], output_dtype)) {
            ESP_LOGE("InverseSigmoid",
                     "%s: quantized X/Y require valid per-tensor exponents in (-31, 31)",
                     node_name.c_str());
            return nullptr;
        }

        std::vector<int> input_shape;
        std::vector<int> output_shape;
        const bool has_input_shape = (fbs_model->get_operation_input_shape(node_name, 0, input_shape) == ESP_OK);
        const bool has_output_shape = (fbs_model->get_operation_output_shape(node_name, 0, output_shape) == ESP_OK);
        if (has_input_shape && !checked_positive_shape_size(input_shape)) {
            ESP_LOGE("InverseSigmoid",
                     "%s: invalid input shape (rank=%d)",
                     node_name.c_str(),
                     static_cast<int>(input_shape.size()));
            return nullptr;
        }
        if (has_output_shape && !checked_positive_shape_size(output_shape)) {
            ESP_LOGE("InverseSigmoid",
                     "%s: invalid output shape (rank=%d)",
                     node_name.c_str(),
                     static_cast<int>(output_shape.size()));
            return nullptr;
        }
        if (has_input_shape && has_output_shape && input_shape != output_shape) {
            ESP_LOGE("InverseSigmoid",
                     "%s: output shape (rank=%d) must match input shape (rank=%d)",
                     node_name.c_str(),
                     static_cast<int>(output_shape.size()),
                     static_cast<int>(input_shape.size()));
            return nullptr;
        }

        return new InverseSigmoid(node_name.c_str(), eps, MODULE_INPLACE_CHANGED_BUFFER, quant_type);
    }
};

} // namespace module
} // namespace dl
