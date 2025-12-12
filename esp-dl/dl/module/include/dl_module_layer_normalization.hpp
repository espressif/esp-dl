#pragma once

#include "dl_base_add.hpp"
#include "dl_base_mul.hpp"
#include "dl_base_sub.hpp"
#include "dl_math.hpp"
#include "dl_module_base.hpp"
#include "dl_module_reduce_mean.hpp"
#include "dl_tool.hpp"
#include "esp_log.h"
#include <cassert>
#include <cmath>
#include <type_traits>

namespace dl {
namespace module {

/**
 * @brief: LayerNormalization operator as defined in ONNX.
 *
 * The overall computation can be split into two stages.
 * The first stage is standardization, which makes the
 * normalized elements have zero mean and unit variances.
 * The computation required by standardization can be
 * described by the following equations.
 * ```
 * Mean = ReduceMean(X)
 * D = Sub(X, Mean)
 * DD = Mul(D, D)
 * Var = ReduceMean(DD)
 * VarEps = Add(Var, epsilon)
 * StdDev = Sqrt(VarEps)
 * InvStdDev = Reciprocal(StdDev)
 * Normalized = Mul(D, InvStdDev)
 * ```
 * The second stage then scales and shifts the outcome of the
 * first stage using
 * ```
 * NormalizedScaled = Mul(Normalized, Scale)
 * Y = Add(NormalizedScaled, B)
 * ```
 *
 * Reference: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
 */
class LayerNormalization : public Module {
private:
    int m_axis;       ///< The axis to normalize, default is -1
    float m_epsilon;  ///< Small value to avoid division by zero, default is 1e-5
    int m_stash_type; ///< Whether to stash intermediate values in higher precision (0 or 1)

public:
    /**
     * @brief Construct a new LayerNormalization object.
     *
     * @param axis           The axis to normalize (default -1)
     * @param epsilon        Small value to avoid division by zero (default 1e-5)
     * @param stash_type     Whether to stash intermediate values in higher precision (default false)
     * @param name           Name of module
     * @param inplace        Inplace type
     * @param quant_type     Quantization type
     */
    LayerNormalization(int axis = -1,
                       float epsilon = 1e-5f,
                       int stash_type = 0,
                       const char *name = NULL,
                       module_inplace_t inplace = MODULE_NON_INPLACE,
                       quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_axis(axis), m_epsilon(epsilon), m_stash_type(stash_type)
    {
    }

    /**
     * @brief Destroy the LayerNormalization object.
     */
    ~LayerNormalization() {}

    /**
     * @brief Calculate output shape based on input shapes.
     *
     * @param input_shapes   Input shapes [X_shape, (optional) Scale_shape, (optional) B_shape]
     * @return std::vector<std::vector<int>> Output shapes [Y_shape, Mean_shape, InvStdDev_shape]
     */
    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) override
    {
        assert(input_shapes.size() >= 1); // At least X input

        std::vector<int> x_shape = input_shapes[0];
        std::vector<int> y_shape = x_shape; // Y has same shape as X

        // Calculate normalized shape for Mean and InvStdDev
        int valid_axis = get_valid_axis(x_shape);
        std::vector<int> normalized_shape = calculate_normalized_shape(x_shape, valid_axis);

        // Return shapes for Y, Mean, and InvStdDev
        return {y_shape, normalized_shape, normalized_shape};
    }

    /**
     * @brief Forward pass for LayerNormalization.
     *
     * @param context   Model context
     * @param mode      Runtime mode
     */
    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO) override
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            forward_template<float>(context, mode);
        }
    }

    /**
     * @brief Low-level forward interface.
     *
     * @param args   Arguments for the operation
     */
    void forward_args(void *args) override
    {
        // LayerNormalization is complex and uses template forwarding
        // Actual implementation is in forward_template
    }

    /**
     * @brief Template forward implementation.
     *
     * @tparam T     Data type (int8_t, int16_t, float)
     * @param context Model context
     * @param mode    Runtime mode
     */
    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *x = context->get_tensor(m_inputs_index[0]);
        TensorBase *scale = (m_inputs_index.size() > 1) ? context->get_tensor(m_inputs_index[1]) : nullptr;
        TensorBase *bias = (m_inputs_index.size() > 2) ? context->get_tensor(m_inputs_index[2]) : nullptr;

        TensorBase *y = context->get_tensor(m_outputs_index[0]);
        TensorBase *mean = (m_outputs_index.size() > 1) ? context->get_tensor(m_outputs_index[1]) : nullptr;
        TensorBase *inv_std_dev = (m_outputs_index.size() > 2) ? context->get_tensor(m_outputs_index[2]) : nullptr;

        // Get pointers to data
        T *x_ptr = (T *)x->get_element_ptr();
        T *y_ptr = (T *)y->get_element_ptr();
        T *scale_ptr = scale ? (T *)scale->get_element_ptr() : nullptr;
        T *bias_ptr = bias ? (T *)bias->get_element_ptr() : nullptr;
        T *mean_ptr = mean ? (T *)mean->get_element_ptr() : nullptr;
        T *inv_std_dev_ptr = inv_std_dev ? (T *)inv_std_dev->get_element_ptr() : nullptr;

        // Get shapes
        std::vector<int> x_shape = x->shape;
        int valid_axis = get_valid_axis(x_shape);

        // Calculate dimensions
        int outer_dim = 1;
        for (int i = 0; i < valid_axis; i++) {
            outer_dim *= x_shape[i];
        }

        int inner_dim = 1;
        for (size_t i = valid_axis; i < x_shape.size(); i++) {
            inner_dim *= x_shape[i];
        }

        // Use compile-time branching to eliminate runtime if statements
        if constexpr (std::is_same_v<T, float>) {
            // Float32 path - direct computation
            for (int i = 0; i < outer_dim; i++) {
                float *x_slice = reinterpret_cast<float *>(x_ptr) + i * inner_dim;
                float *y_slice = reinterpret_cast<float *>(y_ptr) + i * inner_dim;

                // Calculate mean
                float mean_val = 0.0f;
                for (int j = 0; j < inner_dim; j++) {
                    mean_val += x_slice[j];
                }
                mean_val /= inner_dim;

                // Store mean if output tensor exists
                if (mean_ptr && mean) {
                    reinterpret_cast<float *>(mean_ptr)[i] = mean_val;
                }

                // Calculate variance
                float variance = 0.0f;
                for (int j = 0; j < inner_dim; j++) {
                    float diff = x_slice[j] - mean_val;
                    variance += diff * diff;
                }
                variance /= inner_dim;

                // Add epsilon and calculate inverse standard deviation
                float inv_std = 1.0f / dl::math::sqrt_newton(variance + m_epsilon);
                // float inv_std = math::fast_inv_sqrt(variance + m_epsilon); // faster but less accurate

                // Store inverse standard deviation if output tensor exists
                if (inv_std_dev_ptr && inv_std_dev) {
                    reinterpret_cast<float *>(inv_std_dev_ptr)[i] = inv_std;
                }

                // Normalize and apply scale/bias
                float *scale_slice = scale_ptr ? reinterpret_cast<float *>(scale_ptr) : nullptr;
                float *bias_slice = bias_ptr ? reinterpret_cast<float *>(bias_ptr) : nullptr;
                for (int j = 0; j < inner_dim; j++) {
                    float x_val = x_slice[j];
                    float normalized = (x_val - mean_val) * inv_std;

                    float scale_val = scale_slice ? scale_slice[j] : 1.0f;
                    float bias_val = bias_slice ? bias_slice[j] : 0.0f;

                    y_slice[j] = normalized * scale_val + bias_val;
                }
            }
        } else {
            // Quantized path (int8_t or int16_t)
            // Get quantization scales
            float input_scale = DL_SCALE(x->exponent);
            float output_scale = DL_RESCALE(y->exponent);
            float scale_scale = scale ? DL_SCALE(scale->exponent) : 1.0f;
            float bias_scale = bias ? DL_SCALE(bias->exponent) : 0.0f;
            float mean_output_scale = mean ? DL_RESCALE(mean->exponent) : 1.0f;
            float inv_std_dev_output_scale = inv_std_dev ? DL_RESCALE(inv_std_dev->exponent) : 1.0f;

            for (int i = 0; i < outer_dim; i++) {
                T *x_slice = x_ptr + i * inner_dim;
                T *y_slice = y_ptr + i * inner_dim;

                // Calculate mean
                float mean_val = 0.0f;
                int sum_val = 0;
                for (int j = 0; j < inner_dim; j++) {
                    sum_val += x_slice[j];
                }
                mean_val = sum_val * 1.0 / inner_dim;

                // Store mean if output tensor exists
                if (mean_ptr && mean) {
                    if constexpr (std::is_same_v<T, int8_t>) {
                        int8_t &output_ref = reinterpret_cast<int8_t &>(mean_ptr[i]);
                        tool::truncate(output_ref, tool::round(mean_val * mean_output_scale * input_scale));
                    } else if constexpr (std::is_same_v<T, int16_t>) {
                        int16_t &output_ref = reinterpret_cast<int16_t &>(mean_ptr[i]);
                        tool::truncate(output_ref, tool::round(mean_val * mean_output_scale * input_scale));
                    }
                }

                // Calculate variance
                float variance = 0.0f;
                for (int j = 0; j < inner_dim; j++) {
                    float diff = x_slice[j] - mean_val;
                    variance += diff * diff;
                }
                mean_val *= input_scale;
                variance *= input_scale * input_scale;
                variance /= inner_dim;

                // Add epsilon and calculate inverse standard deviation
                float inv_std = 1.0f / dl::math::sqrt_newton(variance + m_epsilon);
                // float inv_std = math::fast_inv_sqrt(variance + m_epsilon); // faster but less accurate

                // Store inverse standard deviation if output tensor exists
                if (inv_std_dev_ptr && inv_std_dev) {
                    if constexpr (std::is_same_v<T, int8_t>) {
                        int8_t &output_ref = reinterpret_cast<int8_t &>(inv_std_dev_ptr[i]);
                        tool::truncate(output_ref, tool::round(inv_std * inv_std_dev_output_scale));
                    } else if constexpr (std::is_same_v<T, int16_t>) {
                        int16_t &output_ref = reinterpret_cast<int16_t &>(inv_std_dev_ptr[i]);
                        tool::truncate(output_ref, tool::round(inv_std * inv_std_dev_output_scale));
                    }
                }

                // Normalize and apply scale/bias
                for (int j = 0; j < inner_dim; j++) {
                    float x_val = x_slice[j] * input_scale;
                    float normalized = (x_val - mean_val) * inv_std;

                    float scale_val = scale_ptr ? scale_ptr[j] * scale_scale : 1.0f;
                    float bias_val = bias_ptr ? bias_ptr[j] * bias_scale : 0.0f;

                    float result = normalized * scale_val + bias_val;

                    // Use appropriate truncate based on quantization type
                    if constexpr (std::is_same_v<T, int8_t>) {
                        int8_t &output_ref = reinterpret_cast<int8_t &>(y_slice[j]);
                        tool::truncate(output_ref, tool::round(result * output_scale));
                    } else if constexpr (std::is_same_v<T, int16_t>) {
                        int16_t &output_ref = reinterpret_cast<int16_t &>(y_slice[j]);
                        tool::truncate(output_ref, tool::round(result * output_scale));
                    }
                }
            }
        }
    }

    /**
     * @brief Calculate normalized shape based on input shape and axis.
     *
     * @param shape   Input shape
     * @param axis    Normalization axis
     * @return std::vector<int> Normalized shape
     */
    static std::vector<int> calculate_normalized_shape(const std::vector<int> &shape, int axis)
    {
        if (axis < 0) {
            axis = shape.size() + axis;
        }

        std::vector<int> normalized_shape(shape.size(), 1);
        for (int i = 0; i < axis; i++) {
            normalized_shape[i] = shape[i];
        }

        return normalized_shape;
    }

    /**
     * @brief Deserialize LayerNormalization module instance.
     *
     * @param fbs_model  Flatbuffer model
     * @param node_name  Node name
     * @return Module*   Deserialized module
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;

        int axis = -1;
        float epsilon = 1e-5f;
        int stash_type = 0;
        quant_type_t quant_type;

        fbs_model->get_operation_attribute(node_name, "axis", axis);
        fbs_model->get_operation_attribute(node_name, "epsilon", epsilon);
        fbs_model->get_operation_attribute(node_name, "stash_type", stash_type);
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        // Create module
        op = new LayerNormalization(
            axis, epsilon, stash_type, node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }

    /**
     * @brief Print module information.
     */
    void print() override
    {
        ESP_LOGI("LayerNormalization",
                 "axis: %d, epsilon: %f, stash_type: %d, quant_type: %s.",
                 m_axis,
                 m_epsilon,
                 m_stash_type,
                 quant_type_to_string(quant_type));
    }

private:
    /**
     * @brief Helper function to get axis with proper bounds.
     *
     * @param shape   Input shape
     * @return int    Valid axis value
     */
    int get_valid_axis(const std::vector<int> &shape)
    {
        if (m_axis < 0) {
            return shape.size() + m_axis;
        }
        return m_axis;
    }
};

} // namespace module
} // namespace dl
