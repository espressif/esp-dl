#pragma once

#include "dl_base_norm.hpp"
#include "dl_base_reduce.hpp"
#include "dl_math.hpp"
#include "dl_module_base.hpp"
#include "esp_log.h"
#include <cassert>
#include <cmath>

namespace dl {
namespace module {

/**
 * @brief: RMSNormalization (Root Mean Square Normalization).
 *
 * Computes:
 *   rms = sqrt(mean(X^2) + epsilon)
 *   Y = X / rms * Scale
 *
 * This is simpler than LayerNormalization: no mean subtraction, no bias term.
 * Only a learnable scale (weight) parameter is applied after normalization.
 *
 * Reference: https://arxiv.org/abs/1910.07467
 */
class RMSNormalization : public Module {
private:
    int m_axis;      ///< The axis to normalize, default is -1
    float m_epsilon; ///< Small value to avoid division by zero, default is 1e-5

public:
    RMSNormalization(int axis = -1,
                     float epsilon = 1e-5f,
                     const char *name = NULL,
                     module_inplace_t inplace = MODULE_NON_INPLACE,
                     quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_axis(axis), m_epsilon(epsilon)
    {
    }

    ~RMSNormalization() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) override
    {
        assert(input_shapes.size() >= 2);
        // Y has same shape as X; scale is required per ONNX RMSNormalization
        return {input_shapes[0]};
    }

    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO) override
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_int8(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_int16(context, mode);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            forward_float(context, mode);
        }
    }

    void forward_args(void *args) override {}

    void forward_int8(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *x = context->get_tensor(m_inputs_index[0]);
        TensorBase *scale = context->get_tensor(m_inputs_index[1]);
        TensorBase *y = context->get_tensor(m_outputs_index[0]);

        int8_t *x_ptr = x->get_element_ptr<int8_t>();
        int8_t *y_ptr = y->get_element_ptr<int8_t>();
        // scale (weight) is stored as float to preserve precision
        float *scale_ptr = scale->get_element_ptr<float>();

        int outer_dim = 0;
        int inner_dim = 0;
        get_outer_inner_dims(x->shape, outer_dim, inner_dim);

        float input_scale = DL_SCALE(x->exponent);
        float input_scale_sq = input_scale * input_scale;
        float output_scale = DL_RESCALE(y->exponent);
        float scale_factor = input_scale * output_scale;

        for (int i = 0; i < outer_dim; i++) {
            int8_t *x_slice = x_ptr + i * inner_dim;
            int8_t *y_slice = y_ptr + i * inner_dim;

            int32_t sum_sq = dl::base::reduce_l2(x_slice, inner_dim, 1);
            float mean_sq = static_cast<float>(sum_sq) * input_scale_sq / inner_dim;
            float rms = dl::math::fast_inv_sqrt(mean_sq + m_epsilon);
            rms *= scale_factor;

            base::rms_norm(y_slice, x_slice, scale_ptr, &rms, inner_dim);
        }
    }

    void forward_int16(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *x = context->get_tensor(m_inputs_index[0]);
        TensorBase *scale = context->get_tensor(m_inputs_index[1]);
        TensorBase *y = context->get_tensor(m_outputs_index[0]);

        int16_t *x_ptr = x->get_element_ptr<int16_t>();
        int16_t *y_ptr = y->get_element_ptr<int16_t>();
        // scale (weight) is stored as float to preserve precision
        float *scale_ptr = scale->get_element_ptr<float>();

        int outer_dim = 0;
        int inner_dim = 0;
        get_outer_inner_dims(x->shape, outer_dim, inner_dim);

        float input_scale = DL_SCALE(x->exponent);
        float input_scale_sq = input_scale * input_scale;
        float output_scale = DL_RESCALE(y->exponent);
        float scale_factor = input_scale * output_scale;

        for (int i = 0; i < outer_dim; i++) {
            int16_t *x_slice = x_ptr + i * inner_dim;
            int16_t *y_slice = y_ptr + i * inner_dim;

            int64_t sum_sq = dl::base::reduce_l2(x_slice, inner_dim, 1);
            float mean_sq = static_cast<float>(sum_sq) * input_scale_sq / inner_dim;
            float rms = dl::math::fast_inv_sqrt(mean_sq + m_epsilon);
            rms *= scale_factor;

            base::rms_norm(y_slice, x_slice, scale_ptr, &rms, inner_dim);
        }
    }

    void forward_float(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *x = context->get_tensor(m_inputs_index[0]);
        TensorBase *scale = context->get_tensor(m_inputs_index[1]);
        TensorBase *y = context->get_tensor(m_outputs_index[0]);

        float *x_ptr = x->get_element_ptr<float>();
        float *y_ptr = y->get_element_ptr<float>();
        float *scale_ptr = scale->get_element_ptr<float>();

        int outer_dim = 0;
        int inner_dim = 0;
        get_outer_inner_dims(x->shape, outer_dim, inner_dim);

        for (int i = 0; i < outer_dim; i++) {
            float *x_slice = x_ptr + i * inner_dim;
            float *y_slice = y_ptr + i * inner_dim;

            float mean_sq = 0.0f;
            for (int j = 0; j < inner_dim; j++) {
                mean_sq += x_slice[j] * x_slice[j];
            }
            mean_sq /= inner_dim;

            float rms = dl::math::fast_inv_sqrt(mean_sq + m_epsilon);
            for (int j = 0; j < inner_dim; j++) {
                y_slice[j] = x_slice[j] * rms * scale_ptr[j];
            }
        }
    }

    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;

        int axis = -1;
        float epsilon = 1e-5f;
        quant_type_t quant_type;

        fbs_model->get_operation_attribute(node_name, "axis", axis);
        fbs_model->get_operation_attribute(node_name, "epsilon", epsilon);
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        op = new RMSNormalization(axis, epsilon, node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }

    void print() override
    {
        ESP_LOGI("RMSNormalization",
                 "axis: %d, epsilon: %f, quant_type: %s.",
                 m_axis,
                 m_epsilon,
                 quant_type_to_string(quant_type));
    }

private:
    int get_valid_axis(const std::vector<int> &shape)
    {
        if (m_axis < 0) {
            return shape.size() + m_axis;
        }
        return m_axis;
    }

    void get_outer_inner_dims(const std::vector<int> &shape, int &outer_dim, int &inner_dim)
    {
        int valid_axis = get_valid_axis(shape);
        outer_dim = 1;
        for (int i = 0; i < valid_axis; i++) {
            outer_dim *= shape[i];
        }
        inner_dim = 1;
        for (size_t i = valid_axis; i < shape.size(); i++) {
            inner_dim *= shape[i];
        }
    }
};

} // namespace module
} // namespace dl
