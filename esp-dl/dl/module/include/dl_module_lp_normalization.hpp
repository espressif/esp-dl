#pragma once

#include "dl_base_reduce.hpp"
#include "dl_math.hpp"
#include "dl_module_base.hpp"
#include "dl_tool.hpp"
#include "esp_log.h"
#include <cassert>
#include <cmath>

namespace dl {
namespace module {

/**
 * @brief LpNormalization operator as defined in ONNX.
 *
 * Given a matrix, apply Lp-normalization along the provided axis.
 * output = input / Lp_norm(input, axis).
 * When the Lp norm is zero, the output is defined to be zero.
 *
 * Reference: https://onnx.ai/onnx/operators/onnx__LpNormalization.html
 */
class LpNormalization : public Module {
private:
    int m_axis;
    int m_p;

public:
    LpNormalization(int axis = -1,
                    int p = 2,
                    const char *name = NULL,
                    module_inplace_t inplace = MODULE_NON_INPLACE,
                    quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_axis(axis), m_p(p)
    {
    }

    ~LpNormalization() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) override
    {
        assert(input_shapes.size() >= 1);
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

    void forward_float(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        float *input_ptr = input->get_element_ptr<float>();
        float *output_ptr = output->get_element_ptr<float>();

        std::vector<int> &shape = input->shape;
        int dims = shape.size();
        if (dims == 0) {
            output_ptr[0] = 0;
            return;
        }

        int positive_axis = m_axis < 0 ? dims + m_axis : m_axis;
        int len = shape[positive_axis];

        if (input_ptr != output_ptr) {
            for (size_t i = 0; i < input->size; i++) {
                output_ptr[i] = input_ptr[i];
            }
        }

        if (positive_axis == dims - 1) {
            int outer_loop = input->size / len;
            for (int o = 0; o < outer_loop; o++) {
                float inv_norm = compute_inv_norm(output_ptr, len);
                if (inv_norm > 0.0f) {
                    for (int i = 0; i < len; i++) {
                        output_ptr[i] *= inv_norm;
                    }
                } else {
                    for (int i = 0; i < len; i++) {
                        output_ptr[i] = 0.0f;
                    }
                }
                output_ptr += len;
            }
        } else {
            int outer_loop = 1;
            int inner_loop = 1;
            get_outer_inner_loops(shape, dims, positive_axis, outer_loop, inner_loop);

            for (int o = 0; o < outer_loop; o++) {
                for (int k = 0; k < inner_loop; k++) {
                    float inv_norm = compute_inv_norm_strided(output_ptr, len, inner_loop);
                    if (inv_norm > 0.0f) {
                        for (int i = 0; i < len; i++) {
                            output_ptr[i * inner_loop] *= inv_norm;
                        }
                    } else {
                        for (int i = 0; i < len; i++) {
                            output_ptr[i * inner_loop] = 0.0f;
                        }
                    }
                    output_ptr += 1;
                }
                output_ptr += inner_loop * (len - 1);
            }
        }
    }

    void forward_int8(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        int8_t *input_ptr = input->get_element_ptr<int8_t>();
        int8_t *output_ptr = output->get_element_ptr<int8_t>();

        std::vector<int> &shape = input->shape;
        int dims = shape.size();
        if (dims == 0) {
            output_ptr[0] = 0;
            return;
        }

        int positive_axis = m_axis < 0 ? dims + m_axis : m_axis;
        int len = shape[positive_axis];

        // input_scale cancels out: output_float = x_int / norm_int
        // only output_scale is needed for requantization
        float output_scale = DL_RESCALE(output->exponent);

        if (input_ptr != output_ptr) {
            for (size_t i = 0; i < input->size; i++) {
                output_ptr[i] = input_ptr[i];
            }
        }

        if (positive_axis == dims - 1) {
            int outer_loop = input->size / len;
            for (int o = 0; o < outer_loop; o++) {
                float inv_norm = compute_quant_inv_norm(output_ptr, len);
                inv_norm *= output_scale;

                if (inv_norm > 0.0f) {
                    for (int i = 0; i < len; i++) {
                        float result = output_ptr[i] * inv_norm;
                        tool::truncate(output_ptr[i], tool::round(result));
                    }
                } else {
                    for (int i = 0; i < len; i++) {
                        output_ptr[i] = 0;
                    }
                }
                output_ptr += len;
            }
        } else {
            int outer_loop = 1;
            int inner_loop = 1;
            get_outer_inner_loops(shape, dims, positive_axis, outer_loop, inner_loop);

            for (int o = 0; o < outer_loop; o++) {
                for (int k = 0; k < inner_loop; k++) {
                    float inv_norm = compute_quant_inv_norm_strided(output_ptr, len, inner_loop);
                    if (inv_norm > 0.0f) {
                        for (int i = 0; i < len; i++) {
                            float result = static_cast<float>(output_ptr[i * inner_loop]) * inv_norm;
                            tool::truncate(output_ptr[i * inner_loop], tool::round(result * output_scale));
                        }
                    } else {
                        for (int i = 0; i < len; i++) {
                            output_ptr[i * inner_loop] = 0;
                        }
                    }
                    output_ptr += 1;
                }
                output_ptr += inner_loop * (len - 1);
            }
        }
    }

    void forward_int16(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        int16_t *input_ptr = input->get_element_ptr<int16_t>();
        int16_t *output_ptr = output->get_element_ptr<int16_t>();

        std::vector<int> &shape = input->shape;
        int dims = shape.size();
        if (dims == 0) {
            output_ptr[0] = 0;
            return;
        }

        int positive_axis = m_axis < 0 ? dims + m_axis : m_axis;
        int len = shape[positive_axis];

        // input_scale cancels out: output_float = x_int / norm_int
        // only output_scale is needed for requantization
        float output_scale = DL_RESCALE(output->exponent);

        if (input_ptr != output_ptr) {
            for (size_t i = 0; i < input->size; i++) {
                output_ptr[i] = input_ptr[i];
            }
        }

        if (positive_axis == dims - 1) {
            int outer_loop = input->size / len;
            for (int o = 0; o < outer_loop; o++) {
                float inv_norm = compute_quant_inv_norm(output_ptr, len);
                inv_norm *= output_scale;

                if (inv_norm > 0.0f) {
                    for (int i = 0; i < len; i++) {
                        float result = output_ptr[i] * inv_norm;
                        tool::truncate(output_ptr[i], tool::round(result));
                    }
                } else {
                    for (int i = 0; i < len; i++) {
                        output_ptr[i] = 0;
                    }
                }
                output_ptr += len;
            }
        } else {
            int outer_loop = 1;
            int inner_loop = 1;
            get_outer_inner_loops(shape, dims, positive_axis, outer_loop, inner_loop);

            for (int o = 0; o < outer_loop; o++) {
                for (int k = 0; k < inner_loop; k++) {
                    float inv_norm = compute_quant_inv_norm_strided(output_ptr, len, inner_loop);
                    if (inv_norm > 0.0f) {
                        for (int i = 0; i < len; i++) {
                            float result = static_cast<float>(output_ptr[i * inner_loop]) * inv_norm;
                            tool::truncate(output_ptr[i * inner_loop], tool::round(result * output_scale));
                        }
                    } else {
                        for (int i = 0; i < len; i++) {
                            output_ptr[i * inner_loop] = 0;
                        }
                    }
                    output_ptr += 1;
                }
                output_ptr += inner_loop * (len - 1);
            }
        }
    }

    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;

        int axis = -1;
        int p = 2;
        quant_type_t quant_type;

        fbs_model->get_operation_attribute(node_name, "axis", axis);
        fbs_model->get_operation_attribute(node_name, "p", p);
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        op = new LpNormalization(axis, p, node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }

    void print() override
    {
        ESP_LOGI("LpNormalization", "axis: %d, p: %d, quant_type: %s.", m_axis, m_p, quant_type_to_string(quant_type));
    }

private:
    static void get_outer_inner_loops(
        const std::vector<int> &shape, int dims, int positive_axis, int &outer_loop, int &inner_loop)
    {
        outer_loop = 1;
        inner_loop = 1;
        for (int i = 0; i < dims; i++) {
            if (i < positive_axis) {
                outer_loop *= shape[i];
            } else if (i > positive_axis) {
                inner_loop *= shape[i];
            }
        }
    }

    float compute_inv_norm(float *data, int len)
    {
        if (m_p == 1) {
            float sum_abs = 0.0f;
            for (int i = 0; i < len; i++) {
                sum_abs += fabsf(data[i]);
            }
            return sum_abs > 0.0f ? 1.0f / sum_abs : 0.0f;
        } else {
            float sum_sq = dl::base::reduce_l2(data, len, 1);
            return sum_sq > 0.0f ? math::fast_inv_sqrt(sum_sq) : 0.0f;
        }
    }

    float compute_inv_norm_strided(float *data, int len, int stride)
    {
        if (m_p == 1) {
            float sum_abs = 0.0f;
            for (int i = 0; i < len; i++) {
                sum_abs += fabsf(data[i * stride]);
            }
            return sum_abs > 0.0f ? 1.0f / sum_abs : 0.0f;
        } else {
            float sum_sq = dl::base::reduce_l2(data, len, stride);
            return sum_sq > 0.0f ? math::fast_inv_sqrt(sum_sq) : 0.0f;
        }
    }

    template <typename U>
    float compute_quant_inv_norm(U *data, int len)
    {
        if (m_p == 1) {
            float sum_abs = 0.0f;
            for (int i = 0; i < len; i++) {
                sum_abs += fabsf(static_cast<float>(data[i]));
            }
            return sum_abs > 0.0f ? 1.0f / sum_abs : 0.0f;
        } else {
            float sum_sq = static_cast<float>(dl::base::reduce_l2(data, len, 1));
            return sum_sq > 0.0f ? math::fast_inv_sqrt(sum_sq) : 0.0f;
        }
    }

    template <typename U>
    float compute_quant_inv_norm_strided(U *data, int len, int stride)
    {
        if (m_p == 1) {
            float sum_abs = 0.0f;
            for (int i = 0; i < len; i++) {
                sum_abs += fabsf(static_cast<float>(data[i * stride]));
            }
            return sum_abs > 0.0f ? 1.0f / sum_abs : 0.0f;
        } else {
            float sum_sq = static_cast<float>(dl::base::reduce_l2(data, len, stride));
            return sum_sq > 0.0f ? math::fast_inv_sqrt(sum_sq) : 0.0f;
        }
    }
};

} // namespace module
} // namespace dl
