#pragma once

#include "dl_module_base.hpp"

namespace dl {
namespace module {
/**
 * NOTE:
 *
 * @tparam feature_t supports int16_t and int8_t,
 *         - int16_t: stands for operation in int16_t quantize
 *         - int8_t: stands for operation in int8_t quantize
 */
class LogSoftmax : public Module {
private:
    int m_axis;
    float *m_exp_table;

public:
    /**
     * @brief Construct a new LogSoftmax object.
     *
     * @param name            name of module
     * @param axis            the axis along which LogSoftmax will be computed. Accepted range is [-r, r-1] where r =
     * rank(input).
     * @param inplace         inplace type.
     * @param quant_type      quantization type.
     */
    LogSoftmax(const char *name = NULL,
               int axis = -1,
               module_inplace_t inplace = MODULE_NON_INPLACE,
               quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_axis(axis)
    {
        m_exp_table = nullptr;
    }

    /**
     * @brief Destroy the LogSoftmax object.
     */
    ~LogSoftmax()
    {
        if (m_exp_table != nullptr) {
            free(m_exp_table);
        }
    }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        return {input_shapes[0]};
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_lut(input, output);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            int16_t *input_element = (int16_t *)input->get_element_ptr();
            float *output_element = (float *)output->get_element_ptr();

            float scale = DL_SCALE(input->exponent);
            for (int i = 0; i < input->get_size(); i++) {
                output_element[i] = scale * input_element[i];
            }
            forward_float(output_element, output->get_size(), output->get_shape(), m_axis);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            float *input_element = (float *)input->get_element_ptr();
            float *output_element = (float *)output->get_element_ptr();

            if (input_element != output_element) {
                memcpy(output_element, input_element, input->get_bytes());
            }
            forward_float(output_element, output->get_size(), output->get_shape(), m_axis);
        }
    }

    /**
     * @brief LogSoftmax float implementation.
     *        LogSoftmax(x_i) = (x_i - max) - log(sum(exp(x_j - max)))
     */
    void forward_float(float *output_element, int size, std::vector<int> shape, int axis)
    {
        int dims = shape.size();
        int positive_axis = axis < 0 ? dims + axis : axis;
        int len = shape[positive_axis];

        if (positive_axis == dims - 1) {
            int outer_loop = 1;
            if (dims > 1) {
                outer_loop = size / len;
            }

            for (int o = 0; o < outer_loop; o++) {
                float max = output_element[0];
                for (int i = 1; i < len; i++) {
                    max = DL_MAX(max, output_element[i]);
                }

                float sum = 0.f;
                for (int i = 0; i < len; i++) {
                    sum += expf(output_element[i] - max);
                }
                float log_sum = logf(sum);

                for (int i = 0; i < len; i++) {
                    output_element[i] = output_element[i] - max - log_sum;
                }
                output_element += len;
            }
        } else {
            int outer_loop = 1;
            int inner_loop = 1;
            for (int i = 0; i < dims; i++) {
                if (i < positive_axis) {
                    outer_loop *= shape[i];
                } else if (i > positive_axis) {
                    inner_loop *= shape[i];
                }
            }

            for (int o = 0; o < outer_loop; o++) {
                for (int k = 0; k < inner_loop; k++) {
                    float max = output_element[0];
                    for (int i = 1; i < len; i++) {
                        max = DL_MAX(max, output_element[i * inner_loop]);
                    }

                    float sum = 0.f;
                    for (int i = 0; i < len; i++) {
                        sum += expf(output_element[i * inner_loop] - max);
                    }
                    float log_sum = logf(sum);

                    for (int i = 0; i < len; i++) {
                        output_element[i * inner_loop] = output_element[i * inner_loop] - max - log_sum;
                    }
                    output_element += 1;
                }
                output_element += inner_loop * (len - 1);
            }
        }
    }

    /**
     * @brief LogSoftmax LUT implementation for int8 quantized input.
     *        Uses exp LUT to compute sum(exp(x)), then output = scale * x - log(sum).
     */
    void forward_lut(TensorBase *input, TensorBase *output)
    {
        if (m_exp_table == nullptr) {
            m_exp_table = (float *)tool::malloc_aligned(256 * sizeof(float), MALLOC_CAP_DEFAULT);
            tool::gen_lut_8bit(m_exp_table, input->exponent, expf);
        }

        int dims = input->get_shape().size();
        int positive_axis = m_axis < 0 ? dims + m_axis : m_axis;
        int len = input->get_shape()[positive_axis];
        int8_t *input_element = (int8_t *)input->get_element_ptr();
        assert(output->get_dtype() == DATA_TYPE_FLOAT);
        float *output_element = (float *)output->get_element_ptr();
        float scale = DL_SCALE(input->exponent);

        if (positive_axis == dims - 1) {
            int outer_loop = 1;
            if (dims > 1) {
                outer_loop = input->get_size() / len;
            }

            for (int o = 0; o < outer_loop; o++) {
                float sum = 0.f;
                for (int i = 0; i < len; i++) {
                    sum += m_exp_table[input_element[i] + 128];
                }
                float log_sum = logf(sum);

                for (int i = 0; i < len; i++) {
                    output_element[i] = scale * input_element[i] - log_sum;
                }
                input_element += len;
                output_element += len;
            }
        } else {
            int outer_loop = 1;
            int inner_loop = 1;
            for (int i = 0; i < dims; i++) {
                if (i < positive_axis) {
                    outer_loop *= input->get_shape()[i];
                } else if (i > positive_axis) {
                    inner_loop *= input->get_shape()[i];
                }
            }

            for (int o = 0; o < outer_loop; o++) {
                for (int k = 0; k < inner_loop; k++) {
                    float sum = 0.f;
                    for (int i = 0; i < len; i++) {
                        sum += m_exp_table[input_element[i * inner_loop] + 128];
                    }
                    float log_sum = logf(sum);

                    for (int i = 0; i < len; i++) {
                        output_element[i * inner_loop] = scale * input_element[i * inner_loop] - log_sum;
                    }
                    input_element += 1;
                    output_element += 1;
                }
                input_element += inner_loop * (len - 1);
                output_element += inner_loop * (len - 1);
            }
        }
    }

    /**
     * @brief deserialize LogSoftmax module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        int axis = -1;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "axis", axis);

        op = new LogSoftmax(node_name.c_str(), axis, MODULE_NON_INPLACE, quant_type);
        return op;
    }

    void print() { ESP_LOGI("LogSoftmax", "quant_type: %s. axis:%d", quant_type_to_string(quant_type), m_axis); }
};
} // namespace module
} // namespace dl
