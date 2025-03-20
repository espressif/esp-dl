#pragma once

#include "dl_module_base.hpp"
#include "dl_module_lut.hpp"

namespace dl {
namespace module {

// https://onnx.ai/onnx/operators/onnx__Elu.html
class Elu : public Module {
private:
    float m_alpha;

public:
    /**
     * @brief Construct a new Elu object.
     *
     * @param name          name of module
     * @param alpha         coefficient of Elu.
     * @param inplace       inplace type.
     * @param quant_type    quant type.
     */
    Elu(const char *name = NULL,
        float alpha = 0.01,
        module_inplace_t inplace = MODULE_NON_INPLACE,
        quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
        m_alpha = alpha;
    }

    /**
     * @brief Destroy the Elu object.
     */
    ~Elu() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<std::vector<int>> output_shapes(1, input_shapes[0]);
        return output_shapes;
    }

    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            TensorBase *input = context->get_tensor(m_inputs_index[0]);
            TensorBase *output = context->get_tensor(m_outputs_index[0]);
            float *input_ptr = (float *)input->get_element_ptr();
            float *output_ptr = (float *)output->get_element_ptr();

            for (size_t i = 0; i < input->size; i++) {
                if (input_ptr[i] < 0) {
                    output_ptr[i] = m_alpha * (expf(input_ptr[i]) - 1);
                } else {
                    output_ptr[i] = input_ptr[i];
                }
            }
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        T *input_ptr = (T *)input->get_element_ptr();
        T *output_ptr = (T *)output->get_element_ptr();

        float input_scale = DL_SCALE(input->exponent);
        float output_scale = DL_RESCALE(output->exponent);
        for (size_t i = 0; i < input->size; i++) {
            float temp = input_ptr[i] * input_scale;
            if (temp >= 0) {
                tool::truncate(output_ptr[i], tool::round(temp * output_scale));
            } else {
                tool::truncate(output_ptr[i], tool::round(m_alpha * (expf(temp) - 1) * output_scale));
            }
        }
    }

    /**
     * @brief deserialize Elu module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        float alpha = 1.0;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "alpha", alpha);

        // Create module
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            TensorBase *table = fbs_model->get_operation_lut(node_name);
            if (table) {
                op = new LUT(node_name.c_str(), table, MODULE_INPLACE_CHANGED_BUFFER, quant_type);
            } else {
                op = new Elu(node_name.c_str(), alpha, MODULE_INPLACE_CHANGED_BUFFER, quant_type);
            }
        } else {
            op = new Elu(node_name.c_str(), alpha, MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        }

        return op;
    }

    void print() { ESP_LOGI("Elu", "quant_type: %s, alpha: %f.", quant_type_to_string(quant_type), m_alpha); }
};
} // namespace module
} // namespace dl
