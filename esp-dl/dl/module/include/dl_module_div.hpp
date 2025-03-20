#pragma once
#include "dl_base_div.hpp"
#include "dl_base_shape.hpp"
#include "dl_module_base.hpp"

namespace dl {
namespace module {
/**
 * @brief: Performs element-wise binary subtraction (with Numpy-style broadcasting support).
 *         Please refer to https://onnx.ai/onnx/operators/onnx__Div.html for more details
 *
 */

class Div : public Module {
private:
    void *m_args;

public:
    /**
     * @brief Construct a new Div object.
     *
     * @param name              name of module
     * @param inplace           inplace type.
     * @param quant_type        quantize type.
     * @param inputs_constant   if there are constant inputs, pass them through this parameter.
     */
    Div(const char *name = NULL,
        module_inplace_t inplace = MODULE_NON_INPLACE,
        quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
        m_args = nullptr;
    }

    /**
     * @brief Destroy the Div object.
     */
    ~Div()
    {
        if (m_args) {
            if (quant_type == QUANT_TYPE_SYMM_8BIT) {
                base::elemwiseArgsType<int8_t> *args = (base::elemwiseArgsType<int8_t> *)m_args;
                if (args->table) {
                    free(args->table);
                }
                free(args);
            } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
                base::elemwiseArgsType<int16_t> *args = (base::elemwiseArgsType<int16_t> *)m_args;
                if (args->table) {
                    free(args->table);
                }
                free(args);
            }
        }
    }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<int> output_shape = base::get_multidirectional_broadcasting_shape(input_shapes[0], input_shapes[1]);

        return std::vector<std::vector<int>>(1, output_shape);
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        }
    }

    void forward_args(void *args)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            base::elemwise_div((base::elemwiseArgsType<int8_t> *)args);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            base::elemwise_div((base::elemwiseArgsType<int16_t> *)args);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input0 = context->get_tensor(m_inputs_index[0]);
        TensorBase *input1 = context->get_tensor(m_inputs_index[1]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        if (m_args) {
            forward_args(m_args);
        } else {
            m_args =
                (void *)base::get_elemwise_div_args<T>(output, input0, input1, mode); // get element-wise operation args
            forward_args(m_args);
        }
    }

    /**
     * @brief deserialize Div module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        //
        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            op = new Div(node_name.c_str(), MODULE_NON_INPLACE, quant_type);
        }
        return op;
    }

    void print()
    {
        ESP_LOGI("Div",
                 "quant_type: %s, input feature map size: %d.",
                 quant_type_to_string(quant_type),
                 m_inputs_index.size());
    }
};

} // namespace module
} // namespace dl
