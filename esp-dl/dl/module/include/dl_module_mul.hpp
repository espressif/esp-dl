#pragma once
#include "dl_base_mul.hpp"
#include "dl_base_shape.hpp"
#include "dl_module_base.hpp"

namespace dl {
namespace module {
/**
 * @brief: Performs element-wise binary subtraction (with Numpy-style broadcasting support).
 *         Please refer to https://onnx.ai/onnx/operators/onnx__Mul.html for more details
 *
 */

class Mul : public Module {
public:
    /**
     * @brief Construct a new Mul object.
     *
     * @param name              name of module
     * @param inplace           inplace type.
     * @param quant_type        quantize type.
     */
    Mul(const char *name = NULL,
        module_inplace_t inplace = MODULE_NON_INPLACE,
        quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
    }

    /**
     * @brief Destroy the Mul object.
     */
    ~Mul() {}

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
            base::elemwise_mul((base::elemwiseArgsType<int8_t> *)args);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            base::elemwise_mul((base::elemwiseArgsType<int16_t> *)args);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input0 = context->get_tensor(m_inputs_index[0]);
        TensorBase *input1 = context->get_tensor(m_inputs_index[1]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        std::vector<base::elemwiseArgsType<T>> m_args =
            base::get_elemwise_operation_args<T>(output, input0, input1, mode); // get element-wise operation args
        int task_size = m_args.size();
        if (task_size == 1) {
            forward_args((void *)&m_args[0]); // single task
        } else if (task_size == 2) {
            module_forward_dual_core(this, (void *)&m_args[0], (void *)&m_args[1]); // two task
        } else {
            ESP_LOGE("Mul", "Only support task size is 1 or 2, currently task size is %d", task_size);
        }
    }

    /**
     * @brief deserialize Mul module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            op = new Mul(node_name.c_str(), MODULE_NON_INPLACE, quant_type);
        }
        return op;
    }

    void print()
    {
        ESP_LOGI("Mul",
                 "quant_type: %s, input feature map size: %d.",
                 quant_type_to_string(quant_type),
                 m_inputs_index.size());
    }
};

} // namespace module
} // namespace dl
