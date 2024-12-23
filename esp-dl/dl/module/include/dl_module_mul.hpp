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
 * @tparam feature_t supports int16_t and int8_t,
 *         - int16_t: stands for operation in int16_t quantize
 *         - int8_t: stands for operation in int8_t quantize
 */

class Mul : public Module {
private:
    TensorBase *m_input0_constant; /*<! input0 of MatMul. If matmul has a constant input0, the value is not NULL;
                                      otherwise, it is NULL >*/
    TensorBase *m_input1_constant; /*<! input1 of MatMul. If matmul has a constant input1, the value is not NULL;
                                      otherwise, it is NULL >*/

public:
    /**
     * @brief Construct a new Mul object.
     *
     * @param name            name of module
     * @param inplace         inplace type.
     */
    Mul(TensorBase *input0_constant,
        TensorBase *input1_constant,
        const char *name = NULL,
        module_inplace_t inplace = MODULE_NON_INPLACE,
        quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_input0_constant(input0_constant), m_input1_constant(input1_constant)
    {
    }

    /**
     * @brief Destroy the Mul object.
     */
    ~Mul()
    {
        if (m_input0_constant) {
            delete m_input0_constant;
            m_input0_constant = nullptr;
        }

        if (m_input1_constant) {
            delete m_input1_constant;
            m_input1_constant = nullptr;
        }
    }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<int> input0_shape;
        std::vector<int> input1_shape;
        if (input_shapes.size() == 2) {
            input0_shape = input_shapes[0];
            input1_shape = input_shapes[1];
        } else if (input_shapes.size() == 1 && m_input0_constant) {
            input0_shape = m_input0_constant->get_shape();
            input1_shape = input_shapes[0];
        } else if (input_shapes.size() == 1 && m_input1_constant) {
            input0_shape = input_shapes[0];
            input1_shape = m_input1_constant->get_shape();
        } else {
            ESP_LOGE("Mul", "%s, input is error.", __FUNCTION__);
            assert(false);
        }

        std::vector<int> output_shape = base::get_multidirectional_broadcasting_shape(input0_shape, input1_shape);

        return std::vector<std::vector<int>>(1, output_shape);
    }

    void forward(std::vector<dl::TensorBase *> &tensors, runtime_mode_t mode)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(tensors, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(tensors, mode);
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
    void forward_template(std::vector<TensorBase *> &tensors, runtime_mode_t mode)
    {
        TensorBase *input0 = nullptr;
        TensorBase *input1 = nullptr;

        if (m_inputs_index.size() == 2) {
            input0 = tensors[m_inputs_index[0]];
            input1 = tensors[m_inputs_index[1]];
        } else if (m_inputs_index.size() == 1 && m_input0_constant) {
            input0 = m_input0_constant;
            input1 = tensors[m_inputs_index[0]];
        } else if (m_inputs_index.size() == 1 && m_input1_constant) {
            input0 = tensors[m_inputs_index[0]];
            input1 = m_input1_constant;
        } else {
            ESP_LOGE("Mul", "%s, input is error.", __FUNCTION__);
        }

        TensorBase *output = tensors[m_outputs_index[0]];

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
            TensorBase *input0_constant = fbs_model->get_operation_parameter(node_name, 0);
            TensorBase *input1_constant = fbs_model->get_operation_parameter(node_name, 1);
            op = new Mul(input0_constant, input1_constant, NULL, MODULE_NON_INPLACE, quant_type);
        }
        return op;
    }

    void print()
    {
        ESP_LOGI("Mul",
                 "quant_type: %s, m_input0_constant: %p, m_input1_constant: %p.",
                 quant_type_to_string(quant_type),
                 m_input0_constant,
                 m_input1_constant);
    }
};

} // namespace module
} // namespace dl
