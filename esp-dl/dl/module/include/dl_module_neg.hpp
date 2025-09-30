#pragma once

#include "dl_module_base.hpp"
#include "dl_tool.hpp"

namespace dl {
namespace module {

/**
 * @brief: Neg takes one input data (Tensor) and produces one output data (Tensor)
 *         where each element flipped sign, y = -x, is applied to the tensor elementwise.
 *         Please refer to https://onnx.ai/onnx/operators/onnx__Neg.html for more details
 *
 */
class Neg : public Module {
public:
    /**
     * @brief Construct a new Neg object.
     *
     * @param name              name of module
     * @param inplace           inplace type.
     * @param quant_type        quantize type.
     */
    Neg(const char *name = NULL,
        module_inplace_t inplace = MODULE_NON_INPLACE,
        quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
    }

    /**
     * @brief Destroy the Neg object.
     */
    ~Neg() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        // Output shape is the same as input shape
        return {input_shapes[0]};
    }

    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_int8(input, output);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_int16(input, output);
        } else {
            forward_float(input, output);
        }
    }

    void forward_args(void *args) {}

    void forward_int8(TensorBase *input, TensorBase *output)
    {
        int8_t *input_ptr = input->get_element_ptr<int8_t>();
        int8_t *output_ptr = output->get_element_ptr<int8_t>();
        size_t size = input->get_size();

        for (size_t i = 0; i < size; i++) {
            if (input_ptr[i] == INT8_MIN) {
                output_ptr[i] = INT8_MAX;
            } else {
                output_ptr[i] = -input_ptr[i];
            }
        }
    }

    void forward_int16(TensorBase *input, TensorBase *output)
    {
        int16_t *input_ptr = input->get_element_ptr<int16_t>();
        int16_t *output_ptr = output->get_element_ptr<int16_t>();
        size_t size = input->get_size();

        for (size_t i = 0; i < size; i++) {
            if (input_ptr[i] == INT16_MIN) {
                output_ptr[i] = INT16_MAX;
            } else {
                output_ptr[i] = -input_ptr[i];
            }
        }
    }

    void forward_float(TensorBase *input, TensorBase *output)
    {
        float *input_ptr = input->get_element_ptr<float>();
        float *output_ptr = output->get_element_ptr<float>();
        size_t size = input->get_size();

        for (size_t i = 0; i < size; i++) {
            output_ptr[i] = -input_ptr[i];
        }
    }

    /**
     * @brief deserialize Neg module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        // Create module
        op = new Neg(node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }

    void print() { ESP_LOGI("Neg", "quant_type: %s.", quant_type_to_string(quant_type)); }
};

} // namespace module
} // namespace dl
