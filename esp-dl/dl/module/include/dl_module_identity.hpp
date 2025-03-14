#pragma once

#include "dl_module_base.hpp"

namespace dl {
namespace module {

/**
 * @brief: Identity operator
 *         Supports float, int16_t and int8_t
 */
class Identity : public Module {
public:
    /**
     * @brief Construct a new Identity object.
     *
     * @param name            name of module
     * @param inplace         inplace type.
     */
    Identity(const char *name = NULL,
             module_inplace_t inplace = MODULE_NON_INPLACE,
             quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
    }

    /**
     * @brief Destroy the Identity object.
     */
    ~Identity() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        return {input_shapes[0]};
    }

    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        assert(input->get_size() == output->get_size());
        if (output->get_element_ptr() != input->get_element_ptr()) {
            output->assign(input);
        }
    }

    void forward_args(void *args) {}

    /**
     * @brief deserialize Identity module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        // Create module
        op = new Identity(node_name.c_str(), MODULE_INPLACE_UNCHANGED_BUFFER, quant_type);
        return op;
    }

    void print() { ESP_LOGI("Identity", "quant_type: %s.", quant_type_to_string(quant_type)); }
};
} // namespace module
} // namespace dl
