#pragma once

#include "dl_module_base.hpp"

namespace dl {
namespace module {

class RequantizeLinear : public Module {
public:
    /**
     * @brief Construct a new RequantizeLinear object.
     *
     * @param name            name of module
     * @param inplace         inplace type.
     */
    RequantizeLinear(const char *name = NULL,
                     module_inplace_t inplace = MODULE_NON_INPLACE,
                     quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
    }

    /**
     * @brief Destroy the RequantizeLinear object.
     */
    ~RequantizeLinear() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<std::vector<int>> output_shapes(1, input_shapes[0]);
        return output_shapes;
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        assert(input->get_size() == output->get_size());
        output->assign(input);
    }

    void forward_args(void *args) {}

    /**
     * @brief deserialize RequantizeLinear module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        // Create module
        op = new RequantizeLinear(node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }

    void print() { ESP_LOGI("RequantizeLinear", "quant_type: %s.", quant_type_to_string(quant_type)); }
};
} // namespace module
} // namespace dl
