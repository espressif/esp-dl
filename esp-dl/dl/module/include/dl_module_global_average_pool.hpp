#pragma once

#include "dl_base_avg_pool2d.hpp"
#include "dl_module_base.hpp"

namespace dl {
namespace module {
class GlobalAveragePool : public Module {
public:
    /**
     * @brief Construct a new GlobalAveragePool object.
     *
     * @param name            name of module
     */
    GlobalAveragePool(const char *name = NULL, quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, MODULE_NON_INPLACE, quant_type)
    {
    }

    /**
     * @brief Destroy the GlobalAveragePool object.
     */
    ~GlobalAveragePool() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        assert(input_shapes[0].size() == 3 || input_shapes[0].size() == 4);
        std::vector<int> input_shape = input_shapes[0];
        std::vector<int> output_shape(input_shape.size(), 1);
        if (input_shape.size() == 3) {
            output_shape[2] = input_shape[2];
        } else if (input_shape.size() == 4) {
            output_shape[3] = input_shape[3];
        }

        return {output_shape};
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
            base::avg_pool2d<int8_t>(args);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            base::avg_pool2d<int16_t>(args);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        std::vector<base::PoolArgsType<T>> m_args;

        if (input->shape.size() == 3) {
            m_args = base::get_pool_args<T>(output, input, {0, 0}, {input->shape[1]}, {1}, mode);
        } else if (input->shape.size() == 4) {
            m_args =
                base::get_pool_args<T>(output, input, {0, 0, 0, 0}, {input->shape[1], input->shape[2]}, {1, 1}, mode);
        }
        int task_size = m_args.size();
        if (task_size == 1) { // single task
            forward_args((void *)&m_args[0]);
        } else if (task_size == 2) { // multi task, use semaphore to maintain synchronization.
            module_forward_dual_core(this, (void *)&m_args[0], (void *)&m_args[1]);
        } else {
            ESP_LOGE("GlobalAveragePool", "Only support task size is 1 or 2, currently task size is %d", task_size);
        }
    }

    /**
     * @brief deserialize GlobalAveragePool module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        // Create module
        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            op = new GlobalAveragePool(node_name.c_str(), quant_type);
        }
        return op;
    }

    void print() { ESP_LOGI("GlobalAveragePool", "quant_type: %s.", quant_type_to_string(quant_type)); }
};
} // namespace module
} // namespace dl
