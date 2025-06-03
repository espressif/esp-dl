#pragma once

#include "dl_base_max_pool2d.hpp"
#include "dl_module_base.hpp"

namespace dl {
namespace module {
class MaxPool : public Module {
private:
    std::vector<int> m_filter_shape; /*!< filter shape in [height, width] */
    std::vector<int> m_padding;      /*!< padding size needed in [top, bottom, left, right] of this operation */
    std::vector<int> m_strides;      /*!< stride along each spatial axis. [height, width] */
public:
    /**
     * @brief Construct a new MaxPool object.
     *
     * @param name            name of module
     * @param filter_shape    filter shape in [height, width]
     * @param padding         padding size needed in [top, bottom, left, right] of this operation
     * @param strides         stride along each spatial axis. [height, width]
     */
    MaxPool(const char *name = NULL,
            const std::vector<int> &filter_shape = {2, 2},
            const std::vector<int> &padding = {},
            const std::vector<int> &strides = {1, 1},
            quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, MODULE_NON_INPLACE, quant_type),
        m_filter_shape(filter_shape),
        m_padding(padding),
        m_strides(strides)
    {
    }

    /**
     * @brief Destroy the MaxPool object.
     */
    ~MaxPool() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        assert(input_shapes[0].size() == 3 || input_shapes[0].size() == 4);
        std::vector<int> input_shape = input_shapes[0];
        std::vector<int> output_shape = input_shape;

        output_shape[1] = (input_shape[1] + m_padding[0] + m_padding[1] - m_filter_shape[0]) / m_strides[0] + 1;
        if (input_shape.size() == 3) {
            output_shape[2] = input_shape[2];
        } else if (input_shape.size() == 4) {
            output_shape[2] = (input_shape[2] + m_padding[2] + m_padding[3] - m_filter_shape[1]) / m_strides[1] + 1;
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
            base::max_pool2d<int8_t>(args);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            base::max_pool2d<int16_t>(args);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        std::vector<base::PoolArgsType<T>> m_args =
            base::get_pool_args<T>(output, input, m_padding, m_filter_shape, m_strides, mode);
        int task_size = m_args.size();
        if (task_size == 1) { // single task
            forward_args((void *)&m_args[0]);
        } else if (task_size == 2) { // multi task, use semaphore to maintain synchronization.
            module_forward_dual_core(this, (void *)&m_args[0], (void *)&m_args[1]);
        } else {
            ESP_LOGE("MaxPool", "Only support task size is 1 or 2, currently task size is %d", task_size);
        }
    }

    /**
     * @brief deserialize MaxPool module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        std::vector<int> kernel_shape;
        std::vector<int> pads;
        std::vector<int> strides;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "kernel_shape", kernel_shape);
        fbs_model->get_operation_attribute(node_name, "pads", pads);
        fbs_model->get_operation_attribute(node_name, "strides", strides);
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        // Create module
        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            if (pads.size() == 4) {
                pads = {pads[0], pads[2], pads[1], pads[3]};
            }

            op = new MaxPool(node_name.c_str(), kernel_shape, pads, strides, quant_type);
        }
        return op;
    }

    void print()
    {
        ESP_LOGI("MaxPool",
                 "quant_type: %s, kernel size: %s, pads size: %s, strides size: %s",
                 quant_type_to_string(quant_type),
                 vector_to_string(m_filter_shape).c_str(),
                 vector_to_string(m_padding).c_str(),
                 vector_to_string(m_strides).c_str());
    }
};
} // namespace module
} // namespace dl
