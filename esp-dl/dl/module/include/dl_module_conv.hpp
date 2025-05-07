#pragma once

#include "dl_base_conv2d.hpp"
#include "dl_base_depthwise_conv2d.hpp"
#include "dl_module_base.hpp"
#include <typeinfo>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace dl {
namespace module {

/**
 * @brief Activation(Conv(input, filter) + bias).
 *
 */
class Conv : public Module {
private:
    std::vector<int> m_dilations;
    std::vector<int> m_strides;
    const int m_group;
    activation_type_t activation; /*!< activation of Conv, if you don't specify anything, no activation is applied */
    std::vector<int> m_pads;      /*!< pads size needed in [top, bottom, left, right] of this operation */
    bool is_bias_reseted;

    void reset_bias(ModelContext *context)
    {
        if (is_bias_reseted == false) {
            if (m_inputs_index.size() == 3) {
                TensorBase *bias = context->get_tensor(m_inputs_index[2]);
                if (bias) {
                    bias->reset_bias_layout(quant_type, m_group != 1);
                }
            }
            is_bias_reseted = true;
        }
    }

public:
    /**
     * @brief Construct a new Conv object.
     *
     * @param activation      activation of Conv, if you don't specify anything, no activation is applied
     * @param pads            the shape must be 2 or 4, the value of each position is: [pads top, pads bottom, pads
     * left, pads right]
     * @param dilations       dilation value along each spatial axis of the filter
     * @param strides         stride along each spatial axis
     * @param group           group of Conv
     * @param name            name of module
     */
    Conv(activation_type_t activation = Linear,
         std::vector<int> pads = {},
         std::vector<int> dilations = {},
         std::vector<int> strides = {},
         const char *name = NULL,
         const int group = 1,
         quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, MODULE_NON_INPLACE, quant_type),
        m_dilations(dilations),
        m_strides(strides),
        m_group(group),
        activation(activation),
        m_pads(pads)
    {
        is_bias_reseted = false;
    }

    /**
     * @brief Destroy the Conv object.
     *
     */
    ~Conv() {}

    /**
     * @brief Calculate the output shape
     *
     * @param input_shape The shape of inputs
     *
     * @return output shape
     */
    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        assert(input_shapes.size() >= 2);
        assert(input_shapes[0].size() == 3 || input_shapes[0].size() == 4);
        std::vector<int> input_shape = input_shapes[0];
        std::vector<int> filter_shape = input_shapes[1];
        std::vector<int> output_shape = input_shape;

        // refer to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        output_shape[1] =
            (input_shape[1] + m_pads[0] + m_pads[1] - m_dilations[0] * (filter_shape[0] - 1) - 1) / m_strides[0] + 1;
        if (input_shape.size() == 3) {
            output_shape[2] = m_group == 1 ? filter_shape[2] : input_shape[2];
        } else if (input_shape.size() == 4) {
            output_shape[2] =
                (input_shape[2] + m_pads[2] + m_pads[3] - m_dilations[1] * (filter_shape[1] - 1) - 1) / m_strides[1] +
                1;
            output_shape[3] = m_group == 1 ? filter_shape[3] : input_shape[3];
        }

        return {output_shape};
    }

    void forward_args(void *args)
    {
        if (m_group == 1) {
            if (quant_type == QUANT_TYPE_SYMM_8BIT) {
                base::conv2d<int8_t, int32_t, int32_t>(args);
            } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
                base::conv2d<int16_t, int32_t, int64_t>(args);
            }
        } else {
            if (quant_type == QUANT_TYPE_SYMM_8BIT) {
                base::depthwise_conv2d<int8_t, int32_t, int32_t>(args);
            } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
                base::depthwise_conv2d<int16_t, int32_t, int64_t>(args);
            }
        }
    }

    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO)
    {
        reset_bias(context);

        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *filter = context->get_tensor(m_inputs_index[1]);
        TensorBase *bias = nullptr;
        if (m_inputs_index.size() == 3) {
            bias = context->get_tensor(m_inputs_index[2]);
        }
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        std::vector<base::ArgsType<T>> m_args =
            base::get_conv_operation_args<T>(output,
                                             input,
                                             m_pads,
                                             filter,
                                             m_strides,
                                             m_dilations,
                                             m_group,
                                             bias,
                                             this->activation,
                                             nullptr,
                                             mode); // do not support RReLU and Leaky RelU
        int task_size = m_args.size();
        if (task_size == 1) { // single task
            forward_args((void *)&m_args[0]);
        } else if (task_size == 2) { // multi task, use semaphore to maintain synchronization.
            module_forward_dual_core(this, (void *)&m_args[0], (void *)&m_args[1]);
        } else {
            ESP_LOGE("Conv", "Only support task size is 1 or 2, currently task size is %d", task_size);
        }
    }

    /**
     * @brief deserialize Conv module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *conv_op = nullptr;

        std::vector<int> pads;
        std::vector<int> strides;
        std::vector<int> dilations;
        int group = 1;
        activation_type_t activation_type;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "pads", pads);
        fbs_model->get_operation_attribute(node_name, "strides", strides);
        fbs_model->get_operation_attribute(node_name, "dilations", dilations);
        fbs_model->get_operation_attribute(node_name, "group", group);
        fbs_model->get_operation_attribute(node_name, "activation", activation_type);
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        // Create module
        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            if (pads.size() == 4) {
                pads = {pads[0], pads[2], pads[1], pads[3]};
            }
            conv_op = new Conv(activation_type, pads, dilations, strides, node_name.c_str(), group, quant_type);
        }

        return conv_op;
    }

    void print()
    {
        ESP_LOGI("Conv",
                 "pads: %s, strides: %s, dilations: %s, group: %d, activation: %s, "
                 "quant_type: %s.",
                 vector_to_string(m_pads).c_str(),
                 vector_to_string(m_strides).c_str(),
                 vector_to_string(m_dilations).c_str(),
                 m_group,
                 activation_type_to_string(activation),
                 quant_type_to_string(quant_type));
    }

    // void set_preload_addr(void *addr, size_t size)
    // {
    //     size_t offset = 0;
    //     if (this->filter) {
    //         offset = this->filter->set_preload_addr(addr, size);
    //     }
    //     if (this->bias) {
    //         this->bias->set_preload_addr((void *)((char *)addr + offset), size - offset);
    //     }
    // }

    // void preload()
    // {
    //     // printf("preload filter and bias!");
    //     if (filter)
    //         filter->preload();
    //     if (bias)
    //         bias->preload();
    // }

    // void reset()
    // {
    //     this->m_inputs_index.clear();
    //     this->m_outputs_index.clear();
    //     this->filter->cache = nullptr;
    //     if (this->bias != nullptr) {
    //         this->bias->cache = nullptr;
    //     }
    // }
};
} // namespace module
} // namespace dl
