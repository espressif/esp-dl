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
 * @brief Activation(Gemm(input, filter) + bias).
 *
 */
class Gemm : public Module {
private:
    activation_type_t activation; /*!< activation of Gemm, if you don't specify anything, no activation is applied */
    bool is_bias_reseted;

    void reset_bias(ModelContext *context)
    {
        if (is_bias_reseted == false) {
            if (m_inputs_index.size() == 3) {
                TensorBase *bias = context->get_tensor(m_inputs_index[2]);
                if (bias) {
                    bias->reset_bias_layout(quant_type, false);
                }
            }
            is_bias_reseted = true;
        }
    }

public:
    /**
     * @brief Construct a new Gemm object.
     *
     * @param activation      activation of Gemm, if you don't specify anything, no activation is applied
     * @param name            name of module
     */
    Gemm(activation_type_t activation = Linear, const char *name = nullptr, quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, MODULE_NON_INPLACE, quant_type), activation(activation)
    {
        is_bias_reseted = false;
    }

    /**
     * @brief Destroy the Gemm object.
     *
     */
    ~Gemm() {}

    /**
     * @brief Calculate the output shape
     *
     * @param input_shape The shape of inputs
     *
     * @return output shape
     */
    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<int> input0_shape = input_shapes[0];
        std::vector<int> filter_shape = input_shapes[1];
        assert(filter_shape.size() == 4);
        assert(filter_shape[0] == 1);
        assert(filter_shape[1] == 1);
        assert(input0_shape[input0_shape.size() - 1] == filter_shape[2]);

        // refer to https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        std::vector<int> output_shape = input_shapes[0];
        output_shape[output_shape.size() - 1] = filter_shape[3];
        return {output_shape};
    }

    void forward_args(void *args)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            base::conv2d<int8_t, int32_t, int32_t>(args);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            base::conv2d<int16_t, int32_t, int64_t>(args);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        std::vector<int> padding(4, 0);
        TensorBase *input0 = context->get_tensor(m_inputs_index[0]);
        TensorBase *filter = context->get_tensor(m_inputs_index[1]);
        TensorBase *bias = nullptr;
        if (m_inputs_index.size() == 3) {
            bias = context->get_tensor(m_inputs_index[2]);
        }
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        std::vector<int> origin_input_shape = input0->get_shape();
        std::vector<int> origin_output_shape = output->get_shape();
        input0->set_shape({1, 1, input0->get_size() / origin_input_shape.back(), origin_input_shape.back()});
        output->set_shape({1, 1, output->get_size() / origin_output_shape.back(), origin_output_shape.back()});

        std::vector<base::ArgsType<T>> m_args =
            base::get_conv_operation_args<T>(output,
                                             input0,
                                             padding,
                                             filter,
                                             {1, 1} /*strides*/,
                                             {1, 1} /*dilations*/,
                                             1 /*group*/,
                                             bias,
                                             this->activation,
                                             nullptr,
                                             mode); // do not support PReLU and Leaky RelU
        int task_size = m_args.size();
        if (task_size == 1) { // single task
            forward_args((void *)&m_args[0]);
        } else if (task_size == 2) { // multi task, use semaphore to maintain synchronization.
            ESP_LOGI("Gemm", "two task...");
            module_forward_dual_core(this, (void *)&m_args[0], (void *)&m_args[1]);
        } else {
            ESP_LOGE("Gemm", "Only support task size is 1 or 2, currently task size is %d", task_size);
        }
        input0->set_shape(origin_input_shape);
        output->set_shape(origin_output_shape);
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

    /**
     * @brief deserialize Conv2d module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *gemm_op = nullptr;

        activation_type_t activation_type;
        quant_type_t quant_type;
        int transA = -1, transB = -1;
        fbs_model->get_operation_attribute(node_name, "activation", activation_type);
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "transA", transA);
        fbs_model->get_operation_attribute(node_name, "transB", transB);
        assert(transA == -1 || transA == 0);
        assert(transB == -1 || transB == 0);

        // Create module
        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            gemm_op = new Gemm(activation_type, node_name.c_str(), quant_type);
        }

        return gemm_op;
    }

    void print()
    {
        ESP_LOGI("Gemm",
                 "activation: %s, quant_type: %s.",
                 activation_type_to_string(activation),
                 quant_type_to_string(quant_type));
    }
};
} // namespace module
} // namespace dl
