#pragma once

#include "dl_base_conv2d.hpp"
#include "dl_base_conv_args.hpp"
#include "dl_base_conv_select.hpp"
#include "dl_base_depthwise_conv2d.hpp"
#include "dl_module_base.hpp"
#include <string>
#include <typeinfo>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <climits>
#include <cstdint>

namespace dl {
namespace module {

/**
 * MAC floor for RUNTIME_MODE_AUTO width splitting of a batched Gemm (B*K*N).
 *
 * Measured with -O3 -ffast-math on ESP32-S3 rev2 at 240 MHz and ESP32-P4 rev103
 * at 360 MHz, ESP-IDF 5.5.6. Shapes below the floor stay on one core.
 * MULTI_CORE keeps the conv floor. A single-core build, any other target, and
 * any quant type other than int8, int16, and w8a16 return INT64_MAX.
 */
inline int64_t gemm_auto_width_min_macs(quant_type_t quant)
{
#if CONFIG_FREERTOS_NUMBER_OF_CORES < 2
    (void)quant;
    return INT64_MAX;
#elif CONFIG_IDF_TARGET_ESP32S3
    // int8: 524288 still regresses (32x256x64, 0.957). 589824 wins (24x128x192, 1.126).
    // int16: 245760 is still a small loss. 262144 no longer regresses.
    // w8a16: 65536 regresses. 98304 is the first bucket whose bands all stay faster.
    if (quant == QUANT_TYPE_SYMM_8BIT) {
        return 576 * 1024;
    }
    if (quant == QUANT_TYPE_SYMM_W8A16) {
        return 96 * 1024;
    }
    if (quant == QUANT_TYPE_SYMM_16BIT) {
        return 256 * 1024;
    }
    return INT64_MAX;
#elif CONFIG_IDF_TARGET_ESP32P4
    // int8: through 655360 the split is noise or a regression (8x256x256, 0.956).
    // 786432 is the first bucket with a separated gain on every measured shape.
    // int16: 196608 is under 3%. 245760 is a separated 8% gain and the losses are gone.
    // w8a16: 102400 still regresses. 126976 and above do not.
    if (quant == QUANT_TYPE_SYMM_8BIT) {
        return 768 * 1024;
    }
    if (quant == QUANT_TYPE_SYMM_W8A16) {
        return 112 * 1024;
    }
    if (quant == QUANT_TYPE_SYMM_16BIT) {
        return 240 * 1024;
    }
    return INT64_MAX;
#else
    (void)quant;
    return INT64_MAX;
#endif
}

/**
 * @brief Activation(Gemm(input, filter) + bias).
 *
 */
class Gemm : public Module {
private:
    activation_type_t activation; /*!< activation of Gemm, if you don't specify anything, no activation is applied */
    bool is_bias_reseted;
    dl_kernel_erased_t m_fn[3] = {};

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

    void forward_args(void *args) { base::conv2d(args, quant_type, m_fn[0], m_fn[1], m_fn[2]); }

    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO)
    {
        reset_bias(context);

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

        // Only AUTO consults the Gemm floor. SINGLE_CORE never splits. MULTI_CORE keeps the conv floor.
        const int64_t auto_width_min_macs =
            (mode == RUNTIME_MODE_AUTO) ? gemm_auto_width_min_macs(quant_type) : static_cast<int64_t>(-1);
        base::ConvOpArgs m_args(output,
                                input0,
                                padding,
                                filter,
                                {1, 1} /*strides*/,
                                {1, 1} /*dilations*/,
                                1 /*group*/,
                                bias,
                                this->activation,
                                nullptr,
                                mode,
                                quant_type,
                                false,
                                auto_width_min_macs); // do not support PReLU and Leaky RelU
        int task_size = m_args.size();
        if (task_size < 1 || !base::dl_conv_ensure_kernels(m_fn, m_args.get_args(0), 0, quant_type, "Gemm")) {
            input0->set_shape(origin_input_shape);
            output->set_shape(origin_output_shape);
            return;
        }
        if (task_size == 1) { // single task
            forward_args((void *)&m_args.get_args(0));
        } else if (task_size == 2) { // multi task, use semaphore to maintain synchronization.
            module_forward_dual_core(this, (void *)&m_args.get_args(0), (void *)&m_args.get_args(1));
        } else {
            ESP_LOGE("Gemm", "Only support task size is 1 or 2, currently task size is %d", task_size);
        }
        input0->set_shape(origin_input_shape);
        output->set_shape(origin_output_shape);
    }

    /**
     * @brief deserialize Conv2d module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *gemm_op = nullptr;

        activation_type_t activation_type;
        quant_type_t quant_type = QUANT_TYPE_NONE;
        std::string quant_type_str;
        int transA = -1, transB = -1;
        fbs_model->get_operation_attribute(node_name, "activation", activation_type);
        // Prebuilt fbs_model only maps S8/S16/F32. Read the raw string first so
        // W8A16 is not lost if the library treats it as unknown.
        if (fbs_model->get_operation_attribute(node_name, "quant_type", quant_type_str) == ESP_OK &&
            quant_type_str == "W8A16") {
            quant_type = QUANT_TYPE_SYMM_W8A16;
        } else {
            fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        }
        fbs_model->get_operation_attribute(node_name, "transA", transA);
        fbs_model->get_operation_attribute(node_name, "transB", transB);
        assert(transA == -1 || transA == 0);
        assert(transB == -1 || transB == 0);

        // Create module
        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT ||
            quant_type == QUANT_TYPE_SYMM_W8A16) {
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
