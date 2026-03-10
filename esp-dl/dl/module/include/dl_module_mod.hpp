#pragma once

#include "dl_base_mod.hpp"
#include "dl_base_shape.hpp"
#include "dl_module_base.hpp"

namespace dl {
namespace module {
/**
 * @brief: Performs element-wise modulus operation (with Numpy-style broadcasting support).
 *         Please refer to https://onnx.ai/onnx/operators/onnx__Mod.html for more details.
 *         Implements fmod=1 semantics: C-style fmod where result sign follows dividend.
 *         Per ONNX spec, fmod=1 is required for float types; fmod=0 is for integer-only types.
 *         Since ESP-DL works with quantized float models, fmod=1 is the applicable mode.
 */
class Mod : public Module {
private:
    int m_fmod; ///< fmod attribute from ONNX (1: C-style fmod for float types)

public:
    /**
     * @brief Construct a new Mod object.
     *
     * @param name            name of module
     * @param inplace         inplace type.
     * @param quant_type      quantize type.
     * @param fmod            fmod attribute (1 for C-style fmod, required for float types per ONNX spec)
     */
    Mod(const char *name = NULL,
        module_inplace_t inplace = MODULE_NON_INPLACE,
        quant_type_t quant_type = QUANT_TYPE_NONE,
        int fmod = 1) :
        Module(name, inplace, quant_type), m_fmod(fmod)
    {
    }

    /**
     * @brief Destroy the Mod object.
     */
    ~Mod() {}

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
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            forward_template<float>(context, mode);
        }
    }

    void forward_args(void *args)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            base::elemwise_mod((base::elemwiseArgsType<int8_t> *)args);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            base::elemwise_mod((base::elemwiseArgsType<int16_t> *)args);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            base::elemwise_mod((base::elemwiseArgsType<float> *)args);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input0 = context->get_tensor(m_inputs_index[0]);
        TensorBase *input1 = context->get_tensor(m_inputs_index[1]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        std::vector<base::elemwiseArgsType<T>> m_args =
            base::get_elemwise_operation_args<T>(output, input0, input1, mode);
        int task_size = m_args.size();
        if (task_size == 1) {
            forward_args((void *)&m_args[0]);
        } else if (task_size == 2) {
            module_forward_dual_core(this, (void *)&m_args[0], (void *)&m_args[1]);
        } else {
            ESP_LOGE("Mod", "Only support task size is 1 or 2, currently task size is %d", task_size);
        }
    }

    /**
     * @brief deserialize Mod module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        int fmod = 1;
        fbs_model->get_operation_attribute(node_name, "fmod", fmod);

        op = new Mod(node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type, fmod);
        return op;
    }

    void print()
    {
        ESP_LOGI("Mod",
                 "quant_type: %s, fmod: %d, input feature map size: %d.",
                 quant_type_to_string(quant_type),
                 m_fmod,
                 m_inputs_index.size());
    }
};

} // namespace module
} // namespace dl
