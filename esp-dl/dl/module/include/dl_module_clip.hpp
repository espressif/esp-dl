#pragma once

#include "dl_module_base.hpp"

namespace dl {
namespace module {

class Clip : public Module {
private:
    TensorBase *m_min; /*<! Minimum value of Clip >*/
    TensorBase *m_max; /*<! Maximum value of Clip >*/

public:
    /**
     * @brief Construct a new Clip object.
     *
     * @param name            name of module
     * @param inplace         inplace type.
     */
    Clip(TensorBase *min,
         TensorBase *max,
         const char *name = NULL,
         module_inplace_t inplace = MODULE_NON_INPLACE,
         quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_min(min), m_max(max)
    {
    }

    /**
     * @brief Destroy the Clip object.
     */
    ~Clip()
    {
        if (m_min) {
            delete m_min;
        }

        if (m_max) {
            delete m_max;
        }
    }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<std::vector<int>> output_shapes(1, input_shapes[0]);
        return output_shapes;
    }

    void forward(std::vector<dl::TensorBase *> &tensors, runtime_mode_t mode)
    {
        DL_LOG_LAYER_LATENCY_INIT();
        DL_LOG_LAYER_LATENCY_START();
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(tensors, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(tensors, mode);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            forward_template<float>(tensors, mode);
        }
        DL_LOG_LAYER_LATENCY_END(this->name, "Clip");
    }

    void forward_args(void *args) {}

    template <typename T>
    void forward_template(std::vector<dl::TensorBase *> &tensors, runtime_mode_t mode)
    {
        TensorBase *input = tensors[m_inputs_index[0]];
        TensorBase *output = tensors[m_outputs_index[0]];
        assert(input->get_size() == output->get_size());
        T min_value = *static_cast<T *>(m_min->get_element_ptr());
        T max_value = *static_cast<T *>(m_max->get_element_ptr());

        T *src_data = static_cast<T *>(input->get_element_ptr());
        T *data = static_cast<T *>(output->get_element_ptr());
        for (int i = 0; i < input->get_size(); i++) {
            data[i] = DL_CLIP(src_data[i], min_value, max_value);
        }
    }

    /**
     * @brief deserialize Clip module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        TensorBase *min = fbs_model->get_operation_parameter(node_name, 1);
        TensorBase *max = fbs_model->get_operation_parameter(node_name, 2);

        // Create module
        op = new Clip(min, max, node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        return op;
    }

    void print() { ESP_LOGI("Clip", "quant_type: %s.", quant_type_to_string(quant_type)); }
};
} // namespace module
} // namespace dl
