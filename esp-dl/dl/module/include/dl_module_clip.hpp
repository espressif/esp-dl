#include "dl_module_base.hpp"
#include "dl_module_lut.hpp"
namespace dl {
namespace module {

/**
 * @brief: Please refer to https://onnx.ai/onnx/operators/onnx__Clip.html for more details
 *
 * @tparam feature_t supports float, int16_t and int8_t,
 *         - int16_t: stands for operation in int16_t quantize
 *         - int8_t: stands for operation in int8_t quantize
 */
class Clip : public Module {
public:
    /**
     * @brief Construct a new Clip object.
     *
     * @param name            name of module
     * @param inplace         inplace type.
     */
    Clip(const char *name = NULL,
         module_inplace_t inplace = MODULE_NON_INPLACE,
         quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
    }

    /**
     * @brief Destroy the Clip object.
     */
    ~Clip() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<std::vector<int>> output_shapes(1, input_shapes[0]);
        return output_shapes;
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        TensorBase *min = nullptr, *max = nullptr;
        if (m_inputs_index.size() == 3) {
            min = context->get_tensor(m_inputs_index[1]);
            max = context->get_tensor(m_inputs_index[2]);
        } else if (m_inputs_index.size() == 2) {
            min = context->get_tensor(m_inputs_index[1]);
        } else {
            ESP_LOGW("Clip", "Clip min and max are not set!");
            output->assign(input);
            return;
        }

        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(input, min, max, output);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(input, min, max, output);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            float *input_ptr = (float *)input->get_element_ptr();
            float *output_ptr = (float *)output->get_element_ptr();
            float min_value = min == nullptr ? std::numeric_limits<float>::min() : min->get_element<float>(0);
            float max_value = max == nullptr ? std::numeric_limits<float>::max() : max->get_element<float>(0);

            for (size_t i = 0; i < input->size; i++) {
                output_ptr[i] = DL_CLIP(input_ptr[i], min_value, max_value);
            }
        }
    }

    template <typename T>
    void forward_template(TensorBase *input, TensorBase *min, TensorBase *max, TensorBase *output)
    {
        T *input_ptr = (T *)input->get_element_ptr();
        T *output_ptr = (T *)output->get_element_ptr();
        T min_value = min == nullptr ? std::numeric_limits<T>::min() : min->get_element<T>(0);
        T max_value = max == nullptr ? std::numeric_limits<T>::max() : max->get_element<T>(0);

        float rescale = DL_SCALE(input->exponent) * DL_RESCALE(output->exponent);
        for (size_t i = 0; i < input->size; i++) {
            T temp = DL_CLIP(input_ptr[i], min_value, max_value);
            tool::truncate(output_ptr[i], tool::round(temp * rescale));
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

        // Create module
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            TensorBase *table = fbs_model->get_operation_lut(node_name);
            if (table) {
                op = new LUT(node_name.c_str(), table, MODULE_INPLACE_CHANGED_BUFFER, quant_type);
            }
        }
        if (op == nullptr) {
            op = new Clip(node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        }
        return op;
    }

    void print() { ESP_LOGI("Clip", "quant_type: %s.", quant_type_to_string(quant_type)); }
};
} // namespace module
} // namespace dl
