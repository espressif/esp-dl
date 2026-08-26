#pragma once

#include "dl_base_lut.hpp"
#include "dl_module_base.hpp"

namespace dl {
namespace module {
/**
 * NOTE:int16 using linear interpolation + lookup table.
 *
 * @tparam feature_t supports int16_t and int8_t,
 *         - int16_t: stands for operation in int16_t quantize
 *         - int8_t: stands for operation in int8_t quantize
 */
class LUT : public Module {
public:
    /**
     * @brief Construct a new LUT object.
     *
     * @param name            name of module
     * @param inplace         inplace type.
     */
    LUT(const char *name = NULL,
        module_inplace_t inplace = MODULE_INPLACE_CHANGED_BUFFER,
        quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type)
    {
    }

    bool is_lut_module() const override { return true; }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<std::vector<int>> output_shapes(1, input_shapes[0]);
        return output_shapes;
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        assert(m_inputs_index.size() >= 2);
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *table = context->get_tensor(m_inputs_index.back());
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        assert(table != nullptr);
        assert(output->exponent == table->exponent);

        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            base::lut_s8((int8_t *)output->get_element_ptr(),
                         (int8_t *)input->get_element_ptr(),
                         input->size,
                         (int8_t *)table->get_element_ptr());
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            int16_t *input_ptr = (int16_t *)input->get_element_ptr();
            int16_t *output_ptr = (int16_t *)output->get_element_ptr();
            int16_t *table_ptr = (int16_t *)(table->get_element_ptr());
            assert(table->get_size() > 1);
            int step = 65536 / (table->get_size() - 1);

            if (step > 1) {
                assert((step & (step - 1)) == 0);
                base::lut_s16_nearest_neighbor(output_ptr, input_ptr, input->size, table_ptr, step);

            } else {
                for (size_t i = 0; i < input->size; i++) {
                    output_ptr[i] = table_ptr[input_ptr[i] + 32768];
                }
            }
        }
    }

    void forward_args(void *args) {}

    static bool has_lut(fbs::FbsModel *fbs_model, const std::string &node_name)
    {
        std::string lut_name;
        return fbs_model->get_operation_lut_name(node_name, lut_name) == ESP_OK;
    }

    /**
     * @brief deserialize LUT module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        if (!has_lut(fbs_model, node_name)) {
            ESP_LOGE("LUT", "Table is null!");
            return nullptr;
        }

        // Create module
        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            op = new LUT(node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
        } else {
            ESP_LOGE("LUT", "Only support QUANT_TYPE_SYMM_8BIT or QUANT_TYPE_SYMM_16BIT!");
        }
        return op;
    }

    void print() { ESP_LOGI("LUT", "quant_type: %s.", quant_type_to_string(quant_type)); }
};
} // namespace module
} // namespace dl
