#pragma once

#include "dl_module_base.hpp"

namespace dl {
namespace module {

class ReverseSequence : public Module {
private:
    int m_batch_axis;
    int m_time_axis;

    template <typename T>
    void reverse_sequence_impl(TensorBase *input, TensorBase *sequence_lens, TensorBase *output)
    {
        // Copy input to output first
        T *input_data = input->get_element_ptr<T>();
        T *output_data = output->get_element_ptr<T>();
        if (input_data != output_data) {
            memcpy(output_data, input_data, input->get_bytes());
        }

        // Validate parameters
        std::vector<int> input_shape = input->get_shape();
        // Get sequence lengths
        int64_t *seq_lens = sequence_lens->get_element_ptr<int64_t>();
        int batch_size = input_shape[m_batch_axis];
        int batch_stride = input->axis_offset[m_batch_axis];
        int time_stride = input->axis_offset[m_time_axis];

        // Calculate feature size (non-time/batch dimensions)
        int feature_size = 1;
        for (size_t i = 2; i < input_shape.size(); ++i) {
            feature_size *= input_shape[i];
        }

        // Dispatch based on quantization type
        for (int b = 0; b < batch_size; ++b) {
            int len = seq_lens[b];
            if (len <= 1)
                continue;
            int base_offset = b * batch_stride;

            for (int t = 0; t < len / 2; ++t) {
                const int t1 = t;
                const int t2 = len - 1 - t;

                T *ptr1 = output_data + base_offset + t1 * time_stride;
                T *ptr2 = output_data + base_offset + t2 * time_stride;

                for (int i = 0; i < feature_size; ++i) {
                    int temp = ptr1[i];
                    ptr1[i] = ptr2[i];
                    ptr2[i] = temp;
                }
            }
        }
    }

public:
    ReverseSequence(int batch_axis,
                    int time_axis,
                    const char *name = nullptr,
                    module_inplace_t inplace = MODULE_NON_INPLACE,
                    quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_batch_axis(batch_axis), m_time_axis(time_axis)
    {
        assert(m_batch_axis != m_time_axis);
    }

    ~ReverseSequence() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        return {input_shapes[0]}; // Output shape same as input
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *sequence_lens = context->get_tensor(m_inputs_index[1]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        switch (this->quant_type) {
        case QUANT_TYPE_SYMM_8BIT:
            reverse_sequence_impl<int8_t>(input, sequence_lens, output);
            break;

        case QUANT_TYPE_SYMM_16BIT:
            reverse_sequence_impl<int16_t>(input, sequence_lens, output);
            break;

        case QUANT_TYPE_FLOAT32:
            reverse_sequence_impl<float>(input, sequence_lens, output);
            break;

        default:
            ESP_LOGE("ReverseSequence", "Unsupported quantization type");
            assert(false);
        }
    }

    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        quant_type_t quant_type;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        int batch_axis = 1;
        fbs_model->get_operation_attribute(node_name, "batch_axis", batch_axis);

        int time_axis = 0;
        fbs_model->get_operation_attribute(node_name, "time_axis", time_axis);

        return new ReverseSequence(batch_axis, time_axis, node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
    }

    void print()
    {
        ESP_LOGI("ReverseSequence",
                 "quant_type: %s, batch_axis: %d, time_axis: %d",
                 quant_type_to_string(quant_type),
                 m_batch_axis,
                 m_time_axis);
    }
};

} // namespace module
} // namespace dl
