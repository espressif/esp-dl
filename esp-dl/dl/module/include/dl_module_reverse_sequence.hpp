#pragma once

#include "dl_module_base.hpp"

namespace dl {
namespace module {

class ReverseSequence : public Module {
private:
    int m_batch_axis;
    int m_time_axis;
    TensorBase *m_sequence_lens;

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
                    TensorBase *sequence_lens = nullptr,
                    const char *name = nullptr,
                    module_inplace_t inplace = MODULE_NON_INPLACE,
                    quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_batch_axis(batch_axis), m_time_axis(time_axis)
    {
        m_sequence_lens = sequence_lens;
        if (m_sequence_lens) {
            m_sequence_lens->print();
        }
        assert(m_batch_axis != m_time_axis);
    }

    ~ReverseSequence()
    {
        if (m_sequence_lens) {
            delete m_sequence_lens;
        }
    }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        assert(input_shapes.size() >= 1);
        return {input_shapes[0]}; // Output shape same as input
    }

    void forward(std::vector<dl::TensorBase *> &tensors, runtime_mode_t mode)
    {
        TensorBase *input = tensors[m_inputs_index[0]];

        TensorBase *sequence_lens = m_sequence_lens;
        if (sequence_lens == nullptr) {
            m_sequence_lens = tensors[m_inputs_index[1]];
        }

        TensorBase *output = tensors[m_outputs_index[0]];

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

        TensorBase *sequence_lens = fbs_model->get_operation_parameter(node_name, 1);

        return new ReverseSequence(
            batch_axis, time_axis, sequence_lens, node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);
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
