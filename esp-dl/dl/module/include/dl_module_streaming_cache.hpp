#pragma once

#include "dl_module_base.hpp"

namespace dl {
namespace module {
/**
 * @brief: Performs StreamingCachetraction operation.
 * memcpy(output, cache, cache_size)
 * memcpy(output+cache_size, input, input_size)
 * memcpy(cache, output+input_size, cache_size)
 */
class StreamingCache : public Module {
private:
    int m_window_size;
    int m_frame_axis;
    int m_cache_bytes;
    TensorBase *m_cache;

public:
    /**
     * @brief Construct a new StreamingCache object.
     *
     * @param name              name of module
     * @param inplace           inplace type.
     * @param quant_type        quantize type.
     */
    StreamingCache(const char *name = NULL,
                   int window_size = 1,
                   int frame_axis = 1,
                   module_inplace_t inplace = MODULE_NON_INPLACE,
                   quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_window_size(window_size), m_frame_axis(frame_axis)
    {
        m_cache = nullptr;
    }

    /**
     * @brief Destroy the StreamingCache object.
     */
    ~StreamingCache()
    {
        if (m_cache) {
            delete m_cache;
            m_cache = nullptr;
        }
    }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<int> output_shape = input_shapes[0];
        if (output_shape.size() <= m_frame_axis) {
            ESP_LOGE("StreamingCache", "frame_axis is out of input tensor dimension.");
        }
        if (m_frame_axis < 0) {
            m_frame_axis = output_shape.size() + m_frame_axis;
        }
        output_shape[m_frame_axis] += m_window_size - 1;

        return std::vector<std::vector<int>>(1, output_shape);
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        int input_bytes = input->get_bytes();

        if (m_cache == nullptr) {
            std::vector<int> cache_shape = output->get_shape();
            cache_shape[m_frame_axis] = m_window_size - 1;
            m_cache = new TensorBase(cache_shape, nullptr, input->get_exponent(), input->get_dtype());
            m_cache_bytes = m_cache->get_bytes();
            memset(m_cache->get_element_ptr(), 0, m_cache_bytes);
        }

        // Copy cache to output
        tool::copy_memory(output->get_element_ptr(), m_cache->get_element_ptr(), m_cache_bytes);

        // Copy input to output
        tool::copy_memory(output->get_element_ptr<uint8_t>() + m_cache_bytes, input->get_element_ptr(), input_bytes);

        // Update cache
        tool::copy_memory(m_cache->get_element_ptr(), output->get_element_ptr<uint8_t>() + input_bytes, m_cache_bytes);
    }

    /**
     * @brief deserialize StreamingCache module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        int window_size = 1;
        int frame_axis = 1;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "window_size", window_size);
        fbs_model->get_operation_attribute(node_name, "frame_axis", frame_axis);

        op = new StreamingCache(node_name.c_str(), window_size, frame_axis, MODULE_NON_INPLACE, quant_type);
        return op;
    }

    void print()
    {
        ESP_LOGI("StreamingCache",
                 "quant_type: %s, window_size: %d, frame_axis: %d",
                 quant_type_to_string(quant_type),
                 m_window_size,
                 m_frame_axis);
    }
};

} // namespace module
} // namespace dl
