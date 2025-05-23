#pragma once

#include "dl_model_base.hpp"
#include "dl_tool.hpp"
#include <vector>

extern const uint8_t model_espdl[] asm("_binary_streaming_models_espdl_start");

class StreamingModel {
private:
    dl::Model *m_p_model = nullptr;
    dl::TensorBase *m_p_input = nullptr;
    // The characteristic of initializing the buffer to zero will be utilized directly.
    dl::TensorBase *m_p_input_cache = nullptr;
    dl::TensorBase *m_p_output = nullptr;
    dl::TensorBase *m_p_output_cache = nullptr;

public:
    StreamingModel(std::string model_name,
                   std::string input_name,
                   std::string output_name,
                   std::string input_cache_name = "",
                   std::string output_cache_name = "",
                   fbs::model_location_type_t location = fbs::MODEL_LOCATION_IN_FLASH_RODATA)
    {
        m_p_model = new dl::Model((const char *)model_espdl, model_name.c_str(), location);
        m_p_input = m_p_model->get_input(input_name);
        if (!input_cache_name.empty()) {
            m_p_input_cache = m_p_model->get_input(input_cache_name);
        }
        m_p_output = m_p_model->get_output(output_name);
        if (!output_cache_name.empty()) {
            m_p_output_cache = m_p_model->get_output(output_cache_name);
        }
    }

    ~StreamingModel() { delete m_p_model; }

    dl::TensorBase *operator()(dl::TensorBase *input_p) const
    {
        if (!m_p_input || !m_p_model) {
            ESP_LOGE("StreamingModel", "The member variable is invalid");
            return nullptr;
        }

        m_p_input->assign(input_p);
        m_p_model->run();
        if (m_p_input_cache && m_p_output_cache) {
            m_p_input_cache->assign(m_p_output_cache);
        }
        return m_p_output;
    }

    dl::TensorBase *get_inputs() const { return m_p_input; }

    dl::TensorBase *get_outputs() const { return m_p_output; }
};
