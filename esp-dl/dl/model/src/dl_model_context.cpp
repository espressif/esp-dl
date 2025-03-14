#include <stdint.h>

#include "dl_model_context.hpp"
static const char *TAG = "dl::ModelContext";

namespace dl {

int ModelContext::add_tensor(const std::string name, bool is_paramter, TensorBase *tensor)
{
    auto iter = m_name2index.find(name);
    int index = 0;
    if (iter == m_name2index.end()) {
        if (is_paramter) {
            m_paramters.push_back(tensor);
            index = pi2ti(m_paramters.size() - 1);
            m_name2index.emplace(name, index);
        } else {
            m_variables.push_back(tensor);
            index = m_variables.size() - 1;
            m_name2index.emplace(name, index);
        }
    } else {
        index = iter->second;
        if (is_paramter) {
            m_paramters[ti2pi(index)] = tensor;
        } else {
            m_variables[index] = tensor;
        }
    }

    return index;
}

void ModelContext::update_tensor(int index, TensorBase *tensor)
{
    if (index < m_variables.size() && index >= 0)
        m_variables[index] = tensor;
    else {
        index = ti2pi(index);
        if (index < m_paramters.size() && index >= 0) {
            m_paramters[ti2pi(index)] = tensor;
        } else {
            ESP_LOGE(TAG, "Tensor index %d not found", index);
        }
    }
}

TensorBase *ModelContext::get_tensor(int index)
{
    if (index >= 0 && index < m_variables.size())
        return m_variables[index];
    else {
        index = ti2pi(index);
        if (index < m_paramters.size() && index >= 0) {
            return m_paramters[index];
        } else {
            ESP_LOGE(TAG, "Tensor index %d not found", index);
        }
    }

    return nullptr;
}

TensorBase *ModelContext::get_tensor(const std::string &name)
{
    if (m_name2index.find(name) != m_name2index.end()) {
        return get_tensor(m_name2index[name]);
    } else {
        ESP_LOGE(TAG, "Tensor %s not found", name.c_str());
    }

    return nullptr;
}

int ModelContext::get_tensor_index(const std::string &name)
{
    if (m_name2index.find(name) != m_name2index.end()) {
        return m_name2index[name];
    } else {
        ESP_LOGE(TAG, "Tensor %s not found", name.c_str());
    }

    return -1;
}

int ModelContext::get_variable_index(const std::string &name)
{
    if (m_name2index.find(name) != m_name2index.end()) {
        int index = m_name2index[name];
        if (index < CONTEXT_PARAMETER_OFFSET) {
            return index;
        } else {
            return -1;
        }
    } else {
        ESP_LOGE(TAG, "Tensor %s not found", name.c_str());
    }

    return -1;
}

bool ModelContext::root_alloc(size_t internal_size, size_t psram_size, int alignment)
{
    m_internal_size = internal_size;
    m_psram_size = psram_size;
    if (m_psram_size > 0) {
        m_psram_root = tool::malloc_aligned(alignment, m_psram_size, MALLOC_CAP_SPIRAM);
        if (!m_psram_root) {
            ESP_LOGE(TAG,
                     "Failed to alloc %.2fKB PSRAM, largest available PSRAM block size %.2fKB",
                     m_psram_size / 1024.f,
                     heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024.f);
            return false;
        }
    }

    if (m_internal_size > 0) {
        m_internal_root = tool::malloc_aligned(alignment, m_internal_size, MALLOC_CAP_INTERNAL);

        if (!m_internal_root) {
            ESP_LOGE(TAG,
                     "Failed to alloc %.2fKB internal RAM, largest available internal RAM block size %.2fKB",
                     m_internal_size / 1024.f,
                     heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024.f);
            return true;
        }
    }
    return true;
}

} // namespace dl
