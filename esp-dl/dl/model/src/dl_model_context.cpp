#include <stdint.h>

#include "dl_model_context.hpp"
#include "dl_tool.hpp"
static const char *TAG = "dl::ModelContext";

namespace dl {

int ModelContext::add_tensor(const std::string name, bool is_paramter, TensorBase *tensor)
{
    auto iter = m_name2index.find(name);
    int index = 0;
    if (iter == m_name2index.end()) {
        if (is_paramter) {
            m_parameters.push_back(tensor);
            index = pi2ti(m_parameters.size() - 1);
            m_name2index.emplace(name, index);
        } else {
            m_variables.push_back(tensor);
            index = m_variables.size() - 1;
            m_name2index.emplace(name, index);
        }
    } else {
        index = iter->second;
        if (is_paramter) {
            m_parameters[ti2pi(index)] = tensor;
        } else {
            m_variables[index] = tensor;
        }
    }

    return index;
}

int ModelContext::push_back_tensor(TensorBase *tensor, bool is_paramter)
{
    if (is_paramter) {
        m_parameters.push_back(tensor);
        return m_parameters.size() - 1;
    } else {
        m_variables.push_back(tensor);
        return m_variables.size() - 1;
    }
    return -1;
}

void ModelContext::update_tensor(int index, TensorBase *tensor)
{
    if (index < m_variables.size() && index >= 0)
        m_variables[index] = tensor;
    else {
        index = ti2pi(index);
        if (index < m_parameters.size() && index >= 0) {
            m_parameters[ti2pi(index)] = tensor;
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
        if (index < m_parameters.size() && index >= 0) {
            return m_parameters[index];
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

size_t ModelContext::get_parameter_memory_size(mem_info_t &mem_info, bool copy)
{
    size_t total_size = 0;
    mem_info = {};
    for (int i = 0; i < m_parameters.size(); i++) {
        if (m_parameters[i] && !(m_parameters[i]->auto_free ^ copy)) {
            memory_addr_type_t mem_type = dl::tool::memory_addr_type(m_parameters[i]->data);
            switch (mem_type) {
            case MEMORY_ADDR_INTERNAL:
                mem_info.internal += m_parameters[i]->get_aligned_bytes();
                break;
            case MEMORY_ADDR_PSRAM:
                mem_info.psram += m_parameters[i]->get_aligned_bytes();
                break;
            case MEMORY_ADDR_FLASH:
                mem_info.flash += m_parameters[i]->get_aligned_bytes();
                break;
            default:
                ESP_LOGE(TAG, "Wrong memory addr type.");
            }
        }
    }
    total_size = mem_info.internal + mem_info.psram + mem_info.flash;
    return total_size;
}

size_t ModelContext::get_variable_memory_size(mem_info_t &mem_info)
{
    mem_info.flash = 0;
    mem_info.internal = m_internal_size;
    mem_info.psram = m_psram_size;
    size_t total_size = m_internal_size + m_psram_size;
    return total_size;
}

bool ModelContext::root_alloc(size_t internal_size, size_t psram_size, int alignment)
{
    m_internal_size = internal_size;
    m_psram_size = psram_size;
    if (m_psram_size > 0) {
        m_psram_root = tool::calloc_aligned(alignment, m_psram_size, 1, MALLOC_CAP_SPIRAM);
        if (!m_psram_root) {
            ESP_LOGE(TAG,
                     "Failed to alloc %.2fKB PSRAM, largest available PSRAM block size %.2fKB",
                     m_psram_size / 1024.f,
                     heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024.f);
            return false;
        }
    }

    if (m_internal_size > 0) {
        m_internal_root = tool::calloc_aligned(alignment, m_internal_size, 1, MALLOC_CAP_INTERNAL);

        if (!m_internal_root) {
            ESP_LOGE(TAG,
                     "Failed to alloc %.2fKB internal RAM, largest available internal RAM block size %.2fKB",
                     m_internal_size / 1024.f,
                     heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024.f);
            return false;
        }
    }
    return true;
}

} // namespace dl
