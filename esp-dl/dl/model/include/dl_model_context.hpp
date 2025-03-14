#pragma once

#include "dl_tensor_base.hpp"
#include "esp_log.h"
#include <map>
namespace dl {

#define CONTEXT_PARAMETER_OFFSET 10000000 /*!< Offset for parameter tensors */
class ModelContext {
public:
    std::vector<TensorBase *> m_variables; /*!< Variable tensors of model, the first one is nullptr */
    std::vector<TensorBase *> m_paramters; /*!< Parameters of model, the first one is nullptr */

private:
    void *m_psram_root;                      /*!< PSRAM root pointer */
    void *m_internal_root;                   /*!< Internal root pointer */
    int m_psram_size;                        /*!< In bytes. PSRAM size usage. Only take effect when there's a PSRAM */
    int m_internal_size;                     /*!< In bytes. Internal size usage. */
    std::map<std::string, int> m_name2index; /*!< Tensor name to index map
                                               >=0: variable tensor
                                               <0: parameter tensor */
    /**
     * @brief Gets the parameter tensor index by global tensor index.
     *
     * @param index The index of the tensor.
     * @return int Returns parameter index for m_paramters.
     */
    inline int ti2pi(int index) { return index - CONTEXT_PARAMETER_OFFSET; }

    /**
     * @brief Gets grobal tensor index by parameter tensor index.
     *
     * @param index The index of grobal tensor.
     * @return int Returns grobal index for name2index.
     */
    inline int pi2ti(int index) { return index + CONTEXT_PARAMETER_OFFSET; }

public:
    /**
     * @brief Constructor for ModelContext.
     * Initializes the PSRAM and internal root pointers to nullptr.
     */
    ModelContext()
    {
        m_psram_root = nullptr;
        m_internal_root = nullptr;
        m_psram_size = 0;
        m_internal_size = 0;
    }

    /**
     * @brief Destructor for ModelContext.
     * Clears all resources and tensors.
     */
    ~ModelContext() { clear(); }

    /**
     * @brief Adds a tensor to the parameter or variable list.
     *
     * @param name The name of the tensor.
     * @param is_paramter Whether the tensor is a parameter (default: false).
     * @param tensor Pointer to the TensorBase object (default: nullptr).
     *
     * @return int Returns the index of the added tensor.
     */
    int add_tensor(const std::string name, bool is_paramter = false, TensorBase *tensor = nullptr);

    /**
     * @brief Updates the tensor at the specified index.
     *
     * @param index The index of the tensor to update.
     * @param tensor Pointer to the new TensorBase object.
     */
    void update_tensor(int index, TensorBase *tensor);

    /**
     * @brief Gets the tensor by its index.
     *
     * @param index The index of the tensor.
     * @return TensorBase* Returns the pointer to the TensorBase object, or nullptr if the index is invalid.
     */
    TensorBase *get_tensor(int index);

    /**
     * @brief Gets the tensor by its name.
     *
     * @param name The name of the tensor.
     * @return TensorBase* Returns the pointer to the TensorBase object, or nullptr if the name is not found.
     */
    TensorBase *get_tensor(const std::string &name);

    /**
     * @brief Gets the tensor index by its name.
     *
     * @param name The name of the tensor.
     * @return int Returns index if the name is found, else -1
     */
    int get_tensor_index(const std::string &name);

    /**
     * @brief Gets the variable tensor index by its name.
     *
     * @param name The name of the tensor.
     * @param index The index of the tensor.
     * @return int Returns index if the name is found and is variable tensor, else -1
     */
    int get_variable_index(const std::string &name);

    /**
     * @brief Gets the count of variable tensors.
     *
     * @return int Returns the number of variable tensors.
     */
    int get_variable_count() { return m_variables.size(); }

    /**
     * @brief Gets the count of parameter tensors.
     *
     * @return int Returns the number of parameter tensors.
     */
    int get_parameter_count() { return m_paramters.size(); }

    /**
     * @brief Minimizes the context by clearing the name-to-index map.
     * This is used to free unnecessary intermediate variables during the inference.
     */
    void minimize() { m_name2index.clear(); }

    /**
     * @brief Allocates memory for PSRAM and internal roots.
     *
     * @param internal_size The size of the internal memory in bytes.
     * @param psram_size The size of the PSRAM memory in bytes.
     * @param alignment The alignment of the memory in bytes.
     * @return Bool Return true if the allocation is successful, false otherwise.
     */
    bool root_alloc(size_t internal_size, size_t psram_size, int alignment = 16);

    /**
     * @brief Gets the pointer to the PSRAM root.
     *
     * @return Void* Returns the pointer to the PSRAM root.
     */
    void *get_psram_root() { return m_psram_root; }

    /**
     * @brief Gets the pointer to the internal root.
     *
     * @return Void* Returns the pointer to the internal root.
     */
    void *get_internal_root() { return m_internal_root; }

    /**
     * @brief Gets the size of the PSRAM memory in bytes.
     *
     * @return int Returns the size of the PSRAM memory in bytes.
     */
    int get_psram_size() { return m_psram_size; }

    /**
     * @brief Gets the size of the internal memory in bytes.
     *
     * @return int Returns the size of the internal memory in bytes.
     */
    int get_internal_size() { return m_internal_size; }

    /**
     * @brief Frees the memory allocated for PSRAM and internal roots.
     * This function ensures proper cleanup of allocated memory.
     */
    void root_free()
    {
        // In IDF, free(p) is equivalent to heap_caps_free(p).
        if (m_internal_root) {
            free(m_internal_root);
            m_internal_root = nullptr;
        }
        if (m_psram_root) {
            free(m_psram_root);
            m_psram_root = nullptr;
        }
    }

    /**
     * @brief Clears all resources and tensors in the context.
     * This includes clearing variables, parameters, name-to-index map, and freeing memory.
     */
    void clear()
    {
        m_variables.clear();
        for (int i = 0; i < m_paramters.size(); i++) {
            delete m_paramters[i];
        }
        m_paramters.clear();
        m_name2index.clear();
        root_free();
    }
};

} // namespace dl
