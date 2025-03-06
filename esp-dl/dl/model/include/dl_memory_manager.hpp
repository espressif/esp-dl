#pragma once

#include "dl_module_base.hpp"
#include "esp_heap_caps.h"
#include "fbs_model.hpp"
#include <list>

namespace dl {
namespace memory {

/**
 * @brief Memory manager base class, each model has its own memory manager
 * TODO: share memory manager with different models
 */
class MemoryManagerBase {
protected:
    void *psram_root;    /*!< PSRAM root pointer */
    void *internal_root; /*!< Internal ram pointer */

public:
    std::vector<TensorBase *> tensors;     /*!< All tensors in the model */
    int alignment;                         /*!< The root pointer needs to be aligned must be a power of two */
    std::map<std::string, int> name2index; /*!< Tensor name to index map */
    size_t internal_size;                  /*!< The bytes of internal ram */
    size_t psram_size;                     /*!< The bytes of psram */

    /**
     * @brief Construct a new Memory Manager Base object
     *
     * @param alignment Memory address alignment
     */
    MemoryManagerBase(int alignment = 16) :
        psram_root(nullptr), internal_root(nullptr), tensors({}), alignment(alignment), internal_size(0), psram_size(0)
    {
    }

    /**
     * @brief Destroy the MemoryManager object. Return resource.
     */
    virtual ~MemoryManagerBase() { this->reset(); }

    /**
     * @brief Allocate memory for each tensor, include all input and output tensors
     *
     * @param fbs_model       FlatBuffer's Model
     * @param execution_plan  Topological sorted module list
     */
    virtual void alloc(fbs::FbsModel *fbs_model, std::vector<dl::module::Module *> &execution_plan) {};

    /**
     * @brief Set preload address for module's parameters
     *
     * @param execution_plan   Topological sorted module list
     */
    virtual void set_preload_addr(std::vector<dl::module::Module *> execution_plan) {}

    /**
     * @brief Reset the memory manager, free all memory for each tensor, include all input and output tensors
     */
    virtual void reset();

    /**
     * @brief Get tensor by index
     *
     * @param index The tensor index, type: int
     *
     * @return The TensorBase pointer
     */
    TensorBase *get_tensor(int index);

    /**
     * @brief Get tensor by name
     *
     * @param name The tensor name, type: std::string
     *
     * @return The TensorBase pointer
     */
    TensorBase *get_tensor(const std::string &name);

    /**
     * @brief Get tensor index by name
     *
     * @param name The tensor name, type: std::string
     *
     * @return The TensorBase index
     */
    int get_tensor_index(const std::string &name);

    /**
     * @brief Free psram root and internal root pointer
     */
    void root_free();

    /**
     * @brief Get PSRAM root pointer
     *
     * @return void*
     */
    void *get_psram_root() { return this->psram_root; }

    /**
     * @brief Get internal ram root pointer
     *
     * @return void*
     */
    void *get_internal_root() { return this->internal_root; }

    /**
     * @brief Get PSRAM size allocated by memory manager.
     *
     * @return size_t
     */
    size_t get_psram_size() { return this->psram_size; }

    /**
     * @brief Get internal RAM size allocated by memory manager.
     *
     * @return size_t
     */
    size_t get_internal_size() { return this->internal_size; }
};

/**
 * @brief Tensor info, include tensor name, shape, dtype, size, time range
 * and call times, which is used to plan model memory
 */
class TensorInfo {
private:
    std::string name;
    int time_begin;
    int time_end;
    std::vector<int> shape;
    dtype_t dtype;
    int exponent;
    size_t size; // Size, in bytes
    uint32_t call_times;
    uint32_t offset;          // PSRAM offset
    uint32_t internal_offset; // Internal ram offset, used to allocate tensor on both PSRAM and internal ram
    bool is_internal;
    TensorInfo *m_leader_tensor;
    TensorInfo
        *m_follower_dirty_tensor; // Only reference the follower tensor which will modify the data of leader tensor.

public:
    /**
     * @brief Construct a new Tensor Info object
     *
     * @param name          Tensor name
     * @param time_begin    Tensor lifetime begin
     * @param time_end      Tensor lifetime end
     * @param shape         Tensor shape
     * @param dtype         Tensor dtype
     * @param exponent      Tensor exponent
     * @param is_internal   Is tensor in internal RAM or not
     */
    TensorInfo(std::string &name,
               int time_begin,
               int time_end,
               std::vector<int> shape,
               dtype_t dtype,
               int exponent,
               bool is_internal = false);

    /**
     * @brief Destroy the Tensor Info object
     *
     */
    ~TensorInfo() {}

    /**
     * @brief Set the inplace leader tensor object
     *
     * @param tensor Inplace leader tensor
     */
    void set_inplace_leader_tensor(TensorInfo *tensor);

    /**
     * @brief Set the inplace follower tensor object
     *
     * @param tensor Inplace follower tensor
     */
    void set_inplace_follower_tensor(TensorInfo *tensor) { m_follower_dirty_tensor = tensor; }

    /**
     * @brief Get the inplace follower tensor object
     *
     * @return TensorInfo* Inplace follower tensor
     */
    TensorInfo *get_inplace_follower_tensor() { return m_follower_dirty_tensor; }

    /**
     * @brief Update Tensor lifetime
     *
     * @param new_time new tensor lifetime
     */
    void update_time(int new_time);

    /**
     * @brief Create a TensorBase object according to TensorInfo
     *
     * @param internal_root Internal RAM root pointer
     * @param psram_root    PSRAM root pointer
     * @return TensorBase*
     */
    TensorBase *create_tensor(void *internal_root, void *psram_root);

    /**
     * @brief Is inplaced or not
     *
     * @return true if inplaced else false
     */
    bool is_inplaced() { return this->m_leader_tensor != nullptr; }

    /**
     * @brief Get the tensor offset
     *
     * @return uint32_t
     */
    uint32_t get_offset()
    {
        if (m_leader_tensor) {
            return m_leader_tensor->get_offset();
        }
        return this->offset;
    }

    /**
     * @brief Set the tensor offset
     *
     * @param offset
     */
    void set_offset(uint32_t offset)
    {
        if (m_leader_tensor) {
            m_leader_tensor->set_offset(offset);
        }
        this->offset = offset;
    }

    /**
     * @brief Get the internal offset
     *
     * @return uint32_t
     */
    uint32_t get_internal_offset()
    {
        if (m_leader_tensor) {
            return m_leader_tensor->get_internal_offset();
        }
        return this->internal_offset;
    }

    /**
     * @brief Get the internal state
     *
     * @return true if is internal else false
     */
    bool get_internal_state()
    {
        if (m_leader_tensor) {
            return m_leader_tensor->get_internal_state();
        }
        return this->is_internal;
    }

    /**
     * @brief Set the internal state
     *
     * @param is_internal
     */
    void set_internal_state(bool is_internal)
    {
        if (m_leader_tensor) {
            m_leader_tensor->set_internal_state(is_internal);
        }
        this->is_internal = is_internal;
    }

    /**
     * @brief Set the internal offset
     *
     * @param offset
     */
    void set_internal_offset(uint32_t offset)
    {
        if (m_leader_tensor) {
            m_leader_tensor->set_internal_offset(offset);
            m_leader_tensor->set_internal_state(true);
        }
        this->is_internal = true;
        this->internal_offset = offset;
    }

    /**
     * @brief Get the liftetime end
     *
     * @return int
     */
    int get_time_end()
    {
        if (m_leader_tensor) {
            return m_leader_tensor->get_time_end();
        }
        return this->time_end;
    }

    /**
     * @brief Get the liftetime begin
     *
     * @return int
     */
    int get_time_begin()
    {
        if (m_leader_tensor) {
            return m_leader_tensor->get_time_begin();
        }
        return this->time_begin;
    }

    /**
     * @brief Get the tensor size
     *
     * @return size_t
     */
    size_t get_size() { return this->size; }

    /**
     * @brief Get the tensor name
     *
     * @return std::string
     */
    std::string get_name() { return this->name; }

    /**
     * @brief Get the tensor shape
     *
     * @return std::vector<int>
     */
    std::vector<int> get_shape() { return this->shape; }

    /**
     * @brief print tensor info
     *
     */
    void print()
    {
        printf("name:%s, from %d to %d, size:%d, offset:(%ld, %ld)\n",
               name.c_str(),
               time_begin,
               time_end,
               size,
               offset,
               internal_offset);
    }
};

/**
 * @brief Memory chunk, include size, is free, offset, alignment and tensor, which is used to simulate memory allocation
 */
class MemoryChunk {
public:
    size_t size;        /*!< Memeory chunk size */
    bool is_free;       /*!< Whether memory chunk is free or not */
    int offset;         /*!< Offset relative to root pointer */
    int alignment;      /*!< Memory address alignment */
    TensorInfo *tensor; /*!< Info of the tensor which occupies the memory */

    /**
     * @brief Construct a new Memory Chunk object
     *
     * @param size          Memory chunk size
     * @param is_free       Whether free or not
     * @param alignment     Memory chunk alignment
     */
    MemoryChunk(size_t size, int is_free, int alignment = 16);

    /**
     * @brief Construct a new Memory Chunk object
     *
     * @param tensor        TensorInfo
     * @param alignment     Memory chunk alignment
     */
    MemoryChunk(TensorInfo *tensor, int alignment = 16);

    /**
     * @brief Destroy the Memory Chunk object
     *
     */
    ~MemoryChunk() {}

    /**
     * @brief Merge continuous free chunk
     *
     * @param chunk
     * @return MemoryChunk*
     */
    MemoryChunk *merge_free_chunk(MemoryChunk *chunk);

    /**
     * @brief Insert tensor into free chunk
     *
     * @param tensor
     * @return MemoryChunk*
     */
    MemoryChunk *insert(TensorInfo *tensor);

    /**
     * @brief Extend free chunk and insert tensor
     *
     * @param tensor
     * @return MemoryChunk*
     */
    MemoryChunk *extend(TensorInfo *tensor);

    /**
     * @brief Free memory chunk, set is_free to true and set tensor to nullptr
     */
    void free()
    {
        this->is_free = true;
        this->tensor = nullptr;
    }

    /**
     * @brief get aligned size, which is 16/alignemt bytes aligned
     *
     * @param size
     * @return size_t
     */
    size_t get_aligned_size(size_t size);
};

/**
 * @brief print memory list
 */
void print_memory_list(const char *tag, std::list<MemoryChunk *> &memory_list);

/**
 * @brief sort memory list by memory chunk size
 */
void sort_memory_list(std::list<MemoryChunk *> &memory_list);

}; // namespace memory
} // namespace dl
