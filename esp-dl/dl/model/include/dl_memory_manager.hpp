#pragma once

#include "dl_model_context.hpp"
#include "dl_module_base.hpp"
#include "esp_heap_caps.h"
#include "fbs_model.hpp"
#include <list>
#include <tuple>

namespace dl {
/**
 * @brief Memory manager base class, each model has its own memory manager
 * TODO: share memory manager with different models
 */
class MemoryManagerBase {
public:
    int alignment; /*!< The root pointer needs to be aligned must be a power of two */

    /**
     * @brief Construct a new Memory Manager Base object
     *
     * @param alignment Memory address alignment
     */
    MemoryManagerBase(int alignment = 16) : alignment(alignment) {}

    /**
     * @brief Destroy the MemoryManager object. Return resource.
     */
    virtual ~MemoryManagerBase() {}

    /**
     * @brief Allocate memory for each tensor, include all input and output tensors
     *
     * @param fbs_model       FlatBuffer's Model
     * @param execution_plan  Topological sorted module list
     * @param context         Model context
     * @param input_shapes    Optional map to override graph input shapes. The key is the graph input name and the
     *                        value is the shape. If empty, the shapes stored in fbs_model are used.
     * @return Bool Return true if the allocation is successful, false otherwise.
     */
    virtual bool alloc(fbs::FbsModel *fbs_model,
                       std::vector<dl::module::Module *> &execution_plan,
                       ModelContext *context,
                       const std::map<std::string, std::vector<int>> &input_shapes = {}) = 0;
};

/**
 * @brief Common lifetime and placement metadata for planner-managed memory.
 *
 * Both tensors and native workspaces use this representation so they can share
 * the same greedy arena and lifetime-reuse algorithm.
 */
class MemoryInfo {
private:
    std::string name;
    int time_begin;
    int time_end;
    size_t size; // Size, in bytes
    uint32_t call_times;
    uint32_t offset;          // PSRAM offset
    uint32_t internal_offset; // Internal ram offset, used to allocate tensor on both PSRAM and internal ram
    bool is_internal;
    MemoryInfo *m_leader;

protected:
    /**
     * @brief Update the number of bytes required by this planner allocation.
     *
     * @param size Allocation size in bytes.
     */
    void set_size(size_t size) { this->size = size; }

    /**
     * @brief Share placement and lifetime metadata with an in-place leader.
     *
     * @param leader Leader allocation, or nullptr to remove the relationship.
     */
    void set_leader(MemoryInfo *leader) { m_leader = leader; }

public:
    /**
     * @brief Construct planner metadata for a tensor or native workspace.
     *
     * @param name Allocation name used in planner diagnostics.
     * @param time_begin First execution step that uses the allocation.
     * @param time_end Last execution step that uses the allocation.
     * @param size Allocation size in bytes.
     * @param is_internal Whether the allocation is initially assigned to internal RAM.
     */
    MemoryInfo(const std::string &name, int time_begin, int time_end, size_t size, bool is_internal = false);
    virtual ~MemoryInfo() {}

    /**
     * @brief Extend the allocation lifetime to include a new execution step.
     *
     * @param new_time New lifetime end; -1 marks a persistent allocation.
     */
    void update_time(int new_time);

    /**
     * @brief Check whether this allocation shares storage with a leader.
     *
     * @return true when an in-place leader is assigned, otherwise false.
     */
    bool is_inplaced() { return m_leader != nullptr; }

    /**
     * @brief Check whether the allocation must stay in PSRAM.
     *
     * @return true when internal RAM placement is forbidden, otherwise false.
     */
    virtual bool is_psram_only() const { return false; }

    /**
     * @brief Get the offset in the planner's PSRAM arena.
     *
     * @return PSRAM arena offset in bytes.
     */
    uint32_t get_offset() { return m_leader ? m_leader->get_offset() : offset; }

    /**
     * @brief Set the offset in the planner's PSRAM arena.
     *
     * @param offset PSRAM arena offset in bytes.
     */
    void set_offset(uint32_t offset)
    {
        if (m_leader) {
            m_leader->set_offset(offset);
        }
        this->offset = offset;
    }

    /**
     * @brief Get the offset in the planner's internal-RAM arena.
     *
     * @return Internal-RAM arena offset in bytes.
     */
    uint32_t get_internal_offset() { return m_leader ? m_leader->get_internal_offset() : internal_offset; }

    /**
     * @brief Check whether the allocation is assigned to internal RAM.
     *
     * @return true for internal RAM, otherwise false.
     */
    bool get_internal_state() { return m_leader ? m_leader->get_internal_state() : is_internal; }

    /**
     * @brief Set whether the allocation is assigned to internal RAM.
     *
     * @param is_internal true for internal RAM, false for PSRAM.
     */
    void set_internal_state(bool is_internal)
    {
        if (m_leader) {
            m_leader->set_internal_state(is_internal);
        }
        this->is_internal = is_internal;
    }

    /**
     * @brief Assign an offset in the internal-RAM arena.
     *
     * @param offset Internal-RAM arena offset in bytes.
     */
    void set_internal_offset(uint32_t offset)
    {
        if (m_leader) {
            m_leader->set_internal_offset(offset);
            m_leader->set_internal_state(true);
        }
        is_internal = true;
        internal_offset = offset;
    }

    /**
     * @brief Get the last execution step in the allocation lifetime.
     *
     * @return Lifetime end, or -1 for a persistent allocation.
     */
    int get_time_end() { return m_leader ? m_leader->get_time_end() : time_end; }

    /**
     * @brief Get the first execution step in the allocation lifetime.
     *
     * @return Lifetime begin.
     */
    int get_time_begin() { return m_leader ? m_leader->get_time_begin() : time_begin; }

    /**
     * @brief Get the allocation size.
     *
     * @return Allocation size in bytes.
     */
    size_t get_size() { return size; }

    /**
     * @brief Get the planner allocation name.
     *
     * @return Allocation name.
     */
    std::string get_name() { return name; }

    /**
     * @brief Print lifetime, size, and arena offsets for diagnostics.
     */
    void print()
    {
        printf("name:%s, from %d to %d, size:%zu, offset:(%lu, %lu)\n",
               name.c_str(),
               time_begin,
               time_end,
               size,
               static_cast<unsigned long>(offset),
               static_cast<unsigned long>(internal_offset));
    }
};

/**
 * @brief Tensor-specific planner metadata.
 */
class TensorInfo : public MemoryInfo {
private:
    std::vector<int> shape;
    dtype_t dtype;
    int exponent;
    TensorInfo
        *m_follower_dirty_tensor; // Only reference the follower tensor which will modify the data of leader tensor.
    TensorInfo
        *m_follower_clean_tensor; // Only reference the follower tensor which don't modify the data of leader tensor.

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
     * @brief Set the inplace follower dirty tensor object
     *
     * @param tensor Inplace follower dirty tensor
     */
    void set_inplace_follower_dirty_tensor(TensorInfo *tensor) { m_follower_dirty_tensor = tensor; }

    /**
     * @brief Set the inplace follower clean tensor object
     *
     * @param tensor Inplace follower clean tensor
     */
    void set_inplace_follower_clean_tensor(TensorInfo *tensor) { m_follower_clean_tensor = tensor; }

    /**
     * @brief Get the inplace follower tensor object
     *
     * @return std::pair<TensorInfo *, TensorInfo *>
     */
    std::pair<TensorInfo *, TensorInfo *> get_inplace_follower_tensor()
    {
        return {m_follower_dirty_tensor, m_follower_clean_tensor};
    }

    /**
     * @brief Create a TensorBase view over the planner-owned tensor storage.
     *
     * @param internal_root Base address of the internal-RAM arena.
     * @param psram_root Base address of the PSRAM arena.
     * @return TensorBase view bound to the selected planner arena.
     */
    TensorBase *create_tensor(void *internal_root, void *psram_root);

    /**
     * @brief Get the tensor shape
     *
     * @return std::vector<int>
     */
    std::vector<int> get_shape() { return this->shape; }
};

/**
 * @brief Native workspace planner metadata.
 */
class WorkspaceInfo : public MemoryInfo {
private:
    WorkspaceBindingId workspace_binding_id;
    bool psram_only;

public:
    /**
     * @brief Construct metadata for a module-native workspace.
     *
     * @param name Workspace name used in planner diagnostics.
     * @param time_begin First execution step that uses the workspace.
     * @param time_end Last execution step that uses the workspace.
     * @param size Workspace size in bytes.
     * @param workspace_binding_id Binding used to publish the planned address.
     * @param psram_only Whether internal RAM placement is forbidden.
     */
    WorkspaceInfo(const std::string &name,
                  int time_begin,
                  int time_end,
                  size_t size,
                  WorkspaceBindingId workspace_binding_id,
                  bool psram_only) :
        MemoryInfo(name, time_begin, time_end, size), workspace_binding_id(workspace_binding_id), psram_only(psram_only)
    {
    }

    /**
     * @brief Get the model-context binding for this workspace.
     *
     * @return Workspace binding identifier.
     */
    WorkspaceBindingId get_workspace_binding_id() const { return workspace_binding_id; }

    /**
     * @brief Check whether the workspace must stay in PSRAM.
     *
     * @return true when internal RAM placement is forbidden, otherwise false.
     */
    bool is_psram_only() const override { return psram_only; }

    /**
     * @brief Resolve the workspace address from the planner root arenas.
     *
     * @param internal_root Base address of the internal-RAM arena.
     * @param psram_root Base address of the PSRAM arena.
     * @return Pointer to the planned workspace storage.
     */
    void *get_data(void *internal_root, void *psram_root);
};

/**
 * @brief Memory chunk used to simulate allocation of tensors and workspaces.
 */
class MemoryChunk {
public:
    size_t size;        /*!< Memory chunk size */
    bool is_free;       /*!< Whether memory chunk is free or not */
    int offset;         /*!< Offset relative to root pointer */
    int alignment;      /*!< Memory address alignment */
    MemoryInfo *memory; /*!< Info of the planner allocation occupying this chunk */

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
     * @param memory        Planner allocation metadata
     * @param alignment     Memory chunk alignment
     */
    MemoryChunk(MemoryInfo *memory, int alignment = 16);

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
     * @brief Insert a planner allocation into a free chunk
     *
     * @param memory
     * @return MemoryChunk*
     */
    MemoryChunk *insert(MemoryInfo *memory);

    /**
     * @brief Extend free chunk and insert a planner allocation
     *
     * @param memory
     * @return MemoryChunk*
     */
    MemoryChunk *extend(MemoryInfo *memory);

    /**
     * @brief Free memory chunk.
     */
    void free()
    {
        this->is_free = true;
        this->memory = nullptr;
    }
};

/**
 * @brief Prints detailed information about memory chunks in the specified list
 * @param tag          Character string identifying the memory pool type (e.g., "PSRAM" or "INTERNAL")
 * @param memory_list  Reference to list of MemoryChunk objects to display allocation details
 */
void print_memory_list(const char *tag, std::list<MemoryChunk *> &memory_list);

/**
 * @brief Sorts memory chunks in ascending order by their allocated size
 * @param memory_list  Reference to list of MemoryChunk objects to be sorted
 */
void sort_memory_list(std::list<MemoryChunk *> &memory_list);

} // namespace dl
