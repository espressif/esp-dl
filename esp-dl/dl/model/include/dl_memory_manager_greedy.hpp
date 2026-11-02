#pragma once

#include "dl_memory_manager.hpp"

namespace dl {

/**
 * @brief Greedy memory manager that allocates memory for tensors in execution order,
 * prioritizing internal RAM allocation first.
 */
class MemoryManagerGreedy : public MemoryManagerBase {
private:
    size_t max_internal_size;                      /*!< Maximum allowed internal RAM usage in bytes.
                                                      Effective only when PSRAM is available */
    std::list<MemoryChunk *> psram_memory_list;    /*!< List of allocated PSRAM memory blocks */
    std::list<MemoryChunk *> psram_free_list;      /*!< List of free PSRAM memory blocks */
    std::list<MemoryChunk *> internal_memory_list; /*!< List of allocated internal RAM memory blocks */
    std::list<MemoryChunk *> internal_free_list;   /*!< List of free internal RAM memory blocks */

    /**
     * @brief Extracts tensor metadata (shape, data type, size) from FlatBuffer model
     * and execution plan for memory planning
     * @param fbs_model FlatBuffer representation of the neural network model
     * @param execution_plan Topologically sorted list of computation modules
     * @param context Runtime context containing device-specific configurations
     * @param tensor_info Output vector to store TensorInfo objects for all tensors
     * @param workspace_info Output vector to store native WorkspaceInfo objects
     * @param input_shapes Optional map to override graph input shapes
     */
    void get_tensor_info_from_fbs(fbs::FbsModel *fbs_model,
                                  std::vector<dl::module::Module *> execution_plan,
                                  ModelContext *context,
                                  std::vector<TensorInfo *> &tensor_info,
                                  std::vector<WorkspaceInfo *> &workspace_info,
                                  const std::map<std::string, std::vector<int>> &input_shapes = {});

    /**
     * @brief Simulates memory allocation process for given tensor information
     * @param memory_info Vector containing metadata for tensors and workspaces
     * @param node_num Total number of nodes in the execution plan
     */
    void simulate(std::vector<MemoryInfo *> &memory_info, int node_num);

    /**
     * @brief Simulates memory allocation with priority to internal RAM
     * @param memory_info Vector containing tensor and workspace metadata
     * @param node_num Total computation nodes in the network
     */
    void simulate_with_internal_memory(std::vector<MemoryInfo *> &memory_info, int node_num);

    /**
     * @brief Releases a planner allocation and returns its chunk to the free list
     * @param memory Planner allocation whose memory needs to be freed
     * @param memory_list List containing all memory blocks of the memory type
     * @param free_list List to which freed memory blocks should be added
     * @return MemoryChunk* Pointer to the freed memory block
     */
    MemoryChunk *free_memory(MemoryInfo *memory,
                             std::list<MemoryChunk *> &memory_list,
                             std::list<MemoryChunk *> &free_list);

    /**
     * @brief Allocates storage for a planner allocation from the selected memory pool
     * @param memory Planner allocation containing size and placement requirements
     * @param mode Allocation mode (0=standard, other values reserved for future use)
     * @return MemoryChunk* Pointer to newly allocated memory block or nullptr on failure
     */
    MemoryChunk *alloc_memory(MemoryInfo *memory, int mode = 0);

    /**
     * @brief Allocates storage for a planner allocation from internal RAM
     * @param memory Planner allocation specifying memory requirements
     * @param mode Allocation mode (0=standard, other values reserved)
     * @return MemoryChunk* Pointer to allocated internal memory block or nullptr
     */
    MemoryChunk *alloc_internal_memory(MemoryInfo *memory, int mode = 0);

    /**
     * @brief Releases all allocated memory blocks in both PSRAM and internal memory pools
     */
    void free_memory_list();

public:
    /**
     * @brief Constructs a greedy memory manager with specified constraints
     * @param max_internal_size Maximum allowed internal RAM usage in bytes
     * @param alignment Memory address alignment requirement (default: 16 bytes)
     */
    MemoryManagerGreedy(int max_internal_size, int alignment = 16) :
        MemoryManagerBase(alignment),
        max_internal_size(max_internal_size > 0 ? static_cast<size_t>(max_internal_size) : 0)
    {
        size_t largest_internal_size = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
        if (this->max_internal_size > largest_internal_size) {
            this->max_internal_size = largest_internal_size;
        }
    }

    /**
     * @brief Destructor that releases all managed memory resources
     */
    ~MemoryManagerGreedy() { this->free_memory_list(); }

    /**
     * @brief Allocates memory for all network tensors following greedy strategy
     * @param fbs_model FlatBuffer model containing network architecture
     * @param execution_plan Execution graph ordered by computation dependencies
     * @param context Device-specific runtime configuration
     * @param input_shapes Optional map to override shapes of graph inputs for allocation
     * @return bool True if successful allocation, false if memory insufficient
     */
    bool alloc(fbs::FbsModel *fbs_model,
               std::vector<dl::module::Module *> &execution_plan,
               ModelContext *context,
               const std::map<std::string, std::vector<int>> &input_shapes = {}) override;

    /**
     * @brief Releases all allocated memory including tensor buffers and memory pools
     */
    void free();
};
} // namespace dl
