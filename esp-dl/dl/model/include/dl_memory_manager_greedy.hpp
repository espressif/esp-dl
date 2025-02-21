#pragma once

#include "dl_memory_manager.hpp"

namespace dl {
namespace memory {

/**
 * @brief Greedy memory manager, allocate memory for each tensor in the order of the execution plan.
 */
class MemoryManagerGreedy : public MemoryManagerBase {
    // refer to https://zhuanlan.zhihu.com/p/423688020
private:
    size_t max_internal_size; /*!< In bytes. Limit the max internal size usage. Only take effect when there's a PSRAM,
                                 and you want to alloc memory on internal RAM first */
    std::list<MemoryChunk *> psram_memory_list;    /*!< All PSRAM memory chunk list */
    std::list<MemoryChunk *> psram_free_list;      /*!< Free PSRAM memory chunk list */
    std::list<MemoryChunk *> internal_memory_list; /*!< All internal memory chunk list */
    std::list<MemoryChunk *> internal_free_list;   /*!< Free internal memory chunk list */

    void get_tensor_info_from_fbs(fbs::FbsModel *fbs_model,
                                  std::vector<dl::module::Module *> execution_plan,
                                  std::vector<TensorInfo *> &tensor_info);

    void simulate(std::vector<TensorInfo *> &tensor_info, int node_num);

    void simulate_with_internal_memory(std::vector<TensorInfo *> &tensor_info, int node_num);

    MemoryChunk *free_tensor(TensorInfo *tensor,
                             std::list<MemoryChunk *> &memory_list,
                             std::list<MemoryChunk *> &free_list);

    MemoryChunk *alloc_tensor(TensorInfo *tensor, int mode = 0);

    MemoryChunk *alloc_internal_tensor(TensorInfo *tensor, int mode = 0);

    void free_memory_list();

public:
    /**
     * @brief Construct a new Memory Manager Greedy object
     *
     * @param max_internal_size In bytes. Limit the max internal size usage. Only take effect when there's a PSRAM,
                                and you want to alloc memory on internal RAM first
     * @param alignment         Memory address alignment
     */
    MemoryManagerGreedy(int max_internal_size, int alignment = 16) :
        MemoryManagerBase(alignment), max_internal_size(max_internal_size)
    {
    }

    /**
     * @brief Destroy the Memory Manager Greedy object
     *
     */
    ~MemoryManagerGreedy() { this->free_memory_list(); }

    /**
     * @brief Allocate memory for each tensor, include all input and output tensors
     *
     * @param fbs_model       FlatBuffer's Model
     * @param execution_plan  Topological sorted module list
     */
    void alloc(fbs::FbsModel *fbs_model, std::vector<dl::module::Module *> &execution_plan);

    /**
     * @brief Set preload address for module's parameters
     *
     * @param execution_plan   Topological sorted module list
     */
    void set_preload_addr(std::vector<dl::module::Module *> execution_plan);

    /**
     * @brief Free memory, include all tensors and memory list
     */
    void free();
};
} // namespace memory
} // namespace dl
