#include <stdint.h>

#include "dl_memory_manager_greedy.hpp"
#include "esp_log.h"
#include <algorithm>

static const char *TAG = "MemoryManagerGreedy";

namespace dl {

bool MemoryManagerGreedy::alloc(fbs::FbsModel *fbs_model,
                                std::vector<dl::module::Module *> &execution_plan,
                                ModelContext *context,
                                const std::map<std::string, std::vector<int>> &input_shapes)
{
    std::vector<TensorInfo *> tensor_info;
    std::vector<WorkspaceInfo *> workspace_info;
    std::vector<MemoryInfo *> memory_info;

    // Replanning replaces all variable views and both root arenas, but retains
    // the graph's variable indices, names, and parameter tensors.
    context->reset_variable_arena();
    get_tensor_info_from_fbs(fbs_model, execution_plan, context, tensor_info, workspace_info, input_shapes);
    memory_info.reserve(tensor_info.size() + workspace_info.size());
    memory_info.insert(memory_info.end(), tensor_info.begin(), tensor_info.end());
    memory_info.insert(memory_info.end(), workspace_info.begin(), workspace_info.end());

    // simulate the memory allocation
#if CONFIG_SPIRAM
    if (this->max_internal_size > this->alignment) {
        simulate_with_internal_memory(memory_info, execution_plan.size());
    } else {
        simulate(memory_info, execution_plan.size());
    }
#else
    simulate(memory_info, execution_plan.size());
#endif

    void *psram_root = nullptr;
    void *internal_root = nullptr;
    int psram_size = 0;
    int internal_size = 0;

    if (!this->psram_memory_list.empty()) {
        psram_size = psram_memory_list.back()->offset + psram_memory_list.back()->size;
    }

    if (!this->internal_memory_list.empty()) {
        internal_size = internal_memory_list.back()->offset + internal_memory_list.back()->size;
    }

    // Allocate the roots transactionally before publishing any tensor or
    // workspace view into the context.
    const bool roots_allocated = context->root_alloc(internal_size, psram_size, this->alignment);
    if (roots_allocated) {
        psram_root = context->get_psram_root();
        internal_root = context->get_internal_root();

        // start to allocate tensors
        for (int i = 0; i < tensor_info.size(); i++) {
            context->update_tensor(i, tensor_info[i]->create_tensor(internal_root, psram_root));
        }
        // Native workspaces point directly into the same planner-owned arenas.
        for (WorkspaceInfo *info : workspace_info) {
            context->bind_workspace(
                info->get_workspace_binding_id(), info->get_data(internal_root, psram_root), info->get_size());
        }
    } else {
        ESP_LOGE(TAG, "root_alloc failed");
    }

    // free TensorInfo vector
    for (int i = 0; i < tensor_info.size(); i++) {
        delete tensor_info[i];
    }
    for (WorkspaceInfo *info : workspace_info) {
        delete info;
    }

    // free memory list
    this->free_memory_list();

    return roots_allocated;
}

void MemoryManagerGreedy::free()
{
    this->free_memory_list();
}

void MemoryManagerGreedy::get_tensor_info_from_fbs(fbs::FbsModel *fbs_model,
                                                   std::vector<dl::module::Module *> execution_plan,
                                                   ModelContext *context,
                                                   std::vector<TensorInfo *> &tensor_info,
                                                   std::vector<WorkspaceInfo *> &workspace_info,
                                                   const std::map<std::string, std::vector<int>> &input_shapes)
{
    tensor_info.resize(context->get_variable_count());
    // 1. add graph inputs
    std::vector<std::string> graph_inputs = fbs_model->get_graph_inputs();
    int index = -1;
    std::string name;

    for (int i = 0; i < graph_inputs.size(); i++) {
        name = graph_inputs[i];
        index = context->get_variable_index(name);

        if (index >= 0) {
            auto shape_iter = input_shapes.find(name);
            std::vector<int> shape =
                (shape_iter != input_shapes.end()) ? shape_iter->second : fbs_model->get_value_info_shape(name);
            TensorInfo *info = new TensorInfo(
                name, 0, -1, shape, fbs_model->get_value_info_dtype(name), fbs_model->get_value_info_exponent(name));
            tensor_info[index] = info;
        }
    }

    // 2. add tensor outputs and update time line of tensors
    std::vector<std::string> graph_outputs = fbs_model->get_graph_outputs();
    std::vector<std::string> sorted_nodes = fbs_model->topological_sort();
    std::vector<std::string> op_inputs;
    std::vector<std::string> op_outputs;
    for (int i = 0; i < execution_plan.size(); i++) {
        dl::module::Module *module = execution_plan[i];
        if (!module) {
            ESP_LOGE(__FUNCTION__, "module %d is nullptr\n", i);
            break;
        }

        // update the time of tensor by node's inputs
        std::vector<std::vector<int>> op_input_shapes;
        fbs_model->get_operation_inputs_and_outputs(sorted_nodes[i], op_inputs, op_outputs);

        for (int j = 0; j < op_inputs.size(); j++) {
            name = op_inputs[j];
            index = context->get_variable_index(name);
            if (index >= 0) {
                // The previously existing tensor will dirty the input. Must disconnect the inplace link.
                auto [pre_follower_dirty_tensor, pre_follower_clean_tensor] =
                    tensor_info[index]->get_inplace_follower_tensor();
                if (pre_follower_dirty_tensor) {
                    tensor_info[index]->set_inplace_follower_dirty_tensor(nullptr);
                    pre_follower_dirty_tensor->set_inplace_leader_tensor(nullptr);
                }

                auto out_iter = std::find(graph_outputs.begin(), graph_outputs.end(), name);
                if (out_iter == graph_outputs.end())
                    tensor_info[index]->update_time(i + 1); // free this tensor next step
                op_input_shapes.push_back(tensor_info[index]->get_shape());
            } else {
                TensorBase *tensor = context->get_tensor(name);
                if (tensor) {
                    op_input_shapes.push_back(tensor->get_shape());
                } else {
                    op_input_shapes.push_back({});
                }
            }
        }

        // add output tensors
        std::vector<std::vector<int>> output_shapes = module->get_output_shape(op_input_shapes);
        if ((module->inplace == MODULE_INPLACE_UNCHANGED_BUFFER || module->inplace == MODULE_INPLACE_CHANGED_BUFFER) &&
            op_outputs.size() == 1) {
            name = op_outputs[0];
            TensorInfo *inplace_tensor = nullptr;
            TensorInfo *info = new TensorInfo(name,
                                              i,
                                              -1,
                                              output_shapes[0],
                                              fbs_model->get_value_info_dtype(name),
                                              fbs_model->get_value_info_exponent(name));
            index = context->get_variable_index(name);
            tensor_info[index] = info;

            // inplace, loop all inputs and find a suitable inplace tensor
            for (int j = 0; j < op_inputs.size(); j++) {
                name = op_inputs[j];
                index = context->get_variable_index(name);
                if (index >= 0) {
                    inplace_tensor = tensor_info[index];
                    if (inplace_tensor->get_size() >= info->get_size()) {
                        auto out_iter = std::find(graph_outputs.begin(), graph_outputs.end(), name);
                        if (out_iter == graph_outputs.end()) {
                            break;
                        } else {
                            // If op_input is graph output. It can't be set inplace.
                            inplace_tensor = nullptr;
                        }
                    } else {
                        // If op_input size is less than output. It can't be set inplace.
                        inplace_tensor = nullptr;
                    }
                }
            }
            if (inplace_tensor) {
                auto [pre_follower_dirty_tensor, pre_follower_clean_tensor] =
                    inplace_tensor->get_inplace_follower_tensor();
                // The previously existing tensor will dirty the input. Must disconnect the inplace link.
                if (pre_follower_dirty_tensor) {
                    inplace_tensor->set_inplace_follower_dirty_tensor(nullptr);
                    pre_follower_dirty_tensor->set_inplace_leader_tensor(nullptr);
                }

                // Relink the inplace.
                /*********************
                 * 1.If the tensor already has a clean follower, setting it to dirty will skip the inplace
                 *   operation to avoid contaminating the current clean follower tensor.
                 * 2.If the tensor already has a clean follower, setting it to clean again has no effect
                 *   and proceeds normally.
                 * 3.If the tensor already has a dirty follower, the code first disconnects the existing
                 *   dirty follower, then proceeds normally when setting a new dirty follower.
                 * 4.If the tensor already has a dirty follower, the code first disconnects the existing
                 *   dirty follower, then proceeds normally when setting a new clean follower.
                 *********************/
                info->set_inplace_leader_tensor(inplace_tensor);
                if (module->inplace == MODULE_INPLACE_CHANGED_BUFFER) {
                    if (pre_follower_clean_tensor) {
                        info->set_inplace_leader_tensor(nullptr);
                    } else {
                        inplace_tensor->set_inplace_follower_dirty_tensor(info);
                    }
                } else if (module->inplace == MODULE_INPLACE_UNCHANGED_BUFFER) {
                    inplace_tensor->set_inplace_follower_clean_tensor(info);
                } else {
                    // Nothing to do.
                }
            }
        } else {
            for (int j = 0; j < op_outputs.size(); j++) {
                name = op_outputs[j];
                TensorInfo *info = new TensorInfo(name,
                                                  i,
                                                  -1,
                                                  output_shapes[j],
                                                  fbs_model->get_value_info_dtype(name),
                                                  fbs_model->get_value_info_exponent(name));
                index = context->get_variable_index(name);
                tensor_info[index] = info;
            }
        }

        // Add native workspace buffers, which only live while this node runs.
        std::vector<dl::module::WorkspaceRequest> requests = module->get_workspace_requests(op_input_shapes);
        module->m_workspace_binding_ids.assign(requests.size(), kInvalidWorkspaceBindingId);
        for (int j = 0; j < requests.size(); j++) {
            name = sorted_nodes[i] + ":workspace" + std::to_string(j);
            module->m_workspace_binding_ids[j] = context->create_workspace_binding(name);
            if (requests[j].size == 0) {
                continue;
            }
            workspace_info.push_back(new WorkspaceInfo(
                name, i, i + 1, requests[j].size, module->m_workspace_binding_ids[j], requests[j].psram_only));
        }
    }
}

void MemoryManagerGreedy::simulate(std::vector<MemoryInfo *> &memory_info, int node_num)
{
    std::vector<std::vector<MemoryInfo *>> node_alloc_memory(node_num);
    std::vector<std::vector<MemoryInfo *>> node_free_memory(node_num);

    for (int i = 0; i < node_num; i++) {
        node_alloc_memory[i] = {};
        node_free_memory[i] = {};
    }

    for (MemoryInfo *info : memory_info) {
        if (info->is_inplaced()) {
            continue;
        }

        int time_begin = info->get_time_begin();
        int time_end = info->get_time_end();

        if (time_begin >= 0 && time_begin < node_num) {
            node_alloc_memory[time_begin].push_back(info);
        }

        if (time_end >= 0 && time_end < node_num) {
            node_free_memory[time_end].push_back(info);
        }
    }

    for (int i = 0; i < node_num; i++) {
        for (MemoryInfo *info : node_free_memory[i]) {
#if CONFIG_SPIRAM
            free_memory(info, this->psram_memory_list, this->psram_free_list);
#else
            free_memory(info, this->internal_memory_list, this->internal_free_list);
#endif
        }

        for (MemoryInfo *info : node_alloc_memory[i]) {
            alloc_memory(info);
        }

        // print_memory_list("psram memory list:", this->memory_list);
        // print_memory_list("psram free list:", this->free_list);
    }
}

void MemoryManagerGreedy::simulate_with_internal_memory(std::vector<MemoryInfo *> &memory_info, int node_num)
{
    MemoryChunk *internal_chunk = new MemoryChunk(this->max_internal_size, true, this->alignment);
    this->internal_memory_list.push_back(internal_chunk);
    this->internal_free_list.push_back(internal_chunk);

    std::vector<std::vector<MemoryInfo *>> node_alloc_memory(node_num);
    std::vector<std::vector<MemoryInfo *>> node_free_memory(node_num);

    for (int i = 0; i < node_num; i++) {
        node_alloc_memory[i] = {};
        node_free_memory[i] = {};
    }

    for (MemoryInfo *info : memory_info) {
        if (info->is_inplaced()) {
            continue;
        }

        int time_begin = info->get_time_begin();
        int time_end = info->get_time_end();

        if (time_begin >= 0 && time_begin < node_num) {
            node_alloc_memory[time_begin].push_back(info);
        }

        if (time_end >= 0 && time_end < node_num) {
            node_free_memory[time_end].push_back(info);
        }
    }

    for (int i = 0; i < node_num; i++) {
        for (MemoryInfo *info : node_free_memory[i]) {
            if (info->get_internal_state()) {
                free_memory(info, this->internal_memory_list, this->internal_free_list);
            } else {
                free_memory(info, this->psram_memory_list, this->psram_free_list);
            }
        }

        for (MemoryInfo *info : node_alloc_memory[i]) {
            MemoryChunk *chunk = info->is_psram_only() ? nullptr : alloc_internal_memory(info);
            if (chunk == nullptr) {
                chunk = alloc_memory(info);
            }
        }

        // print_memory_list("psram memory list", this->memory_list);
        // print_memory_list("psram free list", this->free_list);
        // print_memory_list("internal memory list", this->internal_memory_list);
        // print_memory_list("internal free list", this->internal_free_list);
    }
}

MemoryChunk *MemoryManagerGreedy::free_memory(MemoryInfo *memory,
                                              std::list<MemoryChunk *> &memory_list,
                                              std::list<MemoryChunk *> &free_list)
{
    MemoryChunk *chunk = nullptr;
    for (auto it = memory_list.begin(); it != memory_list.end(); ++it) {
        chunk = *it;
        if (chunk->memory == memory) {
            chunk->free();

            // merge with the next chunk
            auto next_it = std::next(it, 1);
            if (next_it != memory_list.end()) {
                MemoryChunk *next_chunk = *next_it;
                if (chunk->merge_free_chunk(next_chunk)) {
                    auto free_it = std::find(free_list.begin(), free_list.end(), next_chunk);
                    memory_list.erase(next_it);
                    free_list.erase(free_it);
                    delete next_chunk;
                }
            }

            // merge with the previous chunk
            if (it != memory_list.begin()) {
                auto prev_it = std::prev(it);
                MemoryChunk *prev_chunk = *prev_it;
                if (chunk->merge_free_chunk(prev_chunk)) {
                    auto free_it = std::find(free_list.begin(), free_list.end(), prev_chunk);
                    memory_list.erase(prev_it);
                    free_list.erase(free_it);
                    delete prev_chunk;
                }
            }

            // sort free list
            free_list.push_back(chunk);
            sort_memory_list(free_list);
            return chunk;
        }
    }
    return chunk;
}

MemoryChunk *MemoryManagerGreedy::alloc_memory(MemoryInfo *memory, int mode)
{
#if CONFIG_SPIRAM
    std::list<MemoryChunk *> &memory_list = this->psram_memory_list;
    std::list<MemoryChunk *> &free_list = this->psram_free_list;
#else
    std::list<MemoryChunk *> &memory_list = this->internal_memory_list;
    std::list<MemoryChunk *> &free_list = this->internal_free_list;
#endif
    // printf("alloc tensor:%s\n", tensor->name.c_str());
    MemoryChunk *chunk = nullptr;
    for (auto it = free_list.begin(); it != free_list.end(); ++it) {
        if ((*it)->size >= memory->get_size()) {
            // find a valid free memory chunk, split it and put the tensor into it
            chunk = *it;
            auto mem_it = std::find(memory_list.begin(), memory_list.end(), chunk);
            MemoryChunk *split_chunk = chunk->insert(memory);
            free_list.erase(it); // remove this memory chunk in free list
            if (split_chunk != nullptr) {
                memory_list.insert(std::next(mem_it, 1), split_chunk); // add split memory chunk in memory list
                free_list.push_back(split_chunk);                      // add split memory chunk in free list
                sort_memory_list(free_list);                           // sort free list
            }
            break;
        }
    }

    if (chunk == nullptr) {
        if (!memory_list.empty()) {
            MemoryChunk *last_chunk = memory_list.back();
            if (last_chunk->is_free) {
                auto last_it = std::find(free_list.begin(), free_list.end(), last_chunk);
                free_list.erase(last_it);
                sort_memory_list(free_list);
                chunk = last_chunk->extend(memory);
            }
        }

        if (chunk == nullptr) {
            // add a new memory chunk
            chunk = new MemoryChunk(memory, this->alignment);
            if (!memory_list.empty()) {
                MemoryChunk *last = memory_list.back();
                chunk->offset = last->offset + last->size;
            }
            memory_list.push_back(chunk);
        }
    }
#if CONFIG_SPIRAM
    memory->set_offset(chunk->offset);
#else
    memory->set_internal_offset(chunk->offset);
#endif
    return chunk;
}

MemoryChunk *MemoryManagerGreedy::alloc_internal_memory(MemoryInfo *memory, int mode)
{
    // printf("alloc tensor:%s\n", tensor->name.c_str());
    MemoryChunk *chunk = nullptr;
    for (auto it = internal_free_list.begin(); it != internal_free_list.end(); ++it) {
        if ((*it)->size >= memory->get_size()) {
            // find a valid free memory chunk, split it and put the tensor into it
            chunk = *it;
            auto mem_it = std::find(internal_memory_list.begin(), internal_memory_list.end(), chunk);
            MemoryChunk *split_chunk = chunk->insert(memory);
            internal_free_list.erase(it); // remove this memory chunk in free list
            if (split_chunk != nullptr) {
                internal_memory_list.insert(std::next(mem_it, 1), split_chunk); // add split memory chunk in memory list
                internal_free_list.push_back(split_chunk);                      // add split memory chunk in free list
                sort_memory_list(internal_free_list);                           // sort free list
            }
            memory->set_internal_offset(chunk->offset);
            break;
        }
    }

    return chunk;
}

void MemoryManagerGreedy::free_memory_list()
{
    if (!psram_memory_list.empty()) {
        for (auto it = psram_memory_list.begin(); it != psram_memory_list.end(); ++it) {
            MemoryChunk *chunk = *it;
            delete chunk;
        }
        psram_memory_list.clear();
        psram_free_list.clear();
    }

    if (!internal_memory_list.empty()) {
        for (auto it = internal_memory_list.begin(); it != internal_memory_list.end(); ++it) {
            MemoryChunk *chunk = *it;
            delete chunk;
        }
        internal_memory_list.clear();
        internal_free_list.clear();
    }
}

} // namespace dl
