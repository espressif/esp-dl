#include <stdint.h>

#include "dl_memory_manager_greedy.hpp"
#include "esp_log.h"

static const char *TAG = "MemoryManagerGreedy";

namespace dl {

bool MemoryManagerGreedy::alloc(fbs::FbsModel *fbs_model,
                                std::vector<dl::module::Module *> &execution_plan,
                                ModelContext *context)
{
    std::vector<TensorInfo *> tensor_info;
    // get all tensor info from flatbuffers
    get_tensor_info_from_fbs(fbs_model, execution_plan, context, tensor_info);

    // simulate the memory allocation
#if CONFIG_SPIRAM
    if (this->max_internal_size > this->alignment) {
        simulate_with_internal_memory(tensor_info, execution_plan.size());
    } else {
        simulate(tensor_info, execution_plan.size());
    }
#else
    simulate(tensor_info, execution_plan.size());
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

    // alloc memory for tensors
    if (context->root_alloc(internal_size, psram_size, this->alignment)) {
        psram_root = context->get_psram_root();
        internal_root = context->get_internal_root();

        // start to allocate tensors
        for (int i = 0; i < tensor_info.size(); i++) {
            context->update_tensor(i, tensor_info[i]->create_tensor(internal_root, psram_root));
        }
    } else {
        ESP_LOGE(TAG, "root_alloc failed");
    }

    // free TensorInfo vector
    for (int i = 0; i < tensor_info.size(); i++) {
        delete tensor_info[i];
    }

    // free memory list
    this->free_memory_list();

    if (psram_root || internal_root) {
        return true;
    }

    return false;
}

void MemoryManagerGreedy::free()
{
    this->free_memory_list();
}

void MemoryManagerGreedy::get_tensor_info_from_fbs(fbs::FbsModel *fbs_model,
                                                   std::vector<dl::module::Module *> execution_plan,
                                                   ModelContext *context,
                                                   std::vector<TensorInfo *> &tensor_info)
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
            TensorInfo *info = new TensorInfo(name,
                                              0,
                                              -1,
                                              fbs_model->get_value_info_shape(name),
                                              fbs_model->get_value_info_dtype(name),
                                              fbs_model->get_value_info_exponent(name));
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
        std::vector<std::vector<int>> input_shapes;
        fbs_model->get_operation_inputs_and_outputs(sorted_nodes[i], op_inputs, op_outputs);

        for (int j = 0; j < op_inputs.size(); j++) {
            name = op_inputs[j];
            index = context->get_variable_index(name);
            if (index >= 0) {
                // The previously existing tensor will dirty the input. Must disconnect the inplace link.
                TensorInfo *follower_tensor = tensor_info[index]->get_inplace_follower_tensor();
                if (follower_tensor) {
                    tensor_info[index]->set_inplace_follower_tensor(nullptr);
                    follower_tensor->set_inplace_leader_tensor(nullptr);
                }

                auto out_iter = std::find(graph_outputs.begin(), graph_outputs.end(), name);
                if (out_iter == graph_outputs.end())
                    tensor_info[index]->update_time(i + 1); // free this tensor next step
                input_shapes.push_back(tensor_info[index]->get_shape());
            } else {
                TensorBase *tensor = context->get_tensor(name);
                if (tensor) {
                    input_shapes.push_back(tensor->get_shape());
                } else {
                    input_shapes.push_back({});
                }
            }
        }

        // add output tensors
        std::vector<std::vector<int>> output_shapes = module->get_output_shape(input_shapes);
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
                TensorInfo *pre_follower_tensor = inplace_tensor->get_inplace_follower_tensor();
                // The previously existing tensor will dirty the input. Must disconnect the inplace link.
                if (pre_follower_tensor) {
                    inplace_tensor->set_inplace_follower_tensor(nullptr);
                    pre_follower_tensor->set_inplace_leader_tensor(nullptr);
                }

                // Relink the inplace.
                info->set_inplace_leader_tensor(inplace_tensor);
                if (module->inplace == MODULE_INPLACE_CHANGED_BUFFER) {
                    inplace_tensor->set_inplace_follower_tensor(info);
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
    }
}

void MemoryManagerGreedy::simulate(std::vector<TensorInfo *> &tensor_info, int node_num)
{
    std::vector<std::vector<TensorInfo *>> node_alloc_tensors(node_num);
    std::vector<std::vector<TensorInfo *>> node_free_tensors(node_num);

    for (int i = 0; i < node_num; i++) {
        node_alloc_tensors[i] = {};
        node_free_tensors[i] = {};
    }

    for (int i = 0; i < tensor_info.size(); i++) {
        // If this tensor is inplaced by other tensor, skip it
        if (tensor_info[i]->is_inplaced()) {
            continue;
        }

        int time_begin = tensor_info[i]->get_time_begin();
        int time_end = tensor_info[i]->get_time_end();

        if (time_begin >= 0 && time_begin < node_num) {
            node_alloc_tensors[time_begin].push_back(tensor_info[i]);
        }

        if (time_end >= 0 && time_end < node_num) {
            node_free_tensors[time_end].push_back(tensor_info[i]);
        }
    }

    for (int i = 0; i < node_num; i++) {
        for (auto it = node_free_tensors[i].begin(); it != node_free_tensors[i].end(); it++) {
#if CONFIG_SPIRAM
            free_tensor(*it, this->psram_memory_list, this->psram_free_list);
#else
            free_tensor(*it, this->internal_memory_list, this->internal_free_list);
#endif
        }

        for (auto it = node_alloc_tensors[i].begin(); it != node_alloc_tensors[i].end(); it++) {
            alloc_tensor(*it);
        }

        // print_memory_list("psram memory list:", this->memory_list);
        // print_memory_list("psram free list:", this->free_list);
    }
}

void MemoryManagerGreedy::simulate_with_internal_memory(std::vector<TensorInfo *> &tensor_info, int node_num)
{
    MemoryChunk *internal_chunk = new MemoryChunk(this->max_internal_size, true, this->alignment);
    this->internal_memory_list.push_back(internal_chunk);
    this->internal_free_list.push_back(internal_chunk);

    std::vector<std::vector<TensorInfo *>> node_alloc_tensors(node_num);
    std::vector<std::vector<TensorInfo *>> node_free_tensors(node_num);

    for (int i = 0; i < node_num; i++) {
        node_alloc_tensors[i] = {};
        node_free_tensors[i] = {};
    }

    for (int i = 0; i < tensor_info.size(); i++) {
        // If this tensor is inplaced by other tensor, skip it
        if (tensor_info[i]->is_inplaced()) {
            continue;
        }

        int time_begin = tensor_info[i]->get_time_begin();
        int time_end = tensor_info[i]->get_time_end();

        if (time_begin >= 0 && time_begin < node_num) {
            node_alloc_tensors[time_begin].push_back(tensor_info[i]);
        }

        if (time_end >= 0 && time_end < node_num) {
            node_free_tensors[time_end].push_back(tensor_info[i]);
        }
    }

    for (int i = 0; i < node_num; i++) {
        for (auto it = node_free_tensors[i].begin(); it != node_free_tensors[i].end(); it++) {
            if ((*it)->get_internal_state()) {
                free_tensor(*it, this->internal_memory_list, this->internal_free_list);
            } else {
                free_tensor(*it, this->psram_memory_list, this->psram_free_list);
            }
        }

        for (auto it = node_alloc_tensors[i].begin(); it != node_alloc_tensors[i].end(); it++) {
            MemoryChunk *chunk = alloc_internal_tensor(*it);
            if (chunk == nullptr) {
                chunk = alloc_tensor(*it);
            }
        }

        // print_memory_list("psram memory list", this->memory_list);
        // print_memory_list("psram free list", this->free_list);
        // print_memory_list("internal memory list", this->internal_memory_list);
        // print_memory_list("internal free list", this->internal_free_list);
    }
}

MemoryChunk *MemoryManagerGreedy::free_tensor(TensorInfo *tensor,
                                              std::list<MemoryChunk *> &memory_list,
                                              std::list<MemoryChunk *> &free_list)
{
    MemoryChunk *chunk = nullptr;
    for (auto it = memory_list.begin(); it != memory_list.end(); ++it) {
        chunk = *it;
        if (chunk->tensor == tensor) {
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

MemoryChunk *MemoryManagerGreedy::alloc_tensor(TensorInfo *tensor, int mode)
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
        if ((*it)->size >= tensor->get_size()) {
            // find a valid free memory chunk, split it and put the tensor into it
            chunk = *it;
            auto mem_it = std::find(memory_list.begin(), memory_list.end(), chunk);
            MemoryChunk *split_chunk = chunk->insert(tensor);
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
                chunk = last_chunk->extend(tensor);
            }
        }

        if (chunk == nullptr) {
            // add a new memory chunk
            chunk = new MemoryChunk(tensor, this->alignment);
            if (!memory_list.empty()) {
                MemoryChunk *last = memory_list.back();
                chunk->offset = last->offset + last->size;
            }
            memory_list.push_back(chunk);
        }
    }
    tensor->set_offset(chunk->offset);
    return chunk;
}

MemoryChunk *MemoryManagerGreedy::alloc_internal_tensor(TensorInfo *tensor, int mode)
{
    // printf("alloc tensor:%s\n", tensor->name.c_str());
    MemoryChunk *chunk = nullptr;
    for (auto it = internal_free_list.begin(); it != internal_free_list.end(); ++it) {
        if ((*it)->size >= tensor->get_size()) {
            // find a valid free memory chunk, split it and put the tensor into it
            chunk = *it;
            auto mem_it = std::find(internal_memory_list.begin(), internal_memory_list.end(), chunk);
            MemoryChunk *split_chunk = chunk->insert(tensor);
            internal_free_list.erase(it); // remove this memory chunk in free list
            if (split_chunk != nullptr) {
                internal_memory_list.insert(std::next(mem_it, 1), split_chunk); // add split memory chunk in memory list
                internal_free_list.push_back(split_chunk);                      // add split memory chunk in free list
                sort_memory_list(internal_free_list);                           // sort free list
            }
            tensor->set_internal_offset(chunk->offset);
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
