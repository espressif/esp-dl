#include "dl_memory_manager.hpp"

namespace dl {
/*oooooooooooooooooo00000000000000000000 MemoryInfo 00000000000000000000ooooooooooooooooo*/

MemoryInfo::MemoryInfo(const std::string &name, int time_begin, int time_end, size_t size, bool is_internal) :
    name(name),
    time_begin(time_begin),
    time_end(time_end),
    size(size),
    call_times(0),
    offset(0),
    internal_offset(0),
    is_internal(is_internal),
    m_leader(nullptr)
{
}

void MemoryInfo::update_time(int new_time)
{
    if (m_leader) {
        m_leader->update_time(new_time);
        time_end = m_leader->get_time_end();
    } else if (new_time == -1) {
        time_end = -1;
    } else if (new_time > time_end) {
        time_end = new_time;
    }
    call_times++;
}

/*oooooooooooooooooo00000000000000000000 TensorInfo 00000000000000000000ooooooooooooooooo*/

TensorInfo::TensorInfo(std::string &name,
                       int time_begin,
                       int time_end,
                       std::vector<int> shape,
                       dtype_t dtype,
                       int exponent,
                       bool is_internal) :
    MemoryInfo(name, time_begin, time_end, 0, is_internal),
    dtype(dtype),
    exponent(exponent),
    m_follower_dirty_tensor(nullptr),
    m_follower_clean_tensor(nullptr)
{
    size_t bytes = 1;
    if (shape.size() > 0) {
        this->shape.push_back(shape[0]);
        bytes = shape[0];
        for (int i = 1; i < shape.size(); i++) {
            this->shape.push_back(shape[i]);
            bytes *= shape[i];
        }
    }

    if (dtype == DATA_TYPE_FLOAT || dtype == DATA_TYPE_INT32 || dtype == DATA_TYPE_UINT32) {
        bytes *= 4;
    } else if (dtype == DATA_TYPE_UINT8 || dtype == DATA_TYPE_INT8) {
        bytes *= 1;
    } else if (dtype == DATA_TYPE_UINT16 || dtype == DATA_TYPE_INT16) {
        bytes *= 2;
    } else if (dtype == DATA_TYPE_DOUBLE || dtype == DATA_TYPE_INT64 || dtype == DATA_TYPE_UINT64) {
        bytes *= 8;
    }
    set_size(bytes);
}

void TensorInfo::set_inplace_leader_tensor(TensorInfo *tensor)
{
    int follower_time_end = get_time_end();
    set_leader(tensor);
    if (tensor) {
        if (tensor->get_time_end() < follower_time_end || follower_time_end == -1) {
            tensor->update_time(follower_time_end);
        }
    }
}

TensorBase *TensorInfo::create_tensor(void *internal_root, void *psram_root)
{
    TensorBase *tensor = nullptr;
    uint8_t *element = nullptr;

#if CONFIG_SPIRAM
    if (get_internal_state()) {
        element = (uint8_t *)internal_root + this->get_internal_offset();
    } else {
        element = (uint8_t *)psram_root + this->get_offset();
    }
#else
    element = (uint8_t *)internal_root + this->get_internal_offset();
#endif

    tensor = new TensorBase(shape, element, exponent, dtype, false);
    return tensor;
}

void *WorkspaceInfo::get_data(void *internal_root, void *psram_root)
{
#if CONFIG_SPIRAM
    if (get_internal_state()) {
        return static_cast<uint8_t *>(internal_root) + get_internal_offset();
    }
    return static_cast<uint8_t *>(psram_root) + get_offset();
#else
    return static_cast<uint8_t *>(internal_root) + get_internal_offset();
#endif
}

/*oooooooooooooooooo00000000000000000000 MemoryChunk 00000000000000000000ooooooooooooooooo*/

MemoryChunk::MemoryChunk(size_t size, int is_free, int alignment) : size(size), is_free(is_free), alignment(alignment)
{
    this->memory = nullptr;
    this->offset = 0;
}

MemoryChunk::MemoryChunk(MemoryInfo *memory, int alignment)
{
    this->alignment = alignment;
    this->size = dl::tool::get_aligned_size(memory->get_size(), this->alignment);
    this->is_free = false;
    this->memory = memory;
    this->offset = 0;
}

MemoryChunk *MemoryChunk::merge_free_chunk(MemoryChunk *chunk)
{
    if (chunk != nullptr) {
        if (chunk->is_free) {
            this->size = this->size + chunk->size;
            if (chunk->offset < this->offset) {
                this->offset = chunk->offset;
            }
            return this;
        }
    }
    return nullptr;
}

MemoryChunk *MemoryChunk::insert(MemoryInfo *memory)
{
    int aligned_size = dl::tool::get_aligned_size(memory->get_size(), this->alignment);

    if (this->is_free && this->size >= aligned_size) {
        this->memory = memory;
        this->is_free = false;
        if (this->size > aligned_size) {
            MemoryChunk *chunk = new MemoryChunk(this->size - aligned_size, true, this->alignment);
            this->size = aligned_size;
            chunk->offset = this->offset + aligned_size;
            return chunk;
        }
    }
    return nullptr;
}

MemoryChunk *MemoryChunk::extend(MemoryInfo *memory)
{
    int aligned_size = dl::tool::get_aligned_size(memory->get_size(), this->alignment);

    // only extend the size of memory chunk
    if (this->is_free && this->size < aligned_size) {
        this->memory = memory;
        this->is_free = false;
        this->size = aligned_size;
        return this;
    }
    return nullptr;
}

void print_memory_list(const char *tag, std::list<MemoryChunk *> &memory_list)
{
    for (auto it = memory_list.begin(); it != memory_list.end(); ++it) {
        std::string name = "";
        std::string state = "false";
        if ((*it)->memory) {
            name = (*it)->memory->get_name();
        }
        if ((*it)->is_free) {
            state = "true";
        }
        ESP_LOGI(tag,
                 "[size:%d, offset:%d, free:%s, tensor:%s] -> ",
                 (*it)->size,
                 (*it)->offset,
                 state.c_str(),
                 name.c_str());
    }
    printf("\n");
}

void sort_memory_list(std::list<MemoryChunk *> &memory_list)
{
    // sort free list by size
    memory_list.sort([](MemoryChunk *a, MemoryChunk *b) {
        return a->size < b->size; // 升序排序
    });
}

} // namespace dl
