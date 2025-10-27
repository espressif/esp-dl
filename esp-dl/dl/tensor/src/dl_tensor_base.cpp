#include "dl_tensor_base.hpp"
#include "dl_base_pad.hpp"
#include "dl_base_requantize_linear.hpp"
#include <iostream>
namespace dl {

template <typename RT, typename T>
RT quantize(T input, float inv_scale)
{
    int output = tool::round(input * inv_scale);
    output = DL_CLIP(
        output, sizeof(RT) == 1 ? DL_QUANT8_MIN : DL_QUANT16_MIN, sizeof(RT) == 1 ? DL_QUANT8_MAX : DL_QUANT16_MAX);
    return static_cast<RT>(output);
}

template int8_t quantize<int8_t, float>(float input, float scale);
template int16_t quantize<int16_t, float>(float input, float scale);
template int8_t quantize<int8_t, double>(double input, float scale);
template int16_t quantize<int16_t, double>(double input, float scale);

template <typename T, typename RT>
RT dequantize(T input, float scale)
{
    RT output = static_cast<RT>(input) * scale;
    return output;
}

template float dequantize<int8_t, float>(int8_t input, float scale);
template float dequantize<int16_t, float>(int16_t input, float scale);
template double dequantize<int8_t, double>(int8_t input, float scale);
template double dequantize<int16_t, double>(int16_t input, float scale);

size_t dtype_sizeof(dtype_t dtype)
{
    switch (dtype) {
    case DATA_TYPE_FLOAT:
        return sizeof(float);
    case DATA_TYPE_INT8:
        return sizeof(int8_t);
    case DATA_TYPE_UINT8:
        return sizeof(uint8_t);
    case DATA_TYPE_INT16:
        return sizeof(int16_t);
    case DATA_TYPE_UINT16:
        return sizeof(uint16_t);
    case DATA_TYPE_INT32:
        return sizeof(int32_t);
    case DATA_TYPE_UINT32:
        return sizeof(uint32_t);
    case DATA_TYPE_INT64:
        return sizeof(int64_t);
    case DATA_TYPE_UINT64:
        return sizeof(uint64_t);
    case DATA_TYPE_BOOL:
        return sizeof(bool);
    case DATA_TYPE_DOUBLE:
        return sizeof(double);
    case DATA_TYPE_FLOAT16:
        return 2;
    default:
        return 1;
    }
    return 1;
}

const char *dtype_to_string(dtype_t dtype)
{
    switch (dtype) {
    case DATA_TYPE_FLOAT:
        return "float";
    case DATA_TYPE_UINT8:
        return "uint8";
    case DATA_TYPE_INT8:
        return "int8";
    case DATA_TYPE_UINT16:
        return "uint16";
    case DATA_TYPE_INT16:
        return "int16";
    case DATA_TYPE_INT32:
        return "int32";
    case DATA_TYPE_UINT32:
        return "unit32";
    case DATA_TYPE_DOUBLE:
        return "double";
    case DATA_TYPE_STRING:
        return "string";
    case DATA_TYPE_BOOL:
        return "bool";
    case DATA_TYPE_FLOAT16:
        return "float16";
    case DATA_TYPE_INT64:
        return "int64";
    case DATA_TYPE_UINT64:
        return "uint64";
    case DATA_TYPE_UNDEFINED:
        return "undefined";
    default:
        return "undefined";
    }
    return "undefined";
}

const char *activation_type_to_string(activation_type_t type)
{
    switch (type) {
    case Linear:
        return "None";
    case ReLU:
        return "ReLU";
    case LeakyReLU:
        return "LeakyReLU";
    case PReLU:
        return "PReLU";
    default:
        return "None";
    }
    return "None";
}

const char *quant_type_to_string(quant_type_t type)
{
    switch (type) {
    case QUANT_TYPE_SYMM_8BIT:
        return "symm 8bit";
    case QUANT_TYPE_SYMM_16BIT:
        return "symm 16bit";
    case QUANT_TYPE_SYMM_32BIT:
        return "symm 32bit";
    case QUANT_TYPE_FLOAT32:
        return "float";
    default:
        return "None";
    }
    return "None";
}

TensorBase::TensorBase(
    std::vector<int> shape, const void *element, int exponent, dtype_t dtype, bool deep, uint32_t caps)
{
    this->set_shape(shape);
    this->exponent = exponent;
    this->dtype = dtype;
    this->cache = nullptr;
    size_t dtype_bytes = this->get_dtype_bytes();
    size_t aligned_size = this->get_aligned_size();
    if (element) {
        if (deep) {
            this->auto_free = true;
            this->data = tool::calloc_aligned(16, aligned_size, dtype_bytes, caps);
            tool::copy_memory(this->data, const_cast<void *>(element), this->get_size() * dtype_bytes);
        } else {
            this->auto_free = false;
            this->data = const_cast<void *>(element);
        }
    } else {
        this->auto_free = true;
        this->data = tool::calloc_aligned(16, aligned_size, dtype_bytes, caps);
    }
    if ((!element || deep) && !this->data) {
        ESP_LOGE(
            "TensorBase",
            "Failed to alloc %.2fKB RAM, largest available PSRAM block size %.2fKB, internal RAM block size %.2fKB",
            size / 1024.f,
            heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024.f,
            heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024.f);
    }
    this->caps = caps;
}

bool TensorBase::assign(TensorBase *tensor, int start_position)
{
    if (tensor == nullptr || this->get_size() != tensor->get_size()) {
        return false;
    }

    if (this->exponent == tensor->exponent && this->dtype == tensor->dtype) {
        tool::copy_memory(this->data, tensor->data, this->get_bytes());
    } else if (tensor->dtype == DATA_TYPE_FLOAT) {
        float *src_data = (float *)tensor->data;
        float inv_scale = DL_RESCALE(this->exponent);

        if (this->dtype == DATA_TYPE_INT8) {
            int8_t *data = (int8_t *)this->data;
            int data_size = this->get_size();
            // Ensure start_position is within valid range [0, data_size)
            if (start_position < 0 || start_position >= data_size)
                return false;

            int i = 0;
            // Copy from start_position to end of src_data
            for (int j = start_position; j < data_size; i++, j++) {
                data[i] = quantize<int8_t>(src_data[j], inv_scale);
            }
            // Wrap around and copy from start of src_data to start_position - 1
            for (int j = 0; j < start_position; ++i, ++j) {
                data[i] = quantize<int8_t>(src_data[j], inv_scale);
            }

        } else if (this->dtype == DATA_TYPE_INT16) {
            int16_t *data = (int16_t *)this->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = quantize<int16_t>(src_data[i], inv_scale);
            }
        } else {
            return false;
        }
    } else if (tensor->dtype == DATA_TYPE_DOUBLE) {
        double *src_data = (double *)tensor->data;
        float inv_scale = DL_RESCALE(this->exponent);

        if (this->dtype == DATA_TYPE_INT8) {
            int8_t *data = (int8_t *)this->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = quantize<int8_t, double>(src_data[i], inv_scale);
            }
        } else if (this->dtype == DATA_TYPE_INT16) {
            int16_t *data = (int16_t *)this->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = quantize<int16_t, double>(src_data[i], inv_scale);
            }
        } else {
            return false;
        }
    } else if (this->dtype == DATA_TYPE_FLOAT) {
        float *data = (float *)this->data;
        float scale = DL_SCALE(tensor->exponent);

        if (tensor->dtype == DATA_TYPE_INT8 || tensor->dtype == DATA_TYPE_UINT8) {
            int8_t *src_data = (int8_t *)tensor->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = dequantize(src_data[i], scale);
            }
        } else if (tensor->dtype == DATA_TYPE_INT16 || tensor->dtype == DATA_TYPE_UINT16) {
            int16_t *src_data = (int16_t *)tensor->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = dequantize(src_data[i], scale);
            }
        } else {
            return false;
        }
    } else if (this->dtype == DATA_TYPE_DOUBLE) {
        double *data = (double *)this->data;
        float scale = DL_SCALE(tensor->exponent);

        if (tensor->dtype == DATA_TYPE_INT8 || tensor->dtype == DATA_TYPE_UINT8) {
            int8_t *src_data = (int8_t *)tensor->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = dequantize<int8_t, double>(src_data[i], scale);
            }
        } else if (tensor->dtype == DATA_TYPE_INT16 || tensor->dtype == DATA_TYPE_UINT16) {
            int16_t *src_data = (int16_t *)tensor->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = dequantize<int16_t, double>(src_data[i], scale);
            }
        } else {
            return false;
        }
    } else if (this->exponent != tensor->exponent || this->dtype != tensor->dtype) {
        // quantize(dequtize())
        if (this->exponent == tensor->exponent) {
            if (this->dtype == DATA_TYPE_INT8 && tensor->dtype == DATA_TYPE_INT16) {
                int16_t *src_data = static_cast<int16_t *>(tensor->data);
                int8_t *data = static_cast<int8_t *>(this->data);
                for (int i = 0; i < this->get_size(); i++) {
                    data[i] = static_cast<int8_t>(DL_CLIP(src_data[i], DL_QUANT8_MIN, DL_QUANT8_MAX));
                }
            } else if (this->dtype == DATA_TYPE_INT16 && tensor->dtype == DATA_TYPE_INT8) {
                int8_t *src_data = static_cast<int8_t *>(tensor->data);
                int16_t *data = static_cast<int16_t *>(this->data);
                for (int i = 0; i < this->get_size(); i++) {
                    data[i] = static_cast<int16_t>(src_data[i]);
                }
            } else {
                return false;
            }
        } else {
#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32P4
            if (this->get_size() > 50) {
                std::vector<base::requantizeArgsType> args =
                    base::get_requantize_operation_args(this, tensor, RUNTIME_MODE_SINGLE_CORE);
                if (!(reinterpret_cast<uintptr_t>(args[0].input_element) & 0xf) &&
                    !(reinterpret_cast<uintptr_t>(args[0].output_element) & 0xf)) {
                    if (this->dtype == DATA_TYPE_INT8 && tensor->dtype == DATA_TYPE_INT8) {
                        base::requantize_linear<int8_t, int8_t>(&args[0]);
                        return true;
                    } else if (this->dtype == DATA_TYPE_INT8 && tensor->dtype == DATA_TYPE_INT16) {
                        base::requantize_linear<int8_t, int16_t>(&args[0]);
                        return true;
                    } else if (this->dtype == DATA_TYPE_INT16 && tensor->dtype == DATA_TYPE_INT16 &&
                               !(args[0].output_shift == 0 && args[0].output_scale > 0x4000)) {
                        base::requantize_linear<int16_t, int16_t>(&args[0]);
                        return true;
                    } else if (this->dtype == DATA_TYPE_INT16 && tensor->dtype == DATA_TYPE_INT8) {
                        base::requantize_linear<int16_t, int8_t>(&args[0]);
                        return true;
                    }
                }
            }
#endif

            float src_scale = DL_SCALE(tensor->exponent);
            float inv_scale = 1.0 / (DL_SCALE(this->exponent));

            if (this->dtype == DATA_TYPE_INT8 && tensor->dtype == DATA_TYPE_INT8) {
                int8_t *src_data = static_cast<int8_t *>(tensor->data);
                int8_t *data = static_cast<int8_t *>(this->data);
                for (int i = 0; i < this->get_size(); i++) {
                    float tmp = dequantize(src_data[i], src_scale);
                    data[i] = quantize<int8_t>(tmp, inv_scale);
                }
            } else if (this->dtype == DATA_TYPE_INT16 && tensor->dtype == DATA_TYPE_INT16) {
                int16_t *src_data = static_cast<int16_t *>(tensor->data);
                int16_t *data = static_cast<int16_t *>(this->data);
                for (int i = 0; i < this->get_size(); i++) {
                    float tmp = dequantize(src_data[i], src_scale);
                    data[i] = quantize<int16_t>(tmp, inv_scale);
                }
            } else if (this->dtype == DATA_TYPE_INT8 && tensor->dtype == DATA_TYPE_INT16) {
                int16_t *src_data = static_cast<int16_t *>(tensor->data);
                int8_t *data = static_cast<int8_t *>(this->data);
                for (int i = 0; i < this->get_size(); i++) {
                    float tmp = dequantize(src_data[i], src_scale);
                    data[i] = quantize<int8_t>(tmp, inv_scale);
                }
            } else if (this->dtype == DATA_TYPE_INT16 && tensor->dtype == DATA_TYPE_INT8) {
                int8_t *src_data = static_cast<int8_t *>(tensor->data);
                int16_t *data = static_cast<int16_t *>(this->data);
                for (int i = 0; i < this->get_size(); i++) {
                    float tmp = dequantize(src_data[i], src_scale);
                    data[i] = quantize<int16_t>(tmp, inv_scale);
                }
            } else {
                return false;
            }
        }
    } else {
        return false;
    }
    return true;
}

bool TensorBase::assign(std::vector<int> shape, const void *element, int exponent, dtype_t dtype)
{
    TensorBase tensor(shape, element, exponent, dtype, false);
    return this->assign(&tensor);
}

TensorBase &TensorBase::set_element_ptr(void *data)
{
    this->data = data;

    return *this;
}

TensorBase &TensorBase::set_shape(const std::vector<int> shape)
{
    assert(shape.size() > 0);
    this->size = 1;
    for (int i = 0; i < shape.size(); ++i) {
        assert(shape[i] >= 1);
        this->size *= shape[i];
    }
    this->shape = shape;

    std::vector<int> axis_offset(this->shape.size(), 1);
    for (int i = shape.size() - 2; i > -1; --i) {
        axis_offset[i] = axis_offset[i + 1] * this->shape[i + 1];
    }
    this->axis_offset = axis_offset;
    return *this;
}

size_t TensorBase::set_preload_addr(void *addr, size_t size)
{
    size_t aligned_size = this->get_aligned_size();
    if (addr && size >= aligned_size) {
        this->cache = addr;
        return aligned_size;
    }
    this->cache = nullptr;
    return 0;
}

void TensorBase::reset_bias_layout(quant_type_t op_quant_type, bool is_depthwise)
{
    // The bias needs to be quantized to 32 bits.
    if (op_quant_type == QUANT_TYPE_SYMM_8BIT) {
        assert(this->dtype == DATA_TYPE_INT32);
    } else {
        if (this->dtype != DATA_TYPE_INT64) {
            ESP_LOGE(__FUNCTION__,
                     "The quantization bit width of the bias has been modified. Please update esp-ppq and then "
                     "re-quantize the model.");
        }
        assert(this->dtype == DATA_TYPE_INT64);
    }

#if CONFIG_IDF_TARGET_ESP32S3
    // Reset bias layout for esp32s3
    if (op_quant_type == QUANT_TYPE_SYMM_8BIT) {
        // 0x000AAAAA000BBBBB ==> 0xAAAAABBBBB
        size_t dtype_bytes = 1;
        size_t align = 16 / dtype_bytes;
        size_t data_num = this->get_size();
        size_t align_num = ((size_t)(data_num / align)) * align;
        size_t remain_num = data_num - align_num;
        if (is_depthwise) {
            align_num = data_num;
            remain_num = 0;
        }
        // QACC, EE.LD.QACC_L.L.128.IP / EE.LD.QACC_H.L.128.IP requires 16-byte address alignment.
        //      When the bias is stored with a size of 4 bytes, the address is exactly 16-byte aligned
        //      when used in EE.LD.QACC_H.L.128.IP, so the size of the aligned portion of memory here
        //      is calculated based on 4 bytes.
        // ACCX, EE.LD.ACCX.IP requires 8-byte address alignment.
        size_t memory_size_needed = align_num * 4 + remain_num * 8;
        // get the aligned size
        memory_size_needed = memory_size_needed % align == 0 ? memory_size_needed
                                                             : memory_size_needed + align - memory_size_needed % align;
        int32_t *src_ptr = static_cast<int32_t *>(this->data);
        int8_t *dst_ptr = static_cast<int8_t *>(tool::calloc_aligned(16, memory_size_needed, dtype_bytes, this->caps));
        assert(dst_ptr);
        int8_t *dst_ptr_head = dst_ptr;

        // 0x000AAAAA000BBBBB ==> 0xAAAAABBBBB
        int i = 0;
        for (; i < align_num; i++) {
            int32_t src_data = src_ptr[i] & 0xfffff;
            if (i & 1) {
                int8_t src_least_4bit = src_data & 0xf;
                (*(--dst_ptr_head)) |= (src_least_4bit << 4);
                src_data >>= 4;
            } else {
                *dst_ptr_head = src_data & 0xff;
                src_data >>= 8;
            }
            dst_ptr_head++;
            *(reinterpret_cast<int16_t *>(dst_ptr_head)) = static_cast<int16_t>(src_data);
            dst_ptr_head += 2;

            // Move to the 16-byte memory address alignment.
            if (((i + 1) % (align >> 1) == 0) && (reinterpret_cast<uintptr_t>(dst_ptr_head) & 0xf)) {
                dst_ptr_head = dst_ptr_head + 16 - (reinterpret_cast<uintptr_t>(dst_ptr_head) & 0xf);
            }
        }

        for (int j = 0; j < remain_num; j++, i++) {
            (reinterpret_cast<int64_t *>(dst_ptr_head))[j] = src_ptr[i];
        }

        if (this->auto_free) {
            heap_caps_free(this->data);
        }
        this->data = dst_ptr;
        this->auto_free = true;

    } else if (op_quant_type == QUANT_TYPE_SYMM_16BIT) {
        // 0x000000AAAAAAAAAA000000BBBBBBBBBB ==> 0xAAAAAAAAAABBBBBBBBBB
        size_t dtype_bytes = 2;
        size_t align = 16 / dtype_bytes;
        size_t data_num = this->get_size();
        size_t align_num = ((size_t)(data_num / align)) * align;
        size_t remain_num = data_num - align_num;
        if (is_depthwise) {
            align_num = data_num;
            remain_num = 0;
        }

        int64_t *src_ptr = static_cast<int64_t *>(this->data);
        int8_t *dst_ptr =
            static_cast<int8_t *>(tool::calloc_aligned(16, data_num, this->get_dtype_bytes(), this->caps));
        int8_t *dst_ptr_head = dst_ptr;
        // 0x000000AAAAAAAAAA000000BBBBBBBBBB ==> 0xAAAAAAAAAABBBBBBBBBB
        for (int i = 0; i < align_num; i++) {
            dl::tool::copy_memory(dst_ptr, src_ptr, 5);
            dst_ptr += 5;
            src_ptr++;

            // Move to the 16-byte memory address alignment.
            if (((i + 1) % (align >> 1) == 0) && (reinterpret_cast<uintptr_t>(dst_ptr) & 0xf)) {
                dst_ptr = dst_ptr + 16 - (reinterpret_cast<uintptr_t>(dst_ptr) & 0xf);
            }
        }

        for (int j = 0; j < remain_num; j++) {
            (reinterpret_cast<int64_t *>(dst_ptr))[j] = src_ptr[j];
        }

        if (this->auto_free) {
            heap_caps_free(this->data);
        }

        this->data = dst_ptr_head;
        this->auto_free = true;
    }
#endif
}

void TensorBase::push(TensorBase *new_tensor, int dim)
{
    if (dim < 0 || dim >= this->shape.size()) {
        ESP_LOGE(__FUNCTION__, "The dim is out of range.");
        return;
    }

    if (this->dtype != new_tensor->dtype) {
        ESP_LOGE(__FUNCTION__, "this->dtype != new_tensor->dtype");
        return;
    }

    if (new_tensor->shape[dim] > this->shape[dim]) {
        ESP_LOGE(__FUNCTION__,
                 "The time series dimension size of new tensor "
                 "is greater than that of the current tensor.");
        return;
    }

    // NCW or NWC
    int loop_num = 1;
    int move_chunk_size = 1;
    int copy_chunk_size = 1;
    for (int i = this->shape.size() - 1; i >= 0; i--) {
        if (i != dim) {
            assert(this->shape[i] == new_tensor->shape[i]);
        }

        if (i < dim) {
            loop_num *= this->shape[i];
        } else if (i == dim) {
            move_chunk_size *= (this->shape[i] - new_tensor->shape[i]);
            copy_chunk_size *= new_tensor->shape[i];
        } else if (i > dim) {
            move_chunk_size *= this->shape[i];       // The move size of the current tensor
            copy_chunk_size *= new_tensor->shape[i]; // The copy size of the new tensor
        }
    }

    // Since the old one is removed to free up space for the new one, their sizes are equal.
    int move_offset = copy_chunk_size;
    int current_loop_stride = move_chunk_size + copy_chunk_size;

    for (int i = 0; i < loop_num; i++) {
        if (this->dtype == DATA_TYPE_INT8) {
            // move chunk
            tool::copy_memory(get_element_ptr<int8_t>() + i * current_loop_stride,
                              get_element_ptr<int8_t>() + i * current_loop_stride + move_offset,
                              move_chunk_size);
            // copy new tensor
            tool::copy_memory(get_element_ptr<int8_t>() + i * current_loop_stride + move_chunk_size,
                              new_tensor->get_element_ptr<int8_t>() + i * copy_chunk_size,
                              copy_chunk_size);
        } else if (this->dtype == DATA_TYPE_INT16) {
            // move chunk
            tool::copy_memory(get_element_ptr<int16_t>() + i * current_loop_stride,
                              get_element_ptr<int16_t>() + i * current_loop_stride + move_offset,
                              move_chunk_size);
            // copy new tensor
            tool::copy_memory(get_element_ptr<int16_t>() + i * current_loop_stride + move_chunk_size,
                              new_tensor->get_element_ptr<int16_t>() + i * copy_chunk_size,
                              copy_chunk_size);
        } else {
            ESP_LOGE(__FUNCTION__, "Don't support dtype: %d", this->dtype);
        }
    }
    return;
}

void TensorBase::print(bool print_data)
{
    ESP_LOGI(__FUNCTION__,
             "shape: %s, dtype: %s, exponent: %d, auto_free: %d, data: %p",
             vector_to_string(get_shape()).c_str(),
             dtype_to_string(get_dtype()),
             this->exponent,
             this->auto_free,
             this->data);

    if (this->data && print_data) {
        ESP_LOGI(__FUNCTION__, "The data of TensorBase is:");
        if (get_dtype() == DATA_TYPE_FLOAT) {
            for (int i = 0; i < get_size(); i++) {
                printf("%f, ", get_element<float>(i));
                if ((i + 1) % 16 == 0) {
                    printf("\n");
                }
            }
        } else if (get_dtype() == DATA_TYPE_DOUBLE) {
            for (int i = 0; i < get_size(); i++) {
                printf("%f, ", get_element<double>(i));
                if ((i + 1) % 16 == 0) {
                    printf("\n");
                }
            }
        } else if (get_dtype() == DATA_TYPE_INT8) {
            for (int i = 0; i < get_size(); i++) {
                printf("%d, ", get_element<int8_t>(i));
                if ((i + 1) % 16 == 0) {
                    printf("\n");
                }
            }
        } else if (get_dtype() == DATA_TYPE_INT16) {
            for (int i = 0; i < get_size(); i++) {
                printf("%d, ", get_element<int16_t>(i));
                if ((i + 1) % 16 == 0) {
                    printf("\n");
                }
            }
        } else if (get_dtype() == DATA_TYPE_BOOL) {
            for (int i = 0; i < get_size(); i++) {
                printf("%d, ", get_element<bool>(i));
                if ((i + 1) % 16 == 0) {
                    printf("\n");
                }
            }
        } else {
            ESP_LOGE(__FUNCTION__, "The dtype don't support now!");
        }
        printf("\n");
    }
}

TensorBase *TensorBase::reshape(std::vector<int> shape)
{
    int size_gt = this->get_size();
    int index = -1;
    for (int i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            assert(index == -1);
            index = i;
        } else {
            assert(shape[i] > 0);
        }
    }
    int size = 1;
    if (index == -1) {
        for (int i = 0; i < shape.size(); ++i) {
            size *= shape[i];
        }
        assert(size == size_gt);
        this->set_shape(shape);
    } else {
        for (int i = 0; i < shape.size(); ++i) {
            if (shape[i] > 0) {
                size *= shape[i];
            }
        }
        assert((size_gt % size) == 0);
        shape[index] = size_gt / size;
        this->set_shape(shape);
    }
    return this;
}

template <typename T>
TensorBase *TensorBase::transpose(T *input_element,
                                  std::vector<int> &input_shape,
                                  std::vector<int> &input_axis_offset,
                                  std::vector<int> &perm)
{
    if (perm.size() == 0) {
        for (int i = shape.size() - 1; i >= 0; i--) {
            perm.push_back(i);
        }
    }
    int dims = perm.size();

    for (int i = 0; i < dims; ++i) {
        if (perm[i] < 0)
            perm[i] = dims + perm[i];
        this->shape[i] = input_shape[perm[i]];
    }

    this->axis_offset[dims - 1] = 1;
    for (int i = dims - 2; i > -1; --i) {
        this->axis_offset[i] = this->axis_offset[i + 1] * this->shape[i + 1];
    }
    T *output_element = (T *)this->get_element_ptr();

    std::vector<int> input_axis_index(dims);
    if (dims == 4) {
        uint32_t input_idx = 0, output_idx = 0;
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                for (int k = 0; k < input_shape[2]; k++) {
                    for (int l = 0; l < input_shape[3]; l++) {
                        input_axis_index = {i, j, k, l};
                        input_idx = l + k * input_axis_offset[2] + j * input_axis_offset[1] + i * input_axis_offset[0];
                        output_idx = input_axis_index[perm[3]] * this->axis_offset[3] +
                            input_axis_index[perm[2]] * this->axis_offset[2] +
                            input_axis_index[perm[1]] * this->axis_offset[1] +
                            input_axis_index[perm[0]] * this->axis_offset[0];
                        output_element[output_idx] = input_element[input_idx];
                    }
                }
            }
        }
    } else if (dims == 3) {
        uint32_t input_idx = 0, output_idx = 0;
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                for (int k = 0; k < input_shape[2]; k++) {
                    input_axis_index = {i, j, k};
                    input_idx = k + j * input_axis_offset[1] + i * input_axis_offset[0];
                    output_idx = input_axis_index[perm[2]] * this->axis_offset[2] +
                        input_axis_index[perm[1]] * this->axis_offset[1] +
                        input_axis_index[perm[0]] * this->axis_offset[0];
                    output_element[output_idx] = input_element[input_idx];
                }
            }
        }
    } else if (dims == 2) {
        uint32_t input_idx = 0, output_idx = 0;
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                input_axis_index = {i, j};
                input_idx = j + i * input_axis_offset[0];
                output_idx =
                    input_axis_index[perm[1]] * this->axis_offset[1] + input_axis_index[perm[0]] * this->axis_offset[0];
                output_element[output_idx] = input_element[input_idx];
            }
        }
    } else {
        // for any dims
        std::vector<int> index_old(dims, 0);
        for (int i = 0; i < size; ++i) {
            int dim_div_value = i;
            int index_new = 0;
            for (int j = dims - 1; j > -1; --j) {
                index_old[j] = dim_div_value % input_shape[j];
                dim_div_value /= input_shape[j];
            }
            for (int j = dims - 1; j > -1; --j) {
                index_new += index_old[perm[j]] * this->axis_offset[j];
            }
            output_element[index_new] = input_element[i];
        }
    }

    return this;
}

TensorBase *TensorBase::transpose(TensorBase *input, std::vector<int> perm)
{
    assert(this->get_size() == input->get_size());
    assert(this->dtype == input->dtype);

    if (this->dtype == DATA_TYPE_INT8) {
        transpose<int8_t>((int8_t *)input->get_element_ptr(), input->shape, input->axis_offset, perm);
    } else if (this->dtype == DATA_TYPE_UINT8) {
        transpose<uint8_t>((uint8_t *)input->get_element_ptr(), input->shape, input->axis_offset, perm);
    } else if (this->dtype == DATA_TYPE_INT16) {
        transpose<int16_t>((int16_t *)input->get_element_ptr(), input->shape, input->axis_offset, perm);
    } else if (this->dtype == DATA_TYPE_INT32) {
        transpose<int32_t>((int32_t *)input->get_element_ptr(), input->shape, input->axis_offset, perm);
    } else if (this->dtype == DATA_TYPE_UINT16) {
        transpose<uint16_t>((uint16_t *)input->get_element_ptr(), input->shape, input->axis_offset, perm);
    } else if (this->dtype == DATA_TYPE_INT32) {
        transpose<uint32_t>((uint32_t *)input->get_element_ptr(), input->shape, input->axis_offset, perm);
    } else if (this->dtype == DATA_TYPE_FLOAT) {
        transpose<float>((float *)input->get_element_ptr(), input->shape, input->axis_offset, perm);
    }

    return this;
}

int TensorBase::get_element_index(const std::vector<int> &axis_index)
{
    int element_index = 0;
    for (int i = 0; i < axis_index.size(); i++) {
        element_index += axis_index[i] * this->axis_offset[i];
    }
    return element_index;
}

std::vector<int> TensorBase::get_element_coordinates(int index)
{
    int ndim = this->shape.size();
    std::vector<int> coords(ndim);
    for (int k = 0; k < ndim; ++k) {
        coords[k] = index / this->axis_offset[k];
        index %= this->axis_offset[k];
    }
    return coords;
}

template <typename T>
T TensorBase::get_element(int index)
{
    if (index < 0) {
        index += this->size;
    }

    return ((T *)this->data)[index];
}
template int8_t TensorBase::get_element<int8_t>(int index);
template uint8_t TensorBase::get_element<uint8_t>(int index);
template int16_t TensorBase::get_element<int16_t>(int index);
template uint16_t TensorBase::get_element<uint16_t>(int index);
template int32_t TensorBase::get_element<int32_t>(int index);
template uint32_t TensorBase::get_element<uint32_t>(int index);
template int64_t TensorBase::get_element<int64_t>(int index);
template float TensorBase::get_element<float>(int index);

template <typename T>
T TensorBase::get_element(const std::vector<int> &axis_index)
{
    int index = this->get_element_index(axis_index);
    return ((T *)this->data)[index];
}
template int8_t TensorBase::get_element<int8_t>(const std::vector<int> &axis_index);
template uint8_t TensorBase::get_element<uint8_t>(const std::vector<int> &axis_index);
template int16_t TensorBase::get_element<int16_t>(const std::vector<int> &axis_index);
template uint16_t TensorBase::get_element<uint16_t>(const std::vector<int> &axis_index);
template int32_t TensorBase::get_element<int32_t>(const std::vector<int> &axis_index);
template uint32_t TensorBase::get_element<uint32_t>(const std::vector<int> &axis_index);
template float TensorBase::get_element<float>(const std::vector<int> &axis_index);

template <typename T>
bool TensorBase::compare_elements(const T *gt_elements, float epsilon, bool verbose)
{
    T *elements = (T *)this->get_element_ptr();
    if (elements == gt_elements) {
        return true;
    }

    for (int i = 0; i < this->get_size(); i++) {
        if (elements[i] - gt_elements[i] > epsilon || elements[i] - gt_elements[i] < -epsilon) {
            if (verbose) {
                ESP_LOGE(__FUNCTION__,
                         "Inconsistent values, ground truth: %.10f, infer: %.10f, epsilon:%.10f",
                         gt_elements[i] * 1.0,
                         elements[i] * 1.0,
                         epsilon);
                std::vector<int> position = this->get_element_coordinates(i);
                ESP_LOGE(__FUNCTION__, "The position is: %s", vector_to_string(position).c_str());
            }
            return false;
        }
    }

    return true;
}
template bool TensorBase::compare_elements<int8_t>(const int8_t *gt_elements, float epsilon, bool verbose);
template bool TensorBase::compare_elements<uint8_t>(const uint8_t *gt_elements, float epsilon, bool verbose);
template bool TensorBase::compare_elements<int16_t>(const int16_t *gt_elements, float epsilon, bool verbose);
template bool TensorBase::compare_elements<uint16_t>(const uint16_t *gt_elements, float epsilon, bool verbose);
template bool TensorBase::compare_elements<int32_t>(const int32_t *gt_elements, float epsilon, bool verbose);
template bool TensorBase::compare_elements<uint32_t>(const uint32_t *gt_elements, float epsilon, bool verbose);
template bool TensorBase::compare_elements<float>(const float *gt_elements, float epsilon, bool verbose);
template bool TensorBase::compare_elements<double>(const double *gt_elements, float epsilon, bool verbose);
template bool TensorBase::compare_elements<bool>(const bool *gt_elements, float epsilon, bool verbose);

bool TensorBase::is_same_shape(TensorBase *tensor)
{
    if (this->shape.size() != tensor->shape.size()) {
        return false;
    }
    for (int i = 0; i < this->shape.size(); i++) {
        if (this->shape[i] != tensor->shape[i]) {
            return false;
        }
    }
    return true;
}

bool TensorBase::equal(TensorBase *tensor, float epsilon, bool verbose)
{
    if (tensor == nullptr) {
        return false;
    }

    // compare data type
    dtype_t type1 = this->get_dtype();
    dtype_t type2 = tensor->get_dtype();
    if (type1 != type2) {
        if (verbose) {
            ESP_LOGE(__FUNCTION__, "data type not equal: %s != %s", dtype_to_string(type1), dtype_to_string(type2));
        }
        return false;
    }

    // compare shape
    if (!this->is_same_shape(tensor)) {
        if (verbose) {
            ESP_LOGE(__FUNCTION__,
                     "shape not equal: %s != %s",
                     vector_to_string(this->shape).c_str(),
                     vector_to_string(tensor->shape).c_str());
        }
        return false;
    }

    // compare tensor element
    if (this->exponent != tensor->exponent) {
        if (verbose) {
            ESP_LOGE(__FUNCTION__, "exponent not equal: %d != %d", this->exponent, tensor->exponent);
        }
        return false;
    }

    if (verbose || epsilon > 0) {
        if (type1 == DATA_TYPE_INT8) {
            return this->compare_elements<int8_t>((int8_t *)tensor->get_element_ptr(), epsilon, verbose);
        } else if (type1 == DATA_TYPE_INT16) {
            return this->compare_elements<int16_t>((int16_t *)tensor->get_element_ptr(), epsilon, verbose);
        } else if (type1 == DATA_TYPE_FLOAT) {
            return this->compare_elements<float>((float *)tensor->get_element_ptr(), epsilon, verbose);
        } else if (type1 == DATA_TYPE_UINT8) {
            return this->compare_elements<uint8_t>((uint8_t *)tensor->get_element_ptr(), epsilon, verbose);
        } else if (type1 == DATA_TYPE_UINT16) {
            return this->compare_elements<uint16_t>((uint16_t *)tensor->get_element_ptr(), epsilon, verbose);
        } else if (type1 == DATA_TYPE_INT32) {
            return this->compare_elements<int32_t>((int32_t *)tensor->get_element_ptr(), epsilon, verbose);
        } else if (type1 == DATA_TYPE_UINT32) {
            return this->compare_elements<uint32_t>((uint32_t *)tensor->get_element_ptr(), epsilon, verbose);
        } else if (type1 == DATA_TYPE_DOUBLE) {
            return this->compare_elements<double>((double *)tensor->get_element_ptr(), epsilon, verbose);
        } else if (type1 == DATA_TYPE_BOOL) {
            return this->compare_elements<bool>((bool *)tensor->get_element_ptr(), epsilon, verbose);
        }
    } else {
        return (memcmp(this->get_element_ptr(), tensor->get_element_ptr(), this->get_bytes()) == 0);
    }

    return false;
}

template <typename T>
void _flip(TensorBase *input, const std::vector<int> &axes)
{
    if (axes.empty()) {
        return;
    }
    T *input_element = (T *)input->get_element_ptr();

    for (int i = 0; i < input->get_size(); ++i) {
        // Compute original coordinates
        std::vector<int> coords = input->get_element_coordinates(i);

        // Flip coordinates
        for (int axis : axes) {
            coords[axis] = input->shape[axis] - 1 - coords[axis];
        }

        // Compute flipped index
        int j = input->get_element_index(coords);

        // Swap elements if needed
        if (i < j) {
            T temp = input_element[i];
            input_element[i] = input_element[j];
            input_element[j] = temp;
        }
    }
}

template <typename T>
void _slice(TensorBase *input,
            TensorBase *output,
            const std::vector<int> &start,
            const std::vector<int> &end,
            const std::vector<int> &axes,
            const std::vector<int> &step)
{
    T *input_element = input->get_element_ptr<T>();
    T *output_element = output->get_element_ptr<T>();
    std::vector<int> input_shape = input->get_shape();
    int dims = input_shape.size();
    std::vector<int> loop_start(dims, 0);
    std::vector<int> loop_end = input_shape;
    std::vector<int> loop_step(dims, 1);
    std::vector<int> fliped;

    int last_axis = start.size() - 1;
    for (int i = 0; i < start.size(); i++) {
        int axis = i;
        int step_i = 1;
        int start_i = start[i];
        int end_i = end[i];
        if (!axes.empty()) {
            axis = (axes[i] < 0) ? (axes[i] + dims) : axes[i];
            if (axis > last_axis) {
                last_axis = axis;
            }
        }
        if (!step.empty()) {
            step_i = step[i];
        }
        if (step_i < 0) {
            fliped.push_back(axis);
            start_i = -start_i - 1;
            end_i = -end_i - 1;
            step_i = -step_i;
        }

        if (end_i > input_shape[axis]) {
            end_i = input_shape[axis];
        }

        loop_start[axis] = start_i < 0 ? (start_i + input_shape[axis]) : (start_i % (input_shape[axis] + 1));
        loop_end[axis] = end_i < 0 ? (end_i + input_shape[axis]) : (end_i % (input_shape[axis] + 1));
        loop_step[axis] = step_i;
        assert(loop_start[axis] < loop_end[axis]);
    }
    int min_offset = loop_end[last_axis] - loop_start[last_axis];
    for (int i = last_axis + 1; i < dims; i++) {
        min_offset *= input_shape[i];
    }
    T *slice_ptr = nullptr;
    int min_offset_bytes = min_offset * sizeof(T);

    if (step.empty()) {
        if (dims == 1 || last_axis == 0) {
            slice_ptr = input_element + input->get_element_index(loop_start);
            tool::copy_memory(output_element, slice_ptr, min_offset_bytes);
        } else {
            std::vector<int> loop_index = loop_start;
            while (loop_index[0] < loop_end[0]) {
                slice_ptr = input_element + input->get_element_index(loop_index);
                tool::copy_memory(output_element, slice_ptr, min_offset_bytes);
                output_element += min_offset;

                loop_index[last_axis - 1] += 1;
                for (int i = last_axis - 1; i > 0; --i) {
                    if (loop_index[i] == loop_end[i]) {
                        loop_index[i] = loop_start[i];
                        loop_index[i - 1] += 1;
                    } else
                        break;
                }
            }
        }
    } else {
        if (dims == 1 || last_axis == 0) {
            slice_ptr = input_element + input->get_element_index(loop_start);
            if (loop_step[0] == 1) {
                tool::copy_memory(output_element, slice_ptr, min_offset_bytes);
            } else {
                for (int i = 0; i < min_offset; i += loop_step[0]) {
                    *output_element = *slice_ptr;
                    slice_ptr += loop_step[0];
                    output_element += 1;
                }
            }
        } else {
            std::vector<int> loop_index = loop_start;
            while (loop_index[0] < loop_end[0]) {
                slice_ptr = input_element + input->get_element_index(loop_index);
                if (loop_step[last_axis] == 1) {
                    tool::copy_memory(output_element, slice_ptr, min_offset_bytes);
                    output_element += min_offset;
                } else {
                    for (int i = 0; i < min_offset; i += loop_step[last_axis]) {
                        *output_element = *slice_ptr;
                        slice_ptr += loop_step[last_axis];
                        output_element += 1;
                    }
                }

                loop_index[last_axis - 1] += loop_step[last_axis - 1];
                for (int i = last_axis - 1; i > 0; --i) {
                    if (loop_index[i] >= loop_end[i]) {
                        loop_index[i] = loop_start[i];
                        loop_index[i - 1] += loop_step[i - 1];
                    } else
                        break;
                }
            }
        }
    }

    if (!fliped.empty()) {
        _flip<T>(output, fliped);
    }
}

void TensorBase::slice(TensorBase *input,
                       TensorBase *output,
                       const std::vector<int> &start,
                       const std::vector<int> &end,
                       const std::vector<int> &axes,
                       const std::vector<int> &step)
{
    dtype_t dtype = input->get_dtype();
    assert(dtype == output->get_dtype());

    if (dtype == DATA_TYPE_INT8) {
        _slice<int8_t>(input, output, start, end, axes, step);
    } else if (dtype == DATA_TYPE_INT16) {
        _slice<int16_t>(input, output, start, end, axes, step);
    } else if (dtype == DATA_TYPE_FLOAT) {
        _slice<float>(input, output, start, end, axes, step);
    } else if (dtype == DATA_TYPE_UINT8) {
        _slice<uint8_t>(input, output, start, end, axes, step);
    } else if (dtype == DATA_TYPE_UINT16) {
        _slice<uint16_t>(input, output, start, end, axes, step);
    } else if (dtype == DATA_TYPE_INT32) {
        _slice<int32_t>(input, output, start, end, axes, step);
    } else if (dtype == DATA_TYPE_UINT32) {
        _slice<uint32_t>(input, output, start, end, axes, step);
    } else {
        ESP_LOGE(__FUNCTION__, "Unsupported data type");
    }
}

template <typename T>
TensorBase *TensorBase::pad(T *input_element,
                            const std::vector<int> &input_shape,
                            const std::vector<int> &pads,
                            const padding_mode_t mode,
                            TensorBase *const_value)
{
    T const_value_element = 0;
    if (const_value) {
        const_value_element = const_value->get_element<T>(0);
    }

    T *output_element = this->get_element_ptr<T>();
    int dims = input_shape.size();

    if (dims == 4) {
        base::pad4D<T>(input_element, output_element, input_shape, pads, mode, const_value_element);
    } else if (dims == 3) {
        base::pad3D<T>(input_element, output_element, input_shape, pads, mode, const_value_element);
    } else if (dims == 2) {
        base::pad2D<T>(input_element, output_element, input_shape, pads, mode, const_value_element);
    } else if (dims == 1) {
        base::pad1D<T>(input_element, output_element, input_shape, pads, mode, const_value_element);
    } else if (dims == 5 && mode != PADDING_CONSTANT) {
        if (pads[0] == 0 && pads[5] == 0) {
            std::vector<int> new_pads = {pads[1], pads[2], pads[3], pads[4], pads[6], pads[7], pads[8], pads[9]};
            std::vector<int> new_shape = {
                input_shape[0] * input_shape[1], input_shape[2], input_shape[3], input_shape[4]};
            base::pad4D<T>(input_element, output_element, new_shape, new_pads, mode, const_value_element);
        }
    } else if (dims > 4 && mode == PADDING_CONSTANT) {
        std::vector<int> loop_start(dims, 0);
        std::vector<int> loop_end(dims, 0);
        T *slice_ptr = nullptr;
        int last_axis = dims - 1;
        int min_offset = input_shape[last_axis];
        int min_offset_bytes = min_offset * sizeof(T);

        for (int i = 0; i < dims; i++) {
            loop_start[i] = pads[i];
            loop_end[i] = loop_start[i] + input_shape[i];
        }

        // step1: set value
        if (const_value_element == 0) {
            tool::set_zero(output_element, this->get_bytes());
        } else {
            tool::set_value<T>(output_element, const_value_element, this->get_size());
        }

        // step2: copy input into output, like slice op
        if (dims == 1) {
            slice_ptr = output_element + this->get_element_index(loop_start);
            tool::copy_memory(slice_ptr, input_element, min_offset_bytes);
        } else {
            std::vector<int> loop_index = loop_start;
            while (loop_index[0] < loop_end[0]) {
                slice_ptr = output_element + this->get_element_index(loop_index);
                tool::copy_memory(slice_ptr, input_element, min_offset_bytes);
                input_element += min_offset;

                loop_index[last_axis - 1] += 1;
                for (int i = last_axis - 1; i > 0; --i) {
                    if (loop_index[i] == loop_end[i]) {
                        loop_index[i] = loop_start[i];
                        loop_index[i - 1] += 1;
                    } else
                        break;
                }
            }
        }
    } else {
        ESP_LOGE("Pad", "Not implemented dim: %d", dims);
    }

    return this;
}
template TensorBase *TensorBase::pad(int8_t *input_element,
                                     const std::vector<int> &input_shape,
                                     const std::vector<int> &pads,
                                     const padding_mode_t mode,
                                     TensorBase *const_value);
template TensorBase *TensorBase::pad(uint8_t *input_element,
                                     const std::vector<int> &input_shape,
                                     const std::vector<int> &pads,
                                     const padding_mode_t mode,
                                     TensorBase *const_value);
template TensorBase *TensorBase::pad(int16_t *input_element,
                                     const std::vector<int> &input_shape,
                                     const std::vector<int> &pads,
                                     const padding_mode_t mode,
                                     TensorBase *const_value);
template TensorBase *TensorBase::pad(uint16_t *input_element,
                                     const std::vector<int> &input_shape,
                                     const std::vector<int> &pads,
                                     const padding_mode_t mode,
                                     TensorBase *const_value);
template TensorBase *TensorBase::pad(int32_t *input_element,
                                     const std::vector<int> &input_shape,
                                     const std::vector<int> &pads,
                                     const padding_mode_t mode,
                                     TensorBase *const_value);
template TensorBase *TensorBase::pad(uint32_t *input_element,
                                     const std::vector<int> &input_shape,
                                     const std::vector<int> &pads,
                                     const padding_mode_t mode,
                                     TensorBase *const_value);
template TensorBase *TensorBase::pad(float *input_element,
                                     const std::vector<int> &input_shape,
                                     const std::vector<int> &pads,
                                     const padding_mode_t mode,
                                     TensorBase *const_value);

TensorBase *TensorBase::pad(TensorBase *input,
                            const std::vector<int> &pads,
                            const padding_mode_t mode,
                            TensorBase *const_value)
{
    dtype_t dtype = this->get_dtype();
    assert(dtype == input->get_dtype());

    if (dtype == DATA_TYPE_INT8) {
        return this->pad<int8_t>(input->get_element_ptr<int8_t>(), input->get_shape(), pads, mode, const_value);
    } else if (dtype == DATA_TYPE_INT16) {
        return this->pad<int16_t>(input->get_element_ptr<int16_t>(), input->get_shape(), pads, mode, const_value);
    } else if (dtype == DATA_TYPE_FLOAT) {
        return this->pad<float>(input->get_element_ptr<float>(), input->get_shape(), pads, mode, const_value);
    } else if (dtype == DATA_TYPE_UINT8) {
        return this->pad<uint8_t>(input->get_element_ptr<uint8_t>(), input->get_shape(), pads, mode, const_value);
    } else if (dtype == DATA_TYPE_UINT16) {
        return this->pad<uint16_t>(input->get_element_ptr<uint16_t>(), input->get_shape(), pads, mode, const_value);
    } else if (dtype == DATA_TYPE_INT32) {
        return this->pad<int32_t>(input->get_element_ptr<int32_t>(), input->get_shape(), pads, mode, const_value);
    } else if (dtype == DATA_TYPE_UINT32) {
        return this->pad<uint32_t>(input->get_element_ptr<uint32_t>(), input->get_shape(), pads, mode, const_value);
    } else {
        ESP_LOGE(__FUNCTION__, "Unsupported data type");
    }

    return this;
}

} // namespace dl
