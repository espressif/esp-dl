#include "dl_tensor_base.hpp"

namespace dl {
int quantize(float input, float scale, float quant_min, float quant_max)
{
    int output = tool::round(input * scale);
    output = DL_CLIP(output, quant_min, quant_max);
    return output;
}

float dequantize(int input, float scale)
{
    float output = input * scale;
    return output;
}

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

std::string shape_to_string(std::vector<int> shape)
{
    if (shape.size() == 0) {
        return "[]";
    }

    std::string str = "[";
    for (int i = 0; i < shape.size(); i++) {
        str += std::to_string(shape[i]);
        if (i != shape.size() - 1) {
            str += ", ";
        }
    }
    str += "]";
    return str;
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
            this->data = tool::calloc_aligned(aligned_size, dtype_bytes, 16, caps);
            tool::copy_memory(this->data, const_cast<void *>(element), this->get_size() * dtype_bytes);
        } else {
            this->auto_free = false;
            this->data = const_cast<void *>(element);
        }
    } else {
        this->auto_free = true;
        this->data = tool::calloc_aligned(aligned_size, dtype_bytes, 16, caps);
    }
    this->caps = caps;
}

bool TensorBase::assign(TensorBase *tensor)
{
    if (tensor == nullptr || this->get_size() != tensor->get_size()) {
        return false;
    }

    if (this->exponent == tensor->exponent && this->dtype == tensor->dtype) {
        tool::copy_memory(this->data, tensor->data, this->get_bytes());
    } else if (tensor->dtype == DATA_TYPE_FLOAT) {
        float *src_data = (float *)tensor->data;
        float scale = 1.0 / (DL_SCALE(this->exponent));

        if (this->dtype == DATA_TYPE_INT8) {
            int8_t *data = (int8_t *)this->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = static_cast<int8_t>(quantize(src_data[i], scale, DL_QUANT8_MIN, DL_QUANT8_MAX));
            }
        } else if (this->dtype == DATA_TYPE_INT16) {
            int16_t *data = (int16_t *)this->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = static_cast<int16_t>(quantize(src_data[i], scale, DL_QUANT16_MIN, DL_QUANT16_MAX));
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
            float src_scale = DL_SCALE(tensor->exponent);
            float scale = 1.0 / (DL_SCALE(this->exponent));

            if (this->dtype == DATA_TYPE_INT8 && tensor->dtype == DATA_TYPE_INT8) {
                int8_t *src_data = static_cast<int8_t *>(tensor->data);
                int8_t *data = static_cast<int8_t *>(this->data);
                for (int i = 0; i < this->get_size(); i++) {
                    float tmp = dequantize(src_data[i], src_scale);
                    data[i] = static_cast<int8_t>(quantize(tmp, scale, DL_QUANT8_MIN, DL_QUANT8_MAX));
                }
            } else if (this->dtype == DATA_TYPE_INT16 && tensor->dtype == DATA_TYPE_INT16) {
                int16_t *src_data = static_cast<int16_t *>(tensor->data);
                int16_t *data = static_cast<int16_t *>(this->data);
                for (int i = 0; i < this->get_size(); i++) {
                    float tmp = dequantize(src_data[i], src_scale);
                    data[i] = static_cast<int16_t>(quantize(tmp, scale, DL_QUANT16_MIN, DL_QUANT16_MAX));
                }
            } else if (this->dtype == DATA_TYPE_INT8 && tensor->dtype == DATA_TYPE_INT16) {
                int16_t *src_data = static_cast<int16_t *>(tensor->data);
                int8_t *data = static_cast<int8_t *>(this->data);
                for (int i = 0; i < this->get_size(); i++) {
                    float tmp = dequantize(src_data[i], src_scale);
                    data[i] = static_cast<int8_t>(quantize(tmp, scale, DL_QUANT8_MIN, DL_QUANT8_MAX));
                }
            } else if (this->dtype == DATA_TYPE_INT16 && tensor->dtype == DATA_TYPE_INT8) {
                int8_t *src_data = static_cast<int8_t *>(tensor->data);
                int16_t *data = static_cast<int16_t *>(this->data);
                for (int i = 0; i < this->get_size(); i++) {
                    float tmp = dequantize(src_data[i], src_scale);
                    data[i] = static_cast<int16_t>(quantize(tmp, scale, DL_QUANT16_MIN, DL_QUANT16_MAX));
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

std::vector<int> TensorBase::get_axis_index(int element_index)
{
    std::vector<int> axis_index(this->shape.size(), 0);
    for (int j = this->shape.size() - 1; j > -1; --j) {
        axis_index[j] = element_index % this->shape[j];
        element_index /= this->shape[j];
    }
    return axis_index;
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
    assert(this->dtype == DATA_TYPE_INT32);

#if CONFIG_IDF_TARGET_ESP32P4
    // Reset bias layout for esp32p4
    if (op_quant_type == QUANT_TYPE_SYMM_16BIT) {
        this->dtype = DATA_TYPE_INT64;
        size_t dtype_bytes = this->get_dtype_bytes();
        size_t aligned_size = this->get_aligned_size();

        int32_t *pre_data = static_cast<int32_t *>(this->data);
        int64_t *cur_data = static_cast<int64_t *>(tool::calloc_aligned(aligned_size, dtype_bytes, 16, this->caps));
        for (int i = 0; i < this->get_size(); i++) {
            cur_data[i] = pre_data[i];
        }
        heap_caps_free(this->data);
        this->data = cur_data;
    }
#elif CONFIG_IDF_TARGET_ESP32S3
    // Reset bias layout for esp32s3
    // 0x000AAAAA000BBBBB ==> 0xAAAAABBBBB
    if (op_quant_type == QUANT_TYPE_SYMM_8BIT) {
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
        int8_t *dst_ptr = static_cast<int8_t *>(tool::calloc_aligned(memory_size_needed, dtype_bytes, 16, this->caps));
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

        heap_caps_free(this->data);
        this->data = dst_ptr;
    } else if (op_quant_type == QUANT_TYPE_SYMM_16BIT) {
        // TODO: reset bias layout for esp32s3 s16
    }
#endif
}

TensorBase &TensorBase::reshape(std::vector<int> shape)
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
    return *this;
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

int TensorBase::get_element_index(const std::vector<int> axis_index)
{
    assert(axis_index.size() == this->shape.size());
    int element_index = 0;
    for (int i = 0; i < axis_index.size(); i++) {
        element_index += axis_index[i] * this->axis_offset[i];
    }
    return element_index;
}

} // namespace dl
