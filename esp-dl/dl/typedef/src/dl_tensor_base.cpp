#include "dl_tensor_base.hpp"

namespace dl {
int quantize(float input, float scale, float quant_min, float quant_max)
{
    int output = dl_esp32p4_round_half_even(input * scale);
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
            str += ",";
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
    this->caps = caps;
    size_t dtype_bytes = this->get_dtype_bytes();
    size_t aligned_size = this->get_aligned_size();
    if (element) {
        if (deep) {
            this->auto_free = true;
            this->data = heap_caps_aligned_calloc(16, aligned_size, dtype_bytes, caps);
            if (!this->data && caps != MALLOC_CAP_8BIT) {
                ESP_LOGW(__FUNCTION__, "heap_caps_aligned_calloc failed from caps: %d, retry with MALLOC_CAP_8BIT");
                this->caps = MALLOC_CAP_8BIT;
                this->data = heap_caps_aligned_calloc(16, aligned_size, dtype_bytes, MALLOC_CAP_8BIT);
            }
            tool::copy_memory(this->data, const_cast<void *>(element), this->get_size() * dtype_bytes);
        } else {
            this->auto_free = false;
            this->data = const_cast<void *>(element);
        }
    } else {
        this->auto_free = true;
        this->data = heap_caps_aligned_calloc(16, aligned_size, dtype_bytes, caps);
        if (!this->data && caps != MALLOC_CAP_8BIT) {
            ESP_LOGW(__FUNCTION__, "heap_caps_aligned_calloc failed from caps: %d, retry with MALLOC_CAP_8BIT");
            this->caps = MALLOC_CAP_8BIT;
            this->data = heap_caps_aligned_calloc(16, aligned_size, dtype_bytes, MALLOC_CAP_8BIT);
        }
    }
}

bool TensorBase::assign(TensorBase *tensor)
{
    if (tensor == nullptr || this->get_size() != tensor->get_size()) {
        return false;
    }

    if (this->exponent == tensor->exponent && this->dtype == tensor->dtype) {
        tool::copy_memory(this->data, tensor->data, this->get_bytes());
    } else if (tensor->dtype == DATA_TYPE_FLOAT) {
        float* src_data = (float*)tensor->data;
        float scale = 1.0 / (DL_SCALE(this->exponent));

        if (this->dtype == DATA_TYPE_INT8) {
            int8_t *data = (int8_t*)this->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = static_cast<int8_t>(quantize(src_data[i], scale, DL_QUANT8_MIN, DL_QUANT8_MAX));
            }
        } else if (this->dtype == DATA_TYPE_INT16) {
            int16_t *data = (int16_t*)this->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = static_cast<int16_t>(quantize(src_data[i], scale, DL_QUANT16_MIN, DL_QUANT16_MAX));
            }
        } else {
            return false;
        }
    } else if (this->dtype == DATA_TYPE_FLOAT) {
        float* data = (float*)this->data;
        float scale = DL_SCALE(tensor->exponent);

        if (tensor->dtype == DATA_TYPE_INT8 || tensor->dtype == DATA_TYPE_UINT8) {
            int8_t *src_data = (int8_t*)tensor->data;
            for (int i = 0; i < this->get_size(); i++) {
                data[i] = dequantize(src_data[i], scale);
            }
        } else if (tensor->dtype == DATA_TYPE_INT16 || tensor->dtype == DATA_TYPE_UINT16) {
            int16_t *src_data = (int16_t*)tensor->data;
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
    for (int j = this->shape.size() - 1; j > -1; --j)
    {
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

void TensorBase::reset_bias_layout()
{
#if CONFIG_IDF_TARGET_ESP32P4
    // TODO: reset bias layout for esp32p4
    this->dtype = DATA_TYPE_INT64;
    size_t dtype_bytes = this->get_dtype_bytes();
    size_t aligned_size = this->get_aligned_size();

    int32_t *pre_data = static_cast<int32_t *>(this->data);
    int64_t *cur_data = static_cast<int64_t *>(heap_caps_aligned_calloc(16, aligned_size, dtype_bytes, this->caps));
    for (int i = 0; i < this->get_size(); i++) {
        cur_data[i] = pre_data[i];
    }
    heap_caps_free(this->data);
    this->data = cur_data;
#elif CONFIG_IDF_TARGET_ESP32S3
    // TODO: reset bias layout for esp32s3
#endif
}


} // namespace dl
