#pragma once

#include "dl_tool.hpp"
#include "esp_log.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <variant>
#include <vector>

namespace dl {

/**
 * @brief The data type of esp-dl is same as flatbuffer's data type.
 */
typedef enum {
    DATA_TYPE_UNDEFINED = 0,
    DATA_TYPE_FLOAT = 1,
    DATA_TYPE_UINT8 = 2,
    DATA_TYPE_INT8 = 3,
    DATA_TYPE_UINT16 = 4,
    DATA_TYPE_INT16 = 5,
    DATA_TYPE_INT32 = 6,
    DATA_TYPE_INT64 = 7,
    DATA_TYPE_STRING = 8,
    DATA_TYPE_BOOL = 9,
    DATA_TYPE_FLOAT16 = 10,
    DATA_TYPE_DOUBLE = 11,
    DATA_TYPE_UINT32 = 12,
    DATA_TYPE_UINT64 = 13,
    DATA_TYPE_MIN = DATA_TYPE_UNDEFINED,
    DATA_TYPE_MAX = DATA_TYPE_UINT64
} dtype_t;

/**
 * quantize float data into integer data
 */
int quantize(float input, float scale, float quant_min, float quant_max);

/**
 * @brief dequantize integer data into float data
 */
float dequantize(int input, float scale);

/**
 * @brief Return the bytes of data type
 */
size_t dtype_sizeof(dtype_t dtype);

/**
 * @brief Return the data type string
 */
const char *dtype_to_string(dtype_t dtype);

/**
 * @brief Return acivation type string
 */
const char *activation_type_to_string(activation_type_t type);

/**
 * @brief Return quant type string
 */
const char *quant_type_to_string(quant_type_t type);

/**
 * @brief Convert shape(vector<int>) to string
 */
std::string shape_to_string(std::vector<int> shape);

/**
 * @brief This class is designed according to PyTorch Tensor.
 * TensorBase is required to ensure that the first address are aligned to 16 bytes and the memory size should be a
 * multiple of 16 bytes.
 *
 * TODO:: Implement more functions
 */
class TensorBase {
public:
    int size;                     /*<! size of element including padding */
    std::vector<int> shape;       /*<! shape of Tensor */
    dtype_t dtype;                /*<! data type of element */
    int exponent;                 /*<! exponent of element */
    bool auto_free;               /*<! free element when object destroy */
    std::vector<int> axis_offset; /*<! element offset of each axis */
    void *data;                   /*<! data pointer */
    void *cache;                  /*<! cache pointer， used for preload and do not need to free */
    uint32_t caps;                /*<! flags indicating the type of memory */

    TensorBase()
    {
        this->size = 0;
        this->shape = {};
        this->dtype = DATA_TYPE_FLOAT;
        this->exponent = 0;
        this->auto_free = true;
        this->axis_offset = {};
        this->data = nullptr;
        this->cache = nullptr;
        this->caps = MALLOC_CAP_8BIT;
    }

    TensorBase(std::vector<int> shape,
               const void *element,
               int exponent = 0,
               dtype_t dtype = DATA_TYPE_FLOAT,
               bool deep = true,
               uint32_t caps = MALLOC_CAP_8BIT);

    virtual ~TensorBase()
    {
        if (this->auto_free) {
            heap_caps_free(this->data);
        }
    }

    /**
     * @brief Assign tensor to this tensor
     *
     * @param tensor
     *
     * @return ture if assign successfully, otherwise false.
     */
    bool assign(TensorBase *tensor);

    /**
     * @brief Assign data to this tensor
     *
     * @param shape
     * @param element
     * @param exponent
     * @param dtype
     *
     * @return ture if assign successfully, otherwise false.
     */
    bool assign(std::vector<int> shape, const void *element, int exponent, dtype_t dtype);

    /**
     * @brief Get the size of Tensor.
     *
     * @return  the size of Tensor.
     */
    int get_size() { return this->size; }

    /**
     * @brief Get the aligned size of Tensor.
     *
     * @return  the aligned size of Tensor.
     */
    int get_aligned_size()
    {
        int align = 16 / this->get_dtype_bytes();
        return this->size % align == 0 ? this->size : this->size + align - this->size % align;
    }

    /**
     * @brief Get the dtype size, in bytes.
     *
     * @return  the size of dtype.
     */
    size_t get_dtype_bytes() { return dtype_sizeof(this->dtype); }

    /**
     * @brief Get the dtype string of Tensor.
     *
     * @return  the string of Tensor's dtype.
     */
    const char *get_dtype_string() { return dtype_to_string(this->dtype); }

    /**
     * @brief Get the bytes of Tensor.
     *
     * @return  the bytes of Tensor.
     */
    int get_bytes() { return this->size * this->get_dtype_bytes(); }

    /**
     * @brief Get element pointer. If cache(preload data pointer) is not null, return cache pointer, otherwise return
     * data pointer.
     *
     * @return  the pointer of Tensor's element
     */
    virtual void *get_element_ptr()
    {
        if (this->cache) {
            return this->cache; // If preload cache is not null, use this pointer
        }

        return this->data;
    }

    /**
     * @brief Get the index of each dims
     *
     * @param element_index the index of the element
     * @return std::vector<int> the index of each dims
     */
    virtual std::vector<int> get_axis_index(int element_index);

    virtual void preload()
    {
        if (this->cache) {
            tool::copy_memory(this->cache, this->cache, this->get_bytes());
        }
    }

    /**
     * @brief Set the shape of Tensor.
     * @param shape the shape of Tensor.
     * @return  Tensor.
     */
    TensorBase &set_shape(const std::vector<int> shape);

    std::vector<int> get_shape() { return this->shape; }

    dtype_t get_dtype() { return this->dtype; }

    size_t set_preload_addr(void *addr, size_t size);

    void reset_bias_layout(quant_type_t op_quant_type, bool is_depthwise);

    /**
     * @brief Change a new shape to the Tensor without changing its data.
     *
     * @param shape  the target shape
     * @return TensorBase&  self
     */
    TensorBase &reshape(std::vector<int> shape);

    template <typename T>
    TensorBase *transpose(T *input_element,
                          std::vector<int> &input_shape,
                          std::vector<int> &input_axis_offset,
                          std::vector<int> &perm);

    /**
     * @brief Reverse or permute the axes of the input Tensor
     *
     * @param input the input Tensor
     * @param perm the new arangement of the dims. if perm == {}, the dims arangement will be reversed.
     * @return TensorBase *self
     */
    TensorBase *transpose(TensorBase *input, std::vector<int> perm = {});

    /**
     * @brief Get the index of element
     *
     * @param axis_index the index of each dims
     * @return int the index of element
     */
    int get_element_index(const std::vector<int> axis_index);

    // int& operator[](size_t index) {
    //     if (index >= this->size) {
    //         throw std::out_of_range("Index out of range");
    //     }
    //     return data[index];
    // }
};
} // namespace dl
