#pragma once

#include "dl_tool.hpp"
#include "esp_log.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace dl {

/**
 * @brief Exponent info for per-tensor / per-channel quantization.
 *
 * Per-tensor: m_exponent only, zero heap allocation.
 * Per-channel: heap-allocated m_exponents array.
 * Provides operator int() so existing `DL_SCALE(tensor->exponent)` code works unchanged.
 */
class ExponentInfo {
private:
    int m_exponent;
    int *m_exponents;
    int m_size;

public:
    /**
     * @brief Construct a per-tensor ExponentInfo with a single exponent value.
     * @param exponent The exponent value for per-tensor quantization. Default is 0.
     */
    ExponentInfo(int exponent = 0) : m_exponent(exponent), m_exponents(nullptr), m_size(1) {}

    /**
     * @brief Construct an ExponentInfo from a vector of exponents.
     * @param exponents Vector of exponent values. If size <= 1, uses per-tensor mode;
     *                  otherwise uses per-channel mode with heap allocation.
     */
    ExponentInfo(const std::vector<int> &exponents) : m_exponents(nullptr), m_size(1)
    {
        if (exponents.size() <= 1) {
            m_exponent = exponents.empty() ? 0 : exponents[0];
        } else {
            m_size = exponents.size();
            m_exponents = new int[m_size];
            for (int i = 0; i < m_size; i++) {
                m_exponents[i] = exponents[i];
            }
            m_exponent = 0;
        }
    }

    /**
     * @brief Destroy the ExponentInfo object and free heap-allocated memory.
     */
    ~ExponentInfo()
    {
        delete[] m_exponents;
        m_exponents = nullptr;
    }

    /**
     * @brief Copy constructor.
     * @param other The ExponentInfo object to copy from.
     */
    ExponentInfo(const ExponentInfo &other) : m_exponent(other.m_exponent), m_exponents(nullptr), m_size(other.m_size)
    {
        if (other.m_exponents && m_size > 1) {
            m_exponents = new int[m_size];
            for (int i = 0; i < m_size; i++) {
                m_exponents[i] = other.m_exponents[i];
            }
        }
    }

    /**
     * @brief Copy assignment operator.
     * @param other The ExponentInfo object to assign from.
     * @return Reference to this object.
     */
    ExponentInfo &operator=(const ExponentInfo &other)
    {
        if (this != &other) {
            delete[] m_exponents;
            m_exponent = other.m_exponent;
            m_size = other.m_size;
            if (other.m_exponents && m_size > 1) {
                m_exponents = new int[m_size];
                for (int i = 0; i < m_size; i++) {
                    m_exponents[i] = other.m_exponents[i];
                }
            } else {
                m_exponents = nullptr;
            }
        }
        return *this;
    }

    /**
     * @brief Move constructor.
     * @param other The ExponentInfo object to move from.
     */
    ExponentInfo(ExponentInfo &&other) noexcept :
        m_exponent(other.m_exponent), m_exponents(other.m_exponents), m_size(other.m_size)
    {
        other.m_exponents = nullptr;
        other.m_size = 1;
    }

    /**
     * @brief Move assignment operator.
     * @param other The ExponentInfo object to move from.
     * @return Reference to this object.
     */
    ExponentInfo &operator=(ExponentInfo &&other) noexcept
    {
        if (this != &other) {
            delete[] m_exponents;
            m_exponent = other.m_exponent;
            m_exponents = other.m_exponents;
            m_size = other.m_size;
            other.m_exponents = nullptr;
            other.m_size = 1;
        }
        return *this;
    }

    /**
     * @brief Assign a single integer value as per-tensor exponent.
     * @param value The exponent value to assign.
     * @return Reference to this object.
     */
    ExponentInfo &operator=(int value)
    {
        delete[] m_exponents;
        m_exponents = nullptr;
        m_size = 1;
        m_exponent = value;
        return *this;
    }

    /**
     * @brief Get exponent value.
     * @param ch  Channel index. -1 (default) returns per-tensor value.
     *            ch >= 0 returns per-channel value if available, otherwise per-tensor.
     * @return The exponent value for the specified channel or per-tensor exponent.
     */
    int get(int ch = -1) const
    {
        if (ch < 0 || !m_exponents) {
            return m_exponent;
        }
        return m_exponents[ch];
    }

    /**
     * @brief Implicit conversion to int, returns the per-tensor exponent value.
     * @return The per-tensor exponent value.
     */
    operator int() const { return m_exponent; }

    /**
     * @brief Check if using per-channel quantization.
     * @return true if per-channel exponents are stored, false otherwise.
     */
    bool is_per_channel() const { return m_exponents != nullptr && m_size > 1; }

    /**
     * @brief Get the number of channels.
     * @return Number of channels for per-channel mode, or 1 for per-tensor mode.
     */
    int channel_size() const { return m_size; }

    /**
     * @brief Get pointer to exponent data.
     * @return Pointer to per-channel exponents array if available, otherwise pointer to per-tensor exponent.
     */
    const int *data() const { return m_exponents ? m_exponents : &m_exponent; }

    /**
     * @brief Compare two ExponentInfo objects for equality.
     * @param other The ExponentInfo object to compare with.
     * @return true if both have the same exponent values, false otherwise.
     */
    bool operator==(const ExponentInfo &other) const
    {
        if (!this->is_per_channel() && !other.is_per_channel()) {
            return this->m_exponent == other.m_exponent;
        }
        if (this->is_per_channel() != other.is_per_channel()) {
            return false;
        }
        if (this->m_size != other.m_size) {
            return false;
        }
        for (int i = 0; i < this->m_size; i++) {
            if (this->m_exponents[i] != other.m_exponents[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Compare two ExponentInfo objects for inequality.
     * @param other The ExponentInfo object to compare with.
     * @return true if exponent values differ, false if equal.
     */
    bool operator!=(const ExponentInfo &other) const { return !(*this == other); }
};

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
template <typename RT, typename T = float>
RT quantize(T input, float inv_scale);

/**
 * @brief dequantize integer data into float data
 */
template <typename T, typename RT = float>
RT dequantize(T input, float scale);

/**
 * @brief Return the bytes of data type
 */
size_t dtype_sizeof(dtype_t dtype);

/**
 * @brief Return the data type string
 */
const char *dtype_to_string(dtype_t dtype);

/**
 * @brief Return activation type string
 */
const char *activation_type_to_string(activation_type_t type);

/**
 * @brief Return quant type string
 */
const char *quant_type_to_string(quant_type_t type);

/**
 * @brief Convert vector to string
 */
template <typename T>
std::string vector_to_string(std::vector<T> shape)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type (int, double, etc.)");

    if (shape.empty()) {
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

/**
 * @brief This class is designed according to PyTorch Tensor.
 * TensorBase is required to ensure that the first address are aligned to 16 bytes and the memory size should be a
 * multiple of 16 bytes.
 *
 * TODO:: Implement more functions
 */
class TensorBase {
public:
    int size;                     ///< size of element including padding
    std::vector<int> shape;       ///< shape of Tensor
    dtype_t dtype;                ///< data type of element
    ExponentInfo exponent;        ///< exponent of element (per-tensor or per-channel)
    bool auto_free;               ///< free element when object destroy
    std::vector<int> axis_offset; ///< element offset of each axis
    void *data;                   ///< data pointer
    void *cache;                  ///< cache pointer， used for preload and do not need to free
    uint32_t caps;                ///< flags indicating the type of memory

    /**
     * @brief Construct a TensorBase object
     *
     * @param shape  Shape of tensor
     * @param element  Pointer of data
     * @param exponent Exponent of tensor, default is 0
     * @param dtype    Data type of element, default is float
     * @param deep     True: malloc memory and copy data, false: use the pointer directly
     * @param caps     Bitwise OR of MALLOC_CAP_* flags indicating the type of memory to be returned
     *
     */
    TensorBase(std::vector<int> shape,
               const void *element,
               int exponent = 0,
               dtype_t dtype = DATA_TYPE_FLOAT,
               bool deep = true,
               uint32_t caps = MALLOC_CAP_DEFAULT);

    /**
     * @brief Construct a TensorBase object with per-channel exponents
     *
     * @param shape     Shape of tensor
     * @param element   Pointer of data
     * @param exponents Per-channel exponents
     * @param dtype     Data type of element, default is float
     * @param deep      True: malloc memory and copy data, false: use the pointer directly
     * @param caps      Bitwise OR of MALLOC_CAP_* flags indicating the type of memory to be returned
     *
     */
    TensorBase(std::vector<int> shape,
               const void *element,
               const std::vector<int> &exponents,
               dtype_t dtype = DATA_TYPE_FLOAT,
               bool deep = true,
               uint32_t caps = MALLOC_CAP_DEFAULT);

    /**
     * @brief Destroy the TensorBase object.
     */
    virtual ~TensorBase()
    {
        if (this->auto_free) {
            heap_caps_free(this->data);
            this->data = nullptr;
        }
    }

#if CONFIG_SPIRAM
    void *operator new(size_t size) { return tool::malloc_aligned(size, MALLOC_CAP_SPIRAM); }

    void operator delete(void *ptr) { heap_caps_free(ptr); }
#endif

    /**
     * @brief Assign tensor to this tensor
     *
     * @param tensor
     *
     * @return true if assign successfully, otherwise false.
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
     * @return true if assign successfully, otherwise false.
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
     * @brief Get the bytes of Tensor.
     *
     * @return  the bytes of Tensor.
     */
    int get_aligned_bytes() { return this->get_aligned_size() * this->get_dtype_bytes(); }

    /**
     * @brief Get data pointer. If cache(preload data pointer) is not null, return cache pointer, otherwise return
     * data pointer.
     *
     * @return  the pointer of Tensor's data
     */
    virtual void *get_element_ptr()
    {
        if (this->cache) {
            return this->cache; // If preload cache is not null, use this pointer
        }

        return this->data;
    }

    /**
     * @brief Get data pointer by the specified template.
     * If cache(preload data pointer) is not null, return cache pointer, otherwise return data pointer.
     *
     * @return  the pointer of Tensor's data
     */
    template <typename T>
    T *get_element_ptr()
    {
        if (this->cache) {
            return (T *)this->cache; // If preload cache is not null, use this pointer
        }

        return (T *)this->data;
    }

    /**
     * @brief Set the data pointer of Tensor.
     *
     * @param data point to data memory
     * @return TensorBase&  self
     */
    TensorBase &set_element_ptr(void *data);

    /**
     * @brief Get the shape of Tensor.
     *
     * @return std::vector<int> the shape of Tensor
     */
    std::vector<int> get_shape() { return this->shape; }

    /**
     * @brief Set the shape of Tensor.
     * @param shape the shape of Tensor.
     * @return  Tensor.
     */
    TensorBase &set_shape(const std::vector<int> shape);

    /**
     * @brief Get the exponent of Tensor
     *
     * @return int the exponent of Tensor
     */
    int get_exponent() { return this->exponent; }

    /**
     * @brief Get the data type of Tensor
     *
     * @return dtype_t the data type of Tensor
     */
    dtype_t get_dtype() { return this->dtype; }

    /**
     * @brief Get the memory flags of Tensor
     *
     * @return uint32_t the memory flags of Tensor
     */
    uint32_t get_caps() { return this->caps; }

    /**
     * @brief Change a new shape to the Tensor without changing its data.
     *
     * @param shape  the target shape
     * @return TensorBase  *self
     */
    TensorBase *reshape(std::vector<int> shape);

    /**
     * @brief Flip the input Tensor along the specified axes.
     *
     * @param axes  the specified axes
     * @return TensorBase&  self
     */
    template <typename T>
    TensorBase *flip(const std::vector<int> &axes);

    /**
     * @brief Reverse or permute the axes of the input Tensor
     *
     * @param input the input Tensor
     * @param perm the new arrangement of the dims. if perm == {}, the dims arrangement will be reversed.
     * @return TensorBase *self
     */
    TensorBase *transpose(TensorBase *input, std::vector<int> perm = {});

    /**
     * @brief Reverse or permute the axes of the input Tensor
     *
     * @param input_element the input data pointer
     * @param input_shape   the input data shape
     * @param input_axis_offset the input data axis offset
     * @param perm the new arrangement of the dims. if perm == {}, the dims arrangement will be reversed.
     *
     * @return TensorBase *self
     */
    template <typename T>
    TensorBase *transpose(T *input_element,
                          std::vector<int> &input_shape,
                          std::vector<int> &input_axis_offset,
                          std::vector<int> &perm);

    /**
     * @brief Check the shape is the same as the shape of input.
     *
     * @param tensor Input tensor pointer
     * @return
     *         - true: same shape
     *         - false: not
     */
    bool is_same_shape(TensorBase *tensor);

    /**
     * @brief Compare the shape and data of two Tensor
     *
     * @param tensor Input tensor
     * @param epsilon The max error of two element
     * @param verbose If true, print the detail of results
     *
     * @return true if two tensor is equal otherwise false
     */
    bool equal(TensorBase *tensor, float epsilon = 1e-6, bool verbose = false);

    /**
     * @brief Produces a slice of the this tensor along multiple axes
     *
     * @warning The length of start, end and step must be same as the shape of input tensor
     *
     * @param start Starting indicesd
     * @param end Ending indices
     * @param axes Axes that starts and ends apply to.
     * @param step Slice step, step = 1 if step is not specified
     * @return TensorBase* Output tensor pointer, created by this slice function
     */
    TensorBase *slice(const std::vector<int> &start,
                      const std::vector<int> &end,
                      const std::vector<int> &axes = {},
                      const std::vector<int> &step = {});

    /**
     * @brief Produces a slice along multiple axes
     *
     * @warning The length of start, end and step must be same as the shape of input tensor
     *
     * @param input   Input Tensor
     * @param output  Output Tensor
     * @param start   Starting indicesd
     * @param end     Ending indices
     * @param axes    Axes that starts and ends apply to.
     * @param step    Slice step, step = 1 if step is not specified
     */
    static void slice(TensorBase *input,
                      TensorBase *output,
                      const std::vector<int> &start,
                      const std::vector<int> &end,
                      const std::vector<int> &axes = {},
                      const std::vector<int> &step = {});

    /**
     * @brief Pad input tensor
     *
     * @param input_element Data pointer of input tensor
     * @param input_shape   Shape of input tensor
     * @param pads   The number of padding elements to add, pads format should be: [x1_begin, x2_begin, …, x1_end,
     * x2_end,…]
     * @param mode   Supported modes: constant(default), reflect, edge
     * @param const_value (Optional) A scalar value to be used if the mode chosen is constant
     *
     * @return Output tensor pointer
     */
    template <typename T>
    TensorBase *pad(T *input_element,
                    const std::vector<int> &input_shape,
                    const std::vector<int> &pads,
                    const padding_mode_t mode,
                    TensorBase *const_value = nullptr);

    /**
     * @brief Pad input tensor
     *
     * @param input  Input tensor pointer
     * @param pads   Padding elements to add, pads format should be: [x1_begin, x2_begin, …, x1_end, x2_end,…]
     * @param mode   Supported modes: constant(default), reflect, edge
     * @param const_value (Optional) A scalar value to be used if the mode chosen is constant
     *
     * @return Output tensor pointer
     */
    TensorBase *pad(TensorBase *input,
                    const std::vector<int> &pads,
                    const padding_mode_t mode,
                    TensorBase *const_value = nullptr);

    /**
     * @brief Compare the elements of two Tensor
     *
     * @param gt_elements The ground truth elements
     * @param epsilon The max error of two element
     * @param verbose If true, print the detail of results
     *
     * @return true if all elements are equal otherwise false
     */
    template <typename T>
    bool compare_elements(const T *gt_elements, float epsilon = 1e-6, bool verbose = false);

    /**
     * @brief Get the index of element
     *
     * @param axis_index The coordinates of element
     * @return int the index of element
     */
    int get_element_index(const std::vector<int> &axis_index);

    /**
     * @brief Get the coordinates of element
     *
     * @param index  The index of element
     * @return   The coordinates of element
     */
    std::vector<int> get_element_coordinates(int index);

    /**
     * @brief Get a element of Tensor by index
     *
     * @param index  The index of element
     * @return   The element of tensor
     */
    template <typename T>
    T get_element(int index);

    /**
     * @brief Get a element of Tensor
     *
     * @param axis_index  The index of element
     * @return   The element of tensor
     */
    template <typename T>
    T get_element(const std::vector<int> &axis_index);

    /**
     * @brief Set a element of Tensor by index
     *
     * @param value  The value of element
     */
    void memset(int value);

    /**
     * @brief Fill tensor data with random bytes from hardware RNG.
     *
     * Uses esp_fill_random() to fill the tensor's internal buffer with random bytes.
     * Requires ESP-IDF (esp_random.h).
     */
    void rand();

    /**
     * @brief Set preload address of Tensor
     *
     * @param addr  The address of preload data
     * @param size  Size of preload data
     *
     * @return The size of preload data
     */
    size_t set_preload_addr(void *addr, size_t size);

    /**
     * @brief Preload the data of Tensor
     *
     */
    virtual void preload()
    {
        if (this->cache) {
            tool::copy_memory(this->cache, this->cache, this->get_bytes());
        }
    }

    /**
     * @brief Reset the layout of Tensor
     *
     * @warning Only available for Convolution. Don't use it unless you know exactly what it does.
     *
     * @param op_quant_type  The quant type of operation
     * @param is_depthwise   Whether is depthwise convolution
     *
     */
    void reset_bias_layout(quant_type_t op_quant_type, bool is_depthwise);

    /**
     * @brief Push new_tensor to current tensor. The time series dimension size of new tensor must is lesser or equal
     * than that of the current tensor."
     *
     * @param new_tensor  The new tensor will be pushed
     * @param dim   Specify the dimension on which to perform streaming stack pushes
     *
     */
    void push(TensorBase *new_tensor, int dim);

    /**
     * @brief print the information of TensorBase
     *
     * @param print_data Whether print the data
     */
    virtual void print(bool print_data = false);
};
} // namespace dl
