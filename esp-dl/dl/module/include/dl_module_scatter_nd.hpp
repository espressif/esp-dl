#pragma once

#include "dl_module_base.hpp"
#include "dl_tensor_base.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace dl {
namespace module {

/**
 * @brief ScatterND takes three inputs: data tensor, indices tensor, and updates tensor.
 *        The output is produced by creating a copy of the input data, and then updating
 *        its values to values specified by updates at specific index positions specified by indices.
 *        Supports reduction operations: none (default), add, mul, max, min.
 */
class ScatterND : public Module {
private:
    std::string m_reduction; ///< Reduction operation: "none", "add", "mul", "max", "min"
    int32_t m_slice_size;    ///< Size of each slice to be updated
    int32_t m_n_slices;      ///< Number of slices to be updated

    /**
     * @brief Apply reduction operation
     *
     * @param current Current value in output tensor
     * @param update Update value from updates tensor
     * @return Result after applying reduction operation
     */
    float apply_reduction(float current, float update)
    {
        if (m_reduction == "add") {
            return current + update;
        } else if (m_reduction == "mul") {
            return current * update;
        } else if (m_reduction == "max") {
            return std::max(current, update);
        } else if (m_reduction == "min") {
            return std::min(current, update);
        } else {
            // Default to "none" if unknown reduction
            return update;
        }
    }

    // Specializations for int8_t and int16_t to handle saturation
    int8_t apply_reduction(int8_t current, int8_t update)
    {
        if (m_reduction == "add") {
            // Handle saturation for int8
            int32_t result = (int32_t)current + (int32_t)update;
            if (result > 127)
                return 127;
            if (result < -128)
                return -128;
            return (int8_t)result;
        } else if (m_reduction == "mul") {
            // Handle saturation for int8
            int32_t result = (int32_t)current * (int32_t)update;
            if (result > 127)
                return 127;
            if (result < -128)
                return -128;
            return (int8_t)result;
        } else if (m_reduction == "max") {
            return std::max(current, update);
        } else if (m_reduction == "min") {
            return std::min(current, update);
        } else {
            return update;
        }
    }

    int16_t apply_reduction(int16_t current, int16_t update)
    {
        if (m_reduction == "add") {
            // Handle saturation for int16
            int32_t result = (int32_t)current + (int32_t)update;
            if (result > 32767)
                return 32767;
            if (result < -32768)
                return -32768;
            return (int16_t)result;
        } else if (m_reduction == "mul") {
            // Handle saturation for int16
            int32_t result = (int32_t)current * (int32_t)update;
            if (result > 32767)
                return 32767;
            if (result < -32768)
                return -32768;
            return (int16_t)result;
        } else if (m_reduction == "max") {
            return std::max(current, update);
        } else if (m_reduction == "min") {
            return std::min(current, update);
        } else {
            return update;
        }
    }

public:
    /**
     * @brief Construct a new ScatterND object
     *
     * @param reduction Reduction operation ("none", "add", "mul", "max", "min")
     * @param name Name of module
     * @param inplace Inplace operation mode
     * @param quant_type Quantization type
     */
    ScatterND(const std::string &reduction = "none",
              const char *name = NULL,
              module_inplace_t inplace = MODULE_NON_INPLACE,
              quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_reduction(reduction)
    {
        m_slice_size = -1;
        m_n_slices = -1;
    }

    /**
     * @brief Destroy the ScatterND object
     */
    ~ScatterND() {}

    /**
     * @brief Calculate output shape (same as data shape)
     *
     * @param input_shapes Input shapes [data_shape, indices_shape, updates_shape]
     * @return Output shapes
     */
    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        // ScatterND output shape is the same as data shape
        if (input_shapes.empty() || input_shapes[0].empty()) {
            return {{}};
        }
        return {input_shapes[0]};
    }

    /**
     * @brief Forward implementation for different data types
     */
    void forward(ModelContext *context, runtime_mode_t mode)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            forward_template<float>(context, mode);
        }
    }

    /**
     * @brief Template forward implementation
     */
    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        // Get input tensors
        TensorBase *data = context->get_tensor(m_inputs_index[0]);
        TensorBase *indices = context->get_tensor(m_inputs_index[1]);
        TensorBase *updates = context->get_tensor(m_inputs_index[2]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        // Get tensor properties
        T *data_ptr = (T *)data->get_element_ptr();
        int64_t *indices_ptr = (int64_t *)indices->get_element_ptr();
        T *updates_ptr = (T *)updates->get_element_ptr();
        T *output_ptr = (T *)output->get_element_ptr();

        // Copy data to output (ScatterND creates a copy of data first)
        if (data_ptr != output_ptr)
            tool::copy_memory(output_ptr, data_ptr, data->size * sizeof(T));

        // Get tensor shapes
        std::vector<int> axis_offset = data->axis_offset;
        std::vector<int> updates_shape = updates->shape;
        int indices_nd = indices->shape.back(); // Last dimension of indices

        // Calculate number of update slices
        if (m_n_slices == -1) {
            std::vector<int> indices_shape = indices->shape;
            int indices_rank = indices_shape.size();
            std::vector<int> data_shape = data->shape;
            int data_rank = data_shape.size();

            m_n_slices = 1;
            for (int i = 0; i < indices_rank - 1; ++i) {
                m_n_slices *= indices_shape[i];
            }

            // Calculate slice size (size of each update slice)
            m_slice_size = 1;
            for (int i = indices_nd; i < data_rank; ++i) {
                m_slice_size *= data_shape[i];
            }
        }

        // Process each update slice
        for (int slice_idx = 0; slice_idx < m_n_slices; ++slice_idx) {
            // Get indices for this slice
            int64_t *slice_indices = indices_ptr + slice_idx * indices_nd;

            // Calculate flat index in data tensor
            int64_t flat_index = 0;
            for (int i = 0; i < indices_nd; ++i) {
                flat_index += slice_indices[i] * axis_offset[i];
            }

            // Apply updates with reduction
            T *update_slice = updates_ptr + slice_idx * m_slice_size;
            T *output_slice = output_ptr + flat_index;

            if (m_reduction == "none") {
                // Directly copy update slice to output
                tool::copy_memory(output_slice, update_slice, m_slice_size * sizeof(T));
            } else {
                // Apply reduction element-wise
                for (int i = 0; i < m_slice_size; ++i) {
                    output_slice[i] = apply_reduction(output_slice[i], update_slice[i]);
                }
            }
        }
    }

    /**
     * @brief Low-level interface for base layer and multi-core processing
     */
    void forward_args(void *args) {}

    /**
     * @brief Deserialize ScatterND module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        std::string reduction = "none";

        // Get quantization type
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);

        // Get reduction attribute if available
        fbs_model->get_operation_attribute(node_name, "reduction", reduction);

        // Create module
        op = new ScatterND(reduction, node_name.c_str(), MODULE_INPLACE_CHANGED_BUFFER, quant_type);

        return op;
    }

    /**
     * @brief Print module information
     */
    void print()
    {
        ESP_LOGI("ScatterND", "reduction: %s, quant_type: %s.", m_reduction.c_str(), quant_type_to_string(quant_type));
    }
};

} // namespace module
} // namespace dl
