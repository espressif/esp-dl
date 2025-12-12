#pragma once

#include "dl_module_base.hpp"
#include <vector>

namespace dl {
namespace module {
/**
 * @brief: Insert zeros between spatial dimensions of input tensor.
 *         This operation inserts zeros between spatial dimensions of the input tensor.
 *         For input tensor with shape [N, *spatial_dims, C], output shape will be
 *         [N, *new_spatial_dims, C] where new_spatial_dims[i] = spatial_dims[i] + (stride[i] - 1) * (spatial_dims[i] -
 * 1) + output_padding[i].
 *
 *         This operator is typically used as part of ConvTranspose decomposition.
 */
class InsertZeros : public Module {
private:
    std::vector<int> m_stride;         /*!< Stride for spatial dimensions */
    std::vector<int> m_output_padding; /*!< Output padding for spatial dimensions */

public:
    /**
     * @brief Construct a new InsertZeros object.
     *
     * @param name              name of module
     * @param stride            stride for spatial dimensions
     * @param output_padding    output padding for spatial dimensions (default 0)
     * @param quant_type        quantize type
     */
    InsertZeros(const char *name = NULL,
                const std::vector<int> &stride = {1},
                const std::vector<int> &output_padding = {0},
                quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, MODULE_NON_INPLACE, quant_type), m_stride(stride), m_output_padding(output_padding)
    {
    }

    /**
     * @brief Destroy the InsertZeros object.
     */
    ~InsertZeros() {}

    /**
     * @brief Get output shape based on input shape and stride/output_padding.
     *
     * @param input_shapes Input shapes (only one input)
     * @return std::vector<std::vector<int>> Output shapes (only one output)
     */
    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        assert(input_shapes.size() == 1);
        std::vector<int> input_shape = input_shapes[0];

        // Input must have at least 3 dimensions: [N, *spatial_dims, C]
        assert(input_shape.size() >= 3);

        int num_spatial_dims = input_shape.size() - 2;
        // Normalize stride and output_padding
        if (m_stride.size() == 1 && num_spatial_dims > 1) {
            m_stride = std::vector<int>(num_spatial_dims, m_stride[0]);
        }
        if (m_output_padding.size() == 1 && num_spatial_dims > 1) {
            m_output_padding = std::vector<int>(num_spatial_dims, m_output_padding[0]);
        }

        // Calculate output shape
        std::vector<int> output_shape = input_shape;
        for (int i = 0; i < num_spatial_dims; i++) {
            int input_dim = input_shape[1 + i];
            int s = m_stride[i];
            int op = m_output_padding[i];
            int new_dim = input_dim + (s - 1) * (input_dim - 1) + op;
            output_shape[1 + i] = new_dim;
        }

        return {output_shape};
    }

    /**
     * @brief Forward pass implementation.
     *
     * @param context Model context
     * @param mode Runtime mode
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
     * @brief Template forward pass.
     *
     * @tparam T Data type
     * @param context Model context
     * @param mode Runtime mode
     */
    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);

        T *output_data = output->get_element_ptr<T>();
        T *input_data = input->get_element_ptr<T>();
        const std::vector<int> &input_shape = input->get_shape();
        const std::vector<int> &output_shape = output->get_shape();
        const std::vector<int> &input_strides = input->axis_offset;
        const std::vector<int> &output_strides = output->axis_offset;
        int batch_size = input_shape[0];
        int channels = input_shape[input_shape.size() - 1]; // Last dimension is channels in NHWC
        int num_spatial_dims = input_shape.size() - 2;

        // Normalize stride and output_padding
        if (m_stride.size() == 1 && num_spatial_dims > 1) {
            m_stride = std::vector<int>(num_spatial_dims, m_stride[0]);
        }
        if (m_output_padding.size() == 1 && num_spatial_dims > 1) {
            m_output_padding = std::vector<int>(num_spatial_dims, m_output_padding[0]);
        }

        // Calculate spatial dimensions (skip batch and channel dimensions)
        std::vector<int> input_spatial_dims(input_shape.begin() + 1, input_shape.end() - 1);
        std::vector<int> output_spatial_dims(output_shape.begin() + 1, output_shape.end() - 1);

        // Initialize output to zeros
        memset(output_data, 0, output->get_size() * sizeof(T));

        // Base offsets for this batch
        int input_batch_offset = 0;
        int output_batch_offset = 0;

        // Process each batch
        if (num_spatial_dims == 1) {
            // 1D case (NHWC layout)
            int input_h = input_shape[1];
            int stride_h = m_stride[0];
            for (int n = 0; n < batch_size; n++) {
                for (int h = 0; h < input_h; h++) {
                    int input_idx = input_batch_offset + h * input_strides[1];
                    int output_idx = output_batch_offset + h * stride_h * output_strides[1];
                    // Copy all channels for this spatial position
                    tool::copy_memory(&output_data[output_idx], &input_data[input_idx], channels * sizeof(T));
                }
                input_batch_offset += input_strides[0];
                output_batch_offset += output_strides[0];
            }

        } else if (num_spatial_dims == 2) {
            // 2D case (NHWC layout)
            int input_h = input_shape[1];
            int input_w = input_shape[2];
            int stride_h = m_stride[0];
            int stride_w = m_stride[1];
            for (int n = 0; n < batch_size; n++) {
                for (int h = 0; h < input_h; h++) {
                    for (int w = 0; w < input_w; w++) {
                        int input_idx = input_batch_offset + h * input_strides[1] + w * input_strides[2];
                        int output_idx =
                            output_batch_offset + h * stride_h * output_strides[1] + w * stride_w * output_strides[2];
                        // Copy all channels for this spatial position
                        tool::copy_memory(&output_data[output_idx], &input_data[input_idx], channels * sizeof(T));
                    }
                }
                input_batch_offset += input_strides[0];
                output_batch_offset += output_strides[0];
            }
        } else {
            ESP_LOGE("InsertZeros", "Unsupported number of spatial dimensions: %d", num_spatial_dims);
            return;
        }
    }

    /**
     * @brief Deserialize InsertZeros module instance by node serialization information.
     *
     * @param fbs_model FlatBuffers model
     * @param node_name Node name
     * @return Module* Deserialized module
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        std::vector<int> stride;
        std::vector<int> output_padding;

        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "stride", stride);
        fbs_model->get_operation_attribute(node_name, "output_padding", output_padding);

        // Create module
        op = new InsertZeros(node_name.c_str(), stride, output_padding, quant_type);
        return op;
    }

    /**
     * @brief Print module information.
     */
    void print()
    {
        ESP_LOGI("InsertZeros",
                 "quant_type: %s, stride: %s, output_padding: %s",
                 quant_type_to_string(quant_type),
                 vector_to_string(m_stride).c_str(),
                 vector_to_string(m_output_padding).c_str());
    }
};

} // namespace module
} // namespace dl
