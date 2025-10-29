#pragma once

#include "dl_base.hpp"
#include "dl_define.hpp"
#include "dl_module_base.hpp"
#include "dl_tensor_base.hpp"
#include "dl_tool.hpp"

namespace dl {
namespace module {

/**
 * @brief: SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
 *         this op outputs a copy of the input tensor where values from the height and width dimensions
 *         are moved to the depth dimension. This is the inverse transformation of DepthToSpace.
 *         Supports int8, int16 and float32.
 */
class SpaceToDepth : public Module {
private:
    int m_blocksize;

    template <typename T>
    void space_to_depth(T *input, T *output, int batch_size, int height, int width, int channels, int block_size)
    {
        int output_width = width / block_size;
        int block_squared = block_size * block_size;
        int output_channels = channels * block_squared;
        int h_offset = output_width * output_channels;

        for (int n = 0; n < batch_size; ++n) {
            int batch_offset_out = n * height * width * channels;

            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        // Calculate output indices using the core SpaceToDepth formula
                        int out_c = c * block_squared + ((h % block_size) * block_size + (w % block_size));
                        int out_h = h / block_size;
                        int out_w = w / block_size;

                        // Calculate linear indices (NHWC layout)
                        int output_idx = batch_offset_out + out_h * h_offset + out_w * output_channels + out_c;
                        output[output_idx] = *input++;
                    }
                }
            }
        }
    }

public:
    /**
     * @brief Construct a new SpaceToDepth object.
     *
     * @param blocksize    Size of the spatial block
     * @param name         Name of module
     * @param inplace      Inplace type
     * @param quant_type   Quantization type
     */
    SpaceToDepth(int blocksize = 2,
                 const char *name = NULL,
                 module_inplace_t inplace = MODULE_NON_INPLACE,
                 quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_blocksize(blocksize)
    {
    }

    /**
     * @brief Destroy the SpaceToDepth object.
     */
    ~SpaceToDepth() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<int> input_shape = input_shapes[0];
        std::vector<int> output_shape = input_shape;

        // The input tensor is expected to be 4D: [N, H, W, C].
        if (input_shape.size() < 4) {
            ESP_LOGE("SpaceToDepth", "Input tensor must be 4D [N, H, W, C]");
            return std::vector<std::vector<int>>(1, input_shape); // return original shape on error
        }

        // Validate that height and width are divisible by blocksize
        if (input_shape[1] % m_blocksize != 0 || input_shape[2] % m_blocksize != 0) {
            ESP_LOGE("SpaceToDepth",
                     "Height (%d) and width (%d) must be divisible by blocksize (%d)",
                     input_shape[1],
                     input_shape[2],
                     m_blocksize);
            return std::vector<std::vector<int>>(1, input_shape); // return original shape on error
        }

        output_shape[1] = input_shape[1] / m_blocksize;                 // height
        output_shape[2] = input_shape[2] / m_blocksize;                 // width
        output_shape[3] = input_shape[3] * (m_blocksize * m_blocksize); // channels

        return std::vector<std::vector<int>>(1, output_shape);
    }

    void forward(ModelContext *context, runtime_mode_t mode = RUNTIME_MODE_AUTO)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            forward_template<float>(context, mode);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        std::vector<int> input_shape = input->get_shape();

        // Perform space-to-depth transformation
        space_to_depth<T>(input->get_element_ptr<T>(),
                          output->get_element_ptr<T>(),
                          input_shape[0],
                          input_shape[1],
                          input_shape[2],
                          input_shape[3],
                          m_blocksize);
    }

    void forward_args(void *args) {}

    /**
     * @brief deserialize SpaceToDepth module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        int blocksize;

        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "blocksize", blocksize);

        // Create module
        op = new SpaceToDepth(blocksize, node_name.c_str(), MODULE_NON_INPLACE, quant_type);
        return op;
    }

    void print()
    {
        ESP_LOGI("SpaceToDepth", "blocksize: %d, quant_type: %s", m_blocksize, quant_type_to_string(quant_type));
    }
};

} // namespace module
} // namespace dl
