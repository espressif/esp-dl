#pragma once

#include "dl_base.hpp"
#include "dl_define.hpp"
#include "dl_module_base.hpp"
#include "dl_tensor_base.hpp"
#include "dl_tool.hpp"

namespace dl {
namespace module {

/**
 * @brief: DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
 *         This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
 *         the input tensor where values from the depth dimension are moved in spatial blocks to the height
 *         and width dimensions. Supports int8, int16 and float32.
 */
class DepthToSpace : public Module {
private:
    int m_blocksize;
    std::string m_mode; // "DCR" or "CRD"

    template <typename T>
    void depth_to_space_crd(T *input, T *output, int batch_size, int height, int width, int channels, int block_size)
    {
        int output_channels = channels / (block_size * block_size);
        int h_offset = width * block_size * output_channels;
        int block_2d = block_size * block_size;

        for (int b = 0; b < batch_size; ++b) {
            int b_offset = b * height * width * channels;
            for (int h = 0; h < height; ++h) {
                int h_idx = h * block_size;
                for (int w = 0; w < width; ++w) {
                    int w_idx = w * block_size;
                    for (int c = 0; c < channels; ++c) {
                        int out_c = c / block_2d;
                        int out_h = h_idx + (c % block_2d) / block_size;
                        int out_w = w_idx + c % block_size;
                        int idx = b_offset + out_h * h_offset + out_w * output_channels + out_c;
                        output[idx] = *input++;
                    }
                }
            }
        }
    }

    template <typename T>
    void depth_to_space_dcr(T *input, T *output, int batch_size, int height, int width, int channels, int block_size)
    {
        int output_channels = channels / (block_size * block_size);
        int h_offset = width * block_size * output_channels;
        int block_2d = block_size * block_size;

        for (int b = 0; b < batch_size; ++b) {
            int b_offset = b * height * width * channels;
            for (int h = 0; h < height; ++h) {
                int new_h = h * block_size;
                for (int w = 0; w < width; ++w) {
                    int new_w = w * block_size;
                    for (int c = 0; c < channels; ++c) {
                        int out_c = c % block_size;
                        int h_idx = new_h + c / block_2d;
                        int w_idx = new_w + (c % block_2d) / block_size;
                        int idx = b_offset + h_idx * h_offset + w_idx * output_channels + out_c;
                        output[idx] = *input++;
                    }
                }
            }
        }
    }

public:
    /**
     * @brief Construct a new DepthToSpace object.
     *
     * @param blocksize    Size of the spatial block
     * @param mode         Mode of depth to space, "DCR" or "CRD", default is "DCR"
     * @param name         Name of module
     * @param inplace      Inplace type
     * @param quant_type   Quantization type
     */
    DepthToSpace(int blocksize = 2,
                 std::string mode = "DCR",
                 const char *name = NULL,
                 module_inplace_t inplace = MODULE_NON_INPLACE,
                 quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, inplace, quant_type), m_blocksize(blocksize), m_mode(mode)
    {
    }

    /**
     * @brief Destroy the DepthToSpace object.
     */
    ~DepthToSpace() {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        std::vector<int> input_shape = input_shapes[0];
        std::vector<int> output_shape = input_shape;

        // The input tensor is expected to be 4D: [N, H, W, C]
        if (input_shape.size() < 4) {
            ESP_LOGE("DepthToSpace", "Input tensor must be 4D [N, H, W, C]");
            return std::vector<std::vector<int>>(1, input_shape); // return original shape on error
        }

        output_shape[1] = input_shape[1] * m_blocksize;                 // height
        output_shape[2] = input_shape[2] * m_blocksize;                 // width
        output_shape[3] = input_shape[3] / (m_blocksize * m_blocksize); // channels

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

        // Perform depth-to-space transformation based on the mode
        if (m_mode == "CRD") {
            // CRD mode: Column, Row, Depth
            depth_to_space_crd<T>(input->get_element_ptr<T>(),
                                  output->get_element_ptr<T>(),
                                  input_shape[0],
                                  input_shape[1],
                                  input_shape[2],
                                  input_shape[3],
                                  m_blocksize);
            // output->reshape({output_batch, output_height, output_width, output_channels});
        } else {
            // DCR mode (default): Depth, Column, Row
            // CRD mode: Column, Row, Depth
            depth_to_space_dcr<T>(input->get_element_ptr<T>(),
                                  output->get_element_ptr<T>(),
                                  input_shape[0],
                                  input_shape[1],
                                  input_shape[2],
                                  input_shape[3],
                                  m_blocksize);
        }
    }

    void forward_args(void *args) {}

    /**
     * @brief deserialize DepthToSpace module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        int blocksize;
        std::string mode;

        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "blocksize", blocksize);
        fbs_model->get_operation_attribute(node_name, "mode", mode);

        if (mode.empty()) {
            mode = "DCR"; // default mode
        }

        // Create module
        op = new DepthToSpace(blocksize, mode, node_name.c_str(), MODULE_NON_INPLACE, quant_type);
        return op;
    }

    void print()
    {
        ESP_LOGI("DepthToSpace",
                 "blocksize: %d, mode: %s, quant_type: %s",
                 m_blocksize,
                 m_mode.c_str(),
                 quant_type_to_string(quant_type));
    }
};

} // namespace module
} // namespace dl
