#pragma once

#include "dl_base_resize.hpp"
#include "dl_module_base.hpp"
#include <math.h>

namespace dl {
namespace module {
/**
 * NOTE:
 * The input data layout for `resize` is NWC/NHWC, while the parameter data layout is NCW/NCHW.
 */
class Resize : public Module {
private:
    const resize_mode_t m_resize_mode; /*!< one of RESIZE_NEAREST or RESIZE_LINEAR or RESIZE_CUBIC */
    std::vector<float> m_scales;       /*!< The scale array along each dimension. */
    std::vector<int64_t> m_sizes;      /*!< Target size of the output tensor. */
    bool m_align_corners;              /*!< Whether coordinate_transformation_mode is align_corners */
    float *m_cache = nullptr; /*!< Temporary memory, used to cache the intermediate results of linear operations. */
public:
    /**
     * @brief Construct a new Resize object.
     *
     * @param name                  name of module
     * @param resize_mode           one of RESIZE_NEAREST or RESIZE_LINEAR or RESIZE_CUBIC
     * @param scales                The scale array along each dimension
     * @param sizes                 The size array along each dimension
     * @param align_corners         If true, the corner pixels of the input and output tensors are aligned
     * @param quant_type            quant type
     */
    Resize(const char *name = NULL,
           const resize_mode_t resize_mode = RESIZE_NEAREST,
           const std::vector<float> scales = {},
           const std::vector<int64_t> sizes = {},
           const bool align_corners = false,
           quant_type_t quant_type = QUANT_TYPE_NONE) :
        Module(name, MODULE_NON_INPLACE, quant_type),
        m_resize_mode(resize_mode),
        m_scales(scales),
        m_sizes(sizes),
        m_align_corners(align_corners)
    {
    }

    /**
     * @brief Destroy the Resize object.
     */
    ~Resize()
    {
        if (m_cache) {
            free(m_cache);
        }
    }

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes)
    {
        assert(input_shapes[0].size() == 3 || input_shapes[0].size() == 4);
        std::vector<int> input_shape = input_shapes[0];
        std::vector<int> output_shape = input_shape;

        /*input/output shape: NWC/NHWC, scales/sizes shape: NCW/NCHW
        Currently, 2D resizing to height and width is supported.*/
        if (!m_sizes.empty()) {
            std::vector<float> scales_tmp(input_shape.size(), 1.0f);

            output_shape[1] = static_cast<int>(m_sizes[2]);
            scales_tmp[2] = output_shape[1] / (float)input_shape[1];
            if (input_shape.size() == 4) {
                output_shape[2] = static_cast<int>(m_sizes[3]);
                scales_tmp[3] = output_shape[2] / (float)input_shape[2];
            }
            m_scales.swap(scales_tmp);
        } else {
            output_shape[1] = (int)(input_shape[1] * m_scales[2]);
            if (input_shape.size() == 4) {
                output_shape[2] = (int)(input_shape[2] * m_scales[3]);
            }
        }

        return {output_shape};
    }

    void forward(ModelContext *context, runtime_mode_t mode)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            forward_template<int8_t>(context, mode);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            forward_template<int16_t>(context, mode);
        }
    }

    void forward_args(void *args)
    {
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            base::resize<int8_t>(args);
        } else if (quant_type == QUANT_TYPE_SYMM_16BIT) {
            base::resize<int16_t>(args);
        }
    }

    template <typename T>
    void forward_template(ModelContext *context, runtime_mode_t mode)
    {
        TensorBase *input = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        int dims = input->get_shape().size();

        if ((dims == 3 && m_scales[2] == 1) || (dims == 4 && m_scales[2] == 1 && m_scales[3] == 1)) {
            output->assign(input);
        }

        std::vector<base::resizeArgsType<T>> m_args =
            base::get_resize_operation_args<T>(output, input, m_resize_mode, m_scales, m_align_corners, m_cache);
        int task_size = m_args.size();
        if (task_size == 1) { // single task
            forward_args((void *)&m_args[0]);
        } else if (task_size == 2) { // multi task, use semaphore to maintain synchronization.
            module_forward_dual_core(this, (void *)&m_args[0], (void *)&m_args[1]);
        } else {
            ESP_LOGE("Resize", "Only support task size is 1 or 2, currently task size is %d", task_size);
        }
    }

    /**
     * @brief deserialize Resize module instance by node serialization information
     */
    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        Module *op = nullptr;
        quant_type_t quant_type;
        resize_mode_t resize_mode;
        std::string coordinate_transformation_mode;
        fbs_model->get_operation_attribute(node_name, "quant_type", quant_type);
        fbs_model->get_operation_attribute(node_name, "mode", resize_mode);
        fbs_model->get_operation_attribute(node_name, "coordinate_transformation_mode", coordinate_transformation_mode);
        dl::TensorBase *scales_tensor = fbs_model->get_operation_parameter(node_name, 2);
        std::vector<float> scales;
        if (scales_tensor) {
            float *scales_data = scales_tensor->get_element_ptr<float>();
            scales.assign(scales_data, scales_data + scales_tensor->get_size());
        }

        dl::TensorBase *sizes_tensor = fbs_model->get_operation_parameter(node_name, 3);
        std::vector<int64_t> sizes;
        if (sizes_tensor) {
            int64_t *sizes_data = sizes_tensor->get_element_ptr<int64_t>();
            sizes.assign(sizes_data, sizes_data + sizes_tensor->get_size());
        }

        // Create module
        if (quant_type == QUANT_TYPE_SYMM_8BIT || quant_type == QUANT_TYPE_SYMM_16BIT) {
            op = new Resize(node_name.c_str(),
                            resize_mode,
                            scales,
                            sizes,
                            coordinate_transformation_mode == "align_corners",
                            quant_type);
        }
        delete scales_tensor;
        delete sizes_tensor;
        return op;
    }

    void print()
    {
        ESP_LOGI("Resize",
                 "quant_type: %s, "
                 "resize_mode: %d, "
                 "scales: %s, "
                 "sizes: %s, "
                 "align_corners: %d.",
                 quant_type_to_string(quant_type),
                 m_resize_mode,
                 vector_to_string(m_scales).c_str(),
                 vector_to_string(m_sizes).c_str(),
                 m_align_corners);
    }
};
} // namespace module
} // namespace dl
