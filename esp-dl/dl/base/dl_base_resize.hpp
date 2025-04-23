#pragma once

#include "dl_base.hpp"

namespace dl {
namespace base {

/**
 * @brief Get the resize operation args object
 *
 * @tparam feature_t
 * @param output
 * @param input
 * @param resize_mode
 * @param scales
 * @param runtime_mode
 * @return std::vector<resizeArgsType<feature_t>>
 */
template <typename feature_t>
std::vector<resizeArgsType<feature_t>> get_resize_operation_args(TensorBase *output,
                                                                 TensorBase *input,
                                                                 resize_mode_t resize_mode,
                                                                 const std::vector<float> &scales,
                                                                 const bool align_corners,
                                                                 float *&cache,
                                                                 const runtime_mode_t runtime_mode = RUNTIME_MODE_AUTO)
{
    // input/output shape: NWC/NHWC, scales/sizes shape: NCW/NCHW
    resizeArgsType<feature_t> args;
    args.input_element = (feature_t *)input->get_element_ptr();
    args.output_element = (feature_t *)output->get_element_ptr(); // output
    args.input_exponent = input->get_exponent();
    args.output_exponent = output->get_exponent();
    args.align_corners = static_cast<int>(align_corners);
    args.dims = input->shape.size();

    args.resize_mode = resize_mode;
    if (args.dims == 3) {
        args.input_height = 1; // inputs and output are the same shape
        args.input_width = input->shape[1];
        args.input_channel = input->shape[2];
        args.output_height = 1;
        args.output_width = output->shape[1];
        args.scale_h = 1;
        args.scale_w = scales[2];
        if (args.resize_mode != RESIZE_NEAREST && args.align_corners) {
            args.scale_w = (args.output_width - 1) / (float)(args.input_width - 1);
        }
        args.scale_h_inv = 1;
        args.scale_w_inv = 1 / args.scale_w;
    } else if (args.dims == 4) {
        args.input_height = input->shape[1]; // inputs and output are the same shape
        args.input_width = input->shape[2];
        args.input_channel = input->shape[3];
        args.output_height = output->shape[1];
        args.output_width = output->shape[2];
        args.scale_h = scales[2];
        args.scale_w = scales[3];
        if (args.resize_mode != RESIZE_NEAREST && args.align_corners) {
            args.scale_h = (args.output_height - 1) / (float)(args.input_height - 1);
            args.scale_w = (args.output_width - 1) / (float)(args.input_width - 1);
        }
        args.scale_h_inv = 1 / args.scale_h;
        args.scale_w_inv = 1 / args.scale_w;
    }

    if (!cache && args.resize_mode == RESIZE_LINEAR) {
        int len = 0;
        if (args.dims == 3) {
            // in_x dequantize
            len = args.input_channel * 2;
        } else if (args.dims == 4) {
            // in_x coordinates + in_x ratio + x linear
            len = args.output_width + args.output_width * 2 + args.output_width * args.input_channel * 2;
        }
        cache = static_cast<float *>(tool::calloc_aligned(16, len, sizeof(float), MALLOC_CAP_DEFAULT));
    }
    args.cache = cache;

    args.output_shift = output->exponent - input->exponent;
    args.output_scale = 1;
    if (args.output_shift < 0) { // ( * output_scale ) >> output_shift
        args.output_scale = 1 << (-args.output_shift);
        args.output_shift = 0;
    }

    // for ISA
    int u = 16 / sizeof(feature_t);
    args.c_div_x = args.input_channel / u;
    args.c_remainder = (args.input_channel % u) * sizeof(feature_t);
    if (args.resize_mode == RESIZE_NEAREST) {
        if (args.scale_h == 2 && args.scale_w == 2) {
            args.output_x_offset = args.input_channel;
            args.output_y_offset = args.input_channel * args.input_width * 2;
        }
    }

    // slice
    std::vector<resizeArgsType<feature_t>> m_args(1, args);
    if (runtime_mode == RUNTIME_MODE_MULTI_CORE) {
        // TODO:
    }

    return m_args;
}

/**
 * @brief
 *
 * @tparam feature_t
 * @param args_ptr
 */
template <typename feature_t>
void resize(void *args_ptr);
} // namespace base
} // namespace dl
