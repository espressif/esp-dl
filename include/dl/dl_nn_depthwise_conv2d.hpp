#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_tool.hpp"

namespace dl
{
    namespace nn
    {
        template <typename input_t, typename output_t>
        void depthwise_conv2d(Feature<output_t> &output, Feature<input_t> &input, std::vector<int> &padding, const Filter<output_t> &filter, const int stride_y, const int stride_x);

        template <typename input_t, typename output_t>
        void depthwise_conv2d(Feature<output_t> &output, Feature<input_t> &input, std::vector<int> &padding, const Filter<output_t> &filter, const int stride_y, const int stride_x, const Bias<output_t> &bias);

        template <typename input_t, typename output_t>
        void depthwise_conv2d(Feature<output_t> &output, Feature<input_t> &input, std::vector<int> &padding, const Filter<output_t> &filter, const int stride_y, const int stride_x, const ReLU<output_t> &relu);

        template <typename input_t, typename output_t>
        void depthwise_conv2d(Feature<output_t> &output, Feature<input_t> &input, std::vector<int> &padding, const Filter<output_t> &filter, const int stride_y, const int stride_x, const Bias<output_t> &bias, const ReLU<output_t> &relu);

        template <typename input_t, typename output_t>
        Feature<output_t> depthwise_conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const int stride_y, const int stride_x, const padding_type_t pad_type, const int output_exponent)
        {
            std::vector<int> output_shape = dl::tool2d::get_output_shape(input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type, true);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();

            Feature<input_t> *padded_input = NULL;

            if (pad_type == PADDING_VALID)
            {
                padded_input = &input;
            }
            else
            {
                std::vector<int> padding = dl::tool2d::get_pad_size(output_shape, input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type);
                padded_input = new Feature<input_t>(input, padding);
            }

            depthwise_conv2d(output, *padded_input, padded_input->padding, filter, stride_y, stride_x);

            if (pad_type != PADDING_VALID)
            {
                delete padded_input;
            }
            return output;
        }

        template <typename input_t, typename output_t>
        Feature<output_t> depthwise_conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const int stride_y, const int stride_x, const padding_type_t pad_type, const Bias<output_t> &bias, const int output_exponent)
        {
            std::vector<int> output_shape = dl::tool2d::get_output_shape(input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type, true);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();

            Feature<input_t> *padded_input = NULL;

            if (pad_type == PADDING_VALID)
            {
                padded_input = &input;
            }
            else
            {
                std::vector<int> padding = dl::tool2d::get_pad_size(output_shape, input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type);
                padded_input = new Feature<input_t>(input, padding);
            }

            depthwise_conv2d(output, *padded_input, padded_input->padding, filter, stride_y, stride_x, bias);

            if (pad_type != PADDING_VALID)
            {
                delete padded_input;
            }
            return output;
        }

        template <typename input_t, typename output_t>
        Feature<output_t> depthwise_conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const int stride_y, const int stride_x, const padding_type_t pad_type, const ReLU<output_t> &relu, const int output_exponent)
        {
            std::vector<int> output_shape = dl::tool2d::get_output_shape(input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type, true);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();

            Feature<input_t> *padded_input = NULL;

            if (pad_type == PADDING_VALID)
            {
                padded_input = &input;
            }
            else
            {
                std::vector<int> padding = dl::tool2d::get_pad_size(output_shape, input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type);
                padded_input = new Feature<input_t>(input, padding);
            }

            depthwise_conv2d(output, *padded_input, padded_input->padding, filter, stride_y, stride_x, relu);

            if (pad_type != PADDING_VALID)
            {
                delete padded_input;
            }
            return output;
        }

        template <typename input_t, typename output_t>
        Feature<output_t> depthwise_conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const int stride_y, const int stride_x, const padding_type_t pad_type, const Bias<output_t> &bias, const ReLU<output_t> &relu, const int output_exponent)
        {
            std::vector<int> output_shape = dl::tool2d::get_output_shape(input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type, true);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();

            Feature<input_t> *padded_input = NULL;

            if (pad_type == PADDING_VALID)
            {
                padded_input = &input;
            }
            else
            {
                std::vector<int> padding = dl::tool2d::get_pad_size(output_shape, input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type);
                padded_input = new Feature<input_t>(input, padding);
            }

            depthwise_conv2d(output, *padded_input, padded_input->padding, filter, stride_y, stride_x, bias, relu);

            if (pad_type != PADDING_VALID)
            {
                delete padded_input;
            }
            return output;
        }

        template <typename input_t, typename output_t>
        void global_depthwise_conv2d(Feature<output_t> &output, Feature<input_t> &input, const Filter<input_t> &filter);

        template <typename input_t, typename output_t>
        void global_depthwise_conv2d(Feature<output_t> &output, Feature<input_t> &input, const Filter<input_t> &filter, const Bias<input_t> &bias);

        template <typename input_t, typename output_t>
        Feature<output_t> global_depthwise_conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const Bias<output_t> &bias, const int output_exponent)
        {
            std::vector<int> output_shape = dl::tool2d::get_output_shape(input.shape, filter.shape_with_dilation, 1, 1, PADDING_VALID, true);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();

            global_depthwise_conv2d(output, input, filter, bias);

            return output;
        }

        template <typename input_t, typename output_t>
        Feature<output_t> global_depthwise_conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const int output_exponent)
        {
            std::vector<int> output_shape = dl::tool2d::get_output_shape(input.shape, filter.shape_with_dilation, 1, 1, PADDING_VALID, true);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();

            global_depthwise_conv2d(output, input, filter);

            return output;
        }
    } // namespace nn
} // namespace dl