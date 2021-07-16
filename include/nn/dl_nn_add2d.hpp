#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_nn.hpp"

namespace dl
{
    namespace nn
    {
        /**
         * @brief activation(add2d(input0, input1)).
         * 
         * @param output      as an output
         * @param input0      as one input
         * @param input1      as another input
         * @param activation  activation of add2d, if you don't specify anything, no activation is applied
         * @param assign_core not effective yet
         */
        void add2d(Tensor<int16_t> &output, Tensor<int16_t> &input0, Tensor<int16_t> &input1, const Activation<int16_t> *const activation = NULL, const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE);

        /**
         * @brief activation(add2d(input0, input1)).
         * 
         * @param output      as an output
         * @param input0      as one input
         * @param input1      as another input
         * @param activation  activation of add2d, if you don't specify anything, no activation is applied
         * @param assign_core not effective yet
         */
        void add2d(Tensor<int8_t> &output, Tensor<int8_t> &input0, Tensor<int8_t> &input1, const Activation<int8_t> *const activation = NULL, const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE);

        /**
         * @brief activation(add2d(input0, input1))
         * 
         * @tparam feature_t supports int16_t and int8_t,
         *         - int16_t: stands for operation in int16_t quantize
         *         - int8_t: stands for operation in int8_t quantize
         * @param output_exponent exponent of output
         * @param input0          as one input
         * @param input1          as another input
         * @param activation      activation of add2d, if you don't specify anything, no activation is applied
         * @param assign_core     not effective yet
         * @return add2d result
         */
        template <typename feature_t>
        Tensor<feature_t> add2d(const int output_exponent, Tensor<feature_t> &input0, Tensor<feature_t> &input1, const Activation<feature_t> *activation, const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE)
        {
            assert(input0.is_same_shape(input1));

            DL_LOG_NN_LATENCY_INIT();

            DL_LOG_NN_LATENCY_START();
            Tensor<feature_t> output;
            output.set_exponent(output_exponent).set_shape(input0.shape).apply_element();
            DL_LOG_NN_LATENCY_END("apply");

            DL_LOG_NN_LATENCY_START();
            add2d(output, input0, input1, activation, assign_core);
            DL_LOG_NN_LATENCY_END("add2d");

            return output;
        }
    } // namespace nn
} // namespace dl