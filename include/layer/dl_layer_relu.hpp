#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_tool.hpp"
#include "dl_nn_relu.hpp"
#include "dl_layer_base.hpp"

namespace dl
{
    namespace layer
    {
        /**
         * @brief ReLU(input).
         * 
         * @tparam feature_t supports int16_t and int8_t,
         *         - int16_t: stands for operation in int16_t quantize
         *         - int8_t: stands for operation in int8_t quantize
         */
        template <typename feature_t>
        class ReLU : public Layer
        {
        public:
            Tensor<feature_t> output; /*<! output of relu>*/

            /**
             * @brief Construct a new ReLU object
             * 
             * @param name            name of relu 
             */
            ReLU(const char *name = NULL) : Layer(name)
            {
            }

            /**
             * @brief Destroy the ReLU object
             * 
             */
            ~ReLU() {}

            /**
             * @brief Update output shape and exponent
             * 
             * @param input       as an input
             */
            void build(Tensor<feature_t> &input)
            {
                this->output.set_shape(input.shape);
                this->output.set_exponent(input.exponent);
            }

            /**
             * @brief Call ReLU operation.
             * 
             * @param input       as an input
             * @param assign_core not effective yet
             * @return ReLU result
             */
            Tensor<feature_t> &call(Tensor<feature_t> &input, const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE)
            {
                DL_LOG_LAYER_LATENCY_INIT();

                DL_LOG_LAYER_LATENCY_START();
                this->output.apply_element();
                DL_LOG_LAYER_LATENCY_END(this->name, "apply");

                DL_LOG_LAYER_LATENCY_START();
                nn::relu(this->output, input, assign_core);
                DL_LOG_LAYER_LATENCY_END(this->name, "relu");

                return this->output;
            }
        };
    } // namespace layer
} // namespace dl
