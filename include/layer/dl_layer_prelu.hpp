#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_nn_prelu.hpp"
#include "dl_layer_base.hpp"

namespace dl
{
    namespace layer
    {
        /**
         * @brief PReLU(input).
         * 
         * @tparam feature_t supports int16_t and int8_t,
         *         - int16_t: stands for operation in int16_t quantize
         *         - int8_t: stands for operation in int8_t quantize
         */
        template <typename feature_t>
        class PReLU : public Layer
        {
        private:
            feature_t *activation_element; /*<! quantized alpha elements along channel axis >*/
            int activation_exponent;       /*<! exponent of quantized alpha elements >*/
        public:
            Tensor<feature_t> output; /*<! output of prelu>*/

            /**
             * @brief Construct a new PReLU object
             * 
             * @param activation_element   quantized alpha elements along channel axis
             * @param activation_exponent  exponent of quantized alpha elements
             * @param name                 name of prelu
             */
            PReLU(const feature_t *activation_element, const int activation_exponent = 0, const char *name = NULL) : Layer(name)
            {
                this->activation_element = activation_element;
                this->activation_exponent = activation_exponent;
            }

            /**
             * @brief Destroy the PReLU object
             * 
             */
            ~PReLU() {}

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
             * @brief Call PReLU operation.
             * 
             * @param input       as an input
             * @param assign_core not effective yet
             * @return PReLU result
             */
            Tensor<feature_t> &call(Tensor<feature_t> &input, const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE)
            {
                DL_LOG_LAYER_LATENCY_INIT();

                DL_LOG_LAYER_LATENCY_START();
                this->output.apply_element();
                DL_LOG_LAYER_LATENCY_END(this->name, "apply");

                DL_LOG_LAYER_LATENCY_START();
                nn::prelu(this->output, input, this->activation_element, this->activation_exponent, assign_core);
                DL_LOG_LAYER_LATENCY_END(this->name, "leakyrelu");

                return this->output;
            }
        };
    } // namespace layer
} // namespace dl
