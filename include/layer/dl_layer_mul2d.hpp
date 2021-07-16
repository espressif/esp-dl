#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_nn_mul2d.hpp"
#include "dl_layer_base.hpp"

namespace dl
{
    namespace layer
    {
        /**
         * @brief Activation(Multiply2D(input0, input1)).
         * NOTE: multiplication is element-wise, i.e., output[i,j,k] = input0[i,j,k] * input1[i,j,k]
         * 
         * @tparam feature_t supports int16_t and int8_t,
         *         - int16_t: stands for operation in int16_t quantize
         *         - int8_t: stands for operation in int8_t quantize
         */
        template <typename feature_t>
        class Mul2D : public Layer
        {
        private:
            const Activation<feature_t> *activation; /*<! activation of Mul2D, if you don't specify anything, no activation is applied >*/

        public:
            Tensor<feature_t> output; /*<! output of Mul2D >*/

            /**
             * @brief Construct a new Mul2D object.
             * 
             * @param output_exponent exponent of output
             * @param activation      activation of Mul2D, if you don't specify anything, no activation is applied
             * @param name            name of layer
             */
            Mul2D(const int output_exponent, const Activation<feature_t> *activation = NULL, const char *name = NULL) : Layer(name), activation(activation)
            {
                this->output.set_exponent(output_exponent);
            }

            /**
             * @brief Destroy the Multiply2D object.
             */
            ~Mul2D() {}

            /**
             * @brief Update output shape.
             * NOTE: input0.shape must equal to input1.shape.
             * 
             * @param input0 as one input
             * @param input1 as another input
             */
            void build(Tensor<feature_t> &input0, Tensor<feature_t> &input1)
            {
                assert(input0.is_same_shape(input1));
                this->output.set_shape(input0.shape);
            }

            /**
             * @brief Call Mul2D operation.
             * 
             * @param input0      as one input
             * @param input1      as another input
             * @param assign_core not effective yet
             * @return Mul2D result
             */
            Tensor<feature_t> &call(Tensor<feature_t> &input0, Tensor<feature_t> &input1, const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE)
            {
                DL_LOG_LAYER_LATENCY_INIT();
             
                DL_LOG_LAYER_LATENCY_START();
                this->output.apply_element();
                DL_LOG_LAYER_LATENCY_END(this->name, "apply");

                DL_LOG_LAYER_LATENCY_START();
                nn::mul2d(this->output, input0, input1, this->activation, assign_core);
                DL_LOG_LAYER_LATENCY_END(this->name, "mul2d");

                return this->output;
            }
        };
    } // namespace layer
} // namespace dl
