#pragma once

#include <vector>
#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_nn_global_avg_pool2d.hpp"

namespace dl
{
    namespace layer
    {
        /**
         * @brief GlobalAveragePool2D(input).
         * 
         * @tparam feature_t supports int16_t and int8_t,
         *         - int16_t: stands for operation in int16_t quantize
         *         - int8_t: stands for operation in int8_t quantize
         */
        template <typename feature_t>
        class GlobalAveragePool2D : public Layer
        {
        private:
            const int output_exponent; /*<! exponent of output >*/
        public:
            Tensor<feature_t> output; /*<! output of GlobalAveragePool2D >*/

            /**
             * @brief Construct a new GlobalAveragePool2D object.
             * 
             * @param output_exponent exponent of output
             * @param name            name of layer
             */
            GlobalAveragePool2D(const int output_exponent, const char *name = NULL) : Layer(name),
            {
                this->output.set_exponent(output_exponent);
            }

            /**
            * @brief Destroy the GlobalAveragePool2D object.
            * 
            */
            ~GlobalAveragePool2D() {}

            /**
             * @brief Update output shape.
             * 
             * @param input as an input
             */
            void build(Tensor<feature_t> &input)
            {
                assert(input.shape[0] > 0);
                assert(input.shape[1] > 0);

                vector<int> output_shape(input.shape.size(), 1);
                output_shape[2] = input.shape[2];
                this->output.set_shape(output_shape);
            }

            /**
             * @brief Call GlobalAveragePool2D operation
             * 
             * @param input           as an input
             * @param autoload_enable one of true or false, 
             *                        - true: load input and output from PSRAM to CACHE automatically
             *                        - false: do not
             * @param assign_core     not effective yet
             * @return GlobalAveragePool2D result
             */
            Tensor<feature_t> &call(Tensor<feature_t> &input, uint8_t autoload_enable = 0)
            {
                DL_LOG_LAYER_LATENCY_INIT();

                DL_LOG_LAYER_LATENCY_START();
                this->output.apply_element();
                DL_LOG_LAYER_LATENCY_END(this->name, "apply");

                if (autoload_enable)
                {
                    dl::tool::cache::autoload_func((uint32_t)(this->output.element), this->output.get_size() * sizeof(feature_t),
                                                   (uint32_t)(input.element), input.get_size() * sizeof(feature_t));
                }

                DL_LOG_LAYER_LATENCY_START();
                nn::global_avg_pool2d(output, input);
                DL_LOG_LAYER_LATENCY_END(this->name, "global_avg_pool2d");

                return this->output;
            }
        };
    } // namespace layer
} // namespace dl
