#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"

namespace dl
{
    namespace layer
    {
        template <typename input_t, typename output_t>
        class Model
        {
        private:
            std::vector<int> input_shape;

        public:
            virtual ~Model() {}

            /**
             * @brief 
             * 
             * @param input 
             */
            virtual void build(Feature<input_t> &input) = 0;

            /**
             * @brief 
             * 
             * @param input 
             */
            virtual void call(Feature<input_t> &input) = 0;

            /**
             * @brief 
             * 
             * @param input 
             */
            void forward(Feature<input_t> &input);
        };
    } // namespace layer
} // namespace dl
