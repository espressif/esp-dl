#pragma once

#include <assert.h>
#include <vector>

#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_tool.hpp"
#include "dl_layer_base.hpp"

namespace dl
{
    namespace layer
    {
        template <typename T>
        class Concat2D : Layer
        {
        private:
            std::vector<Feature<T> *> output_vec;
            std::vector<int> offset;
            std::vector<int> channel;

        public:
            Feature<T> output;

            Concat2D(const char *name = NULL) : Layer(name) {}
            ~Concat2D() {}

            void build(std::vector<Feature<T> *> args)
            {
                assert(args.size() > 0);

                this->output_vec = args;

                this->offset = std::vector<int>(args.size());
                this->channel = std::vector<int>(args.size());

                int output_exponent = args[0]->exponent;
                this->offset[0] = 0;
                this->channel[0] = args[0]->shape[2];
                std::vector<int> output_shape = args[0]->shape;

                for (int i = 1; i < args.size(); i++)
                {
                    assert(output_shape[0] == args[i]->shape[0]); // height
                    assert(output_shape[1] == args[i]->shape[1]); // width
                    assert(output_exponent == args[i]->exponent); // exponent

                    this->offset[i] = output_shape[2];
                    this->channel[i] = args[i]->shape[2];
                    output_shape[2] += args[i]->shape[2];
                }
                this->output.set_shape(output_shape);
                this->output.set_exponent(output_exponent);
            }

            void backward()
            {
                std::vector<int> max_padding = this->output.padding;
                int max_channel_with_padding = this->output.shape_with_padding[2];
                for (int i = 0; i < this->output_vec.size(); i++)
                {
                    for (int j = 0; j < max_padding.size(); j++)
                    {
                        max_padding[j] = DL_MAX(max_padding[j], this->output_vec[i]->padding[j]);
                    }
                    max_channel_with_padding = DL_MAX(max_channel_with_padding, this->output_vec[i]->shape_with_padding[2]);
                }

                this->output.set_padding(max_padding);
                this->output.shape_with_padding[2] = max_channel_with_padding;
                for (int i = 0; i < this->output_vec.size(); i++)
                {
                    this->output_vec[i]->set_padding(max_padding);
                    this->output_vec[i]->shape_with_padding[2] = max_channel_with_padding;
                    // this->output_vec[i]->print_shape();
#if CONFIG_DEBUG_MODE
                    assert(this->output.shape_with_padding[0] == this->output_vec[i]->shape_with_padding[0]);
                    assert(this->output.shape_with_padding[1] == this->output_vec[i]->shape_with_padding[1]);
                    assert(this->output.shape_with_padding[2] == this->output_vec[i]->shape_with_padding[2]);
#endif
                }
            }

            void calloc_element()
            {
#if CONFIG_DEBUG_MODE
                printf("%s:\n", this->name);
                dl::tool::Latency latency;
                latency.start();
#endif
                this->output.calloc_element();

                for (int i = 0; i < this->offset.size(); i++)
                {
                    this->output_vec[i]->element = this->output.element + this->offset[i];
                    this->output_vec[i]->set_auto_free(false);
                }
#if CONFIG_DEBUG_MODE
                latency.end();
                latency.print("\toperation");
#endif
            }
        };

    } // namespace layer
} // namespace dl