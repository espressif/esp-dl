#pragma once
#include <stdlib.h>

namespace dl
{
    namespace layer
    {
        class Layer
        {
        public:
            char *name;
            Layer(const char *name = NULL);
            ~Layer();
        };
    } // namespace layer
} // namespace dl
