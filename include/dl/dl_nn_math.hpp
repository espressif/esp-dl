#pragma once

#include <assert.h>
#include <vector>
#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_tool.hpp"

namespace dl
{
    namespace nn
    {
        template <typename T>
        void add(Feature<T> &input1, Feature<T> &input2);

        template <typename T>
        void global_avg_pool(Feature<T> &output, Feature<T> &input);
    } // namespace nn
} // namespace dl