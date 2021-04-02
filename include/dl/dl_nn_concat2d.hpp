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
        void concat2d(Feature<T> &output, std::vector<Feature<T>> features);
    } // namespace nn
} // namespace dl