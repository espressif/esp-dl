#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace mnist_coefficient
{
    const dl::Filter<int16_t> *get_fused_gemm_0_filter();
    const dl::Bias<int16_t> *get_fused_gemm_0_bias();
    const dl::Activation<int16_t> *get_fused_gemm_0_activation();
    const dl::Filter<int16_t> *get_fused_gemm_1_filter();
    const dl::Bias<int16_t> *get_fused_gemm_1_bias();
    const dl::Activation<int16_t> *get_fused_gemm_1_activation();
    const dl::Filter<int16_t> *get_fused_gemm_2_filter();
    const dl::Bias<int16_t> *get_fused_gemm_2_bias();
}
