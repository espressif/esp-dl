#include "dl_base_norm.hpp"
#include "dl_base_isa.hpp"
#include "dl_tool.hpp"

namespace dl {
namespace base {

void rms_norm(int8_t *output, int8_t *input, float *scale, float *rms, int n)
{
#if CONFIG_PIE_V1_BOOST
    dl_tie728_rmsnorm_s8(output, input, scale, rms, n);
#elif CONFIG_PIE_V2_BOOST
    dl_esp32p4_rmsnorm_s8(output, input, scale, rms, n);
#else
    for (int j = 0; j < n; j++) {
        float result = input[j] * (*rms);
        result *= scale[j];
        tool::truncate(output[j], tool::round(result));
    }
#endif
}

void rms_norm(int16_t *output, int16_t *input, float *scale, float *rms, int n)
{
#if CONFIG_PIE_V1_BOOST
    dl_tie728_rmsnorm_s16(output, input, scale, rms, n);
#elif CONFIG_PIE_V2_BOOST
    dl_esp32p4_rmsnorm_s16(output, input, scale, rms, n);
#else
    for (int j = 0; j < n; j++) {
        float result = input[j] * (*rms);
        result *= scale[j];
        tool::truncate(output[j], tool::round(result));
    }
#endif
}

} // namespace base
} // namespace dl
