#include "dl_tool.hpp"
#include <string.h>

extern "C" {
#if CONFIG_XTENSA_BOOST
void dl_xtensa_bzero_32b(void *ptr, const int n);
#endif

#if CONFIG_TIE728_BOOST
void dl_tie728_bzero_128b(void *ptr, const int n);
void dl_tie728_bzero(void *ptr, const int n);
void dl_tie728_memcpy(void *dst, const void *src, const int n);
#endif
int round_half_even(float value)
{
    float rounded;
    if (value < 0) {
        rounded = value - 0.5f;
    } else {
        rounded = value + 0.5f;
    }

    int int_part = (int)rounded;
    if (rounded == (float)int_part) {
        if (int_part % 2 != 0) {
            if (value < 0)
                int_part++;
            else
                int_part--;
        }
    }
    return int_part;
}
}

namespace dl {
namespace tool {
void set_zero(void *ptr, const int n)
{
#if CONFIG_TIE728_BOOST
    dl_tie728_bzero(ptr, n);
#else
    bzero(ptr, n);
#endif
}

void copy_memory(void *dst, void *src, const int n)
{
#if CONFIG_TIE728_BOOST
    dl_tie728_memcpy(dst, src, n);
#else
    memcpy(dst, src, n);
#endif
}
} // namespace tool
} // namespace dl
