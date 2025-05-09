#include "dl_fft_base.h"

bool dl_is_power_of_two(int x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}

int dl_power_of_two(uint32_t n)
{
    int pos = 0;
    if (n >= 1 << 16) {
        n >>= 16;
        pos += 16;
    }
    if (n >= 1 << 8) {
        n >>= 8;
        pos += 8;
    }
    if (n >= 1 << 4) {
        n >>= 4;
        pos += 4;
    }
    if (n >= 1 << 2) {
        n >>= 2;
        pos += 2;
    }
    if (n >= 1 << 1) {
        pos += 1;
    }
    return pos;
}

float *dl_short_to_float(int16_t *x, int len, int exponent, float *y)
{
    float scale = 1.0;
    if (exponent > 0) {
        scale = (1 << exponent);
    } else if (exponent < 0) {
        scale = 1.0 / (1 << (-exponent));
    }
    for (int i = 0; i < len; i++) {
        y[i] = scale * x[i];
    }
    return y;
}
