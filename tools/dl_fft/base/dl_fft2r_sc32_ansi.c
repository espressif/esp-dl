#include "dl_fft_base.h"

static inline int32_t mulh_s32(int32_t a, int32_t b)
{
    return (int32_t)(((int64_t)a * b) >> 32);
}

// round(t / 2), ties to even
static inline int32_t half_even(int32_t t)
{
    return (t + ((t >> 1) & 1)) >> 1;
}

static int32_t to_q31(double v)
{
    double q = round(v * 2147483648.0);
    if (q > 2147483647.0) {
        q = 2147483647.0;
    } else if (q < -2147483648.0) {
        q = -2147483648.0;
    }
    return (int32_t)q;
}

int32_t *dl_gen_fft_table_sc32(int fft_point, uint32_t caps)
{
    int n = fft_point / 2 + 1;
    int32_t *table = (int32_t *)heap_caps_malloc(sizeof(int32_t) * 2 * n, caps);
    if (table) {
        for (int j = 0; j < n; j++) {
            double angle = M_PI * j / fft_point;
            table[2 * j] = to_q31(cos(angle));
            table[2 * j + 1] = to_q31(sin(angle));
        }
    }
    return table;
}

// Halving rounds half to even and each sum of two truncating mulh() gets a +/-1 bias correction.
// A biased rounding accumulates over the stages into the low-frequency bins.
void dl_fft2r_sc32_dif_ansi(int32_t *data, const int32_t *table, int N)
{
    int32_t *end = data + 2 * N;
    int quarter = N / 2; // table index of angle pi/2
    for (int half = N >> 1, stride = 1; half >= 1; half >>= 1, stride <<= 1) {
        int span = 4 * half; // int32 distance between groups
        int off = 2 * half;  // int32 distance between the two butterfly inputs

        for (int32_t *a = data; a < end; a += span) {
            int32_t *b = a + off;
            int32_t ar = a[0], ai = a[1], br = b[0], bi = b[1];
            a[0] = half_even(ar + br);
            a[1] = half_even(ai + bi);
            b[0] = half_even(ar - br);
            b[1] = half_even(ai - bi);
        }

        for (int k = 1; k < half; k++) {
            if (2 * k == half) {
                // W = -j
                for (int32_t *a = data + 2 * k; a < end; a += span) {
                    int32_t *b = a + off;
                    int32_t ar = a[0], ai = a[1], br = b[0], bi = b[1];
                    a[0] = half_even(ar + br);
                    a[1] = half_even(ai + bi);
                    b[0] = half_even(ai - bi);
                    b[1] = half_even(br - ar);
                }
                continue;
            }
            int j = 2 * k * stride;
            int32_t c, s;
            if (j <= quarter) {
                c = table[2 * j];
                s = table[2 * j + 1];
            } else {
                j -= quarter;
                c = -table[2 * j + 1];
                s = table[2 * j];
            }
            for (int32_t *a = data + 2 * k; a < end; a += span) {
                int32_t *b = a + off;
                int32_t ar = a[0], ai = a[1], br = b[0], bi = b[1];
                int32_t dr = ar - br, di = ai - bi;
                a[0] = half_even(ar + br);
                a[1] = half_even(ai + bi);
                b[0] = mulh_s32(dr, c) + mulh_s32(di, s) + 1;
                b[1] = mulh_s32(di, c) - mulh_s32(dr, s);
            }
        }
    }
}

void dl_bitrev2r_sc32_ansi(int32_t *data, int N)
{
    uint64_t *d = (uint64_t *)data;
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            uint64_t t = d[i];
            d[i] = d[j];
            d[j] = t;
        }
    }
}

// Split the packed complex spectrum Z of N points into the real spectrum X:
//   X[k]   = (Z[k] + conj(Z[N-k])) / 2 - j * W^k * (Z[k] - conj(Z[N-k])) / 2
//   X[N-k] = conj((Z[k] + conj(Z[N-k])) / 2 + j * W^k * (Z[k] - conj(Z[N-k])) / 2)
void dl_rfft_post_proc_sc32_ansi(int32_t *data, int N, const int32_t *table)
{
    int32_t z0r = data[0], z0i = data[1];
    data[0] = z0r + z0i;
    data[1] = z0r - z0i;
    for (int k = 1; k < N / 2; k++) {
        int32_t *zk = data + 2 * k;
        int32_t *zm = data + 2 * (N - k);
        int32_t c = table[2 * k], s = table[2 * k + 1];
        int32_t er = half_even(zk[0] + zm[0]);
        int32_t ei = half_even(zk[1] - zm[1]);
        int32_t orr = zk[0] - zm[0];
        int32_t oi = zk[1] + zm[1];
        int32_t tr = mulh_s32(oi, c) - mulh_s32(orr, s);
        int32_t ti = -mulh_s32(orr, c) - mulh_s32(oi, s) - 1;
        zk[0] = er + tr;
        zk[1] = ei + ti;
        zm[0] = er - tr;
        zm[1] = ti - ei;
    }
    // X[N/2] = conj(Z[N/2])
    data[N + 1] = -data[N + 1];
}
