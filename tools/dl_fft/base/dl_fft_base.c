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

float *dl_short_to_float(const int16_t *x, int len, int exponent, float *y)
{
    float scale = powf(2, exponent);
    // printf("scale: %f\n", scale);
    for (int i = 0; i < len; i++) {
        y[i] = scale * x[i];
    }
    return y;
}

int16_t dl_reduce_abs_max_ansi(const int16_t *x, int size)
{
    int16_t max_val = 0;
    for (int i = 0; i < size; i++) {
        int32_t abs_val = x[i] < 0 ? -(int32_t)x[i] : (int32_t)x[i];
        if (abs_val > INT16_MAX)
            abs_val = INT16_MAX;
        if ((int16_t)abs_val > max_val)
            max_val = (int16_t)abs_val;
    }
    return max_val;
}

int16_t dl_array_max_q_s16(const int16_t *x, int size)
{
    int16_t max_abs = dl_reduce_abs_max(x, size);

    if (max_abs == 0) {
        return 1;
    }

    int16_t k = 2;
    while (max_abs > 1) {
        k++;
        max_abs = max_abs >> 1;
    }

    return k;
}

int dl_array_max_q_f32(const float *x, int size, float eps)
{
    float max = 0;
    for (int i = 1; i < size; i++) {
        if (x[i] > max) {
            max = x[i];
        } else if (-x[i] > max) {
            max = -x[i];
        }
    }
    int max_int = ceilf(max + eps);

    return dl_power_of_two(max_int);
}

int dl_float_to_short(const float *x, int len, int16_t *y, int out_exponent)
{
    int exponent = out_exponent - dl_array_max_q_f32(x, len, 1e-8);
    float scale = powf(2, exponent);

    for (int i = 0; i < len; i++) {
        y[i] = (int16_t)roundf(x[i] * scale);
    }

    return -exponent;
}

esp_err_t dl_bitrev2r_sc16_ansi(int16_t *data, int N, int log2N)
{
    esp_err_t result = ESP_OK;

    int j, k;
    uint32_t temp;
    uint32_t *in_data = (uint32_t *)data;
    j = 0;
    for (int i = 1; i < (N - 1); i++) {
        k = N >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
        if (i < j) {
            temp = in_data[j];
            in_data[j] = in_data[i];
            in_data[i] = temp;
        }
    }
    return result;
}

esp_err_t dl_bitrev2r_sc16(int16_t *data, int N, int log2N)
{
    if (N <= 1024 && N >= 8) {
        dl_bitrev2r_sc16_asm(data, N, log2N);
    } else {
        return dl_bitrev2r_sc16_ansi(data, N, log2N);
    }

    return ESP_OK;
}

esp_err_t dl_rfft_post_proc_sc16_ansi(int16_t *data, int cpx_points, int16_t *win)
{
    int32_t dc_val = data[0] + data[1];
    int32_t nq_val = data[0] - data[1];
    data[0] = dc_val;
    data[1] = nq_val;

    int16_t *in_f = data + 2;
    int16_t *in_b = data + cpx_points * 2 - 2;
    int16_t *tw = win;

    int16_t *out_f = data + 2;
    int16_t *out_b = data + cpx_points * 2 - 2;

    int loop_points = cpx_points / 2;

    for (int k = 1; k <= loop_points; k++) {
        int16_t fr = in_f[0], fi = in_f[1];
        int16_t br = in_b[0], bi = in_b[1];
        int16_t wr = tw[0], wi = tw[1];

        in_f += 2;
        in_b -= 2;
        tw += 2;

        int32_t sum_r = fr + br;
        int32_t sum_i = fi - bi;
        int32_t diff_r = fr - br;
        int32_t diff_i = fi + bi;
        int32_t tw_r = (diff_r * wr - diff_i * wi) >> 14;
        int32_t tw_i = (diff_i * wr + diff_r * wi) >> 14;

        out_f[0] = (int16_t)((sum_r + tw_r) >> 1);
        out_f[1] = (int16_t)((sum_i + tw_i) >> 1);
        out_b[0] = (int16_t)((sum_r - tw_r) >> 1);
        out_b[1] = (int16_t)((tw_i - sum_i) >> 1);

        out_f += 2;
        out_b -= 2;
    }

    return ESP_OK;
}

int dl_rfft_pre_proc_sc16(int16_t *data, int cpx_points, int16_t *table)
{
    int shift = 0;
#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32P4
    int max_q = dl_array_max_q_s16(data, cpx_points * 2);
    if (max_q >= 15) {
        for (int i = 0; i < cpx_points * 2; i++) {
            data[i] = data[i] >> 1;
        }
        shift = 1;
    }
#endif
    dl_rfft_pre_proc_sc16_asm(data, cpx_points, table);
    return shift;
}

/*
 * Wrapper that protects dl_rfft_post_proc_sc16_asm from int16 overflow.
 *
 * The underlying _asm/_ansi implementations compute (sum_r + tw_r) >> 1 in
 * 16-bit precision (the SIMD AMS hardware truncates the sum to 16 bits before
 * the built-in `>> 1` store, dropping bit 16). When the input data uses the
 * full int16 range (e.g. high-precision FFT output), |sum_r + tw_r| can exceed
 * 32767 and the truncation produces a result that is 32768 off from the
 * mathematically correct value.
 *
 * To avoid this, we pre-shift the data right by 1 whenever the abs-max
 * indicates the data is in int16 range, so all intermediate (sum_r + tw_r)
 * values stay within int16. The caller must compensate by adding the returned
 * shift to the output exponent.
 */
int dl_rfft_post_proc_sc16(int16_t *data, int cpx_points, int16_t *table)
{
    int shift = 0;
#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32P4
    int max_q = dl_array_max_q_s16(data, cpx_points * 2);
    if (max_q >= 15) {
        for (int i = 0; i < cpx_points * 2; i++) {
            data[i] = data[i] >> 1;
        }
        shift = 1;
    }
#endif
    dl_rfft_post_proc_sc16_asm(data, cpx_points, table);
    return shift;
}

esp_err_t dl_rfft_pre_proc_sc16_ansi(int16_t *data, int cpx_points, int16_t *table)
{
    int32_t dc_val = data[0];
    int32_t nq_val = data[1];
    int32_t a0 = dc_val + nq_val;
    int32_t a1 = dc_val - nq_val;
    data[0] = (int16_t)a0;
    data[1] = (int16_t)a1;

    int16_t *in_f = data + 2;
    int16_t *in_b = data + cpx_points * 2 - 2;
    int16_t *tw = table;
    int16_t *out_f = data + 2;
    int16_t *out_b = data + cpx_points * 2 - 2;
    int loop_points = cpx_points / 2;

    for (int k = 1; k <= loop_points; k++) {
        int16_t pr = in_f[0], pi = in_f[1];
        int16_t qr = in_b[0], qi = in_b[1];
        int16_t wr = tw[0], wi = tw[1];

        in_f += 2;
        in_b -= 2;
        tw += 2;
        int32_t sum_r = pr + qr;
        int32_t sum_i = pi - qi;
        int32_t diff_r = pr - qr;
        int32_t diff_i = pi + qi;
        int32_t tw_r = (diff_r * wr + diff_i * wi) >> 14;
        int32_t tw_i = (diff_i * wr - diff_r * wi) >> 14;

        out_f[0] = (int16_t)(sum_r + tw_r);
        out_f[1] = (int16_t)(sum_i + tw_i);
        out_b[0] = (int16_t)(sum_r - tw_r);
        out_b[1] = (int16_t)(tw_i - sum_i);

        out_f += 2;
        out_b -= 2;
    }

    return ESP_OK;
}
