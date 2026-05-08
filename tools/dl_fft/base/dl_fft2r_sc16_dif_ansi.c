#include "dl_fft_base.h"

int16_t *dl_gen_dif_fft_table(int N, uint32_t caps)
{
    int16_t *table = (int16_t *)heap_caps_aligned_alloc(16, N * 2 * sizeof(int16_t), caps);
    if (!table)
        return NULL;

    float base_angle = (float)(M_PI * 2.0) / N;
    int offset = 0;
    int log2N = dl_power_of_two(N);
    int scale = 1 << DL_FFT_DIF_SC16_TABLE_BITS; // scale twiddle factors to fit in int16 with some headroom for
                                                 // intermediate growth

    for (int stage = 0; stage < log2N; stage++) {
        int stride = 1 << stage;
        int num_tw = N >> (stage + 1);
        for (int k = 0; k < num_tw; k++) {
            float angle = k * stride * base_angle;
            table[offset * 2] = (int16_t)roundf(scale * cosf(angle));
            table[offset * 2 + 1] = (int16_t)roundf(scale * sinf(angle));
            offset++;
        }
    }

    return table;
}

int16_t *dl_gen_dif_rfft_table(int N, uint32_t caps)
{
    int16_t *fft_table = (int16_t *)heap_caps_aligned_alloc(16, N * sizeof(int16_t), caps);
    int scale = 1 << DL_FFT_DIF_SC16_TABLE_BITS;

    if (fft_table) {
        for (int i = 0; i < N / 2; i++) {
            double phase = -M_PI * ((i + 1.0) / N + 0.5);
            fft_table[i * 2 + 0] = (int16_t)round(cos(phase) * scale);
            fft_table[i * 2 + 1] = (int16_t)round(sin(phase) * scale);
        }
    }

    return fft_table;
}

void dl_fft2r_sc16_dif_ansi(int16_t *data, int16_t *table, int shift, int num_stages, int N)
{
    int half_size = N >> 1;
    int num_groups = 1;
    int tw_offset = 0;
    int rnd = 1 << (shift - 1);

    for (int stage = 0; stage < num_stages; stage++) {
        for (int g = 0; g < num_groups; g++) {
            int base = g * (half_size << 1);
            for (int k = 0; k < half_size; k++) {
                int ui = base + k;
                int li = ui + half_size;

                int16_t a_re = data[ui * 2];
                int16_t a_im = data[ui * 2 + 1];
                int16_t b_re = data[li * 2];
                int16_t b_im = data[li * 2 + 1];

                int32_t sum_re = (((int32_t)a_re + b_re) << 14) + rnd;
                int32_t sum_im = (((int32_t)a_im + b_im) << 14) + rnd;
                data[ui * 2] = (int16_t)(sum_re >> shift);
                data[ui * 2 + 1] = (int16_t)(sum_im >> shift);

                // int32_t sum_re = (((int32_t)a_re + b_re));
                // int32_t sum_im = (((int32_t)a_im + b_im));
                // data[ui * 2]     = (int16_t)(sum_re >> 1);
                // data[ui * 2 + 1] = (int16_t)(sum_im >> 1);

                int32_t d_re = (int32_t)a_re - b_re;
                int32_t d_im = (int32_t)a_im - b_im;
                int16_t w_re = table[(tw_offset + k) * 2];
                int16_t w_im = table[(tw_offset + k) * 2 + 1];
                int32_t lo_re = d_re * w_re + d_im * w_im + rnd;
                int32_t lo_im = d_im * w_re - d_re * w_im + rnd;
                data[li * 2] = (int16_t)(lo_re >> shift);
                data[li * 2 + 1] = (int16_t)(lo_im >> shift);
            }
        }
        tw_offset += half_size;
        half_size >>= 1;
        num_groups <<= 1;
    }
}

void dl_fft2r_sc16_dif(int16_t *data, int16_t *table, int shift, int num_stages, int N)
{
    if (num_stages >= 3 && num_stages <= 10) {
        dl_fft2r_sc16_dif_asm(data, table, shift, num_stages, N);
    } else {
        dl_fft2r_sc16_dif_ansi(data, table, shift, num_stages, N);
    }
}

void dl_ifft2r_sc16_dif_ansi(int16_t *data, int16_t *table, int shift, int num_stages, int N)
{
    int half_size = N >> 1;
    int num_groups = 1;
    int tw_offset = 0;

    for (int stage = 0; stage < num_stages; stage++) {
        for (int g = 0; g < num_groups; g++) {
            int base = g * (half_size << 1);
            for (int k = 0; k < half_size; k++) {
                int ui = base + k;
                int li = ui + half_size;

                int16_t a_re = data[ui * 2];
                int16_t a_im = data[ui * 2 + 1];
                int16_t b_re = data[li * 2];
                int16_t b_im = data[li * 2 + 1];

                int32_t sum_re = (int32_t)a_re + b_re;
                int32_t sum_im = (int32_t)a_im + b_im;
                data[ui * 2] = (int16_t)(sum_re >> 1);
                data[ui * 2 + 1] = (int16_t)(sum_im >> 1);

                int32_t d_re = (int32_t)a_re - b_re;
                int32_t d_im = (int32_t)a_im - b_im;
                int16_t w_re = table[(tw_offset + k) * 2];
                int16_t w_im = table[(tw_offset + k) * 2 + 1];
                // For IFFT, multiply by W_N^{-k} = cos + i*sin
                // (d_re + i*d_im) * (w_re + i*w_im) = (d_re*w_re - d_im*w_im) + i*(d_im*w_re + d_re*w_im)
                int32_t lo_re = d_re * w_re - d_im * w_im;
                int32_t lo_im = d_im * w_re + d_re * w_im;
                data[li * 2] = (int16_t)(lo_re >> shift);
                data[li * 2 + 1] = (int16_t)(lo_im >> shift);
            }
        }
        tw_offset += half_size;
        half_size >>= 1;
        num_groups <<= 1;
    }
}

void dl_fft2r_sc16_dif_hp_ansi(int16_t *data, int16_t *table, int num_stages, int N, int *out_shift)
{
    int half_size = N >> 1;
    int num_groups = 1;
    int tw_offset = 0;
    int loop_shift = 0;
    int add_rount_mult = 0;
    out_shift[0] = 0;

    for (int stage = 0; stage < num_stages; stage++) {
        loop_shift = dl_array_max_q_s16(data, N * 2);
        add_rount_mult = 1 << (loop_shift - 1);
        out_shift[0] += loop_shift - DL_FFT_DIF_SC16_TABLE_BITS;

        for (int g = 0; g < num_groups; g++) {
            int base = g * (half_size << 1);
            for (int k = 0; k < half_size; k++) {
                int ui = base + k;
                int li = ui + half_size;

                int32_t a_re = data[ui * 2];
                int32_t a_im = data[ui * 2 + 1];
                int32_t b_re = data[li * 2];
                int32_t b_im = data[li * 2 + 1];

                int32_t sum_re = ((a_re + b_re) << DL_FFT_DIF_SC16_TABLE_BITS) + add_rount_mult;
                int32_t sum_im = ((a_im + b_im) << DL_FFT_DIF_SC16_TABLE_BITS) + add_rount_mult;
                data[ui * 2] = (int16_t)(sum_re >> loop_shift);
                data[ui * 2 + 1] = (int16_t)(sum_im >> loop_shift);

                int32_t d_re = a_re - b_re;
                int32_t d_im = a_im - b_im;
                int32_t w_re = table[(tw_offset + k) * 2];
                int32_t w_im = table[(tw_offset + k) * 2 + 1];
                int32_t lo_re = d_re * w_re + d_im * w_im + add_rount_mult;
                int32_t lo_im = d_im * w_re - d_re * w_im + add_rount_mult;
                data[li * 2] = (int16_t)(lo_re >> loop_shift);
                data[li * 2 + 1] = (int16_t)(lo_im >> loop_shift);
            }
        }
        tw_offset += half_size;
        half_size >>= 1;
        num_groups <<= 1;
    }
}

void dl_fft2r_sc16_dif_hp(int16_t *data, int16_t *table, int num_stages, int N, int *out_shift)
{
    if (num_stages >= 3 && num_stages <= 10) {
        dl_fft2r_sc16_dif_hp_asm(data, table, num_stages, N, out_shift);
    } else {
        dl_fft2r_sc16_dif_hp_ansi(data, table, num_stages, N, out_shift);
    }
}

void dl_ifft2r_sc16_dif_hp_ansi(int16_t *data, int16_t *table, int num_stages, int N, int *out_shift)
{
    int half_size = N >> 1;
    int num_groups = 1;
    int tw_offset = 0;
    int loop_shift = 0;
    int add_rount_mult = 0;
    out_shift[0] = 0;

    for (int stage = 0; stage < num_stages; stage++) {
        loop_shift = dl_array_max_q_s16(data, N * 2);
        add_rount_mult = 1 << (loop_shift - 1);
        out_shift[0] += loop_shift - DL_FFT_DIF_SC16_TABLE_BITS;

        for (int g = 0; g < num_groups; g++) {
            int base = g * (half_size << 1);
            for (int k = 0; k < half_size; k++) {
                int ui = base + k;
                int li = ui + half_size;

                int32_t a_re = data[ui * 2];
                int32_t a_im = data[ui * 2 + 1];
                int32_t b_re = data[li * 2];
                int32_t b_im = data[li * 2 + 1];

                int32_t sum_re = ((a_re + b_re) << DL_FFT_DIF_SC16_TABLE_BITS) + add_rount_mult;
                int32_t sum_im = ((a_im + b_im) << DL_FFT_DIF_SC16_TABLE_BITS) + add_rount_mult;
                data[ui * 2] = (int16_t)(sum_re >> loop_shift);
                data[ui * 2 + 1] = (int16_t)(sum_im >> loop_shift);

                int32_t d_re = a_re - b_re;
                int32_t d_im = a_im - b_im;
                int32_t w_re = table[(tw_offset + k) * 2];
                int32_t w_im = table[(tw_offset + k) * 2 + 1];
                int32_t lo_re = d_re * w_re - d_im * w_im + add_rount_mult;
                int32_t lo_im = d_im * w_re + d_re * w_im + add_rount_mult;
                data[li * 2] = (int16_t)(lo_re >> loop_shift);
                data[li * 2 + 1] = (int16_t)(lo_im >> loop_shift);
            }
        }
        tw_offset += half_size;
        half_size >>= 1;
        num_groups <<= 1;
    }
}
