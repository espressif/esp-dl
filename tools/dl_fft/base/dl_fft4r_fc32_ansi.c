// Copyright 2018-2019 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "dl_fft_base.h"

esp_err_t dl_fft4r_fc32_ansi(float *data, int length, float *table, int table_size)
{
    fc32_t bfly[4];
    int log2N = dl_power_of_two(length);
    int log4N = log2N >> 1;
    if ((log2N & 0x01) != 0) {
        return ESP_FAIL;
    }

    int m = 2;
    int wind_step = table_size / length;
    while (1) { /// radix 4
        if (log4N == 0) {
            break;
        }
        length = length >> 2;
        for (int j = 0; j < m; j += 2) {         // j: which FFT of this step
            int start_index = j * (length << 1); // n: n-point FFT

            fc32_t *ptrc0 = (fc32_t *)data + start_index;
            fc32_t *ptrc1 = ptrc0 + length;
            fc32_t *ptrc2 = ptrc1 + length;
            fc32_t *ptrc3 = ptrc2 + length;

            fc32_t *winc0 = (fc32_t *)table;
            fc32_t *winc1 = winc0;
            fc32_t *winc2 = winc0;

            for (int k = 0; k < length; k++) {
                fc32_t in0 = *ptrc0;
                fc32_t in2 = *ptrc2;
                fc32_t in1 = *ptrc1;
                fc32_t in3 = *ptrc3;

                bfly[0].re = in0.re + in2.re + in1.re + in3.re;
                bfly[0].im = in0.im + in2.im + in1.im + in3.im;

                bfly[1].re = in0.re - in2.re + in1.im - in3.im;
                bfly[1].im = in0.im - in2.im - in1.re + in3.re;

                bfly[2].re = in0.re + in2.re - in1.re - in3.re;
                bfly[2].im = in0.im + in2.im - in1.im - in3.im;

                bfly[3].re = in0.re - in2.re - in1.im + in3.im;
                bfly[3].im = in0.im - in2.im + in1.re - in3.re;

                *ptrc0 = bfly[0];
                ptrc1->re = bfly[1].re * winc0->re + bfly[1].im * winc0->im;
                ptrc1->im = bfly[1].im * winc0->re - bfly[1].re * winc0->im;
                ptrc2->re = bfly[2].re * winc1->re + bfly[2].im * winc1->im;
                ptrc2->im = bfly[2].im * winc1->re - bfly[2].re * winc1->im;
                ptrc3->re = bfly[3].re * winc2->re + bfly[3].im * winc2->im;
                ptrc3->im = bfly[3].im * winc2->re - bfly[3].re * winc2->im;

                winc0 += 1 * wind_step;
                winc1 += 2 * wind_step;
                winc2 += 3 * wind_step;

                ptrc0++;
                ptrc1++;
                ptrc2++;
                ptrc3++;
            }
        }
        m = m << 2;
        wind_step = wind_step << 2;
        log4N--;
    }
    return ESP_OK;
}

esp_err_t dl_bitrev4r_fc32_ansi(float *data, int N, uint16_t *reverse_tab, int reverse_size)
{
    esp_err_t result = ESP_OK;
    if (reverse_tab) {
        float r_temp, i_temp;
        for (int n = 0; n < reverse_size; n++) {
            uint16_t i = reverse_tab[n * 2];
            uint16_t j = reverse_tab[n * 2 + 1];
            r_temp = data[j];
            i_temp = data[j + 1];
            data[j] = data[i];
            data[i] = r_temp;
            data[j + 1] = data[i + 1];
            data[i + 1] = i_temp;
        }
    } else {
        int log2N = dl_power_of_two(N);
        int log4N = log2N >> 1;
        if ((log2N & 0x01) != 0) {
            return ESP_FAIL;
        }
        float r_temp, i_temp;
        for (int i = 0; i < N; i++) {
            int cnt;
            int xx;
            int bits2;
            xx = 0;
            cnt = log4N;
            int j = i;
            while (cnt > 0) {
                bits2 = j & 0x3;
                xx = (xx << 2) + bits2;
                j = j >> 2;
                cnt--;
            }
            if (i < xx) {
                r_temp = data[i * 2 + 0];
                i_temp = data[i * 2 + 1];
                data[i * 2 + 0] = data[xx * 2 + 0];
                data[i * 2 + 1] = data[xx * 2 + 1];
                data[xx * 2 + 0] = r_temp;
                data[xx * 2 + 1] = i_temp;
            }
        }
    }

    return result;
}

esp_err_t dl_cplx2real_fc32_ansi(float *data, int N, float *table, int table_size)
{
    int wind_step = table_size / (N);
    fc32_t *result = (fc32_t *)data;
    // Original formula...
    // result[0].re = result[0].re + result[0].im;
    // result[N].re = result[0].re - result[0].im;
    // result[0].im = 0;
    // result[N].im = 0;
    // Optimized one:
    float tmp_re = result[0].re;
    result[0].re = tmp_re + result[0].im;
    result[0].im = tmp_re - result[0].im;

    fc32_t f1k, f2k;
    for (int k = 1; k <= N / 2; k++) {
        fc32_t fpk = result[k];
        fc32_t fpnk = result[N - k];
        f1k.re = fpk.re + fpnk.re;
        f1k.im = fpk.im - fpnk.im;
        f2k.re = fpk.re - fpnk.re;
        f2k.im = fpk.im + fpnk.im;

        float c = -table[k * wind_step + 1];
        float s = -table[k * wind_step];
        fc32_t tw;
        tw.re = c * f2k.re - s * f2k.im;
        tw.im = s * f2k.re + c * f2k.im;

        result[k].re = 0.5 * (f1k.re + tw.re);
        result[k].im = 0.5 * (f1k.im + tw.im);
        result[N - k].re = 0.5 * (f1k.re - tw.re);
        result[N - k].im = 0.5 * (tw.im - f1k.im);
    }
    return ESP_OK;
}

uint16_t *dl_gen_bitrev4r_table(int N, uint32_t caps, int *reverse_size)
{
    int log2N = dl_power_of_two(N);
    int log4N = log2N >> 1;
    if ((log2N & 0x01) != 0) {
        reverse_size[0] = 0;
        return NULL;
    }
    int count = 0, idx = 0;
    for (int i = 0; i < N; i++) {
        int cnt;
        int xx;
        int bits2;
        xx = 0;
        cnt = log4N;
        int j = i;
        while (cnt > 0) {
            bits2 = j & 0x3;
            xx = (xx << 2) + bits2;
            j = j >> 2;
            cnt--;
        }
        if (i < xx) {
            count++;
        }
    }

    if (count * 2 > UINT16_MAX) {
        return NULL;
    }
    reverse_size[0] = count;
    uint16_t *reverse_tab = (uint16_t *)heap_caps_malloc(2 * count * sizeof(uint16_t), caps);

    if (reverse_tab) {
        for (int i = 0; i < N; i++) {
            int cnt;
            int xx;
            int bits2;
            xx = 0;
            cnt = log4N;
            int j = i;
            while (cnt > 0) {
                bits2 = j & 0x3;
                xx = (xx << 2) + bits2;
                j = j >> 2;
                cnt--;
            }
            if (i < xx) {
                reverse_tab[idx * 2] = i * 2;
                reverse_tab[idx * 2 + 1] = xx * 2;
                idx++;
            }
        }
    }
    return reverse_tab;
}

float *dl_gen_rfft_table_f32(int fft_point, uint32_t caps)
{
    float *fft_table = (float *)heap_caps_aligned_alloc(16, fft_point * sizeof(float) * 2, caps);

    if (fft_table) {
        for (int i = 0; i < fft_point; i++) {
            float angle = 2 * M_PI * i * 1.0 / fft_point;
            fft_table[2 * i] = cosf(angle);
            fft_table[2 * i + 1] = sinf(angle);
        }
    }

    return fft_table;
}
