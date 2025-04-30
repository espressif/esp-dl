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
#include "dsp_types.h"
#include "dsps_fft2r.h"
#include "esp_attr.h"
#include "esp_log.h"
#include <malloc.h>
#include <math.h>
#include <string.h>

static const char *TAG = "fftr2_ansi";

// unsigned short reverse(unsigned short x, unsigned short N, int order);

esp_err_t dl_fft2r_fc32_ansi(float *data, int N, float *w)
{
    esp_err_t result = ESP_OK;

    int ie, ia, m;
    float re_temp, im_temp;
    float c, s;
    ie = 1;
    for (int N2 = N / 2; N2 > 0; N2 >>= 1) {
        ia = 0;
        for (int j = 0; j < ie; j++) {
            c = w[2 * j];
            s = w[2 * j + 1];
            for (int i = 0; i < N2; i++) {
                m = ia + N2;
                re_temp = c * data[2 * m] + s * data[2 * m + 1];
                im_temp = c * data[2 * m + 1] - s * data[2 * m];
                data[2 * m] = data[2 * ia] - re_temp;
                data[2 * m + 1] = data[2 * ia + 1] - im_temp;
                data[2 * ia] = data[2 * ia] + re_temp;
                data[2 * ia + 1] = data[2 * ia + 1] + im_temp;
                ia++;
            }
            ia += N2;
        }
        ie <<= 1;
    }
    return result;
}

// unsigned short reverse(unsigned short x, unsigned short N, int order)
// {
//     unsigned short b = x;

//     b = (b & 0xff00) >> 8 | (b & 0x00fF) << 8;
//     b = (b & 0xf0F0) >> 4 | (b & 0x0f0F) << 4;
//     b = (b & 0xCCCC) >> 2 | (b & 0x3333) << 2;
//     b = (b & 0xAAAA) >> 1 | (b & 0x5555) << 1;
//     return b >> (16 - order);
// }

esp_err_t dl_bit_rev_fc32_ansi(float *data, int N)
{
    esp_err_t result = ESP_OK;

    int j, k;
    float r_temp, i_temp;
    j = 0;
    for (int i = 1; i < (N - 1); i++) {
        k = N >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
        if (i < j) {
            r_temp = data[j * 2];
            data[j * 2] = data[i * 2];
            data[i * 2] = r_temp;
            i_temp = data[j * 2 + 1];
            data[j * 2 + 1] = data[i * 2 + 1];
            data[i * 2 + 1] = i_temp;
        }
    }
    return result;
}

esp_err_t dl_cplx2reC_fc32_ansi(float *data, int N)
{
    esp_err_t result = ESP_OK;

    int i;
    int n2 = N << 1;

    float rkl = 0;
    float rkh = 0;
    float rnl = 0;
    float rnh = 0;
    float ikl = 0;
    float ikh = 0;
    float inl = 0;
    float inh = 0;

    for (i = 0; i < (N / 4); i++) {
        rkl = data[i * 2 + 0 + 2];
        ikl = data[i * 2 + 1 + 2];
        rnl = data[n2 - i * 2 - 2];
        inl = data[n2 - i * 2 - 1];

        rkh = data[i * 2 + 0 + 2 + N];
        ikh = data[i * 2 + 1 + 2 + N];
        rnh = data[n2 - i * 2 - 2 - N];
        inh = data[n2 - i * 2 - 1 - N];

        data[i * 2 + 0 + 2] = rkl + rnl;
        data[i * 2 + 1 + 2] = ikl - inl;

        data[n2 - i * 2 - 1 - N] = inh - ikh;
        data[n2 - i * 2 - 2 - N] = rkh + rnh;

        data[i * 2 + 0 + 2 + N] = ikl + inl;
        data[i * 2 + 1 + 2 + N] = rnl - rkl;

        data[n2 - i * 2 - 1] = rkh - rnh;
        data[n2 - i * 2 - 2] = ikh + inh;
    }
    data[N] = data[1];
    data[1] = 0;
    data[N + 1] = 0;

    return result;
}

esp_err_t dl_gen_bitrev2r_table(int N, int step, char *name_ext)
{
    if (!dsp_is_power_of_two(N)) {
        return ESP_ERR_DSP_INVALID_LENGTH;
    }

    int j, k;
    j = 0;
    int items_count = 0;
    ESP_LOGD(TAG, "const uint16_t bitrev2r_table_%i_%s[] = {        ", N, name_ext);
    for (int i = 1; i < (N - 1); i++) {
        k = N >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
        if (i < j) {
            ESP_LOGD(TAG, "%i, %i, ", i * step, j * step);
            items_count++;
            if ((items_count % 8) == 0) {
                ESP_LOGD(TAG, " ");
            }
        }
    }
    ESP_LOGD(TAG, "};");
    ESP_LOGD(TAG, "const uint16_t bitrev2r_table_%i_%s_size = %i;\n", N, name_ext, items_count);

    ESP_LOGD(TAG, "extern const uint16_t bitrev2r_table_%i_%s[];", N, name_ext);
    ESP_LOGD(TAG, "extern const uint16_t bitrev2r_table_%i_%s_size;\n", N, name_ext);
    return ESP_OK;
}

esp_err_t dl_bit_rev2r_fc32(float *data, int N)
{
    uint16_t *table;
    uint16_t table_size;
    switch (N) {
    case 16:
        table = (uint16_t *)dsps_fft2r_rev_tables_fc32[0];
        table_size = dsps_fft2r_rev_tables_fc32_size[0];
        break;
    case 32:
        table = (uint16_t *)dsps_fft2r_rev_tables_fc32[1];
        table_size = dsps_fft2r_rev_tables_fc32_size[1];
        break;
    case 64:
        table = (uint16_t *)dsps_fft2r_rev_tables_fc32[2];
        table_size = dsps_fft2r_rev_tables_fc32_size[2];
        break;
    case 128:
        table = (uint16_t *)dsps_fft2r_rev_tables_fc32[3];
        table_size = dsps_fft2r_rev_tables_fc32_size[3];
        break;
    case 256:
        table = (uint16_t *)dsps_fft2r_rev_tables_fc32[4];
        table_size = dsps_fft2r_rev_tables_fc32_size[4];
        break;
    case 512:
        table = (uint16_t *)dsps_fft2r_rev_tables_fc32[5];
        table_size = dsps_fft2r_rev_tables_fc32_size[5];
        break;
    case 1024:
        table = (uint16_t *)dsps_fft2r_rev_tables_fc32[6];
        table_size = dsps_fft2r_rev_tables_fc32_size[6];
        break;
    case 2048:
        table = (uint16_t *)dsps_fft2r_rev_tables_fc32[7];
        table_size = dsps_fft2r_rev_tables_fc32_size[7];
        break;
    case 4096:
        table = (uint16_t *)dsps_fft2r_rev_tables_fc32[8];
        table_size = dsps_fft2r_rev_tables_fc32_size[8];
        break;

    default:
        return dl_bit_rev_fc32_ansi(data, N);
        break;
    }
    return dl_bit_rev_lookup_fc32_ansi(data, table_size, table);
}

esp_err_t dl_bit_rev_lookup_fc32_ansi(float *data, int reverse_size, uint16_t *reverse_tab)
{
    float r_temp, i_temp;
    for (int n = 0; n < reverse_size; n++) {
        uint16_t i = reverse_tab[n * 2 + 0] >> 2;
        uint16_t j = reverse_tab[n * 2 + 1] >> 2;
        r_temp = data[j];
        data[j] = data[i];
        data[i] = r_temp;
        i_temp = data[j + 1];
        data[j + 1] = data[i + 1];
        data[i + 1] = i_temp;
    }
    return ESP_OK;
}

float *dl_gen_fftr2_table_f32(int fft_point, uint32_t caps)
{
    float *fft_table = (float *)heap_caps_aligned_alloc(16, fft_point * sizeof(float), caps);

    if (fft_table) {
        float e = M_PI * 2.0 / fft_point;

        for (int i = 0; i < (fft_point >> 1); i++) {
            fft_table[2 * i] = cosf(i * e);
            fft_table[2 * i + 1] = sinf(i * e);
        }

        dl_bit_rev_fc32_ansi(fft_table, fft_point >> 1);
    }

    return fft_table;
}
