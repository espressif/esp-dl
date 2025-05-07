#include "dl_rfft.h"
#include "esp_log.h"
#include <math.h>
#include <string.h>

static const char *TAG = "dl rfft";

dl_fft_s16_t *dl_rfft_s16_init(int fft_point, uint32_t caps)
{
    dl_fft_s16_t *handle = (dl_fft_s16_t *)heap_caps_malloc(sizeof(dl_fft_s16_t), caps);
    if (!handle) {
        ESP_LOGE(TAG, "Failed to allocate FFT handle");
        return NULL;
    }
    handle->fftr2_table = NULL;
    handle->rfft_table = NULL;
    handle->fft_point = fft_point;
    handle->log2n = dl_power_of_two(fft_point);

    // rfft table
    handle->rfft_table = dl_gen_rfft_table_s16(fft_point, caps);
    if (!handle->rfft_table) {
        ESP_LOGE(TAG, "Failed to generate FFT table");
        dl_rfft_s16_deinit(handle);
        return NULL;
    }

    // fft table
    handle->fftr2_table = dl_gen_fftr2_table_sc16(fft_point >> 1, caps);
    if (!handle->fftr2_table) {
        ESP_LOGE(TAG, "Failed to generate FFT table");
        dl_rfft_s16_deinit(handle);
        return NULL;
    }

    return handle;
}

// Free FFT handle
void dl_rfft_s16_deinit(dl_fft_s16_t *handle)
{
    if (handle) {
        if (handle->fftr2_table) {
            free(handle->fftr2_table);
        }
        if (handle->rfft_table) {
            free(handle->rfft_table);
        }
        free(handle);
    }
}

// Perform FFT
esp_err_t dl_rfft_s16_run(dl_fft_s16_t *handle, int16_t *data, int in_exponent, int *out_exponent)
{
    if (!handle || !data) {
        return ESP_FAIL;
    }

    int cpx_point = handle->fft_point >> 1;
    dl_fft2r_sc16(data, cpx_point, handle->fftr2_table);
    dl_bitrev2r_sc16_ansi(data, cpx_point);
    dl_cplx2real_sc16_ansi(data, cpx_point, handle->rfft_table);
    out_exponent[0] = in_exponent + handle->log2n;

    return ESP_OK;
}

esp_err_t dl_rfft_s16_hp_run(dl_fft_s16_t *handle, int16_t *data, int in_exponent, int *out_exponent)
{
    if (!handle || !data) {
        return ESP_FAIL;
    }

    int cpx_point = handle->fft_point >> 1;
    out_exponent[0] = 0;
    dl_fft2r_sc16_hp(data, cpx_point, handle->fftr2_table, out_exponent);
    dl_bitrev2r_sc16_ansi(data, cpx_point);
    // dl_cplx2real_sc16_hp_ansi(data, cpx_point, handle->rfft_table, out_exponent);
    dl_cplx2real_sc16_ansi(data, cpx_point, handle->rfft_table);
    out_exponent[0] = in_exponent + out_exponent[0] + 1;

    return ESP_OK;
}
