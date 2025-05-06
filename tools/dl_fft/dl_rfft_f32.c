#include "dl_rfft.h"
#include "esp_attr.h"
#include "esp_log.h"
#include <math.h>
#include <string.h>

static const char *TAG = "dl rfft";

dl_fft_f32_t *dl_rfft_f32_init(int fft_point, uint32_t caps)
{
    if (!dl_is_power_of_two(fft_point)) {
        ESP_LOGE(TAG, "FFT point must be power of two");
        return NULL;
    }

    dl_fft_f32_t *handle = (dl_fft_f32_t *)heap_caps_malloc(sizeof(dl_fft_f32_t), caps);
    if (!handle) {
        ESP_LOGE(TAG, "Failed to allocate FFT handle");
        return NULL;
    }
    handle->fftr2_table = NULL;
    handle->rfft_table = NULL;
    handle->reverse_table = NULL;
    handle->fft_point = fft_point;
    handle->log2n = dl_power_of_two(fft_point);

    // rfft table
    handle->rfft_table = dl_gen_rfft_table_f32(fft_point, caps);
    if (!handle->rfft_table) {
        ESP_LOGE(TAG, "Failed to generate FFT table");
        dl_rfft_f32_deinit(handle);
        return NULL;
    }

    // fft table
    handle->fftr2_table = dl_gen_fftr2_table_f32(fft_point >> 1, caps);
    if (!handle->fftr2_table) {
        ESP_LOGE(TAG, "Failed to generate FFT table");
        dl_rfft_f32_deinit(handle);
        return NULL;
    }

    if (handle->log2n % 2 == 1) {
        handle->reverse_table = dl_gen_bitrev4r_table(fft_point, caps, &handle->reverse_size);
    } else {
        handle->reverse_table = dl_gen_bitrev2r_table(fft_point>>1, caps, &handle->reverse_size);
    }

    return handle;
}

void dl_rfft_f32_deinit(dl_fft_f32_t *handle)
{
    if (handle) {
        if (handle->fftr2_table) {
            free(handle->fftr2_table);
        }
        if (handle->rfft_table) {
            free(handle->rfft_table);
        }
        if (handle->reverse_table) {
            free(handle->reverse_table);
        }
        free(handle);
    }
}
esp_err_t dl_rfft_f32_run(dl_fft_f32_t *handle, float *data)
{
    if (!handle || !data) {
        return ESP_FAIL;
    }
    int fft_point = handle->fft_point;
    float *fftr2_table = handle->fftr2_table;
    float *rfft_table = handle->rfft_table;

    if (handle->log2n % 2 == 1) {
        dl_fft4r_fc32(data, fft_point >> 1, rfft_table, fft_point);
        dl_bitrev4r_fc32_ansi(data, fft_point >> 1, handle->reverse_table, handle->reverse_size);
    } else {
        dl_fft2r_fc32(data, fft_point >> 1, fftr2_table);
        dl_bitrev2r_fc32_ansi(data, fft_point>>1, handle->reverse_table, handle->reverse_size);
    }

    // Convert one complex vector with length N/2 to one real spectrum vector with length N/2
    dl_cplx2real_fc32_ansi(data, fft_point >> 1, rfft_table, fft_point);

    return ESP_OK;
}
