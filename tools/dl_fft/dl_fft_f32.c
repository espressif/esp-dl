#include "dl_fft.h"
#include "esp_log.h"
#include <math.h>
#include <string.h>

static const char *TAG = "dl fft";

// Create a new FFT handle
dl_fft_f32_t *dl_fft_f32_init(int fft_point, uint32_t caps)
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

    // Allocate and generate FFT table
    handle->fftr2_table = dl_gen_fftr2_table_f32(fft_point, caps);
    if (!handle->fftr2_table) {
        ESP_LOGE(TAG, "Failed to generate FFT table");
        dl_fft_f32_deinit(handle);
        return NULL;
    }
    handle->reverse_table = dl_gen_bitrev2r_table(fft_point, caps, &handle->reverse_size);

    return handle;
}

// Free FFT handle
void dl_fft_f32_deinit(dl_fft_f32_t *handle)
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

// Perform FFT
esp_err_t dl_fft_f32_run(dl_fft_f32_t *handle, float *data)
{
    if (!handle || !data) {
        return ESP_FAIL;
    }

    int fft_point = handle->fft_point;
    dl_fft2r_fc32(data, fft_point, handle->fftr2_table);
    dl_bitrev2r_fc32_ansi(data, fft_point, handle->reverse_table, handle->reverse_size);

    return ESP_OK;
}

esp_err_t dl_ifft_f32_run(dl_fft_f32_t *handle, float *data)
{
    if (!handle || !data) {
        return ESP_FAIL;
    }

    int fft_point = handle->fft_point;
    float scale = 1.0f / fft_point;

    dl_ifft2r_fc32(data, fft_point, handle->fftr2_table);
    dl_bitrev2r_fc32_ansi(data, fft_point, handle->reverse_table, handle->reverse_size);

    // Scale by 1/N
    for (int i = 0; i < fft_point * 2; i++) {
        data[i] *= scale;
    }

    return ESP_OK;
}
