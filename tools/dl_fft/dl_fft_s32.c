#include "dl_fft.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

static const char *TAG = "dl fft";

dl_fft_s32_t *dl_fft_s32_init(int fft_point, uint32_t caps)
{
    if (fft_point < 2 || !dl_is_power_of_two(fft_point)) {
        ESP_LOGE(TAG, "FFT point must be power of two and >= 2");
        return NULL;
    }

    dl_fft_s32_t *handle = (dl_fft_s32_t *)heap_caps_malloc(sizeof(dl_fft_s32_t), caps);
    if (!handle) {
        ESP_LOGE(TAG, "Failed to allocate FFT handle");
        return NULL;
    }
    handle->fft_point = fft_point;
    handle->log2n = dl_power_of_two(fft_point);
    handle->fft_table = dl_fft_table_acquire(DL_FFT_TBL_S32_FFT, fft_point, caps, NULL);
    if (!handle->fft_table) {
        ESP_LOGE(TAG, "Failed to generate FFT table");
        dl_fft_s32_deinit(handle);
        return NULL;
    }

    return handle;
}

void dl_fft_s32_deinit(dl_fft_s32_t *handle)
{
    if (handle) {
        dl_fft_table_release(handle->fft_table);
        heap_caps_free(handle);
    }
}

esp_err_t dl_fft_s32_run(dl_fft_s32_t *handle, int32_t *data, int in_exponent, int *out_exponent)
{
    if (!handle || !data) {
        return ESP_FAIL;
    }

    int fft_point = handle->fft_point;
    dl_fft2r_sc32_dif_ansi(data, handle->fft_table, fft_point);
    dl_bitrev2r_sc32_ansi(data, fft_point);
    out_exponent[0] = in_exponent + handle->log2n;

    return ESP_OK;
}
