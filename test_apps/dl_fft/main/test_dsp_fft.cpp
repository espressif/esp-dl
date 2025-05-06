#include "dl_fft.h"
#include "dl_rfft.h"
#include "esp_dsp.h"
#include "test_fft.h"

static const char *TAG = "TEST DSP FFT";

TEST_CASE("Test dsp fft", "[dsp_fft]")
{
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    dsps_fft2r_init_fc32(NULL, 2048);
    const float *input[5] = {fft_input_128, fft_input_256, fft_input_512, fft_input_1024, fft_input_2048};
    const float *output[5] = {fft_output_128, fft_output_256, fft_output_512, fft_output_1024, fft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 100;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test fft(%d): ", nfft);
        float *x = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * 2 * sizeof(float));
        dsps_fft2r_fc32(x, nfft);
        dsps_bit_rev_fc32(x, nfft);
        TEST_ASSERT_EQUAL(true, check_fft_results(x, output[i], nfft, target_db, 1e-3));
        free(x);
    }

    dsps_fft2r_deinit_fc32();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "internal ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("Test dsp real rfft", "[dsp_fft]")
{
    const float *input[5] = {rfft_input_128, rfft_input_256, rfft_input_512, rfft_input_1024, rfft_input_2048};
    const float *output[5] = {rfft_output_128, rfft_output_256, rfft_output_512, rfft_output_1024, rfft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 100;
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    esp_err_t ret = dsps_fft2r_init_fc32(NULL, 256);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Not possible to initialize FFT2R. Error = %i", ret);
        return;
    }

    ret = dsps_fft4r_init_fc32(NULL, 128);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Not possible to initialize FFT4R. Error = %i", ret);
        return;
    }

    for (int i = 0; i < 2; i++) {
        int nfft = test_nfft[i];
        float *x = (float *)malloc(sizeof(float) * nfft);
        float *gt = (float *)malloc(sizeof(float) * nfft);
        memcpy(x, input[i], nfft * sizeof(float));
        memcpy(gt, output[i], nfft * sizeof(float));
        gt[1] = output[i][nfft];

        if (dl_power_of_two(nfft) % 2 == 1) {
            dsps_fft4r_fc32(x, nfft >> 1);
            // Bit reverse
            dsps_bit_rev4r_fc32(x, nfft >> 1);

        } else {
            dsps_fft2r_fc32(x, nfft >> 1);
            // Bit reverse
            dsps_bit_rev2r_fc32(x, nfft >> 1);
        }

        // Convert one complex vector with length N/2 to one real spectrum vector with length N/2
        dsps_cplx2real_fc32(x, nfft >> 1);
        TEST_ASSERT_EQUAL(true, check_fft_results(x, output[i], nfft, target_db, 1e-3));
        free(x);
        free(gt);
    }

    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("Test dl fft s16", "[dsp_fft]")
{
    const int16_t *input[5] = {
        fft_input_s16_128, fft_input_s16_256, fft_input_s16_512, fft_input_s16_1024, fft_input_s16_2048};
    const float *output[5] = {fft_output_128, fft_output_256, fft_output_512, fft_output_1024, fft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 100;
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test fft(%d) s16: ", nfft);
        int16_t *x = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t) * 2, MALLOC_CAP_8BIT);
        float *y = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * 2 * sizeof(int16_t));

        dsps_fft2r_init_sc16(NULL, nfft);
        dsps_fft2r_sc16_ansi(x, nfft);
        dsps_bit_rev_sc16_ansi(x, nfft);
        dl_short_to_float(x, nfft * 2, -15 + 7 + i, y);

        float mean_db = get_snr(y, output[i], nfft * 2);
        TEST_ASSERT_EQUAL(true, mean_db > target_db);
        printf("snr: %f, exponent:%d pass \n", mean_db, -15 + 6 + i);
        free(x);
    }

    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}
