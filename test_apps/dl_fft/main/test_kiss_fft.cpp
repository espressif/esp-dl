
#include "dl_fft.h"
#include "dl_rfft.h"
#include "kiss_fft.h"
#include "kiss_fftr.h"
#include "test_fft.h"

static const char *TAG = "TEST DL AUDIO";
static int LOOP = 10;

TEST_CASE("Test dl kiss fft s16", "[kiss_fft]")
{
    const int16_t *input[5] = {
        rfft_input_s16_128, rfft_input_s16_256, rfft_input_s16_512, rfft_input_s16_1024, rfft_input_s16_2048};
    const float *output[5] = {rfft_output_128, rfft_output_256, rfft_output_512, rfft_output_1024, rfft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 50;
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test rfft(%d) float: ", nfft);
        int16_t *x1 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
        int16_t *x2 = (int16_t *)heap_caps_aligned_alloc(16, (nfft + 2) * sizeof(int16_t), MALLOC_CAP_8BIT);
        float *y = (float *)malloc((nfft + 2) * sizeof(float));
        float *gt = (float *)malloc(sizeof(float) * nfft);
        memcpy(x1, input[i], nfft * sizeof(int16_t));
        memcpy(gt, output[i], nfft * sizeof(float));
        gt[1] = output[i][nfft];
        for (int j = 0; j < nfft; j++) {
            x1[j] = x1[j] * 2;
        }
        kiss_fftr_cfg fft_handle = kiss_fftr_alloc(nfft, 0, 0, 0);
        kiss_fftr(fft_handle, x1, (kiss_fft_cpx *)x2);
        dl_short_to_float(x2, nfft + 2, -16 + i + 7, y);
        y[1] = y[nfft];
        TEST_ASSERT_EQUAL(true, check_fft_results(y, gt, nfft, target_db, 3e-2));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_array_max_q_s16(x1, nfft);
            for (int j = 0; j < nfft; j++) {
                x1[j] = x1[j] * 2;
            }
            kiss_fftr(fft_handle, x1, (kiss_fft_cpx *)x2);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        free(fft_handle);
        free(x1);
        free(x2);
        free(y);
        free(gt);
    }

    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("Test dl kiss ifft s16", "[kiss_fft]")
{
    const int16_t *input[5] = {
        rfft_input_s16_128, rfft_input_s16_256, rfft_input_s16_512, rfft_input_s16_1024, rfft_input_s16_2048};
    const float *output[5] = {rfft_output_128, rfft_output_256, rfft_output_512, rfft_output_1024, rfft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 50;
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test rfft(%d) float: ", nfft);
        int16_t *x1 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
        int16_t *x2 = (int16_t *)heap_caps_aligned_alloc(16, (nfft + 2) * sizeof(int16_t), MALLOC_CAP_8BIT);
        float *y = (float *)malloc((nfft + 2) * sizeof(float));
        float *gt = (float *)malloc(sizeof(float) * nfft);
        memcpy(x1, input[i], nfft * sizeof(int16_t));
        memcpy(gt, output[i], nfft * sizeof(float));
        gt[1] = output[i][nfft];
        for (int j = 0; j < nfft; j++) {
            x1[j] = x1[j] * 2;
        }
        kiss_fftr_cfg fft_handle = kiss_fftr_alloc(nfft, 0, 0, 0);
        kiss_fftr_cfg ifft_handle = kiss_fftr_alloc(nfft, 1, 0, 0);
        kiss_fftr(fft_handle, x1, (kiss_fft_cpx *)x2);
        kiss_fftri(ifft_handle, (kiss_fft_cpx *)x2, x1);
        dl_short_to_float(x2, nfft + 2, -16 + i + 7, y);
        TEST_ASSERT_EQUAL(true, check_fft_results(y, gt, nfft, target_db, 3e-2));

        dl_short_to_float(x2, nfft + 2, -16 + i + 7 - 1, y);
        TEST_ASSERT_EQUAL(true, check_fft_results(y, gt, nfft, target_db, 3e-2));

        dl_short_to_float(x2, nfft + 2, -16 + i + 7 - 2, y);
        TEST_ASSERT_EQUAL(true, check_fft_results(y, gt, nfft, target_db, 3e-2));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_array_max_q_s16(x1, nfft);
            for (int j = 0; j < nfft; j++) {
                x1[j] = x1[j] * 2;
            }
            kiss_fftr(fft_handle, x1, (kiss_fft_cpx *)x2);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        free(fft_handle);
        free(x1);
        free(x2);
        free(y);
        free(gt);
    }

    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}
