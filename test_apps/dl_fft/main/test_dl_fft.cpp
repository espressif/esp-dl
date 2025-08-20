#include "dl_fft.h"
#include "dl_fft.hpp"
#include "dl_rfft.h"
#include "test_fft.h"

static const char *TAG = "TEST DL AUDIO";
static int LOOP = 10;
using namespace dl;

TEST_CASE("1. test dl fft", "[dl_fft]")
{
    const float *input[5] = {fft_input_128, fft_input_256, fft_input_512, fft_input_1024, fft_input_2048};
    const float *output[5] = {fft_output_128, fft_output_256, fft_output_512, fft_output_1024, fft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 90;
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test fft(%d) float: ", nfft);
        float *x = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        float *x2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * 2 * sizeof(float));
        memcpy(x2, input[i], nfft * 2 * sizeof(float));

        dl_fft_f32_t *fft_handle = dl_fft_f32_init(nfft, MALLOC_CAP_8BIT);

        dl_fft_f32_run(fft_handle, x);
        fft->fft(x2, nfft);
        TEST_ASSERT_EQUAL(true, check_is_same(x, x2, nfft * 2, 1e-6));

        TEST_ASSERT_EQUAL(true, check_fft_results(x, output[i], nfft, target_db, 1e-3));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_fft_f32_run(fft_handle, x);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_fft_f32_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before - ram_size_end < 300);
}

TEST_CASE("2. test dl ifft", "[dl_fft]")
{
    const float *input[5] = {fft_input_128, fft_input_256, fft_input_512, fft_input_1024, fft_input_2048};
    const float *output[5] = {fft_output_128, fft_output_256, fft_output_512, fft_output_1024, fft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 80;
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test fft(%d) float: ", nfft);
        float *x = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        float *x2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        memcpy(x, output[i], nfft * 2 * sizeof(float));
        memcpy(x2, output[i], nfft * 2 * sizeof(float));

        dl_fft_f32_t *fft_handle = dl_fft_f32_init(nfft, MALLOC_CAP_8BIT);

        dl_ifft_f32_run(fft_handle, x);
        fft->ifft(x2, nfft);
        TEST_ASSERT_EQUAL(true, check_is_same(x, x2, nfft * 2, 1e-6));

        TEST_ASSERT_EQUAL(true, check_fft_results(x, input[i], nfft * 2, target_db, 1e-3));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_ifft_f32_run(fft_handle, x);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_fft_f32_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before - ram_size_end < 300);
}

TEST_CASE("3. test dl rfft", "[dl_fft]")
{
    const float *input[5] = {rfft_input_128, rfft_input_256, rfft_input_512, rfft_input_1024, rfft_input_2048};
    const float *output[5] = {rfft_output_128, rfft_output_256, rfft_output_512, rfft_output_1024, rfft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 90;
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test rfft(%d) float: ", nfft);
        float *x = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *x2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *gt = (float *)heap_caps_aligned_alloc(16, sizeof(float) * nfft, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * sizeof(float));
        memcpy(x2, input[i], nfft * sizeof(float));
        memcpy(gt, output[i], nfft * sizeof(float));
        gt[1] = output[i][nfft];

        dl_fft_f32_t *fft_handle = dl_rfft_f32_init(nfft, MALLOC_CAP_8BIT);

        dl_rfft_f32_run(fft_handle, x);
        fft->rfft(x2, nfft);
        TEST_ASSERT_EQUAL(true, check_is_same(x, x2, nfft, 1e-6));

        TEST_ASSERT_EQUAL(true, check_fft_results(x, gt, nfft, target_db, 1e-3));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_rfft_f32_run(fft_handle, x);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_rfft_f32_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
        heap_caps_free(gt);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("4. test dl irfft", "[dl_fft]")
{
    const float *input[5] = {rfft_input_128, rfft_input_256, rfft_input_512, rfft_input_1024, rfft_input_2048};
    // const float *output[5] = {rfft_output_128, rfft_output_256, rfft_output_512, rfft_output_1024, rfft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 84;
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test rfft(%d) float: ", nfft);
        float *x = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *x2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *gt = (float *)heap_caps_aligned_alloc(16, sizeof(float) * nfft, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * sizeof(float));
        memcpy(x2, input[i], nfft * sizeof(float));
        memcpy(gt, input[i], nfft * sizeof(float));

        dl_fft_f32_t *fft_handle = dl_rfft_f32_init(nfft, MALLOC_CAP_8BIT);

        dl_rfft_f32_run(fft_handle, x);
        dl_irfft_f32_run(fft_handle, x);
        fft->rfft(x2, nfft);
        fft->irfft(x2, nfft);
        TEST_ASSERT_EQUAL(true, check_is_same(x, x2, nfft, 1e-6));

        TEST_ASSERT_EQUAL(true, check_fft_results(x, input[i], nfft, target_db, 1e-3));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_irfft_f32_run(fft_handle, x);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_rfft_f32_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
        heap_caps_free(gt);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("5. test dl fft s16", "[dl_fft]")
{
    const int16_t *input[5] = {
        fft_input_s16_128, fft_input_s16_256, fft_input_s16_512, fft_input_s16_1024, fft_input_s16_2048};
    const float *output[5] = {fft_output_128, fft_output_256, fft_output_512, fft_output_1024, fft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 50;
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;
    int out_exponent;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test fft(%d) s16: ", nfft);
        int16_t *x = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t) * 2, MALLOC_CAP_8BIT);
        int16_t *x2 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t) * 2, MALLOC_CAP_8BIT);
        float *y = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        float *y2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * 2 * sizeof(int16_t));
        memcpy(x2, input[i], nfft * 2 * sizeof(int16_t));

        dl_fft_s16_t *fft_handle = dl_fft_s16_init(nfft, MALLOC_CAP_8BIT);
        dl_fft_s16_run(fft_handle, x, -15, &out_exponent); // -15 means x is in Q15 format
        fft->fft(x2, nfft, -15, &out_exponent);

        // check snr
        dl_short_to_float(x, nfft * 2, out_exponent, y);
        dl_short_to_float(x2, nfft * 2, out_exponent, y2);
        TEST_ASSERT_EQUAL(true, check_is_same(y, y2, nfft, 1e-6));
        TEST_ASSERT_EQUAL(true, check_fft_results(y, output[i], nfft, target_db, 3e-2));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_fft_s16_run(fft_handle, x, -15, &out_exponent);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_fft_s16_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
        heap_caps_free(y);
        heap_caps_free(y2);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("6. test dl ifft s16", "[dl_fft]")
{
    const int16_t *input[5] = {
        fft_input_s16_128, fft_input_s16_256, fft_input_s16_512, fft_input_s16_1024, fft_input_s16_2048};
    const float *output[5] = {fft_input_128, fft_input_256, fft_input_512, fft_input_1024, fft_input_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 45;
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;
    int exponent;
    int out_exponent;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test fft(%d) s16: ", nfft);
        int16_t *x = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t) * 2, MALLOC_CAP_8BIT);
        int16_t *x2 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t) * 2, MALLOC_CAP_8BIT);
        float *y = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        float *y2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * 2 * sizeof(int16_t));
        memcpy(x2, input[i], nfft * 2 * sizeof(int16_t));

        dl_fft_s16_t *fft_handle = dl_fft_s16_init(nfft, MALLOC_CAP_8BIT);
        dl_fft_s16_run(fft_handle, x, -15, &exponent);    // -15 means x is in Q15 format
        int shift = 15 - dl_array_max_q_s16(x, nfft * 2); // scale input to INT16_MAX
        for (int j = 0; j < nfft * 2; j++) {
            x[j] = x[j] << (shift);
        }
        exponent -= shift;
        dl_ifft_s16_run(fft_handle, x, exponent, &out_exponent); // -15 means x is in Q15 format

        fft->fft(x2, nfft, -15, &exponent);
        for (int j = 0; j < nfft * 2; j++) {
            x2[j] = x2[j] << (shift);
        }
        exponent -= shift;
        fft->ifft(x2, nfft, exponent, &out_exponent);

        // check snr
        dl_short_to_float(x, nfft * 2, out_exponent, y);
        dl_short_to_float(x2, nfft * 2, out_exponent, y2);
        TEST_ASSERT_EQUAL(true, check_is_same(y, y2, nfft, 1e-6));
        TEST_ASSERT_EQUAL(true, check_fft_results(y, output[i], nfft, target_db, 3e-2));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_ifft_s16_run(fft_handle, x, -15, &out_exponent);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_fft_s16_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
        heap_caps_free(y);
        heap_caps_free(y2);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("7. test dl fft hp s16", "[dl_fft]")
{
    const int16_t *input[5] = {
        fft_input_s16_128, fft_input_s16_256, fft_input_s16_512, fft_input_s16_1024, fft_input_s16_2048};
    const float *output[5] = {fft_output_128, fft_output_256, fft_output_512, fft_output_1024, fft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 60; // high precision int16 fft
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;
    int out_exponent;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test fft(%d) s16: ", nfft);
        int16_t *x = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t) * 2, MALLOC_CAP_8BIT);
        int16_t *x2 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t) * 2, MALLOC_CAP_8BIT);
        float *y = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        float *y2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * 2 * sizeof(int16_t));
        memcpy(x2, input[i], nfft * 2 * sizeof(int16_t));

        dl_fft_s16_t *fft_handle = dl_fft_s16_init(nfft, MALLOC_CAP_8BIT);

        dl_fft_s16_hp_run(fft_handle, x, -15, &out_exponent);
        fft->fft_hp(x2, nfft, -15, &out_exponent);

        // check snr
        dl_short_to_float(x, nfft * 2, out_exponent, y);
        dl_short_to_float(x2, nfft * 2, out_exponent, y2);
        TEST_ASSERT_EQUAL(true, check_is_same(y, y2, nfft * 2, 1e-6));
        TEST_ASSERT_EQUAL(true, check_fft_results(y, output[i], nfft, target_db, 2e-2));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_fft_s16_hp_run(fft_handle, x, -15, &out_exponent);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_fft_s16_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
        heap_caps_free(y);
        heap_caps_free(y2);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("8. test dl ifft hp s16", "[dl_fft]")
{
    const int16_t *input[5] = {
        fft_input_s16_128, fft_input_s16_256, fft_input_s16_512, fft_input_s16_1024, fft_input_s16_2048};
    // const float *output[5] = {fft_input_128, fft_input_256, fft_input_512, fft_input_1024, fft_input_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 60; // high precision int16 fft
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;
    int exponent;
    int out_exponent;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test fft(%d) s16: ", nfft);
        int16_t *x = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t) * 2, MALLOC_CAP_8BIT);
        int16_t *x2 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t) * 2, MALLOC_CAP_8BIT);
        float *y = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        float *y2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        float *z = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float) * 2, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * 2 * sizeof(int16_t));
        memcpy(x2, input[i], nfft * 2 * sizeof(int16_t));

        dl_fft_s16_t *fft_handle = dl_fft_s16_init(nfft, MALLOC_CAP_8BIT);

        dl_fft_s16_hp_run(fft_handle, x, -15, &exponent);
        dl_ifft_s16_hp_run(fft_handle, x, exponent, &out_exponent);

        fft->fft_hp(x2, nfft, -15, &exponent);
        fft->ifft_hp(x2, nfft, exponent, &out_exponent);

        // check snr
        dl_short_to_float(x, nfft * 2, out_exponent, y);
        dl_short_to_float(x2, nfft * 2, out_exponent, y2);
        TEST_ASSERT_EQUAL(true, check_is_same(y, y2, nfft * 2, 1e-6));
        dl_short_to_float(input[i], nfft * 2, -15, z);
        TEST_ASSERT_EQUAL(true, check_fft_results(y, z, nfft, target_db, 1e-2));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_ifft_s16_hp_run(fft_handle, x, -15, &out_exponent);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_fft_s16_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
        heap_caps_free(y);
        heap_caps_free(y2);
        heap_caps_free(z);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("9. test dl rfft s16", "[dl_fft]")
{
    const int16_t *input[5] = {
        rfft_input_s16_128, rfft_input_s16_256, rfft_input_s16_512, rfft_input_s16_1024, rfft_input_s16_2048};
    const float *output[5] = {rfft_output_128, rfft_output_256, rfft_output_512, rfft_output_1024, rfft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 45;
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;
    int out_exponent;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test rfft(%d) float: ", nfft);
        int16_t *x = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
        int16_t *x2 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
        float *y = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *y2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *gt = (float *)heap_caps_aligned_alloc(16, sizeof(float) * nfft, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * sizeof(int16_t));
        memcpy(x2, input[i], nfft * sizeof(int16_t));
        memcpy(gt, output[i], nfft * sizeof(float));
        gt[1] = output[i][nfft];

        dl_fft_s16_t *fft_handle = dl_rfft_s16_init(nfft, MALLOC_CAP_8BIT);

        dl_rfft_s16_run(fft_handle, x, -15, &out_exponent);
        fft->rfft(x2, nfft, -15, &out_exponent);

        dl_short_to_float(x, nfft, out_exponent, y);
        dl_short_to_float(x2, nfft, out_exponent, y2);
        TEST_ASSERT_EQUAL(true, check_is_same(y, y2, nfft, 1e-6));
        TEST_ASSERT_EQUAL(true, check_fft_results(y, gt, nfft, target_db, 5e-2));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_rfft_s16_run(fft_handle, x, -15, &out_exponent);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_rfft_s16_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
        heap_caps_free(y);
        heap_caps_free(y2);
        heap_caps_free(gt);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("10. test dl irfft s16", "[dl_fft]")
{
    const int16_t *input[5] = {
        rfft_input_s16_128, rfft_input_s16_256, rfft_input_s16_512, rfft_input_s16_1024, rfft_input_s16_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 40;
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;
    int exponent;
    int out_exponent;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test fft(%d) s16: ", nfft);
        int16_t *x = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
        int16_t *x2 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
        float *y = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *y2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *gt = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * sizeof(int16_t));
        memcpy(x2, input[i], nfft * sizeof(int16_t));

        dl_fft_s16_t *fft_handle = dl_rfft_s16_init(nfft, MALLOC_CAP_8BIT);

        dl_rfft_s16_run(fft_handle, x, -15, &exponent); // -15 means x is in Q15 format
        int shift = 15 - dl_array_max_q_s16(x, nfft);   // scale input to INT16_MAX
        for (int j = 0; j < nfft; j++) {
            x[j] = x[j] << (shift);
        }
        exponent -= shift;
        dl_irfft_s16_run(fft_handle, x, exponent, &out_exponent); // -15 means x is in Q15 format

        fft->rfft(x2, nfft, -15, &exponent);
        shift = 15 - dl_array_max_q_s16(x2, nfft);
        for (int j = 0; j < nfft; j++) {
            x2[j] = x2[j] << (shift);
        }
        exponent -= shift;
        fft->irfft(x2, nfft, exponent, &out_exponent);

        // check snr
        dl_short_to_float(x, nfft, out_exponent, y);
        dl_short_to_float(x2, nfft, out_exponent, y2);
        TEST_ASSERT_EQUAL(true, check_is_same(y, y2, nfft, 1e-6));
        dl_short_to_float(input[i], nfft, -15, gt);
        TEST_ASSERT_EQUAL(true, check_fft_results(y, gt, nfft, target_db, 3e-2));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_irfft_s16_run(fft_handle, x, -15, &out_exponent);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_rfft_s16_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
        heap_caps_free(y);
        heap_caps_free(y2);
        heap_caps_free(gt);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("11. test dl rfft hp s16", "[dl_fft]")
{
    const int16_t *input[5] = {
        rfft_input_s16_128, rfft_input_s16_256, rfft_input_s16_512, rfft_input_s16_1024, rfft_input_s16_2048};
    const float *output[5] = {rfft_output_128, rfft_output_256, rfft_output_512, rfft_output_1024, rfft_output_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 55;
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;
    int out_exponent;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test rfft(%d) float: ", nfft);
        int16_t *x = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
        int16_t *x2 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
        float *y = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *y2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *gt = (float *)heap_caps_aligned_alloc(16, sizeof(float) * nfft, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * sizeof(int16_t));
        memcpy(x2, input[i], nfft * sizeof(int16_t));
        memcpy(gt, output[i], nfft * sizeof(float));
        gt[1] = output[i][nfft];

        dl_fft_s16_t *fft_handle = dl_rfft_s16_init(nfft, MALLOC_CAP_8BIT);

        dl_rfft_s16_hp_run(fft_handle, x, -15, &out_exponent);
        fft->rfft_hp(x2, nfft, -15, &out_exponent);

        dl_short_to_float(x, nfft, out_exponent, y);
        dl_short_to_float(x2, nfft, out_exponent, y2);
        TEST_ASSERT_EQUAL(true, check_is_same(y, y2, nfft, 1e-6));
        TEST_ASSERT_EQUAL(true, check_fft_results(y, gt, nfft, target_db, 1e-2));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_rfft_s16_hp_run(fft_handle, x, -15, &out_exponent);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_rfft_s16_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
        heap_caps_free(y);
        heap_caps_free(y2);
        heap_caps_free(gt);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("12. test dl irfft hp s16", "[dl_fft]")
{
    const int16_t *input[5] = {
        rfft_input_s16_128, rfft_input_s16_256, rfft_input_s16_512, rfft_input_s16_1024, rfft_input_s16_2048};
    int test_nfft[5] = {128, 256, 512, 1024, 2048};
    float target_db = 55;
    FFT *fft = FFT::get_instance();
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start = 0, end = 0;
    int exponent;
    int out_exponent;

    for (int i = 0; i < 5; i++) {
        int nfft = test_nfft[i];
        printf("test rfft(%d) float: ", nfft);
        int16_t *x = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
        int16_t *x2 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
        float *y = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *y2 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
        float *gt = (float *)heap_caps_aligned_alloc(16, sizeof(float) * nfft, MALLOC_CAP_8BIT);
        memcpy(x, input[i], nfft * sizeof(int16_t));
        memcpy(x2, input[i], nfft * sizeof(int16_t));

        dl_fft_s16_t *fft_handle = dl_rfft_s16_init(nfft, MALLOC_CAP_8BIT);

        dl_rfft_s16_hp_run(fft_handle, x, -15, &exponent);
        dl_irfft_s16_hp_run(fft_handle, x, exponent, &out_exponent);

        fft->rfft_hp(x2, nfft, -15, &exponent);
        fft->irfft_hp(x2, nfft, exponent, &out_exponent);

        dl_short_to_float(x, nfft, out_exponent, y);
        dl_short_to_float(x2, nfft, out_exponent, y2);
        TEST_ASSERT_EQUAL(true, check_is_same(y, y2, nfft, 1e-6));
        dl_short_to_float(input[i], nfft, -15, gt);
        TEST_ASSERT_EQUAL(true, check_fft_results(y, gt, nfft, target_db, 1e-2));

        start = esp_timer_get_time();
        for (int k = 0; k < LOOP; k++) {
            dl_irfft_s16_hp_run(fft_handle, x, -15, &out_exponent);
        }
        end = esp_timer_get_time();
        printf("time:%ld us\n", (end - start) / LOOP);
        dl_rfft_s16_deinit(fft_handle);
        heap_caps_free(x);
        heap_caps_free(x2);
        heap_caps_free(y);
        heap_caps_free(y2);
        heap_caps_free(gt);
    }

    fft->clear();
    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

// Test handle caching
TEST_CASE("13. test FFT class handle caching", "[dl_fft_class]")
{
    FFT *fft = FFT::get_instance();

    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    // Clear any existing handles
    fft->clear();
    TEST_ASSERT_EQUAL(0, fft->get_handle_count());

    // Create handles for different FFT lengths
    float *data = (float *)heap_caps_aligned_alloc(16, 2048 * sizeof(float), MALLOC_CAP_8BIT);
    int16_t *data_s16 = (int16_t *)heap_caps_aligned_alloc(16, 2048 * sizeof(int16_t), MALLOC_CAP_8BIT);
    ;
    int exponent;

    // Test float32 FFT handles
    TEST_ASSERT_EQUAL(ESP_OK, fft->fft(data, 128));
    TEST_ASSERT_EQUAL(1, fft->get_handle_count());

    TEST_ASSERT_EQUAL(ESP_OK, fft->fft(data, 256));
    TEST_ASSERT_EQUAL(2, fft->get_handle_count());

    // Test int16 FFT handles
    TEST_ASSERT_EQUAL(ESP_OK, fft->fft(data_s16, 128, 0, &exponent));
    TEST_ASSERT_EQUAL(3, fft->get_handle_count());

    // Test float32 RFFT handles
    TEST_ASSERT_EQUAL(ESP_OK, fft->rfft(data, 128));
    TEST_ASSERT_EQUAL(4, fft->get_handle_count());

    // Test int16 RFFT handles
    TEST_ASSERT_EQUAL(ESP_OK, fft->rfft(data_s16, 128, 0, &exponent));
    TEST_ASSERT_EQUAL(5, fft->get_handle_count());

    // Reuse existing handles (should not increase count)
    TEST_ASSERT_EQUAL(ESP_OK, fft->fft(data, 128));
    TEST_ASSERT_EQUAL(5, fft->get_handle_count());

    TEST_ASSERT_EQUAL(ESP_OK, fft->rfft(data, 128));
    TEST_ASSERT_EQUAL(5, fft->get_handle_count());

    // Clear all handles
    fft->clear();
    TEST_ASSERT_EQUAL(0, fft->get_handle_count());

    free(data);
    free(data_s16);

    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

// Test singleton pattern
TEST_CASE("14. test FFT class singleton", "[dl_fft_class]")
{
    FFT *fft1 = FFT::get_instance();
    FFT *fft2 = FFT::get_instance();

    // Both pointers should point to the same instance
    TEST_ASSERT_EQUAL(fft1, fft2);

    // Clear handles to ensure clean state
    fft1->clear();
    TEST_ASSERT_EQUAL(0, fft1->get_handle_count());
    TEST_ASSERT_EQUAL(0, fft2->get_handle_count());
}
