#include "dl_audio_wav.hpp"
#include "dl_fft.h"
#include "dl_fft.hpp"
#include "dl_fft_base.h"
#include "dl_rfft.h"
#include "test_fft.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

static const char *TAG = "TEST DL AUDIO";

extern const uint8_t wav_embed_test_wav_start[] asm("_binary_test_wav_start");
extern const uint8_t wav_embed_test_wav_end[] asm("_binary_test_wav_end");

using dl::audio::decode_wav;
using dl::audio::dl_audio_t;

static void free_dl_audio(dl_audio_t *a)
{
    if (!a) {
        return;
    }
    free(a->data);
    free(a);
}

/** Mono frame starting at sample index @a base_off (zero-pad past end). */
static void wav_mono_frame_at(const dl_audio_t *a, int base_off, int nfft, int16_t *s16, float *f32)
{
    int ch = a->channels;
    int nframes = (int)a->length;
    for (int j = 0; j < nfft; ++j) {
        int idx = base_off + j;
        int16_t v = 0;
        if (idx >= 0 && idx < nframes) {
            v = (ch == 1) ? a->data[idx] : a->data[idx * ch];
        }
        s16[j] = v;
        f32[j] = v * (1.f / 32768.f);
    }
}

static void abs_error_max_and_sum(const float *ref, const float *x, int n, float *max_err, float *sum_err)
{
    float sum = 0.f;
    float mx = 0.f;
    for (int i = 0; i < n; ++i) {
        float e = fabsf(x[i] - ref[i]);
        sum += e;
        if (e > mx) {
            mx = e;
        }
    }
    *max_err = mx;
    *sum_err = sum;
}
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
    float target_db = 36;
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
        TEST_ASSERT_EQUAL(true, check_fft_results(y, output[i], nfft, target_db, 4e-2));

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
    float target_db = 37;
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
        TEST_ASSERT_EQUAL(true, check_fft_results(y, output[i], nfft, target_db, 5e-2));

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

TEST_CASE("15. test wav rfft512 irfft roundtrip f32", "[dl_fft]")
{
    const int nfft = 512;
    dl_audio_t *wav = decode_wav(wav_embed_test_wav_start, (int)(wav_embed_test_wav_end - wav_embed_test_wav_start));
    TEST_ASSERT_NOT_NULL(wav);
    TEST_ASSERT_GREATER_OR_EQUAL(1, wav->channels);

    int16_t *mono_s16 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
    float *ref_f32 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
    float *work_f32 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
    TEST_ASSERT_NOT_NULL(mono_s16);
    TEST_ASSERT_NOT_NULL(ref_f32);
    TEST_ASSERT_NOT_NULL(work_f32);

    const int nframes = (int)wav->length;
    const int n_blocks = (nframes + nfft - 1) / nfft;

    dl_fft_f32_t *h_f32 = dl_rfft_f32_init(nfft, MALLOC_CAP_8BIT);
    TEST_ASSERT_NOT_NULL(h_f32);

    float worst_max_f32 = 0.f;
    double total_sum_f32 = 0.0;
    const int total_cmp_samples = n_blocks * nfft;

    for (int b = 0; b < n_blocks; ++b) {
        const int base = b * nfft;
        wav_mono_frame_at(wav, base, nfft, mono_s16, ref_f32);

        float max_f32 = 0.f;
        float sum_f32 = 0.f;
        memcpy(work_f32, ref_f32, nfft * sizeof(float));
        TEST_ASSERT_EQUAL(ESP_OK, dl_rfft_f32_run(h_f32, work_f32));
        TEST_ASSERT_EQUAL(ESP_OK, dl_irfft_f32_run(h_f32, work_f32));
        abs_error_max_and_sum(ref_f32, work_f32, nfft, &max_f32, &sum_f32);
        if (max_f32 > worst_max_f32) {
            worst_max_f32 = max_f32;
        }
        total_sum_f32 += (double)sum_f32;
        ESP_LOGI(
            TAG, "frame %d/%d base=%d f32: max_abs_err=%.6e sum_abs_err=%.6e", b + 1, n_blocks, base, max_f32, sum_f32);
    }

    free_dl_audio(wav);
    dl_rfft_f32_deinit(h_f32);

    const float mean_f32 = (float)(total_sum_f32 / (double)total_cmp_samples);
    ESP_LOGI(TAG,
             "wav f32 all frames: n_blocks=%d total_samples=%d worst_max=%.6e mean_abs=%.6e",
             n_blocks,
             total_cmp_samples,
             worst_max_f32,
             mean_f32);

    TEST_ASSERT_EQUAL(true, worst_max_f32 < 1e-5f);
    TEST_ASSERT_EQUAL(true, mean_f32 < 1e-6f);

    heap_caps_free(mono_s16);
    heap_caps_free(ref_f32);
    heap_caps_free(work_f32);
}

TEST_CASE("16. test wav rfft512 irfft roundtrip hp s16", "[dl_fft]")
{
    const int nfft = 512;
    dl_audio_t *wav = decode_wav(wav_embed_test_wav_start, (int)(wav_embed_test_wav_end - wav_embed_test_wav_start));
    TEST_ASSERT_NOT_NULL(wav);
    TEST_ASSERT_GREATER_OR_EQUAL(1, wav->channels);

    int16_t *mono_s16 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
    float *ref_f32 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
    float *ref_hp = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
    float *out_hp = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
    int16_t *work_s16 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
    TEST_ASSERT_NOT_NULL(mono_s16);
    TEST_ASSERT_NOT_NULL(ref_f32);
    TEST_ASSERT_NOT_NULL(ref_hp);
    TEST_ASSERT_NOT_NULL(out_hp);
    TEST_ASSERT_NOT_NULL(work_s16);

    const int nframes = (int)wav->length;
    const int n_blocks = (nframes + nfft - 1) / nfft;

    dl_fft_s16_t *h_s16 = dl_rfft_s16_init(nfft, MALLOC_CAP_8BIT);
    TEST_ASSERT_NOT_NULL(h_s16);

    float worst_max_hp = 0.f;
    double total_sum_hp = 0.0;
    const int total_cmp_samples = n_blocks * nfft;

    for (int b = 0; b < n_blocks; ++b) {
        const int base = b * nfft;
        wav_mono_frame_at(wav, base, nfft, mono_s16, ref_f32);
        dl_short_to_float(mono_s16, nfft, -15, ref_hp);

        int exponent = 0;
        int out_exponent = 0;
        memcpy(work_s16, mono_s16, nfft * sizeof(int16_t));
        TEST_ASSERT_EQUAL(ESP_OK, dl_rfft_s16_hp_run(h_s16, work_s16, -15, &exponent));
        TEST_ASSERT_EQUAL(ESP_OK, dl_irfft_s16_hp_run(h_s16, work_s16, exponent, &out_exponent));
        dl_short_to_float(work_s16, nfft, out_exponent, out_hp);
        float max_hp = 0.f;
        float sum_hp = 0.f;
        abs_error_max_and_sum(ref_hp, out_hp, nfft, &max_hp, &sum_hp);
        if (max_hp > worst_max_hp) {
            worst_max_hp = max_hp;
        }
        total_sum_hp += (double)sum_hp;
        ESP_LOGI(TAG,
                 "frame %d/%d base=%d hp s16: max_abs_err=%.6e sum_abs_err=%.6e (out_exp=%d)",
                 b + 1,
                 n_blocks,
                 base,
                 max_hp,
                 sum_hp,
                 out_exponent);
    }

    free_dl_audio(wav);
    dl_rfft_s16_deinit(h_s16);

    const float mean_hp = (float)(total_sum_hp / (double)total_cmp_samples);
    ESP_LOGI(TAG,
             "wav hp s16 all frames: n_blocks=%d total_samples=%d worst_max=%.6e mean_abs=%.6e",
             n_blocks,
             total_cmp_samples,
             worst_max_hp,
             mean_hp);

    TEST_ASSERT_EQUAL(true, worst_max_hp < 5e-3f);
    TEST_ASSERT_EQUAL(true, mean_hp < 1e-4f);

    heap_caps_free(mono_s16);
    heap_caps_free(ref_f32);
    heap_caps_free(ref_hp);
    heap_caps_free(out_hp);
    heap_caps_free(work_s16);
}

TEST_CASE("17. test wav rfft512 irfft roundtrip s16", "[dl_fft]")
{
    const int nfft = 512;
    dl_audio_t *wav = decode_wav(wav_embed_test_wav_start, (int)(wav_embed_test_wav_end - wav_embed_test_wav_start));
    TEST_ASSERT_NOT_NULL(wav);
    TEST_ASSERT_GREATER_OR_EQUAL(1, wav->channels);

    int16_t *mono_s16 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
    float *ref_f32 = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
    float *ref_q = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
    float *out_q = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), MALLOC_CAP_8BIT);
    int16_t *work_s16 = (int16_t *)heap_caps_aligned_alloc(16, nfft * sizeof(int16_t), MALLOC_CAP_8BIT);
    TEST_ASSERT_NOT_NULL(mono_s16);
    TEST_ASSERT_NOT_NULL(ref_f32);
    TEST_ASSERT_NOT_NULL(ref_q);
    TEST_ASSERT_NOT_NULL(out_q);
    TEST_ASSERT_NOT_NULL(work_s16);

    const int nframes = (int)wav->length;
    const int n_blocks = (nframes + nfft - 1) / nfft;

    dl_fft_s16_t *h_s16 = dl_rfft_s16_init(nfft, MALLOC_CAP_8BIT);
    TEST_ASSERT_NOT_NULL(h_s16);

    float worst_max = 0.f;
    double total_sum = 0.0;
    const int total_cmp_samples = n_blocks * nfft;

    for (int b = 0; b < n_blocks; ++b) {
        const int base = b * nfft;
        wav_mono_frame_at(wav, base, nfft, mono_s16, ref_f32);
        dl_short_to_float(mono_s16, nfft, -15, ref_q);

        int exponent = 0;
        int out_exponent = 0;
        memcpy(work_s16, mono_s16, nfft * sizeof(int16_t));
        TEST_ASSERT_EQUAL(ESP_OK, dl_rfft_s16_run(h_s16, work_s16, -15, &exponent));
        int shift = 15 - dl_array_max_q_s16(work_s16, nfft);
        for (int j = 0; j < nfft; j++) {
            work_s16[j] = work_s16[j] << shift;
        }
        exponent -= shift;
        TEST_ASSERT_EQUAL(ESP_OK, dl_irfft_s16_run(h_s16, work_s16, exponent, &out_exponent));
        dl_short_to_float(work_s16, nfft, out_exponent, out_q);

        float max_e = 0.f;
        float sum_e = 0.f;
        abs_error_max_and_sum(ref_q, out_q, nfft, &max_e, &sum_e);
        if (max_e > worst_max) {
            worst_max = max_e;
        }
        total_sum += (double)sum_e;
        ESP_LOGI(TAG,
                 "frame %d/%d base=%d s16: max_abs_err=%.6e sum_abs_err=%.6e (out_exp=%d shift=%d)",
                 b + 1,
                 n_blocks,
                 base,
                 max_e,
                 sum_e,
                 out_exponent,
                 shift);
    }

    free_dl_audio(wav);
    dl_rfft_s16_deinit(h_s16);

    const float mean_abs = (float)(total_sum / (double)total_cmp_samples);
    ESP_LOGI(TAG,
             "wav s16 all frames: n_blocks=%d total_samples=%d worst_max=%.6e mean_abs=%.6e",
             n_blocks,
             total_cmp_samples,
             worst_max,
             mean_abs);

    /* Same order as test 10 (target_db 40, rmse 3e-2); WAV is slightly harder — margin on max abs. */
    TEST_ASSERT_EQUAL(true, worst_max < 5e-2f);
    TEST_ASSERT_EQUAL(true, mean_abs < 1e-3f);

    heap_caps_free(mono_s16);
    heap_caps_free(ref_f32);
    heap_caps_free(ref_q);
    heap_caps_free(out_q);
    heap_caps_free(work_s16);
}

// Shared FFT table cache: same (kind, fft_point, caps) is reused; last release frees it.
TEST_CASE("18. test fft table cache acquire reuse", "[dl_fft]")
{
    FFT::get_instance()->clear();
    const uint32_t caps = MALLOC_CAP_8BIT;
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    dl_fft_table_release(NULL);

    void *t1 = dl_fft_table_acquire(DL_FFT_TBL_F32_FFT2R, 128, caps, NULL);
    void *t2 = dl_fft_table_acquire(DL_FFT_TBL_F32_FFT2R, 128, caps, NULL);
    TEST_ASSERT_NOT_NULL(t1);
    TEST_ASSERT_EQUAL(t1, t2);

    void *t256 = dl_fft_table_acquire(DL_FFT_TBL_F32_FFT2R, 256, caps, NULL);
    TEST_ASSERT_NOT_NULL(t256);
    TEST_ASSERT_NOT_EQUAL(t1, t256);

    void *t_s16 = dl_fft_table_acquire(DL_FFT_TBL_S16_DIF_FFT, 128, caps, NULL);
    TEST_ASSERT_NOT_NULL(t_s16);
    TEST_ASSERT_NOT_EQUAL(t1, t_s16);

    void *t_rfft = dl_fft_table_acquire(DL_FFT_TBL_F32_RFFT, 128, caps, NULL);
    TEST_ASSERT_NOT_NULL(t_rfft);
    TEST_ASSERT_NOT_EQUAL(t1, t_rfft);

    int extra1 = -1;
    int extra2 = -1;
    void *b1 = dl_fft_table_acquire(DL_FFT_TBL_F32_BITREV2R, 128, caps, &extra1);
    void *b2 = dl_fft_table_acquire(DL_FFT_TBL_F32_BITREV2R, 128, caps, &extra2);
    TEST_ASSERT_NOT_NULL(b1);
    TEST_ASSERT_EQUAL(b1, b2);
    TEST_ASSERT_EQUAL(extra1, extra2);
    TEST_ASSERT_GREATER_THAN(0, extra1);

    dl_fft_table_release(t1);
    dl_fft_table_release(t2);
    dl_fft_table_release(t256);
    dl_fft_table_release(t_s16);
    dl_fft_table_release(t_rfft);
    dl_fft_table_release(b1);
    dl_fft_table_release(b2);

    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("19. test shared fft tables among handles", "[dl_fft]")
{
    FFT::get_instance()->clear();
    const uint32_t caps = MALLOC_CAP_8BIT;
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    dl_fft_f32_t *f32_a = dl_fft_f32_init(128, caps);
    dl_fft_f32_t *f32_b = dl_fft_f32_init(128, caps);
    TEST_ASSERT_NOT_NULL(f32_a);
    TEST_ASSERT_NOT_NULL(f32_b);
    TEST_ASSERT_EQUAL(f32_a->fft_table, f32_b->fft_table);
    TEST_ASSERT_EQUAL(f32_a->bitrev_table, f32_b->bitrev_table);
    TEST_ASSERT_EQUAL(f32_a->bitrev_size, f32_b->bitrev_size);

    dl_fft_f32_t *r4_a = dl_rfft_f32_init(128, caps); // log2n odd -> FFT4R
    dl_fft_f32_t *r4_b = dl_rfft_f32_init(128, caps);
    TEST_ASSERT_NOT_NULL(r4_a);
    TEST_ASSERT_NOT_NULL(r4_b);
    TEST_ASSERT_EQUAL(r4_a->fft_table, r4_b->fft_table);
    TEST_ASSERT_EQUAL(r4_a->rfft_table, r4_b->rfft_table);
    TEST_ASSERT_EQUAL(r4_a->bitrev_table, r4_b->bitrev_table);

    dl_fft_f32_t *r2_a = dl_rfft_f32_init(256, caps); // log2n even -> FFT2R of 128
    dl_fft_f32_t *r2_b = dl_rfft_f32_init(256, caps);
    TEST_ASSERT_NOT_NULL(r2_a);
    TEST_ASSERT_NOT_NULL(r2_b);
    TEST_ASSERT_EQUAL(r2_a->fft_table, r2_b->fft_table);
    TEST_ASSERT_EQUAL(r2_a->rfft_table, r2_b->rfft_table);
    TEST_ASSERT_EQUAL(r2_a->bitrev_table, r2_b->bitrev_table);

    dl_fft_s16_t *s16_a = dl_fft_s16_init(128, caps);
    dl_fft_s16_t *s16_b = dl_fft_s16_init(128, caps);
    TEST_ASSERT_NOT_NULL(s16_a);
    TEST_ASSERT_NOT_NULL(s16_b);
    TEST_ASSERT_EQUAL(s16_a->fft_table, s16_b->fft_table);

    dl_fft_s16_t *rs16_a = dl_rfft_s16_init(128, caps);
    dl_fft_s16_t *rs16_b = dl_rfft_s16_init(128, caps);
    TEST_ASSERT_NOT_NULL(rs16_a);
    TEST_ASSERT_NOT_NULL(rs16_b);
    TEST_ASSERT_EQUAL(rs16_a->fft_table, rs16_b->fft_table);
    TEST_ASSERT_EQUAL(rs16_a->rfft_table, rs16_b->rfft_table);

    dl_fft_f32_deinit(f32_a);
    dl_fft_f32_deinit(f32_b);
    dl_rfft_f32_deinit(r4_a);
    dl_rfft_f32_deinit(r4_b);
    dl_rfft_f32_deinit(r2_a);
    dl_rfft_f32_deinit(r2_b);
    dl_fft_s16_deinit(s16_a);
    dl_fft_s16_deinit(s16_b);
    dl_rfft_s16_deinit(rs16_a);
    dl_rfft_s16_deinit(rs16_b);

    int ram_mid = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_mid);

    int ram0 = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    dl_fft_f32_t *h1 = dl_fft_f32_init(256, caps);
    TEST_ASSERT_NOT_NULL(h1);
    int ram1 = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    dl_fft_f32_t *h2 = dl_fft_f32_init(256, caps);
    dl_fft_f32_t *h3 = dl_fft_f32_init(256, caps);
    TEST_ASSERT_NOT_NULL(h2);
    TEST_ASSERT_NOT_NULL(h3);
    TEST_ASSERT_EQUAL(h1->fft_table, h2->fft_table);
    TEST_ASSERT_EQUAL(h1->fft_table, h3->fft_table);
    int ram3 = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    int first_cost = ram0 - ram1;
    int extra_cost = ram1 - ram3;
    ESP_LOGI(TAG, "shared handle cost: first=%d extra_two=%d", first_cost, extra_cost);
    TEST_ASSERT_GREATER_THAN(256, first_cost);
    TEST_ASSERT_EQUAL(true, extra_cost < first_cost / 4);
    TEST_ASSERT_EQUAL(true, extra_cost < 256);

    dl_fft_f32_deinit(h1);
    dl_fft_f32_deinit(h2);
    dl_fft_f32_deinit(h3);

    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("20. test shared fft tables survive peer deinit", "[dl_fft]")
{
    FFT::get_instance()->clear();
    const uint32_t caps = MALLOC_CAP_8BIT;
    const int nfft = 128;
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    float *x = (float *)heap_caps_aligned_alloc(16, nfft * 2 * sizeof(float), caps);
    float *y = (float *)heap_caps_aligned_alloc(16, nfft * 2 * sizeof(float), caps);
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(y);

    dl_fft_f32_t *f32_a = dl_fft_f32_init(nfft, caps);
    dl_fft_f32_t *f32_b = dl_fft_f32_init(nfft, caps);
    TEST_ASSERT_NOT_NULL(f32_a);
    TEST_ASSERT_NOT_NULL(f32_b);
    TEST_ASSERT_EQUAL(f32_a->fft_table, f32_b->fft_table);

    memcpy(x, fft_input_128, nfft * 2 * sizeof(float));
    TEST_ASSERT_EQUAL(ESP_OK, dl_fft_f32_run(f32_a, x));
    TEST_ASSERT_EQUAL(true, check_fft_results(x, fft_output_128, nfft, 90, 1e-3));

    dl_fft_f32_deinit(f32_a);

    memcpy(y, fft_input_128, nfft * 2 * sizeof(float));
    TEST_ASSERT_EQUAL(ESP_OK, dl_fft_f32_run(f32_b, y));
    TEST_ASSERT_EQUAL(true, check_is_same(x, y, nfft * 2, 1e-6));
    TEST_ASSERT_EQUAL(true, check_fft_results(y, fft_output_128, nfft, 90, 1e-3));
    dl_fft_f32_deinit(f32_b);

    float *rx = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), caps);
    float *ry = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), caps);
    float *rgt = (float *)heap_caps_aligned_alloc(16, nfft * sizeof(float), caps);
    TEST_ASSERT_NOT_NULL(rx);
    TEST_ASSERT_NOT_NULL(ry);
    TEST_ASSERT_NOT_NULL(rgt);
    memcpy(rgt, rfft_output_128, nfft * sizeof(float));
    rgt[1] = rfft_output_128[nfft];

    dl_fft_f32_t *r_a = dl_rfft_f32_init(nfft, caps);
    dl_fft_f32_t *r_b = dl_rfft_f32_init(nfft, caps);
    TEST_ASSERT_NOT_NULL(r_a);
    TEST_ASSERT_NOT_NULL(r_b);
    TEST_ASSERT_EQUAL(r_a->fft_table, r_b->fft_table);

    memcpy(rx, rfft_input_128, nfft * sizeof(float));
    TEST_ASSERT_EQUAL(ESP_OK, dl_rfft_f32_run(r_a, rx));
    TEST_ASSERT_EQUAL(true, check_fft_results(rx, rgt, nfft, 90, 1e-3));

    dl_rfft_f32_deinit(r_a);

    memcpy(ry, rfft_input_128, nfft * sizeof(float));
    TEST_ASSERT_EQUAL(ESP_OK, dl_rfft_f32_run(r_b, ry));
    TEST_ASSERT_EQUAL(true, check_is_same(rx, ry, nfft, 1e-6));
    TEST_ASSERT_EQUAL(true, check_fft_results(ry, rgt, nfft, 90, 1e-3));
    dl_rfft_f32_deinit(r_b);

    int16_t *sx = (int16_t *)heap_caps_aligned_alloc(16, nfft * 2 * sizeof(int16_t), caps);
    int16_t *sy = (int16_t *)heap_caps_aligned_alloc(16, nfft * 2 * sizeof(int16_t), caps);
    float *sf = (float *)heap_caps_aligned_alloc(16, nfft * 2 * sizeof(float), caps);
    TEST_ASSERT_NOT_NULL(sx);
    TEST_ASSERT_NOT_NULL(sy);
    TEST_ASSERT_NOT_NULL(sf);

    dl_fft_s16_t *s_a = dl_fft_s16_init(nfft, caps);
    dl_fft_s16_t *s_b = dl_fft_s16_init(nfft, caps);
    TEST_ASSERT_NOT_NULL(s_a);
    TEST_ASSERT_NOT_NULL(s_b);
    TEST_ASSERT_EQUAL(s_a->fft_table, s_b->fft_table);

    int out_exp = 0;
    memcpy(sx, fft_input_s16_128, nfft * 2 * sizeof(int16_t));
    TEST_ASSERT_EQUAL(ESP_OK, dl_fft_s16_run(s_a, sx, -15, &out_exp));
    dl_short_to_float(sx, nfft * 2, out_exp, sf);
    TEST_ASSERT_EQUAL(true, check_fft_results(sf, fft_output_128, nfft, 36, 4e-2));

    dl_fft_s16_deinit(s_a);

    memcpy(sy, fft_input_s16_128, nfft * 2 * sizeof(int16_t));
    TEST_ASSERT_EQUAL(ESP_OK, dl_fft_s16_run(s_b, sy, -15, &out_exp));
    dl_short_to_float(sy, nfft * 2, out_exp, sf);
    TEST_ASSERT_EQUAL(true, check_fft_results(sf, fft_output_128, nfft, 36, 4e-2));
    dl_fft_s16_deinit(s_b);

    heap_caps_free(x);
    heap_caps_free(y);
    heap_caps_free(rx);
    heap_caps_free(ry);
    heap_caps_free(rgt);
    heap_caps_free(sx);
    heap_caps_free(sy);
    heap_caps_free(sf);

    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("21. test fft/rfft share compatible tables", "[dl_fft]")
{
    FFT::get_instance()->clear();
    const uint32_t caps = MALLOC_CAP_8BIT;
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    // rfft_f32(256): log2n even -> FFT2R/BITREV2R of 128, same as fft_f32(128)
    dl_fft_f32_t *fft128 = dl_fft_f32_init(128, caps);
    dl_fft_f32_t *rfft256 = dl_rfft_f32_init(256, caps);
    TEST_ASSERT_NOT_NULL(fft128);
    TEST_ASSERT_NOT_NULL(rfft256);
    TEST_ASSERT_EQUAL(fft128->fft_table, rfft256->fft_table);
    TEST_ASSERT_EQUAL(fft128->bitrev_table, rfft256->bitrev_table);
    TEST_ASSERT_EQUAL(fft128->bitrev_size, rfft256->bitrev_size);

    // rfft_f32(512): log2n odd -> FFT4R of 512, not shared with fft_f32(256)
    dl_fft_f32_t *fft256 = dl_fft_f32_init(256, caps);
    dl_fft_f32_t *rfft512 = dl_rfft_f32_init(512, caps);
    TEST_ASSERT_NOT_NULL(fft256);
    TEST_ASSERT_NOT_NULL(rfft512);
    TEST_ASSERT_NOT_EQUAL(fft256->fft_table, rfft512->fft_table);

    // rfft_s16(N) always uses DIF_FFT of N/2, same as fft_s16(N/2)
    dl_fft_s16_t *s16_fft128 = dl_fft_s16_init(128, caps);
    dl_fft_s16_t *s16_rfft256 = dl_rfft_s16_init(256, caps);
    TEST_ASSERT_NOT_NULL(s16_fft128);
    TEST_ASSERT_NOT_NULL(s16_rfft256);
    TEST_ASSERT_EQUAL(s16_fft128->fft_table, s16_rfft256->fft_table);

    float *x = (float *)heap_caps_aligned_alloc(16, 128 * 2 * sizeof(float), caps);
    float *rx = (float *)heap_caps_aligned_alloc(16, 256 * sizeof(float), caps);
    float *rgt = (float *)heap_caps_aligned_alloc(16, 256 * sizeof(float), caps);
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(rx);
    TEST_ASSERT_NOT_NULL(rgt);

    memcpy(x, fft_input_128, 128 * 2 * sizeof(float));
    TEST_ASSERT_EQUAL(ESP_OK, dl_fft_f32_run(fft128, x));
    TEST_ASSERT_EQUAL(true, check_fft_results(x, fft_output_128, 128, 90, 1e-3));

    memcpy(rgt, rfft_output_256, 256 * sizeof(float));
    rgt[1] = rfft_output_256[256];
    memcpy(rx, rfft_input_256, 256 * sizeof(float));
    TEST_ASSERT_EQUAL(ESP_OK, dl_rfft_f32_run(rfft256, rx));
    TEST_ASSERT_EQUAL(true, check_fft_results(rx, rgt, 256, 90, 1e-3));

    dl_fft_f32_deinit(fft128);
    // rfft256 still owns the shared 128-point tables
    memcpy(rx, rfft_input_256, 256 * sizeof(float));
    TEST_ASSERT_EQUAL(ESP_OK, dl_rfft_f32_run(rfft256, rx));
    TEST_ASSERT_EQUAL(true, check_fft_results(rx, rgt, 256, 90, 1e-3));

    dl_rfft_f32_deinit(rfft256);
    dl_fft_f32_deinit(fft256);
    dl_rfft_f32_deinit(rfft512);
    dl_fft_s16_deinit(s16_fft128);
    dl_rfft_s16_deinit(s16_rfft256);
    heap_caps_free(x);
    heap_caps_free(rx);
    heap_caps_free(rgt);

    int ram_size_end = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "ram size before: %d, end:%d", ram_size_before, ram_size_end);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_end);
}

TEST_CASE("22. test FFT class concurrent clear", "[dl_fft]")
{
    FFT *fft = FFT::get_instance();
    fft->clear();

    float *data = (float *)heap_caps_aligned_alloc(16, 128 * 2 * sizeof(float), MALLOC_CAP_8BIT);
    TEST_ASSERT_NOT_NULL(data);

    volatile int fail = 0;
    std::thread t_fft([fft, data, &fail]() {
        for (int i = 0; i < 30; i++) {
            memcpy(data, fft_input_128, 128 * 2 * sizeof(float));
            if (fft->fft(data, 128) != ESP_OK) {
                fail = 1;
                break;
            }
            vTaskDelay(1);
        }
    });
    std::thread t_clear([fft]() {
        for (int i = 0; i < 15; i++) {
            fft->clear();
            vTaskDelay(1);
        }
    });

    t_fft.join();
    t_clear.join();
    TEST_ASSERT_EQUAL(0, fail);

    memcpy(data, fft_input_128, 128 * 2 * sizeof(float));
    TEST_ASSERT_EQUAL(ESP_OK, fft->fft(data, 128));
    TEST_ASSERT_EQUAL(true, check_fft_results(data, fft_output_128, 128, 90, 1e-3));

    fft->clear();
    TEST_ASSERT_EQUAL(0, fft->get_handle_count());
    heap_caps_free(data);
}
