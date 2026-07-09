#include "dl_audio_wav.hpp"
#include "dl_fft.h"
#include "dl_fft.hpp"
#include "dl_fft_base.h"
#include "dl_rfft.h"
#include "test_fft.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

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
