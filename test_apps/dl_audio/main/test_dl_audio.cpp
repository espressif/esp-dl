// fbank_bin_v2.hpp
#include "dl_audio_wav.hpp"
#include "dl_fbank.hpp"
#include "dl_mfcc.hpp"
#include "dl_spectrogram.hpp"
#include "esp_timer.h"
#include "stdio.h"
#include "stdlib.h"
#include "unity.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

extern const uint8_t test_wav_start[] asm("_binary_test_wav_start");
extern const uint8_t test_wav_end[] asm("_binary_test_wav_end");
extern const uint8_t test_fbank_bin_start[] asm("_binary_test_fbank_bin_start");
extern const uint8_t test_fbank_bin_end[] asm("_binary_test_fbank_bin_end");
extern const uint8_t test_spectrogram_bin_start[] asm("_binary_test_spectrogram_bin_start");
extern const uint8_t test_spectrogram_bin_end[] asm("_binary_test_spectrogram_bin_end");
extern const uint8_t test_mfcc_bin_start[] asm("_binary_test_mfcc_bin_start");
extern const uint8_t test_mfcc_bin_end[] asm("_binary_test_mfcc_bin_end");
using namespace dl::audio;

struct SpeechFeatureCase {
    SpeechFeatureConfig cfg;
    uint32_t T;
    uint32_t D;
    float *data;
};

static inline uint32_t read_u32_LE(const uint8_t *&p)
{
    uint32_t v = *reinterpret_cast<const uint32_t *>(p);
    p += 4;
    return v;
}

static inline float read_f32_LE(const uint8_t *&p)
{
    float v = *reinterpret_cast<const float *>(p);
    p += 4;
    return v;
}

uint32_t get_case_num(const uint8_t *start, const uint8_t *end)
{
    const uint8_t *p = start;
    uint32_t magic = read_u32_LE(p);
    if (magic != 0xFBA5FBA5) {
        printf("bad magic\n");
        return 0;
    }
    return read_u32_LE(p);
}

SpeechFeatureCase *load_test_case(const uint8_t *start, int idx)
{
    const uint8_t *p = start;
    uint32_t magic = read_u32_LE(p);
    if (magic != 0xFBA5FBA5) {
        printf("bad magic\n");
        assert(0);
    }
    uint32_t n_cases = read_u32_LE(p);
    if (idx >= n_cases) {
        printf("invalid test case index\n");
        assert(0);
    }

    char window_name[16];
    for (int i = 0; i < idx; ++i) {
        p += 56; // Skip config
        uint32_t T = read_u32_LE(p);
        uint32_t D = read_u32_LE(p);
        p += T * D * sizeof(float); // Skip data
    }

    SpeechFeatureCase *c = new SpeechFeatureCase();
    auto &cfg = c->cfg;
    cfg.frame_shift = read_f32_LE(p);
    cfg.frame_length = read_f32_LE(p);
    cfg.low_freq = read_f32_LE(p);
    cfg.high_freq = read_f32_LE(p);
    cfg.preemphasis = read_f32_LE(p);
    cfg.num_mel_bins = read_u32_LE(p);
    cfg.num_ceps = read_u32_LE(p);
    cfg.use_power = static_cast<bool>(read_u32_LE(p));
    cfg.use_log_fbank = static_cast<bool>(read_u32_LE(p));
    cfg.remove_dc_offset = static_cast<bool>(read_u32_LE(p));
    memcpy(window_name, p, 16);
    cfg.window_type = win_type_from_string(window_name);
    p += 16;

    c->T = read_u32_LE(p);
    c->D = read_u32_LE(p);

    size_t bytes_needed = size_t(c->T) * c->D * sizeof(float);
    c->data = (float *)malloc(bytes_needed);
    if (c->data == nullptr) {
        printf("malloc failed for data\n");
        assert(0);
    }
    memcpy(c->data, p, bytes_needed);

    return c;
}

bool check_is_same(float *x, int size, float *gt, float avg_error = 1e-4, float max_error = 1e-2)
{
    float sum = 0;

    for (int i = 0; i < size; ++i) {
        float err = fabsf(x[i] - gt[i]);
        if (err > max_error) {
            printf("check_is_same: x[%d] = %.6f, gt[%d] = %.6f\n", i, x[i], i, gt[i]);
            return false;
        }
        sum += err;
    }
    sum /= size;

    if (sum > avg_error) {
        printf("check_is_same: avg_error=%.6f\n", sum);
        return false;
    }
    return true;
}

TEST_CASE("1. test dl spectrogram", "[dl_audio]")
{
    int cases_num = get_case_num(test_spectrogram_bin_start, test_spectrogram_bin_end);
    dl_audio_t *input = decode_wav(test_wav_start, test_wav_end - test_wav_start);
    print_audio_info(input);
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start, stop;

    for (size_t i = 0; i < cases_num; ++i) {
        SpeechFeatureCase *c = load_test_case(test_spectrogram_bin_start, i);

        SpeechFeatureBase *handle = new Spectrogram(c->cfg);
        handle->print_config();
        std::vector<int> shape = handle->get_output_shape(input->length);
        int size = shape[0] * shape[1];
        if (size > 0) {
            float *output = (float *)malloc(shape[0] * shape[1] * sizeof(float));
            printf("shape of output feature is %d x %d\n", shape[0], shape[1]);
            if (output == nullptr) {
                printf("error: malloc\n");
                exit(-1);
            }
            start = esp_timer_get_time();
            handle->process(input->data, input->length, output);
            stop = esp_timer_get_time();
            TEST_ASSERT_EQUAL(true, check_is_same(output, size, c->data));
            printf("test %d pass, time:%ld us \n\n", i, stop - start);
            free(output);
        }
        delete handle;
        free(c->data);
        delete c;
    }
    int ram_size_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, ram_size_before - ram_size_after < 240);
    printf("ram size before: %d\n", ram_size_before);
    printf("ram size after: %d\n", ram_size_after);
}

TEST_CASE("2. test dl fbank", "[dl_audio]")
{
    int cases_num = get_case_num(test_fbank_bin_start, test_fbank_bin_end);
    dl_audio_t *input = decode_wav(test_wav_start, test_wav_end - test_wav_start);
    print_audio_info(input);
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start, stop;

    for (size_t i = 0; i < cases_num; ++i) {
        SpeechFeatureCase *c = load_test_case(test_fbank_bin_start, i);

        Fbank *handle = new Fbank(c->cfg);
        handle->print_config();
        std::vector<int> shape = handle->get_output_shape(input->length);
        int size = shape[0] * shape[1];
        if (size > 0) {
            float *output = (float *)malloc(shape[0] * shape[1] * sizeof(float));
            printf("shape of output feature is %d x %d\n", shape[0], shape[1]);
            if (output == nullptr) {
                printf("error: malloc\n");
                exit(-1);
            }

            start = esp_timer_get_time();
            handle->process(input->data, input->length, output);
            stop = esp_timer_get_time();
            TEST_ASSERT_EQUAL(true, check_is_same(output, size, c->data));
            printf("test %d pass, time:%ld us \n\n", i, stop - start);
            free(output);
        }
        delete handle;
        free(c->data);
        delete c;
    }

    int ram_size_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_after);
    printf("ram size before: %d\n", ram_size_before);
    printf("ram size after: %d\n", ram_size_after);
}

TEST_CASE("3. test dl mfcc", "[dl_audio]")
{
    int cases_num = get_case_num(test_mfcc_bin_start, test_mfcc_bin_end);
    dl_audio_t *input = decode_wav(test_wav_start, test_wav_end - test_wav_start);
    print_audio_info(input);
    int ram_size_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    uint32_t start, stop;

    for (size_t i = 0; i < cases_num; ++i) {
        SpeechFeatureCase *c = load_test_case(test_mfcc_bin_start, i);

        MFCC *handle = new MFCC(c->cfg);
        handle->print_config();
        std::vector<int> shape = handle->get_output_shape(input->length);
        int size = shape[0] * shape[1];
        if (size > 0) {
            float *output = (float *)malloc(shape[0] * shape[1] * sizeof(float));
            printf("shape of output feature is %d x %d\n", shape[0], shape[1]);
            if (output == nullptr) {
                printf("error: malloc\n");
                exit(-1);
            }

            start = esp_timer_get_time();
            handle->process(input->data, input->length, output);
            stop = esp_timer_get_time();
            TEST_ASSERT_EQUAL(true, check_is_same(output, size, c->data));
            printf("test %d pass, time:%ld us \n\n", i, stop - start);
            free(output);
        }
        delete handle;
        free(c->data);
        delete c;
    }

    int ram_size_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    TEST_ASSERT_EQUAL(true, ram_size_before - ram_size_after < 240);
    printf("ram size before: %d\n", ram_size_before);
    printf("ram size after: %d\n", ram_size_after);
}

TEST_CASE("4. test dl fbank int16", "[dl_audio]")
{
    dl_audio_t *input = decode_wav(test_wav_start, test_wav_end - test_wav_start);
    print_audio_info(input);

    SpeechFeatureConfig cfg;
    cfg.sample_rate = input->sample_rate;
    cfg.frame_length = 25;
    cfg.frame_shift = 10;
    cfg.num_mel_bins = 80;
    cfg.window_type = WinType::HANNING;
    cfg.preemphasis = 0.97f;
    cfg.low_freq = 20.0f;
    cfg.high_freq = 0.0f;
    cfg.use_log_fbank = 1;
    cfg.use_power = true;
    cfg.remove_dc_offset = true;

    const int output_exponent = -10;
    Fbank *handle = new Fbank(cfg);
    FbankS32 *handle_s32 = new FbankS32(cfg);
    std::vector<int> shape = handle->get_output_shape(input->length);
    int frames = shape[0];
    int dim = shape[1];
    TEST_ASSERT_TRUE(frames > 0);

    float *ref = (float *)malloc(frames * dim * sizeof(float));
    int16_t *out = (int16_t *)malloc(frames * dim * sizeof(int16_t));
    TEST_ASSERT_NOT_NULL(ref);
    TEST_ASSERT_NOT_NULL(out);

    int win_len = cfg.frame_length * cfg.sample_rate / 1000;
    int win_step = cfg.frame_shift * cfg.sample_rate / 1000;

    uint32_t t0 = esp_timer_get_time();
    TEST_ASSERT_EQUAL(ESP_OK, handle->process(input->data, input->length, ref));
    uint32_t t_float = esp_timer_get_time() - t0;

    const int16_t *pcm = input->data;
    t0 = esp_timer_get_time();
    for (int i = 0; i < frames; i++) {
        esp_err_t ret = handle_s32->process_frame_int16(pcm, win_len, out + i * dim, pcm[0], output_exponent);
        TEST_ASSERT_EQUAL(ESP_OK, ret);
        pcm += win_step;
    }
    uint32_t t_int16 = esp_timer_get_time() - t0;

    float scale = ldexpf(1.0f, output_exponent);
    int n = frames * dim;
    float *err_q = (float *)malloc(n * sizeof(float));
    TEST_ASSERT_NOT_NULL(err_q);

    auto quantize = [&](float value) -> int16_t {
        float scaled = ldexpf(value, -output_exponent);
        if (scaled >= 32767.0f) {
            return 32767;
        }
        if (scaled <= -32768.0f) {
            return -32768;
        }
        return (int16_t)rintf(scaled);
    };

    float max_err = 0.0f;
    float max_err_active = 0.0f;
    double sum_err = 0.0;
    int n_active = 0;
    int worst[5] = {-1, -1, -1, -1, -1};
    float worst_err[5] = {0, 0, 0, 0, 0};
    // Bins within 1.0 of log(epsilon) sit on the log floor. The rest carry the feature.
    const float floor_level = logf(cfg.log_epsilon) + 1.0f;
    for (int i = 0; i < n; i++) {
        int16_t qref = quantize(ref[i]);
        float y = (float)out[i] * scale;
        float y_ref = (float)qref * scale;
        float err = fabsf(y - y_ref);
        err_q[i] = err;
        sum_err += err;
        if (err > max_err) {
            max_err = err;
        }
        if (ref[i] > floor_level) {
            n_active++;
            if (err > max_err_active) {
                max_err_active = err;
            }
        }
        for (int k = 0; k < 5; k++) {
            if (err > worst_err[k]) {
                for (int s = 4; s > k; s--) {
                    worst[s] = worst[s - 1];
                    worst_err[s] = worst_err[s - 1];
                }
                worst[k] = i;
                worst_err[k] = err;
                break;
            }
        }
    }

    qsort(err_q, n, sizeof(float), [](const void *a, const void *b) -> int {
        float da = *(const float *)a;
        float db = *(const float *)b;
        return (da > db) - (da < db);
    });
    auto percentile = [&](float p) -> float {
        int idx = (int)((n - 1) * p);
        if (idx < 0) {
            idx = 0;
        }
        if (idx >= n) {
            idx = n - 1;
        }
        return err_q[idx];
    };

    float avg_err = (float)(sum_err / n);
    printf("quantized compare: exponent=%d scale=%.6f floor<%.3f active=%d/%d\n",
           output_exponent,
           scale,
           floor_level,
           n_active,
           n);
    printf("  max_err=%.4f  bottleneck_err(active max)=%.4f  p99=%.4f  avg=%.4f\n",
           max_err,
           max_err_active,
           percentile(0.99f),
           avg_err);
    for (int k = 0; k < 5; k++) {
        int i = worst[k];
        if (i < 0) {
            break;
        }
        int16_t qref = quantize(ref[i]);
        printf("  worst[%d] frame=%d bin=%d float=%.4f qfloat=%.4f int16=%.4f err=%.4f\n",
               k,
               i / dim,
               i % dim,
               ref[i],
               (float)qref * scale,
               (float)out[i] * scale,
               worst_err[k]);
    }
    printf("fbank time float=%lu us  int16=%lu us  speedup=%.2fx\n",
           (unsigned long)t_float,
           (unsigned long)t_int16,
           t_int16 > 0 ? (double)t_float / (double)t_int16 : 0.0);

    // s32 rFFT residual stays inside one output code. scale is 2^output_exponent.
    TEST_ASSERT_TRUE(avg_err < scale);
    TEST_ASSERT_TRUE(max_err <= scale);
    free(err_q);

    free(ref);
    free(out);
    delete handle;
    delete handle_s32;
}

static void run_constant_fbank_s32(const char *name, int16_t sample, bool remove_dc)
{
    const int output_exponent = -10;
    const int win_len = 400;
    const int dim = 80;
    int16_t *input = (int16_t *)malloc(win_len * sizeof(int16_t));
    float *input_f = (float *)malloc(win_len * sizeof(float));
    float *ref = (float *)malloc(dim * sizeof(float));
    int16_t *out = (int16_t *)malloc(dim * sizeof(int16_t));
    TEST_ASSERT_NOT_NULL(input);
    TEST_ASSERT_NOT_NULL(input_f);
    TEST_ASSERT_NOT_NULL(ref);
    TEST_ASSERT_NOT_NULL(out);
    for (int i = 0; i < win_len; i++) {
        input[i] = sample;
        input_f[i] = sample / 32768.0f;
    }

    SpeechFeatureConfig cfg;
    cfg.sample_rate = 16000;
    cfg.frame_length = 25;
    cfg.frame_shift = 10;
    cfg.num_mel_bins = dim;
    cfg.window_type = WinType::HANNING;
    cfg.preemphasis = 0.97f;
    cfg.low_freq = 20.0f;
    cfg.use_log_fbank = 1;
    cfg.use_power = true;
    cfg.remove_dc_offset = remove_dc;

    Fbank *f32 = new Fbank(cfg);
    FbankS32 *s32 = new FbankS32(cfg);
    esp_err_t ret_f = f32->process_frame(input_f, win_len, ref, input_f[0]);
    esp_err_t ret_i = s32->process_frame_int16(input, win_len, out, sample, output_exponent);

    float scale = ldexpf(1.0f, output_exponent);
    float max_err = 0.0f;
    int16_t out_min = out[0];
    int16_t out_max = out[0];
    for (int i = 0; i < dim; i++) {
        float err = fabsf((float)out[i] * scale - ref[i]);
        if (err > max_err) {
            max_err = err;
        }
        if (out[i] < out_min) {
            out_min = out[i];
        }
        if (out[i] > out_max) {
            out_max = out[i];
        }
    }
    printf("%s dc=%d ret_f=%d ret_i=%d out=[%d,%d] ref0=%.4f s32_0=%.4f max_err=%.4f\n",
           name,
           (int)remove_dc,
           (int)ret_f,
           (int)ret_i,
           (int)out_min,
           (int)out_max,
           ref[0],
           (float)out[0] * scale,
           max_err);
    TEST_ASSERT_EQUAL(ESP_OK, ret_f);
    TEST_ASSERT_EQUAL(ESP_OK, ret_i);
    delete f32;
    delete s32;
    free(input);
    free(input_f);
    free(ref);
    free(out);
}

TEST_CASE("5. test fbank s32 constant 0 and 1", "[dl_audio]")
{
    run_constant_fbank_s32("all0", 0, true);
    run_constant_fbank_s32("all0", 0, false);
    run_constant_fbank_s32("all1", 1, true);
    run_constant_fbank_s32("all1", 1, false);
}
