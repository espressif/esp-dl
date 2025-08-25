// fbank_bin_v2.hpp
#include "dl_audio_wav.hpp"
#include "dl_fbank.hpp"
#include "dl_mfcc.hpp"
#include "dl_spectrogram.hpp"
#include "esp_timer.h"
#include "stdio.h"
#include "stdlib.h"
#include "unity.h"
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
    TEST_ASSERT_EQUAL(true, ram_size_before == ram_size_after);
    printf("ram size before: %d\n", ram_size_before);
    printf("ram size after: %d\n", ram_size_after);
}
