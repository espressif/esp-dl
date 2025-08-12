#include "dl_fbank.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace dl {
namespace audio {

int get_frame_num(int input_len, int win_len, int win_step)
{
    int num_frames = 0;
    if (input_len >= win_len) {
        num_frames = 1 + (input_len - win_len) / win_step;
    }
    return num_frames;
}

esp_err_t FBank::process_frame(const float *input, int win_len, float *output, float prev)
{
    if (input == nullptr || output == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    apply_window_and_preemphasis(input, win_len, m_cache, m_config.preemphasis, m_win_func, prev);

    compute_spectrum(m_fft_handle, m_cache, win_len, m_config.use_power);

    mel_dotprod(m_cache, m_mel_filter, output);

    if (m_config.use_log_fbank == 1) {
        float epsilon = m_config.log_epsilon;
        for (int j = 0; j < m_config.num_mel_bins; j++) output[j] = logf(output[j] + epsilon);
    } else if (m_config.use_log_fbank == 2) {
        float epsilon = m_config.log_epsilon;
        for (int j = 0; j < m_config.num_mel_bins; j++) output[j] = logf(MAX(output[j], epsilon));
    }
    return ESP_OK;
}

esp_err_t FBank::process_frame(const int16_t *input, int win_len, float *output, int16_t prev)
{
    if (input == nullptr || output == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    for (int i = 0; i < win_len; i++) {
        m_cache[i] = input[i] / 32768.0f;
    }

    return process_frame(m_cache, win_len, output, prev / 32768.0f);
}

esp_err_t FBank::process(const float *input, int input_len, float *output)
{
    if (input == nullptr || output == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    int feature_dim = m_config.num_mel_bins;

    // Calculate number of frames
    if (input_len < m_win_len) {
        return ESP_ERR_INVALID_ARG;
    }

    int num_frames = get_frame_num(input_len, m_win_len, m_win_step);

    // Process each frame
    for (int i = 0; i < num_frames; i++) {
        int start = i * m_win_step;
        float prev = 0;
        if (start > 0) {
            prev = input[start - 1];
        }
        esp_err_t ret = process_frame(input + start, m_win_len, output + i * feature_dim, prev);
        if (ret != ESP_OK) {
            return ret;
        }
    }

    return ESP_OK;
}

esp_err_t FBank::process(const int16_t *input, int input_len, float *output)
{
    if (input == nullptr || output == nullptr || input_len <= 0) {
        return ESP_ERR_INVALID_ARG;
    }

    int feature_dim = m_config.num_mel_bins;

    // Calculate number of frames
    if (input_len < m_win_len) {
        return ESP_ERR_INVALID_ARG;
    }

    int num_frames = get_frame_num(input_len, m_win_len, m_win_step);

    // Process each frame
    for (int i = 0; i < num_frames; i++) {
        int start = i * m_win_step;
        float prev = 0;
        if (start > 0) {
            prev = input[start - 1];
        }
        esp_err_t ret = process_frame(input + start, m_win_len, output + i * feature_dim, prev);
        if (ret != ESP_OK) {
            return ret;
        }
    }

    return ESP_OK;
}

std::vector<int> FBank::get_output_shape(int input_len)
{
    // Calculate number of frames
    int num_frames = get_frame_num(input_len, m_win_len, m_win_step);

    // Return shape as [num_frames, num_mel_bins]
    return std::vector<int>{num_frames, m_config.num_mel_bins};
}

} // namespace audio
} // namespace dl
