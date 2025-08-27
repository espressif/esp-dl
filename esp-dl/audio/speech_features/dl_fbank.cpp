#include "dl_fbank.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace dl {
namespace audio {

esp_err_t Fbank::process_frame(const float *input, int win_len, float *output, float prev)
{
    if (input == nullptr || output == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    if (m_cache != input) {
        memcpy(m_cache, input, sizeof(float) * win_len);
    }

    if (m_config.remove_dc_offset) {
        remove_dc_offset(m_cache, win_len);
    }

    if (m_config.raw_energy && m_config.use_energy) {
        output[0] = compute_energy(m_cache, win_len, m_config.log_epsilon);
        output += 1;
    }

    apply_preemphasis(m_cache, win_len, m_config.preemphasis, prev);

    apply_window(m_cache, win_len, m_win_func);

    if (!m_config.raw_energy && m_config.use_energy) {
        output[0] = compute_energy(m_cache, win_len, m_config.log_epsilon);
        output += 1;
    }

    if (win_len < m_fft_size) {
        memset(m_cache + win_len, 0, sizeof(float) * (m_fft_size - win_len));
    }

    dl_rfft_f32_run(m_fft_config, m_cache);

    compute_spectrum(m_cache, m_fft_size, m_config.use_power);

    mel_dotprod(m_cache, m_mel_filter, output);

    if (m_config.use_log_fbank == 1) {
        float epsilon = m_config.log_epsilon;
        for (int j = 0; j < m_config.num_mel_bins; j++) output[j] = logf(MAX(output[j], epsilon));
    } else if (m_config.use_log_fbank == 2) {
        float epsilon = m_config.log_epsilon;
        for (int j = 0; j < m_config.num_mel_bins; j++) output[j] = logf(output[j] + epsilon);
    }

    return ESP_OK;
}

esp_err_t Fbank::process_frame(const int16_t *input, int win_len, float *output, int16_t prev)
{
    if (input == nullptr || output == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    for (int i = 0; i < win_len; i++) {
        m_cache[i] = input[i] / 32768.0f;
    }

    return process_frame(m_cache, win_len, output, prev / 32768.0f);
}

} // namespace audio
} // namespace dl
