#include "dl_fbank.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace dl {
namespace audio {

static int16_t quantize_s16(float value, int output_exponent)
{
    float scaled = ldexpf(value, -output_exponent);
    if (scaled >= 32767.0f) {
        return 32767;
    }
    if (scaled <= -32768.0f) {
        return -32768;
    }
    return (int16_t)rintf(scaled);
}

static float log_energy_from_sumsq(int64_t sumsq, float epsilon)
{
    if (sumsq <= 0) {
        return logf(epsilon);
    }
    // Samples are compared against the float path, which divides by 32768 first.
    float energy = ldexpf(static_cast<float>(sumsq), -30);
    return logf(DL_MAX(energy, epsilon));
}

} // namespace audio
} // namespace dl

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

    // Use m_cache[0] as prev to match Kaldi's replicate padding: y[0] = x[0] - α·x[0] = (1-α)·x[0]
    apply_preemphasis(m_cache, win_len, m_config.preemphasis, m_cache[0]);

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
        for (int j = 0; j < m_config.num_mel_bins; j++) output[j] = logf(DL_MAX(output[j], epsilon));
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

esp_err_t FbankS16::process_frame(const float *input, int win_len, float *output, float prev)
{
    (void)input;
    (void)win_len;
    (void)output;
    (void)prev;
    return ESP_ERR_NOT_SUPPORTED;
}

esp_err_t FbankS16::process_frame(const int16_t *input, int win_len, float *output, int16_t prev)
{
    (void)input;
    (void)win_len;
    (void)output;
    (void)prev;
    return ESP_ERR_NOT_SUPPORTED;
}

esp_err_t FbankS16::process_frame_int16(
    const int16_t *input, int win_len, int16_t *output, int16_t prev, int output_exponent)
{
    (void)prev;

    if (input == nullptr || output == nullptr || win_len <= 0 || win_len > m_fft_size) {
        return ESP_ERR_INVALID_ARG;
    }
    if (m_fft_s16 == nullptr || m_cache_i32 == nullptr || m_cache_s16 == nullptr || m_mel_filter == nullptr ||
        m_mel_filter->coeff == nullptr) {
        return ESP_ERR_NO_MEM;
    }
    if (m_win_func && m_win_q15 == nullptr) {
        return ESP_ERR_NO_MEM;
    }

    int32_t *x = m_cache_i32;
    for (int i = 0; i < win_len; i++) {
        x[i] = input[i];
    }

    if (m_config.remove_dc_offset) {
        int32_t sum = 0;
        for (int i = 0; i < win_len; i++) {
            sum += x[i];
        }
        int32_t mean = (int32_t)(sum / win_len);
        for (int i = 0; i < win_len; i++) {
            x[i] -= mean;
        }
    }

    if (m_config.raw_energy && m_config.use_energy) {
        int64_t sumsq = 0;
        for (int i = 0; i < win_len; i++) {
            sumsq += (int64_t)x[i] * x[i];
        }
        output[0] = quantize_s16(log_energy_from_sumsq(sumsq, m_config.log_epsilon), output_exponent);
        output += 1;
    }

    float preemph = m_config.preemphasis;
    if (preemph > 1e-7f) {
        for (int i = win_len - 1; i >= 1; i--) {
            x[i] -= (int32_t)rintf(preemph * (float)x[i - 1]);
        }
        // Replicate the first sample, matching the float path.
        x[0] -= (int32_t)rintf(preemph * (float)x[0]);
    }

    if (m_win_q15) {
        for (int i = 0; i < win_len; i++) {
            x[i] = (int32_t)(((int64_t)x[i] * m_win_q15[i] + 16384) >> 15);
        }
    }

    if (!m_config.raw_energy && m_config.use_energy) {
        int64_t sumsq = 0;
        for (int i = 0; i < win_len; i++) {
            sumsq += (int64_t)x[i] * x[i];
        }
        output[0] = quantize_s16(log_energy_from_sumsq(sumsq, m_config.log_epsilon), output_exponent);
        output += 1;
    }

    int16_t *fft_in = m_cache_s16;
    for (int i = 0; i < win_len; i++) {
        fft_in[i] = (int16_t)x[i];
    }
    if (win_len < m_fft_size) {
        memset(fft_in + win_len, 0, sizeof(int16_t) * (m_fft_size - win_len));
    }

    // Input is int16 and the window is <= 1, so the time signal stays in int16.
    // y = fft_in * 2^(-15), matching the float path's /32768 scaling.
    int in_exponent = -15;
    int fft_exp = 0;
    esp_err_t ret = dl_rfft_s16_hp_run(m_fft_s16, fft_in, in_exponent, &fft_exp);
    if (ret != ESP_OK) {
        return ret;
    }

    int spect_len = m_fft_size / 2 + 1;
    int32_t *spec = m_cache_i32;
    int spec_exp_q15 = 0;
    if (m_config.use_power) {
        uint32_t nyquist = (uint32_t)((int32_t)fft_in[1] * fft_in[1]);
        spec[0] = (int32_t)((int32_t)fft_in[0] * fft_in[0]);
        for (int i = 1; i < spect_len - 1; i++) {
            int32_t re = fft_in[i * 2];
            int32_t im = fft_in[i * 2 + 1];
            uint32_t p = (uint32_t)(re * re) + (uint32_t)(im * im);
            spec[i] = (int32_t)(p > 0x7fffffffu ? 0x7fffffffu : p);
        }
        spec[spect_len - 1] = (int32_t)(nyquist > 0x7fffffffu ? 0x7fffffffu : nyquist);
        spec_exp_q15 = 2 * fft_exp - 15;
    } else {
        uint32_t nyquist = (uint32_t)(fft_in[1] < 0 ? -fft_in[1] : fft_in[1]);
        spec[0] = fft_in[0] < 0 ? -fft_in[0] : fft_in[0];
        for (int i = 1; i < spect_len - 1; i++) {
            int32_t re = fft_in[i * 2];
            int32_t im = fft_in[i * 2 + 1];
            uint32_t p = (uint32_t)(re * re) + (uint32_t)(im * im);
            spec[i] = (int32_t)sqrtf((float)p);
        }
        spec[spect_len - 1] = (int32_t)nyquist;
        spec_exp_q15 = fft_exp - 15;
    }

    const int16_t *coeff = m_mel_filter->coeff;
    const int *bank_pos = m_mel_filter->bank_pos;
    int coeff_shift = 0;
    float epsilon = m_config.log_epsilon;
    for (int j = 0; j < m_mel_filter->nfilter; j++) {
        int start = bank_pos[j * 2];
        int len = bank_pos[j * 2 + 1] - start + 1;
        int64_t acc = 0;
        for (int k = 0; k < len; k++) {
            acc += (int64_t)coeff[coeff_shift + k] * spec[start + k];
        }
        coeff_shift += len;

        float mel = acc > 0 ? ldexpf(static_cast<float>(acc), spec_exp_q15) : 0.0f;
        float value = mel;
        if (m_config.use_log_fbank == 1) {
            value = logf(DL_MAX(mel, epsilon));
        } else if (m_config.use_log_fbank == 2) {
            value = logf(mel + epsilon);
        }
        output[j] = quantize_s16(value, output_exponent);
    }

    return ESP_OK;
}

} // namespace audio
} // namespace dl
