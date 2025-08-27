#include "dl_mfcc.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace dl {
namespace audio {

float *MFCC::gen_dct_matrix(int num_rows, int num_cols, uint32_t caps)
{
    // this function is copied from
    // https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/matrix-functions.cc#L592

    std::vector<float> ans(num_rows * num_cols);

    float *p = (float *)heap_caps_aligned_alloc(16, sizeof(float) * num_rows * num_cols, caps);
    if (p == nullptr) {
        return p;
    }

    float normalizer = sqrtf(1.0 / num_cols); // normalizer for X_0

    for (int32_t i = 0; i != num_cols; ++i) {
        p[i] = normalizer;
    }

    normalizer = sqrtf(2.0 / num_cols); // normalizer for other elements

    for (int32_t k = 1; k != num_rows; ++k) {
        for (int32_t n = 0; n != num_cols; ++n) {
            *(p + k * num_cols + n) = normalizer * cos(static_cast<double>(M_PI) / num_cols * (n + 0.5) * k);
        }
    }

    return p;
}

float *MFCC::gen_lifter_coeffs(float Q, int len, uint32_t caps)
{
    // Compute liftering coefficients (scaling on cepstral coeffs)
    // coeffs are numbered slightly differently from HTK: the zeroth
    // index is C0, which is not affected.
    float *coeffs = (float *)heap_caps_aligned_alloc(16, sizeof(float) * len, caps);

    for (int32_t i = 0; i != len; ++i) {
        coeffs[i] = 1.0 + 0.5 * Q * sin(M_PI * i / Q);
    }

    return coeffs;
}

esp_err_t MFCC::process_frame(const float *input, int win_len, float *output, float prev)
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

    mel_dotprod(m_cache, m_mel_filter, m_cache);

    if (m_config.use_log_fbank == 1) {
        float epsilon = m_config.log_epsilon;
        for (int j = 0; j < m_config.num_mel_bins; j++) m_cache[j] = logf(MAX(m_cache[j], epsilon));
    } else if (m_config.use_log_fbank == 2) {
        float epsilon = m_config.log_epsilon;
        for (int j = 0; j < m_config.num_mel_bins; j++) m_cache[j] = logf(m_cache[j] + epsilon);
    }

    // feature = dct_matrix_ * mel_energies [which now have log]
    for (int32_t i = 0; i != m_config.num_ceps; ++i) {
        output[i] = dotprod_f32(m_dct_matrix + i * m_config.num_mel_bins, m_cache, m_config.num_mel_bins);
    }

    if (m_lifter_coeffs) {
        for (int32_t i = 0; i != m_config.num_ceps; ++i) {
            output[i] *= m_lifter_coeffs[i];
        }
    }

    return ESP_OK;
}

esp_err_t MFCC::process_frame(const int16_t *input, int win_len, float *output, int16_t prev)
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
