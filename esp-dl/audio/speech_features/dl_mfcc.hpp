#pragma once

#include "dl_speech_features.hpp"

namespace dl {
namespace audio {

/**
 * @brief MFCC (Filter Bank) features extraction class
 */
class MFCC : public SpeechFeatureBase {
private:
    // MFCC specific parameters
    dl_fft_f32_t *m_fft_config; /*!< FFT configuration */
    mel_filter_t *m_mel_filter; /*!< Mel filterbank coefficients */
    float *m_win_func;          /*!< Window function coefficients */
    float *m_cache;             /*!< Cache buffer for intermediate computations */
    float *m_dct_matrix;        /*!< DCT matrix */
    float *m_lifter_coeffs;     /*!< Lifter coefficients*/

    float *gen_dct_matrix(int num_rows, int num_cols, uint32_t caps);
    float *gen_lifter_coeffs(float Q, int len, uint32_t caps);

public:
    /**
     * @brief Construct a new MFCC object
     *
     * @param config Speech feature configuration
     * @param caps Memory allocation capabilities
     */
    MFCC(const SpeechFeatureConfig config, uint32_t caps = MALLOC_CAP_DEFAULT) : SpeechFeatureBase(config, caps)
    {
        m_fft_config = dl_rfft_f32_init(m_fft_size, caps);
        m_cache =
            (float *)heap_caps_aligned_alloc(16, sizeof(float) * m_fft_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        m_win_func = win_func_init(m_config.window_type, m_win_len);
        m_mel_filter = mel_filter_init(
            m_fft_size, m_config.num_mel_bins, m_config.low_freq, m_config.high_freq, m_config.sample_rate, m_caps);
        m_feature_dim = m_config.use_energy ? m_config.num_ceps + 1 : m_config.num_ceps;

        if (m_config.num_ceps <= m_config.num_mel_bins) {
            m_dct_matrix = gen_dct_matrix(m_config.num_ceps, m_config.num_mel_bins, caps);
            if (m_config.cepstral_lifter != 0) {
                m_lifter_coeffs = gen_lifter_coeffs(m_config.cepstral_lifter, m_config.num_ceps, caps);
            } else {
                m_lifter_coeffs = nullptr;
            }

        } else {
            m_dct_matrix = nullptr;
            m_lifter_coeffs = nullptr;
            ESP_LOGE("MFCC",
                     "num_ceps (%d) must be less than or equal to num_mel_bins (%d)",
                     m_config.num_ceps,
                     m_config.num_mel_bins);
            assert(0);
        }
        m_config.use_power = true;  // allgned with torchaudio.compliance.kaldi.mfcc
        m_config.use_log_fbank = 1; // allgned with torchaudio.compliance.kaldi.mfcc
    }

    /**
     * @brief Destroy the MFCC object
     */
    ~MFCC()
    {
        if (m_fft_config) {
            dl_rfft_f32_deinit(m_fft_config);
            m_fft_config = nullptr;
        }

        if (m_cache) {
            free(m_cache);
        }

        if (m_win_func) {
            free(m_win_func);
        }

        if (m_mel_filter) {
            mel_filter_deinit(m_mel_filter);
        }

        if (m_dct_matrix) {
            free(m_dct_matrix);
        }

        if (m_lifter_coeffs) {
            free(m_lifter_coeffs);
        }
    }

    /**
     * @brief Process a single frame of float audio data
     *
     * @param input Input audio data
     * @param win_len Window length
     * @param output Output MFCC features
     * @param prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process_frame(const float *input, int win_len, float *output, float prev = 0) override;

    /**
     * @brief Process a single frame of int16 audio data
     *
     * @param input Input audio data
     * @param win_len Window length
     * @param output Output MFCC features
     * @param prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process_frame(const int16_t *input, int win_len, float *output, int16_t prev = 0) override;
};

} // namespace audio
} // namespace dl
