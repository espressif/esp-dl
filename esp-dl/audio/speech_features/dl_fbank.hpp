#pragma once

#include "dl_speech_features.hpp"

namespace dl {
namespace audio {

/**
 * @brief Fbank (Filter Bank) features extraction class
 */
class Fbank : public SpeechFeatureBase {
private:
    // Fbank specific parameters
    dl_fft_f32_t *m_fft_config; /*!< FFT configuration */
    mel_filter_t *m_mel_filter; /*!< Mel filterbank coefficients */
    float *m_win_func;          /*!< Window function coefficients */
    float *m_cache;             /*!< Cache buffer for intermediate computations */

public:
    /**
     * @brief Construct a new Fbank object
     *
     * @param config Speech feature configuration
     * @param caps Memory allocation capabilities
     */
    Fbank(const SpeechFeatureConfig config, uint32_t caps = MALLOC_CAP_DEFAULT) : SpeechFeatureBase(config, caps)
    {
        m_fft_config = dl_rfft_f32_init(m_fft_size, caps);
        m_cache =
            (float *)heap_caps_aligned_alloc(16, sizeof(float) * m_fft_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        m_win_func = win_func_init(config.window_type, m_win_len);
        m_feature_dim = config.use_energy ? config.num_mel_bins + 1 : config.num_mel_bins;
        m_mel_filter = mel_filter_init(
            m_fft_size, config.num_mel_bins, config.low_freq, config.high_freq, config.sample_rate, m_caps);
    }

    /**
     * @brief Destroy the Fbank object
     */
    ~Fbank()
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
    }

    /**
     * @brief Process a single frame of float audio data
     *
     * @param input Input audio data
     * @param win_len Window length
     * @param output Output Fbank features
     * @param prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process_frame(const float *input, int win_len, float *output, float prev = 0) override;

    /**
     * @brief Process a single frame of int16 audio data
     *
     * @param input Input audio data
     * @param win_len Window length
     * @param output Output Fbank features
     * @param prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process_frame(const int16_t *input, int win_len, float *output, int16_t prev = 0) override;
};

} // namespace audio
} // namespace dl
