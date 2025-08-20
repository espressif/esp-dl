#pragma once

#include "dl_speech_features.hpp"

namespace dl {
namespace audio {

/**
 * @brief Spectrogram features extraction class
 *        Allgned with torchaudio.compliance.kaldi.spectrogram
 *        Note: this implementation of spectrogram has fixed energy_floor = 0.0
 */
class Spectrogram : public SpeechFeatureBase {
private:
    // Spectrogram specific parameters
    dl_fft_f32_t *m_fft_config; /*!< FFT configuration */
    float *m_win_func;          /*!< Window function coefficients */
    float *m_cache;             /*!< Cache buffer for intermediate computations */

public:
    /**
     * @brief Construct a new Spectrogram object
     *
     * @param config Speech feature configuration
     * @param caps Memory allocation capabilities
     */
    Spectrogram(const SpeechFeatureConfig config, uint32_t caps = MALLOC_CAP_DEFAULT) : SpeechFeatureBase(config, caps)
    {
        m_fft_config = dl_rfft_f32_init(m_fft_size, caps);
        m_feature_dim = m_fft_size / 2 + 1;
        m_cache =
            (float *)heap_caps_aligned_alloc(16, sizeof(float) * m_fft_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        assert(m_cache != nullptr);
        m_win_func = win_func_init(config.window_type, m_win_len);

        m_config.use_power = true;  // allgned with torchaudio.compliance.kaldi.spectrogram
        m_config.use_log_fbank = 1; // allgned with torchaudio.compliance.kaldi.spectrogram
    }

    /**
     * @brief Destroy the Spectrogram object
     */
    ~Spectrogram()
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
    }

    /**
     * @brief Process a single frame of float audio data
     *
     * @param input Input audio data
     * @param win_len Window length
     * @param output Output Spectrogram features
     * @param prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process_frame(const float *input, int win_len, float *output, float prev = 0) override;

    /**
     * @brief Process a single frame of int16 audio data
     *
     * @param input Input audio data
     * @param win_len Window length
     * @param output Output Spectrogram features
     * @param prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process_frame(const int16_t *input, int win_len, float *output, int16_t prev = 0) override;
};

} // namespace audio
} // namespace dl
