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

public:
    /**
     * @brief Construct a new MFCC object
     *
     * @param config Speech feature configuration
     * @param caps Memory allocation capabilities
     */
    MFCC(const SpeechFeatureConfig config, uint32_t caps = MALLOC_CAP_DEFAULT) : SpeechFeatureBase(config)
    {
        m_fft_config = dl_rfft_f32_init(m_fft_size, caps);
        m_cache =
            (float *)heap_caps_aligned_alloc(16, sizeof(float) * m_fft_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        m_win_func = win_func_init(config.window_type, m_win_len);
        m_mel_filter = mel_filter_init(
            m_fft_size, config.num_mel_bins, config.low_freq, config.high_freq, config.sample_rate, m_caps);
        m_feature_dim = config.use_energy ? config.num_cepstral + 1 : config.num_cepstral;
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

    /**
     * @brief Get the output shape for given input length
     *
     * @param input_len Length of input data
     * @return std::vector<int> Output shape as [num_frames, num_mel_bins]
     */
    std::vector<int> get_output_shape(int input_len);
};

} // namespace audio
} // namespace dl
