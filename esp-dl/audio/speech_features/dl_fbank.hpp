#pragma once

#include "dl_speech_features.hpp"

namespace dl {
namespace audio {

/**
 * @brief FBank (Filter Bank) features extraction class
 */
class FBank : public SpeechFeatureBase {
private:
    // FBank specific parameters
    FFT *m_fft_handle;          /*!< FFT handle for spectrum computation */
    mel_filter_t *m_mel_filter; /*!< Mel filterbank coefficients */
    uint32_t m_caps;            /*!< Memory allocation capabilities */
    float *m_win_func;          /*!< Window function coefficients */
    float *m_cache;             /*!< Cache buffer for intermediate computations */
    int m_win_step;             /*!< Window shift*/
    int m_win_len;              /*!< Window length*/

public:
    /**
     * @brief Construct a new FBank object
     *
     * @param config Speech feature configuration
     * @param caps Memory allocation capabilities
     */
    FBank(const SpeechFeatureConfig config, uint32_t caps = MALLOC_CAP_DEFAULT) : SpeechFeatureBase(config)
    {
        m_fft_handle = FFT::get_instance();
        m_caps = caps;
        m_mel_filter = mel_filter_init(
            config.fft_size, config.num_mel_bins, config.low_freq, config.high_freq, config.sample_rate, m_caps);

        m_win_len = config.frame_length * config.sample_rate / 1000;
        m_win_step = config.frame_shift * config.sample_rate / 1000;
        m_cache = (float *)heap_caps_aligned_alloc(16, sizeof(float) * m_win_len, m_caps);
        m_win_func = win_func_init(config.window_type, m_win_len);
    }

    /**
     * @brief Destroy the FBank object
     */
    ~FBank()
    {
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
     * @param output Output FBank features
     * @param prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process_frame(const float *input, int win_len, float *output, float prev = 0) override;

    /**
     * @brief Process a single frame of int16 audio data
     *
     * @param input Input audio data
     * @param win_len Window length
     * @param output Output FBank features
     * @param prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process_frame(const int16_t *input, int win_len, float *output, int16_t prev = 0) override;

    /**
     * @brief Process entire float audio data
     *
     * @param input Input audio data
     * @param input_len Length of input data
     * @param output Output FBank features
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process(const float *input, int input_len, float *output) override;

    /**
     * @brief Process entire int16 audio data
     *
     * @param input Input audio data
     * @param input_len Length of input data
     * @param output Output FBank features
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process(const int16_t *input, int input_len, float *output) override;

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
