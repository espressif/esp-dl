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
     * @param win_len Number of input samples
     * @param output Output Fbank features
     * @param prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process_frame(const float *input, int win_len, float *output, float prev = 0) override;

    /**
     * @brief Process a single frame of int16 audio data
     *
     * @param input Input audio data
     * @param win_len Number of input samples
     * @param output Output Fbank features
     * @param prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process_frame(const int16_t *input, int win_len, float *output, int16_t prev = 0) override;
};

/**
 * @brief Int16 Fbank using the high-precision int16 real FFT.
 *
 * Time-domain steps and the mel dot-product run in int32. Output samples use
 * the caller-supplied exponent: value = output[i] * 2^output_exponent.
 */
class FbankS16 : public SpeechFeatureBase {
private:
    dl_fft_s16_t *m_fft_s16;        /*!< High-precision int16 real FFT */
    mel_filter_s16_t *m_mel_filter; /*!< Q15 mel filterbank */
    float *m_win_func;              /*!< Window function, used to build Q15 coefficients */
    int16_t *m_win_q15;             /*!< Window coefficients in Q15 */
    int32_t *m_cache_i32;           /*!< Integer working buffer */
    int16_t *m_cache_s16;           /*!< In-place int16 FFT buffer */

public:
    /**
     * @brief Construct a new FbankS16 object
     *
     * @param config Speech feature configuration
     * @param caps Memory allocation capabilities
     */
    FbankS16(const SpeechFeatureConfig config, uint32_t caps = MALLOC_CAP_DEFAULT) : SpeechFeatureBase(config, caps)
    {
        m_fft_s16 = dl_rfft_s16_init(m_fft_size, caps);
        m_cache_i32 =
            (int32_t *)heap_caps_aligned_alloc(16, sizeof(int32_t) * m_fft_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        m_cache_s16 =
            (int16_t *)heap_caps_aligned_alloc(16, sizeof(int16_t) * m_fft_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        m_win_func = win_func_init(config.window_type, m_win_len);
        m_feature_dim = config.use_energy ? config.num_mel_bins + 1 : config.num_mel_bins;
        m_mel_filter = mel_filter_s16_init(
            m_fft_size, config.num_mel_bins, config.low_freq, config.high_freq, config.sample_rate, m_caps);

        m_win_q15 = nullptr;
        if (m_win_func) {
            m_win_q15 = (int16_t *)heap_caps_malloc(sizeof(int16_t) * m_win_len, caps);
            if (m_win_q15) {
                for (int i = 0; i < m_win_len; i++) {
                    int32_t q = (int32_t)(m_win_func[i] * 32768.0f + 0.5f);
                    if (q > 32767) {
                        q = 32767;
                    }
                    m_win_q15[i] = (int16_t)q;
                }
            }
        }
    }

    /**
     * @brief Destroy the FbankS16 object
     */
    ~FbankS16()
    {
        if (m_fft_s16) {
            dl_rfft_s16_deinit(m_fft_s16);
            m_fft_s16 = nullptr;
        }
        if (m_cache_i32) {
            free(m_cache_i32);
        }
        if (m_cache_s16) {
            free(m_cache_s16);
        }
        if (m_win_func) {
            free(m_win_func);
        }
        if (m_win_q15) {
            free(m_win_q15);
        }
        if (m_mel_filter) {
            mel_filter_s16_deinit(m_mel_filter);
        }
    }

    esp_err_t process_frame(const float *input, int win_len, float *output, float prev = 0) override;
    esp_err_t process_frame(const int16_t *input, int win_len, float *output, int16_t prev = 0) override;
    esp_err_t process_frame_int16(
        const int16_t *input, int win_len, int16_t *output, int16_t prev, int output_exponent) override;
};

} // namespace audio
} // namespace dl
