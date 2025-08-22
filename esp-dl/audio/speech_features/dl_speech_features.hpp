#pragma once

#include "dl_audio_common.hpp"
#include "dl_fft.hpp"
#include "esp_err.h"
#include <stdlib.h>
#include <vector>

namespace dl {
namespace audio {

/**
 * @brief Speech features configuration structure
 */
struct SpeechFeatureConfig {
    int sample_rate = 16000;                     /*!< Sample rate in Hz */
    int frame_length = 25;                       /*!< Frame length in milliseconds */
    int frame_shift = 10;                        /*!< Frame step in milliseconds */
    int num_mel_bins = 26;                       /*!< Number of mel filterbank bins */
    int num_ceps = 13;                           /*!< Number of cepstral coefficients (for MFCC) */
    float preemphasis = 0.97f;                   /*!< Preemphasis coefficient */
    float cepstral_lifter = 22.0;                /*！< Constant that controls scaling of MFCCs >*/
    WinType window_type = WinType::HANNING;      /*!< Window type */
    float low_freq = 80.0f;                      /*!< Lower edge of mel filters */
    float high_freq = 7600.0f;                   /*!< Upper edge of mel filters */
    float log_epsilon = 1.1920928955078125e-07f; /*!< log epsilon: torch.finfo(torch.float32).eps */
    int use_log_fbank = 1;   /*!< 0: return fbank, 1: return log(x+log_epsilon), 2: return log(max(x, log_epsilon)) */
    bool raw_energy = true;  /*!< If True, compute energy before preemphasis and windowing (Default: True) */
    bool use_power = true;   /*!< If true, use power, else use magnitude. (Default: True) */
    bool use_energy = false; /*!< Add an extra dimension with energy to the FBANK output. (Default: False) */
    bool use_int16_fft = false;   /*!< If true, use float fft, else use int16 fft for faster speed. (Default: False) */
    bool remove_dc_offset = true; /*!< Subtract mean from waveform on each frame (Default: True) */

    SpeechFeatureConfig() = default;
};

/**
 * @brief Prints the configuration details of the speech feature settings.
 *
 * This function outputs the parameters and settings defined in the
 * SpeechFeatureConfig structure to the console or a log for debugging
 * or informational purposes.
 *
 * @param config A constant reference to a SpeechFeatureConfig object
 *               containing the speech feature configuration to be printed.
 */
void print_speech_feature_config(const SpeechFeatureConfig &config);

/**
 * @brief Base class for speech features extraction
 */
class SpeechFeatureBase {
protected:
    SpeechFeatureConfig m_config;
    uint32_t m_caps;   /*!< Memory allocation capabilities */
    int m_win_len;     /*!< Frame length */
    int m_win_step;    /*!< Frame step size */
    int m_fft_size;    /*!< FFT size, must be power of 2 and larger than frame length */
    int m_feature_dim; /*!< Feature dimension */

public:
    /**
     * @brief Construct a new Speech Feature Base object
     *
     * @param config Speech feature configuration
     */
    explicit SpeechFeatureBase(const SpeechFeatureConfig config, uint32_t caps = MALLOC_CAP_DEFAULT)
    {
        m_config = config;
        if (m_config.high_freq <= 0) {
            m_config.high_freq = config.sample_rate / 2.0;
        }
        assert(m_config.preemphasis >= 0.0f);
        assert(m_config.preemphasis < 1.0f);

        m_caps = caps;
        m_win_len = config.frame_length * config.sample_rate / 1000;
        m_win_step = config.frame_shift * config.sample_rate / 1000;
        m_fft_size = next_power_of_2(m_win_len);
        m_feature_dim = 0;
    }

    /**
     * @brief Destroy the Speech Feature Base object
     */
    virtual ~SpeechFeatureBase() = default;

    // Core interface
    /**
     * @brief Process a single frame of float audio data
     *
     * @param input Input audio data
     * @param win_len Window length
     * @param output Output features
     * @param input_prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    virtual esp_err_t process_frame(const float *input, int win_len, float *output, float input_prev = 0) = 0;

    /**
     * @brief Process a single frame of int16 audio data
     *
     * @param input Input audio data
     * @param win_len Window length
     * @param output Output features
     * @param input_prev Previous sample for pre-emphasis
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    virtual esp_err_t process_frame(const int16_t *input, int win_len, float *output, int16_t input_prev = 0) = 0;

    /**
     * @brief Process entire float audio data
     *
     * @param input Input audio data
     * @param input_len Length of input data
     * @param output Output features
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process(const float *input, int input_len, float *output);

    /**
     * @brief Process entire int16 audio data
     *
     * @param input Input audio data
     * @param input_len Length of input data
     * @param output Output features
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    esp_err_t process(const int16_t *input, int input_len, float *output);

    /**
     * @brief Get the output shape for given input length
     *
     * @param input_len Length of input data
     * @return std::vector<int> Output shape as [num_frames, feature_dim]
     */
    std::vector<int> get_output_shape(int input_len);

    /**
     * @brief Get the configuration object
     *
     * @return const SpeechFeatureConfig& Reference to the configuration
     */
    const SpeechFeatureConfig &config() const { return m_config; }

    void print_config() { print_speech_feature_config(m_config); }
};

} // namespace audio
} // namespace dl
