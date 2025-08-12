#pragma once

#include "dl_audio_common.hpp"
#include "dl_fft.hpp"
#include "esp_err.h"
#include <functional>
#include <stdint.h>
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
    int fft_size = 512;                          /*!< FFT size */
    int num_mel_bins = 26;                       /*!< Number of mel filterbank bins */
    int num_cepstral = 13;                       /*!< Number of cepstral coefficients (for MFCC) */
    float preemphasis = 0.97f;                   /*!< Preemphasis coefficient */
    WinType window_type = WinType::HANNING;      /*!< Window type */
    float low_freq = 80.0f;                      /*!< Lower edge of mel filters */
    float high_freq = 7600.0f;                   /*!< Upper edge of mel filters */
    float log_epsilon = 1.1920928955078125e-07f; /*!< log epsilon: torch.finfo(torch.float32).eps */
    int use_log_fbank = 1;   /*!< 0: return fbank, 1: return log(x+log_epsilon), 2: return log(max(x, log_epsilon)) */
    bool use_power = true;   /*!< If true, use power of fft spectrum, else use magnitude of fft spectrum */
    bool use_energy = false; /*!< Include energy feature */
    bool use_int16_fft = false; /*!< If true, use float fft, else use int16 fft for faster speed */

    SpeechFeatureConfig() = default;
};

void print_speech_feature_config(const SpeechFeatureConfig &config)
{
    printf("SpeechFeatureConfig:\n");
    printf("  sample_rate: %d\n", config.sample_rate);
    printf("  frame_length: %d\n", config.frame_length);
    printf("  frame_shift: %d\n", config.frame_shift);
    printf("  fft_size: %d\n", config.fft_size);
    printf("  num_mel_bins: %d\n", config.num_mel_bins);
    printf("  num_cepstral: %d\n", config.num_cepstral);
    printf("  preemphasis: %.2f\n", config.preemphasis);
    printf("  window_type: %d\n", static_cast<int>(config.window_type));
    printf("  low_freq: %.2f\n", config.low_freq);
    printf("  high_freq: %.2f\n", config.high_freq);
    printf("  log_epsilon: %.2e\n", config.log_epsilon);
    printf("  use_log_fbank: %d\n", config.use_log_fbank);
    printf("  use_power: %s\n", config.use_power ? "true" : "false");
    printf("  use_energy: %s\n", config.use_energy ? "true" : "false");
    printf("  use_int16_fft: %s\n", config.use_int16_fft ? "true" : "false");
}

/**
 * @brief Base class for speech features extraction
 */
class SpeechFeatureBase {
protected:
    SpeechFeatureConfig m_config;

public:
    /**
     * @brief Construct a new Speech Feature Base object
     *
     * @param config Speech feature configuration
     */
    explicit SpeechFeatureBase(const SpeechFeatureConfig config) { m_config = config; }

    /**
     * @brief Destroy the Speech Feature Base object
     */
    virtual ~SpeechFeatureBase();

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
    virtual esp_err_t process(const float *input, int input_len, float *output) = 0;

    /**
     * @brief Process entire int16 audio data
     *
     * @param input Input audio data
     * @param input_len Length of input data
     * @param output Output features
     * @return esp_err_t ESP_OK on success, error code otherwise
     */
    virtual esp_err_t process(const int16_t *input, int input_len, float *output) = 0;

    /**
     * @brief Get the output shape for given input length
     *
     * @param input_len Length of input data
     * @return std::vector<int> Output shape as [num_frames, feature_dim]
     */
    virtual std::vector<int> get_output_shape(int input_len) = 0;

    /**
     * @brief Get the configuration object
     *
     * @return const SpeechFeatureConfig& Reference to the configuration
     */
    const SpeechFeatureConfig &config() const { return m_config; }

    // void print_config()
    // {
    //     printf("SpeechFeatureConfig:\n");
    //     printf("  sample_rate: %d\n", config.sample_rate);
    //     printf("  frame_length: %d\n", config.frame_length);
    //     printf("  frame_shift: %d\n", config.frame_shift);
    //     printf("  fft_size: %d\n", config.fft_size);
    //     printf("  num_mel_bins: %d\n", config.num_mel_bins);
    //     printf("  num_cepstral: %d\n", config.num_cepstral);
    //     printf("  preemphasis: %.2f\n", config.preemphasis);
    //     printf("  window_type: %d\n", static_cast<int>(config.window_type));
    //     printf("  low_freq: %.2f\n", config.low_freq);
    //     printf("  high_freq: %.2f\n", config.high_freq);
    //     printf("  log_epsilon: %.2e\n", config.log_epsilon);
    //     printf("  use_log_fbank: %d\n", config.use_log_fbank);
    //     printf("  use_power: %s\n", config.use_power ? "true" : "false");
    //     printf("  use_energy: %s\n", config.use_energy ? "true" : "false");
    //     printf("  use_int16_fft: %s\n", config.use_int16_fft ? "true" : "false");
    // }
};

} // namespace audio
} // namespace dl
