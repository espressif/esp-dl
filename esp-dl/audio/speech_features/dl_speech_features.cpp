#include "dl_speech_features.hpp"

namespace dl {
namespace audio {

void print_speech_feature_config(const SpeechFeatureConfig &config)
{
    printf("SpeechFeatureConfig:\n");
    printf("  sample_rate: %d\n", config.sample_rate);
    printf("  frame_length: %d\n", config.frame_length);
    printf("  frame_shift: %d\n", config.frame_shift);
    printf("  num_mel_bins: %d\n", config.num_mel_bins);
    printf("  num_cepstral: %d\n", config.num_cepstral);
    printf("  preemphasis: %.2f\n", config.preemphasis);
    printf("  window_type: %s\n", win_type_to_string(config.window_type));
    printf("  low_freq: %.2f\n", config.low_freq);
    printf("  high_freq: %.2f\n", config.high_freq);
    printf("  log_epsilon: %.2e\n", config.log_epsilon);
    printf("  use_log_fbank: %d\n", config.use_log_fbank);
    printf("  use_power: %s\n", config.use_power ? "true" : "false");
    printf("  use_energy: %s\n", config.use_energy ? "true" : "false");
    printf("  raw_energy: %s\n", config.raw_energy ? "true" : "false");
    printf("  use_int16_fft: %s\n", config.use_int16_fft ? "true" : "false");
    printf("  remove_dc_offset: %s\n", config.remove_dc_offset ? "true" : "false");
}

esp_err_t SpeechFeatureBase::process(const float *input, int input_len, float *output)
{
    if (input == nullptr || output == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    // Calculate number of frames
    if (input_len < m_win_len) {
        return ESP_ERR_INVALID_ARG;
    }

    int num_frames = get_frame_num(input_len, m_win_len, m_win_step);

    // Process each frame
    for (int i = 0; i < num_frames; i++) {
        esp_err_t ret = process_frame(input, m_win_len, output, input[0]);
        if (ret != ESP_OK) {
            return ret;
        }
        output += m_feature_dim;
        input += m_win_step;
    }

    return ESP_OK;
}

esp_err_t SpeechFeatureBase::process(const int16_t *input, int input_len, float *output)
{
    if (input == nullptr || output == nullptr || input_len <= 0) {
        return ESP_ERR_INVALID_ARG;
    }

    // Calculate number of frames
    if (input_len < m_win_len) {
        return ESP_ERR_INVALID_ARG;
    }

    int num_frames = get_frame_num(input_len, m_win_len, m_win_step);

    // Process each frame
    for (int i = 0; i < num_frames; i++) {
        esp_err_t ret = process_frame(input, m_win_len, output, input[0]);
        if (ret != ESP_OK) {
            return ret;
        }
        output += m_feature_dim;
        input += m_win_step;
    }

    return ESP_OK;
}

} // namespace audio
} // namespace dl
