#include "dl_speech_features.hpp"

namespace dl {
namespace audio {

static const char *tag = "dl::audio";

void print_speech_feature_config(const SpeechFeatureConfig &config)
{
    ESP_LOGI(tag, "SpeechFeatureConfig");
    ESP_LOGI(tag, "  sample_rate: %d", config.sample_rate);
    ESP_LOGI(tag, "  frame_length: %d", config.frame_length);
    ESP_LOGI(tag, "  frame_shift: %d", config.frame_shift);
    ESP_LOGI(tag, "  num_mel_bins: %d", config.num_mel_bins);
    ESP_LOGI(tag, "  num_ceps: %d", config.num_ceps);
    ESP_LOGI(tag, "  preemphasis: %.2f", config.preemphasis);
    ESP_LOGI(tag, "  window_type: %s", win_type_to_string(config.window_type));
    ESP_LOGI(tag, "  low_freq: %.2f", config.low_freq);
    ESP_LOGI(tag, "  high_freq: %.2f", config.high_freq);
    ESP_LOGI(tag, "  log_epsilon: %.2e", config.log_epsilon);
    ESP_LOGI(tag, "  use_log_fbank: %d", config.use_log_fbank);
    ESP_LOGI(tag, "  use_power: %s", config.use_power ? "true" : "false");
    ESP_LOGI(tag, "  use_energy: %s", config.use_energy ? "true" : "false");
    ESP_LOGI(tag, "  raw_energy: %s", config.raw_energy ? "true" : "false");
    ESP_LOGI(tag, "  use_int16_fft: %s", config.use_int16_fft ? "true" : "false");
    ESP_LOGI(tag, "  remove_dc_offset: %s", config.remove_dc_offset ? "true" : "false");
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

std::vector<int> SpeechFeatureBase::get_output_shape(int input_len)
{
    // Calculate number of frames
    int num_frames = get_frame_num(input_len, m_win_len, m_win_step);

    // Return shape as [num_frames, num_mel_bins]
    return std::vector<int>{num_frames, m_feature_dim};
}

} // namespace audio
} // namespace dl
