// #include "dl_stft.hpp"
// #include <math.h>
// #include <stdio.h>
// #include <vector>
// #include <stdlib.h>

// namespace dl {
// namespace audio {

// esp_err_t STFT::process_frame(const float *input, float *output)
// {
//     if (input == nullptr || output == nullptr) {
//         return ESP_ERR_INVALID_ARG;
//     }

//     // Apply window function
//     for (int i = 0; i < m_win_len; i++) {
//         m_cache[i] = input[i] * m_win_func[i];
//     }

//     // Zero pad to FFT size if needed
//     for (int i = m_win_len; i < m_win_len * 2; i++) {
//         m_cache[i] = 0.0f;
//     }

//     // Compute FFT
//     m_fft_handle->fft(m_cache, m_win_len);

//     // Copy result to output
//     int output_len = m_onesided ? (m_win_len + 2) : (m_win_len * 2);
//     for (int i = 0; i < output_len; i++) {
//         output[i] = m_cache[i];
//     }

//     return ESP_OK;
// }

// esp_err_t STFT::process(const float *input, int input_len, float *output)
// {
//     if (input == nullptr || output == nullptr || input_len <= 0) {
//         return ESP_ERR_INVALID_ARG;
//     }

//     // Calculate number of frames
//     if (input_len < m_win_len) {
//         return ESP_ERR_INVALID_ARG;
//     }

//     int num_frames = 1 + (input_len - m_win_len) / m_win_step;

//     // Process each frame
//     for (int i = 0; i < num_frames; i++) {
//         int start = i * m_win_step;
//         esp_err_t ret = process_frame(&input[start], &output[i * (m_onesided ? (m_win_len + 2) : (m_win_len * 2))]);
//         if (ret != ESP_OK) {
//             return ret;
//         }
//     }

//     return ESP_OK;
// }

// std::vector<int> STFT::get_output_shape(int input_len)
// {
//     // Calculate number of frames
//     int num_frames = 0;
//     if (input_len >= m_win_len) {
//         num_frames = 1 + (input_len - m_win_len) / m_win_step;
//     }

//     // Calculate feature dimension
//     int feature_dim = m_onesided ? (m_win_len + 2) : (m_win_len * 2);

//     // Return shape as [num_frames, feature_dim]
//     return std::vector<int>{num_frames, feature_dim};
// }

// void STFT::run(float *data, int data_len, float *features)
// {
//     process(data, data_len, features);
// }

// std::vector<int> STFT::get_feature_shape()
// {
//     // This is a placeholder implementation
//     // In a real implementation, this would depend on the specific data being processed
//     return std::vector<int>{1, m_onesided ? (m_win_len + 2) : (m_win_len * 2)};
// }

// } // namespace audio
// } // namespace dl
