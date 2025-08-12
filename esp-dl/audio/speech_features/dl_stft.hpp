// #pragma once

// #include "dl_audio_common.hpp"
// #include "dl_speech_features.hpp"
// #include "dl_fft.hpp"

// namespace dl {
// namespace audio {

// /**
//  * @brief Short-Time Fourier Transform (STFT) features extraction class
//  */
// class STFT : public SpeechFeatureBase {
// private:
//     // STFT specific parameters
//     FFT *m_fft_handle;          /*!< FFT handle for spectrum computation */
//     uint32_t m_caps;            /*!< Memory allocation capabilities */
//     float *m_win_func;          /*!< Window function coefficients */
//     int m_win_len;
//     int m_win_step;
//     bool m_onesided;

// public:
//     /**
//      * @brief Construct a new STFT object
//      *
//      * @param config Speech feature configuration
//      * @param caps Memory allocation capabilities
//      */
//     STFT(const SpeechFeatureConfig& config, bool onesided=true, uint32_t caps=MALLOC_CAP_DEFAULT):
//     SpeechFeatureBase(config)
//     {
//         m_fft_handle = FFT::get_instance();
//         m_caps = caps;
//         m_onesided = onesided;

//         m_win_len = config.frame_length * config.sample_rate / 1000;
//         m_win_step = config.frame_shift * config.sample_rate / 1000;
//         m_win_func = win_func_init(config.window_type, m_win_len);
//     }

//     /**
//      * @brief Destroy the STFT object
//      */
//     ~STFT()
//     {
//         if (m_win_func) {
//             free(m_win_func);
//         }
//     }

//     /**
//      * @brief Process a single frame of float audio data
//      *
//      * @param input Input audio data
//      * @param win_len Window length
//      * @param output Output STFT features
//      * @param prev Previous sample for pre-emphasis
//      * @return esp_err_t ESP_OK on success, error code otherwise
//      */
//     esp_err_t process_frame(const float* input, int win_len, float* output, float prev=0);

//     /**
//      * @brief Process a single frame of int16 audio data
//      *
//      * @param input Input audio data
//      * @param win_len Window length
//      * @param output Output STFT features
//      * @param prev Previous sample for pre-emphasis
//      * @return esp_err_t ESP_OK on success, error code otherwise
//      */
//     esp_err_t process_frame(const int16_t* input, int win_len, float* output, int16_t prev=0);

//     /**
//      * @brief Process entire float audio data
//      *
//      * @param input Input audio data
//      * @param input_len Length of input data
//      * @param output Output STFT features
//      * @return esp_err_t ESP_OK on success, error code otherwise
//      */
//     esp_err_t process(const float* input, int input_len, float* output) override;

//     /**
//      * @brief Process entire int16 audio data
//      *
//      * @param input Input audio data
//      * @param input_len Length of input data
//      * @param output Output STFT features
//      * @return esp_err_t ESP_OK on success, error code otherwise
//      */
//     esp_err_t process(const int16_t* input, int input_len, float* output) override;

//     /**
//      * @brief Get the output shape for given input length
//      *
//      * @param input_len Length of input data
//      * @return std::vector<int> Output shape as [num_frames, num_mel_bins]
//      */
//     std::vector<int> get_output_shape(int input_len);
// };

// } // namespace audio
// } // namespace dl
