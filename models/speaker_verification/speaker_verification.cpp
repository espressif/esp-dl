#include "speaker_verification.hpp"
#include "esp_log.h"

static const char *TAG = "speaker_verification";

#if CONFIG_SPEAKER_VERIFICATION_MODEL_IN_FLASH_RODATA
extern const uint8_t sv_model_3s_espdl[] asm("_binary_sv_tdnn_tiny_3s_espdl_start");
extern const uint8_t sv_model_6s_espdl[] asm("_binary_sv_tdnn_tiny_6s_espdl_start");
#elif CONFIG_SPEAKER_VERIFICATION_MODEL_IN_FLASH_PARTITION
// Partition labels are selected at runtime based on target_seconds.
#else
#if !defined(CONFIG_BSP_SD_MOUNT_POINT)
#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"
#endif
#endif

SpeakerVerification::SpeakerVerification(int target_sec) : target_seconds(target_sec)
{
    // Only the 3s and 6s models are shipped; anything else falls back to 6s.
    if (target_seconds != 3 && target_seconds != 6) {
        ESP_LOGE(TAG, "Unsupported target_seconds=%d; only 3 or 6 are supported. Falling back to 6.", target_seconds);
        target_seconds = 6;
    }

#if !CONFIG_SPEAKER_VERIFICATION_MODEL_IN_SDCARD
#if CONFIG_SPEAKER_VERIFICATION_MODEL_IN_FLASH_RODATA
    const char *path = (const char *)(target_seconds == 3 ? sv_model_3s_espdl : sv_model_6s_espdl);
#else // CONFIG_SPEAKER_VERIFICATION_MODEL_IN_FLASH_PARTITION
    const char *path = (target_seconds == 3) ? "sv_model_3s" : "sv_model_6s";
#endif
    model = new dl::Model(path, static_cast<fbs::model_location_type_t>(CONFIG_SPEAKER_VERIFICATION_MODEL_LOCATION));
#else
    char sd_path[256];
    snprintf(sd_path,
             sizeof(sd_path),
             "%s/%s/%s",
             CONFIG_BSP_SD_MOUNT_POINT,
             CONFIG_SPEAKER_VERIFICATION_MODEL_SDCARD_DIR,
             target_seconds == 3 ? "sv_tdnn_tiny_3s.espdl" : "sv_tdnn_tiny_6s.espdl");
    model = new dl::Model(sd_path, static_cast<fbs::model_location_type_t>(CONFIG_SPEAKER_VERIFICATION_MODEL_LOCATION));
#endif

    model->minimize();

    dl::audio::SpeechFeatureConfig config;
    config.sample_rate = 16000;
    config.num_mel_bins = 80;
    config.frame_length = 25.0f;
    config.frame_shift = 10.0f;
    config.window_type = dl::audio::WinType::HAMMING;
    config.use_energy = false;
    config.use_log_fbank = 1;
    config.low_freq = 20.0f;
    config.high_freq = 0.0f;

    fbank = new dl::audio::Fbank(config);

    feature_dim = config.num_mel_bins;
    // Derive the frame count from the loaded model's input so the audio is always
    // cropped/padded to exactly the model window.
    const int frame_shift_samples = (int)(config.frame_shift * config.sample_rate / 1000);   // 160
    const int frame_length_samples = (int)(config.frame_length * config.sample_rate / 1000); // 400
    dl::TensorBase *input_tensor = model->get_inputs().begin()->second;
    num_frames = input_tensor->size / feature_dim;
    target_samples = (num_frames - 1) * frame_shift_samples + frame_length_samples;
    embedding_dim = model->get_outputs().begin()->second->size;
    audio_buffer = (float *)malloc(sizeof(float) * target_samples);
    features_buffer = (float *)malloc(sizeof(float) * num_frames * feature_dim);
}

SpeakerVerification::~SpeakerVerification()
{
    free(audio_buffer);
    free(features_buffer);
    delete fbank;
    delete model;
}

void SpeakerVerification::normalize_audio(const int16_t *src, int src_len)
{
    // Center-crop if too long, right-pad with zeros if too short, and scale int16 -> [-1, 1).
    if (src_len < target_samples) {
        for (int i = 0; i < target_samples; i++) audio_buffer[i] = (i < src_len) ? src[i] / 32768.0f : 0.0f;
    } else {
        int start = (src_len - target_samples) / 2;
        for (int i = 0; i < target_samples; i++) audio_buffer[i] = src[start + i] / 32768.0f;
    }
}

bool SpeakerVerification::preprocess(const uint8_t *wav_start, size_t wav_len)
{
    dl::audio::dl_audio_t *audio = dl::audio::decode_wav(wav_start, wav_len);
    if (!audio) {
        ESP_LOGE(TAG, "Failed to decode WAV.");
        return false;
    }
    normalize_audio(audio->data, audio->length);
    free(audio->data);
    free(audio);

    extract_features();
    return true;
}

bool SpeakerVerification::preprocess(const int16_t *samples, size_t num_samples)
{
    if (!samples || num_samples == 0) {
        ESP_LOGE(TAG, "Invalid PCM input.");
        return false;
    }
    normalize_audio(samples, (int)num_samples);
    extract_features();
    return true;
}

void SpeakerVerification::extract_features()
{
    // FBank
    auto input_tensor = model->get_inputs().begin()->second;
    fbank->process(audio_buffer, target_samples, features_buffer);

    // CMVN (only subtract mean)
    for (int d = 0; d < feature_dim; d++) {
        float mean = 0.0f;
        for (int t = 0; t < num_frames; t++) mean += features_buffer[t * feature_dim + d];
        mean /= num_frames;
        for (int t = 0; t < num_frames; t++) features_buffer[t * feature_dim + d] -= mean;
    }

    int in_dim = input_tensor->size;
    int16_t *quantized_input = (int16_t *)input_tensor->data;
    for (int i = 0; i < in_dim; i++)
        quantized_input[i] = dl::quantize<int16_t>(features_buffer[i], DL_RESCALE(input_tensor->exponent));
}

float *SpeakerVerification::run_model()
{
    model->run();

    dl::TensorBase *output_tensor = model->get_outputs().begin()->second;

    float *embedding = (float *)malloc(sizeof(float) * embedding_dim);
    if (!embedding) {
        ESP_LOGE(TAG, "Failed to allocate embedding buffer.");
        return nullptr;
    }

    int8_t *ptr = (int8_t *)output_tensor->data;
    for (int i = 0; i < embedding_dim; i++) embedding[i] = dl::dequantize(ptr[i], DL_SCALE(output_tensor->exponent));

    return embedding;
}

float *SpeakerVerification::run(const uint8_t *wav_start, size_t wav_len)
{
    if (!preprocess(wav_start, wav_len))
        return nullptr;
    return run_model();
}

float *SpeakerVerification::run(const int16_t *samples, size_t num_samples)
{
    if (!preprocess(samples, num_samples))
        return nullptr;
    return run_model();
}

float SpeakerVerification::compute_similarity(const float *e1, const float *e2)
{
    if (!e1 || !e2) {
        ESP_LOGE(TAG, "Invalid embedding.");
        return 0.0f;
    }

    float dot = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (int i = 0; i < embedding_dim; i++) {
        dot += e1[i] * e2[i];
        norm1 += e1[i] * e1[i];
        norm2 += e2[i] * e2[i];
    }

    if (norm1 == 0.0f || norm2 == 0.0f)
        return 0.0f;

    return dot / (sqrtf(norm1) * sqrtf(norm2));
}
