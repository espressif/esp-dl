#include "speaker_verification.hpp"
#include "esp_log.h"

#if CONFIG_SPEAKER_VERIFICATION_MODEL_IN_FLASH_RODATA
extern const uint8_t sv_model_espdl[] asm("_binary_sv_model_espdl_start");
static const char *path = (const char *)sv_model_espdl;
#elif CONFIG_SPEAKER_VERIFICATION_MODEL_IN_FLASH_PARTITION
static const char *path = "sv_model";
#else
#if !defined(CONFIG_BSP_SD_MOUNT_POINT)
#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"
#endif
#endif

SpeakerVerification::SpeakerVerification(int target_sec)
    : target_seconds(target_sec)
{
#if !CONFIG_SPEAKER_VERIFICATION_MODEL_IN_SDCARD
    model = new dl::Model(path,
                          static_cast<fbs::model_location_type_t>(CONFIG_SPEAKER_VERIFICATION_MODEL_LOCATION));
#else
    char sd_path[256];
    snprintf(sd_path,
             sizeof(sd_path),
             "%s/%s/%s",
             CONFIG_BSP_SD_MOUNT_POINT,
             CONFIG_SPEAKER_VERIFICATION_MODEL_SDCARD_DIR,
             "sv_model.espdl");
    model = new dl::Model(sd_path, 
                          static_cast<fbs::model_location_type_t>(CONFIG_SPEAKER_VERIFICATION_MODEL_LOCATION));
#endif
    
    model->minimize();

    dl::audio::SpeechFeatureConfig config;
    config.sample_rate   = 16000;
    config.num_mel_bins  = 80;
    config.frame_length  = 25.0f;
    config.frame_shift   = 10.0f;
    config.window_type   = dl::audio::WinType::HAMMING;
    config.use_energy    = false;
    config.use_log_fbank = 1;
    config.low_freq      = 20.0f;
    config.high_freq     = 0.0f;

    fbank = new dl::audio::Fbank(config);

    int max_samples = target_seconds * config.sample_rate + 240;
    num_frames = (max_samples - config.frame_length * config.sample_rate / 1000) / 
                 (config.frame_shift * config.sample_rate / 1000) + 1;
                 
    audio_buffer = (float*)malloc(sizeof(float) * max_samples);
    features_buffer = (float*)malloc(sizeof(float) * num_frames * config.num_mel_bins);
}

SpeakerVerification::~SpeakerVerification() {
    free(audio_buffer);
    free(features_buffer);
    delete fbank;
    delete model;
}

dl::TensorBase* SpeakerVerification::preprocess(const uint8_t* wav_start, size_t wav_len)
{
    dl::audio::dl_audio_t* audio = dl::audio::decode_wav(wav_start, wav_len);
    int target_samples = target_seconds * 16000 + 240;

    // Pad or crop audio
    if (audio->length < target_samples) {
        for (int i = 0; i < target_samples; i++)
            audio_buffer[i] = (i < audio->length) ? audio->data[i] / 32768.0f : 0.0f;
    } else {
        int start = (audio->length - target_samples) / 2;
        for (int i = 0; i < target_samples; i++)
            audio_buffer[i] = audio->data[start + i] / 32768.0f;
    }
    free(audio->data);
    free(audio);

    // FBank
    auto input_tensor = model->get_inputs().begin()->second;
    fbank->process(audio_buffer, target_samples, features_buffer);

    // CMVN (only subtract mean)
    int feature_dim = 80;
    for (int d = 0; d < feature_dim; d++) {
        float mean = 0.0f;
        for (int t = 0; t < num_frames; t++) mean += features_buffer[t * feature_dim + d];
        mean /= num_frames;
        for (int t = 0; t < num_frames; t++) features_buffer[t * feature_dim + d] -= mean;
    }

    memcpy(input_tensor->data, features_buffer, sizeof(float) * num_frames * feature_dim);
    return input_tensor;
}

float* SpeakerVerification::run(const uint8_t* wav_start, size_t wav_len)
{
    preprocess(wav_start, wav_len);
    model->run();

    auto outputs = model->get_outputs();
    dl::TensorBase* output_tensor = outputs.begin()->second;
    int out_dim = output_tensor->size;
    float* embedding = (float*)malloc(sizeof(float) * out_dim);
    int8_t* ptr = (int8_t*)output_tensor->data;
    for (int i = 0; i < out_dim; i++)
        embedding[i] = dl::dequantize(ptr[i], DL_SCALE(output_tensor->exponent));

    return embedding;
}

float SpeakerVerification::compute_similarity(const float* e1, const float* e2)
{
    int out_dim = model->get_outputs().begin()->second->size;

    float norm1 = 0.0f, norm2 = 0.0f;
    for (int i = 0; i < out_dim; i++) {
        norm1 += e1[i] * e1[i];
        norm2 += e2[i] * e2[i];
    }
    norm1 = sqrtf(norm1);
    norm2 = sqrtf(norm2);

    float sim = 0.0f;
    for (int i = 0; i < out_dim; i++)
        sim += (e1[i] / norm1) * (e2[i] / norm2);

    return sim;
}
