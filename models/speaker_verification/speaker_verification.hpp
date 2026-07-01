#pragma once
#include "dl_audio_wav.hpp"
#include "dl_fbank.hpp"
#include "dl_model_base.hpp"

#ifndef CONFIG_BSP_SD_MOUNT_POINT
#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"
#endif

class SpeakerVerification {
public:
    // target_seconds selects the model window length. Only 3 or 6 are supported
    // (each maps to a dedicated .espdl); any other value falls back to 6.
    SpeakerVerification(int target_seconds = 6);
    ~SpeakerVerification();

    // Run on WAV files. Returns nullptr on failure; caller must free() the embedding.
    float *run(const uint8_t *wav_start, size_t wav_len);
    // Run on raw PCM samples (16 kHz, mono, int16). Returns nullptr on failure;
    // caller must free() the returned embedding.
    float *run(const int16_t *samples, size_t num_samples);
    float compute_similarity(const float *e1, const float *e2);

    // Output embedding dimensionality.
    int get_embedding_dim() const { return embedding_dim; }

private:
    // Preprocess WAV input. Returns false on failure.
    bool preprocess(const uint8_t *wav_start, size_t wav_len);
    // Preprocess PCM input. Returns false on failure.
    bool preprocess(const int16_t *samples, size_t num_samples);
    // Crop/pad src to target_samples (centered) and write normalized floats into audio_buffer.
    void normalize_audio(const int16_t *src, int src_len);
    void extract_features();
    // Run inference and return a newly allocated embedding (caller frees).
    float *run_model();

    int target_seconds;
    int target_samples;
    int num_frames;
    int feature_dim;
    int embedding_dim;

    dl::Model *model;
    dl::audio::Fbank *fbank;
    float *audio_buffer = nullptr;
    float *features_buffer = nullptr;
};
