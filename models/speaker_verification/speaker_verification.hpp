#pragma once
#include "dl_audio_wav.hpp"
#include "dl_fbank.hpp"
#include "dl_model_base.hpp"

#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"

class SpeakerVerification {
public:
    SpeakerVerification(int target_seconds = 6);
    ~SpeakerVerification();

    float *run(const uint8_t *wav_start, size_t wav_len);
    float compute_similarity(const float *e1, const float *e2);

private:
    dl::TensorBase *preprocess(const uint8_t *wav_start, size_t wav_len);
    int target_seconds;
    int num_frames;

    dl::Model *model;
    dl::audio::Fbank *fbank;
    float *audio_buffer = nullptr;
    float *features_buffer = nullptr;
};
