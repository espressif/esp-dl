# Speaker Verification Model

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"
[no support]: https://img.shields.io/badge/-no_support-red "no support"

| Chip     | SPEAKER_VERIFICATION    | 
|----------|-------------------------| 
| ESP32-P4 | ![alt text][supported]  | 

## Model Latency

| name             | preprocess(us) | model(us) | postprocess(us) |
| ---------------- | ---------------- | ----------- | ----------------- |
|  sv_tdnn_tiny_3s | 61593  | 228370 | 171 |
|  sv_tdnn_tiny_6s | 119947 | 441884 | 171 |

## Model Metrics (after quantization)

| name            | dataset      | EER   |
|-----------------|--------------|-------|
| sv_tdnn_tiny_3s | vox1-O-clean | 4.51% |
| sv_tdnn_tiny_6s | vox1-O-clean | 2.19% |

## Model Usage

### How to Create

Two fixed-length models are shipped: a 6-second window (`sv_tdnn_tiny_6s`, 600 frames)
and a 3-second window (`sv_tdnn_tiny_3s`, 300 frames). The target duration selects which
model is loaded and only accepts `3` or `6` (any other value falls back to `6`).
All input audio is automatically padded or trimmed to the selected model's window.

By default the 6-second model is used:

```cpp
SpeakerVerification* verifier = new SpeakerVerification();   // 6s / 600 frames
```

To use the 3-second model:

```cpp
SpeakerVerification* verifier = new SpeakerVerification(3);  // 3s / 300 frames
```

### How to Use

```cpp
float* embed_a = verifier->run(wav_a_start, wav_a_end - wav_a_start);
float* embed_b = verifier->run(wav_b_start, wav_b_end - wav_b_start);
float similarity = verifier->compute_similarity(embed_a, embed_b);
```

> [!NOTE]
> - This model expects 16kHz, mono-channel WAV audio.
The cosine similarity between the two embedding ranges from -1.0 to 1.0.
Higher value indicates greater speaker similarity.


# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model to Flash

- CONFIG_FLASH_SPEAKER_VERIFICATION

Whether to flash the model when model location is set to FLASH rodata or FLASH partition.

## Model Location

- CONFIG_SPEAKER_VERIFICATION_MODEL_IN_FLASH_RODATA
- CONFIG_SPEAKER_VERIFICATION_MODEL_IN_FLASH_PARTITION
- CONFIG_SPEAKER_VERIFICATION_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE]
> - If model location is set to FLASH partition, `partition.csv` must contain two partitions named `sv_model_3s` and `sv_model_6s`, each big enough to hold the corresponding model file.

## SDCard Directory

- CONFIG_SPEAKER_VERIFICATION_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the model to sdcard.