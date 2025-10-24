# Speaker Verification Model

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"
[no support]: https://img.shields.io/badge/-no_support-red "no support"

| Chip     | SPEAKER_VERIFICATION    | 
|----------|-------------------------| 
| ESP32-P4 | ![alt text][supported]  | 

## Model Latency

| name         | preprocess(us) | model(us) | postprocess(us) |
|--------------|----------------|-----------|-----------------|
| sv_tdnn_tiny | 122951         | 461084    | 68              |

## Model Metrics (after quantization)

| name         | dataset      | EER   |
|--------------|--------------|-------|
| sv_tdnn_tiny | vox1-O-clean | 2.36% |

## Model Usage

### How to Create

By default, the model uses a 6-second target duration.
All input audio will be automatically padded or trimmed to this length.

```cpp
SpeakerVerification* verifier = new SpeakerVerification();
```

You can also specify a custom target duration (e.g., 3 seconds).
In this case, the model will use only 3 seconds of audio as input.

```cpp
SpeakerVerification* verifier = new SpeakerVerification(3);
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
> - If model location is set to FLASH partition, `partition.csv` must contain a partition named `sv_model`, and the partition should be big enough to hold the model file.

## SDCard Directory

- CONFIG_SPEAKER_VERIFICATION_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the model to sdcard.