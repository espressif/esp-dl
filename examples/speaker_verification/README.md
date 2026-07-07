[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           | ESP-IDF v5.5           |
|----------|------------------------|------------------------|------------------------|
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] | ![alt text][supported] |

# Speaker Verification Example

A simple audio inference example. Three audio samples are provided: a and b are from the same speaker, while c is from a different speaker. Audio a is enrolled as `speaker1` in a verification database (stored on SPIFFS), then b and c are verified against it by cosine similarity.

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (13694) speaker_verification: Testing audio_b against enrolled speaker1:
I (13694) dl::audio::VerificationDB: [MAX-COSINE] speaker1: score=0.7300 (threshold=0.2500) -> SAME speaker
I (14254) speaker_verification: Testing audio_c against enrolled speaker1:
I (14254) dl::audio::VerificationDB: [MAX-COSINE] speaker1: score=0.0378 (threshold=0.2500) -> DIFFERENT speaker
I (14264) main_task: Returned from app_main()
```

## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options.

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`

