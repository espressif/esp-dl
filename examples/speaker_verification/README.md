[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           | ESP-IDF v5.5           |
|----------|------------------------|------------------------|------------------------|
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] | ![alt text][supported] |

# Speaker Verification Example

A simple audio inference example. Three audio samples are provided: a and b are from the same speaker, while b and c are from different speakers.

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (3567) speaker_verification: Cosine similarity between audio a and b: 0.7531
I (3567) speaker_verification: Cosine similarity between audio b and c: 0.0874
I (3577) main_task: Returned from app_main()
```

## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options.

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`

