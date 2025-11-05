[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |

# Hand Gesture Recognition Example

A simple image inference example. See full example in [esp-who](https://github.com/espressif/esp-who/tree/master/examples).

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (1942) hand_gesture_recognition: category: one, score: 0.999957
I (1952) main_task: Returned from app_main()

```
## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options. This example includes two models, one for hand detection and another for hand gesture classification. 

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`