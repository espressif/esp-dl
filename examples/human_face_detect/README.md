[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |

# Human Face Detect Example

A simple image inference example. See full example in [esp-who](https://github.com/espressif/esp-who/tree/master/examples/human_face_detect).

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (955) human_face_detect: [score: 0.936285, x1: 100, y1: 64, x2: 193, y2: 191]
I (955) human_face_detect: left_eye: [117, 114], left_mouth: [120, 160], nose: [132, 143], right_eye: [157, 112], right_mouth: [151, 160]]
```

## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options. See [Human Face Detect Model](https://github.com/espressif/esp-dl/blob/master/models/human_face_detect/README.md)。

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`