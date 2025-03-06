[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |


# Pedestrian Detect Example

A simple image inference example. See full example in [esp-who](https://github.com/espressif/esp-who/tree/master/examples/pedestrian_detect).

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (1590) pedestrian_detect: [score: 0.883883, x1: 143, y1: 189, x2: 251, y2: 462]
I (1590) pedestrian_detect: [score: 0.883883, x1: 282, y1: 195, x2: 370, y2: 461]
I (1590) pedestrian_detect: [score: 0.805256, x1: 412, y1: 224, x2: 486, y2: 394]
```

## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options. See [Pedestrian Detect Model](https://github.com/espressif/esp-dl/blob/master/models/pedestrian_detect/README.md)。

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`