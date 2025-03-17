[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |

# IMAGENET classification Example

A simple image inference example.

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (2270) imagenet_cls: category: tabby, score: 0.356466
I (2270) imagenet_cls: category: lynx, score: 0.216208
I (2270) imagenet_cls: category: Egyptian_cat, score: 0.168383
I (2280) imagenet_cls: category: tiger_cat, score: 0.131137
I (2280) imagenet_cls: category: window_screen, score: 0.029261
```

## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options. See [ImageNet Cls Model](https://github.com/espressif/esp-dl/blob/master/models/imagenet_cls/README.md)。

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`