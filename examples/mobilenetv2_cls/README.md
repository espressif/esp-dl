[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.5           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-S31 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |

# IMAGENET classification Example

A simple image inference example. It classifies a built-in cat image and benchmarks the
average inference latency in both **single-core** and **multi-core** runtime modes.

The runtime mode is selected per inference by passing a `dl::runtime_mode_t` to `run()`:

```cpp
cls->run(img, dl::RUNTIME_MODE_SINGLE_CORE); // always single core (default)
cls->run(img, dl::RUNTIME_MODE_MULTI_CORE);  // dual core (ESP32-S3 / ESP32-P4)
```

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output(ESP32P4) in idf monitor:

```
I (5146) mobilenetv2_cls: ===== single-core: avg inference latency 327.884 ms over 10 runs =====
I (5146) mobilenetv2_cls: category: tabby, score: 0.569417
I (5146) mobilenetv2_cls: category: Egyptian_cat, score: 0.163141
I (5156) mobilenetv2_cls: category: tiger_cat, score: 0.098950
I (5156) mobilenetv2_cls: category: lynx, score: 0.046741
I (5166) mobilenetv2_cls: category: window_screen, score: 0.036402
I (7656) mobilenetv2_cls: ===== multi-core: avg inference latency 225.612 ms over 10 runs =====
I (7656) mobilenetv2_cls: category: tabby, score: 0.569417
I (7656) mobilenetv2_cls: category: Egyptian_cat, score: 0.163141
I (7656) mobilenetv2_cls: category: tiger_cat, score: 0.098950
I (7666) mobilenetv2_cls: category: lynx, score: 0.046741
I (7676) mobilenetv2_cls: category: window_screen, score: 0.036402
I (7676) main_task: Returned from app_main()

```

## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options. See [ImageNet Cls Model](https://github.com/espressif/esp-dl/blob/master/models/imagenet_cls/README.md)。

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`