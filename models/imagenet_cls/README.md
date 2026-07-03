# Imagenet Classification Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | MOBILENETV2_S8_V1      |
|----------|------------------------|
| ESP32-S3 | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] |

## Model Latency

| name                                             | input(h*w*c)  | preprocess(ms) | model(ms) | postprocess(ms) |
| ------------------------------------------------ | ------------- | -------------- | --------- | ---------------- |
| mobilenetv2_s8_v1_s3 (RUNTIME_MODE_SINGLE_CORE)  | 224 * 224 * 3 | 9.2            | 1294.5    | 1.1              |
| mobilenetv2_s8_v1_s3 (RUNTIME_MODE_MULTI_CORE)   | 224 * 224 * 3 | 9.2            | 856.0     | 1.1              |
| mobilenetv2_s8_v1_p4 (RUNTIME_MODE_SINGLE_CORE)  | 224 * 224 * 3 | 3.1            | 327.9     | 0.8              |
| mobilenetv2_s8_v1_p4 (RUNTIME_MODE_MULTI_CORE)   | 224 * 224 * 3 | 3.1            | 225.6     | 0.8              |

> Multi-core mode only accelerates inference on dual-core targets (ESP32-S3 / ESP32-P4).
> On a single-core target, `RUNTIME_MODE_MULTI_CORE` falls back to single-core execution.

## Model Usage

### How to New `ImageNetCls`

```cpp
ImageNetCls *cls = new ImageNetCls();
```
### How to Classify

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};

// single-core inference (default)
std::vector<dl::cls::result_t> &res = cls->run(img);
```

### How to Select the Runtime Mode

`run()` accepts an optional `dl::runtime_mode_t` (default `RUNTIME_MODE_SINGLE_CORE`) to choose
between single-core and dual-core execution:

```cpp
// dl::RUNTIME_MODE_SINGLE_CORE - always single core (default)
// dl::RUNTIME_MODE_MULTI_CORE  - dual core (ESP32-S3 / ESP32-P4)
// dl::RUNTIME_MODE_AUTO        - auto select
std::vector<dl::cls::result_t> &res = cls->run(img, dl::RUNTIME_MODE_MULTI_CORE);
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp) and [`dl::cls::result_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/classification/dl_cls_define.hpp).

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model Location

- CONFIG_IMAGENET_CLS_MODEL_IN_FLASH_RODATA
- CONFIG_IMAGENET_CLS_MODEL_IN_FLASH_PARTITION
- CONFIG_IMAGENET_CLS_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE] 
> If model location is set to FLASH partition, `partition.csv` must contain a partition named `imagenet_cls`, and the partition should be big enough to hold the model file.

## SDCard Directory

- CONFIG_IMAGENET_CLS_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the models to sdcard.