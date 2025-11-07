# Hand Detection Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"
[no support]: https://img.shields.io/badge/-no_support-red "no support"

| Chip     | ESPDET_PICO_224_224_HAND |
|----------|-------------------------|
| ESP32-S3 | ![alt text][supported]  |
| ESP32-P4 | ![alt text][supported]  |

## Model Latency

| name                           | input(h*w*c)  | preprocess(us) | model(us) | postprocess(us) | 
|--------------------------------|---------------|----------------|-----------|-----------------|
| espdet_pico_224_224_hand_s8_s3 | 224 * 224 * 3 | 7594           | 123783    | 1196            | 
| espdet_pico_224_224_hand_s8_p4 | 224 * 224 * 3 | 2858           | 50458     | 683             |

## Model Usage

``HandDetect`` accepts a ``HandDetect::model_type_t`` parameter. It has a default value determined by [default model](#default-model) option in menuconfig.

### How to New `HandDetect`

#### Only One Model

```cpp
HandDetect *detect = new HandDetect();
```

### How to Detect

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
std::list<dl::detect::result_t> &res = detect->run(img);
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp) and [`dl::detect::result_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/detect/dl_detect_define.hpp).

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model to Flash

- CONFIG_FLASH_ESPDET_PICO_224_224_HAND

Whether to flash the model when model location is set to FLASH rodata or FLASH partition.

## Default Model

- CONFIG_ESPDET_PICO_224_224_HAND

Default model to use if no parameter is passed to ``HandDetect``.

## Model Location

- CONFIG_HAND_DETECT_MODEL_IN_FLASH_RODATA
- CONFIG_HAND_DETECT_MODEL_IN_FLASH_PARTITION
- CONFIG_HAND_DETECT_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE]
> - If model location is set to FLASH partition, `partition.csv` must contain a partition named `hand_det`, and the partition should be big enough to hold the model file.

## SDCard Directory

- CONFIG_HAND_DETECT_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the models to sdcard.