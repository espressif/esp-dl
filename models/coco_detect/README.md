# COCO Detection Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"
[no support]: https://img.shields.io/badge/-no_support-red "no support"

| Chip     | YOLO11N_S8_V1          |  YOLO11N_320_S8_V1      |
|----------|------------------------|-------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported]  |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported]  |

## Model Latency

| name                 | input(h*w*c)  | Flash(MB) | PSRAM(MB) | preprocess(ms) | model(ms) | postprocess(ms) | mAP50-95 on COCO val2017 |
| ---------------------- | --------------- | ----------- | ----------- | ---------------- | ----------- | ----------------- | -------------------------- |
|  yolo11n_s8_v1_s3 |  640 * 640 * 3 |  8 |  8 | 51.5 | 26014.2 | 58.9 |  0.370 |
|  yolo11n_s8_v1_p4 |  640 * 640 * 3 |  16 |  32 | 16.0 | 2535.3 | 14.3 |  0.373 |
|  yolo11n_320_s8_v1_s3 |  320 * 320 * 3 |  8 |  8 | 15.2 | 6161.8 | 19.3 |  0.276 |
|  yolo11n_320_s8_v1_p4 |  320 * 320 * 3 |  16 |  32 | 5.2 | 550.1 | 5.3 |  0.278 |

## Model Usage

``COCODetect`` accepts a ``COCODetect::model_type_t`` parameter. It has a default value determined by [default model](#default-model) option in menuconfig.

### How to New `COCODetect`

#### Only One Model

```cpp
COCODetect *detect = new COCODetect();
```

#### Multiple Models

```cpp
// use YOLO11N_S8_V1
COCODetect *detect = new COCODetect(COCODetect::YOLO11N_S8_V1);
// use YOLO11N_320_S8_V1
// COCODetect *detect = new COCODetect(COCODetect::YOLO11N_320_S8_V1);
```
> [!NOTE] 
> If multiple models is flashed or stored in sdcard, in addition to the default model, you can pass an explicit parameter to ``COCODetect`` to use one of them.

### How to Detect

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
std::list<dl::detect::result_t> &res = detect->run(img);
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp) and [`dl::detect::result_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/detect/dl_detect_define.hpp).

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model to Flash

- CONFIG_FLASH_COCO_DETECT_YOLO11N_S8_V1
- CONFIG_FLASH_COCO_DETECT_YOLO11N_320_S8_V1

Whether to flash the model when model location is set to FLASH rodata or FLASH partition.

## Default Model

- CONFIG_COCO_DETECT_YOLO11N_S8_V1
- CONFIG_COCO_DETECT_YOLO11N_320_S8_V1

Default model to use if no parameter is passed to ``COCODetect``.

## Model Location

- CONFIG_COCO_DETECT_MODEL_IN_FLASH_RODATA
- CONFIG_COCO_DETECT_MODEL_IN_FLASH_PARTITION
- CONFIG_COCO_DETECT_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE]
> - If model location is set to FLASH partition, `partition.csv` must contain a partition named `coco_det`, and the partition should be big enough to hold the model file.
> - When using YOLO11n, do not set model location to sdcard if your PSRAM size is lower than 16MB. Because when loading YOLO11n, memory manager takes 6MB, and the model parameters is nearly 3MB. So we must ensure model parameters stay in FLASH or the PSRAM size is not enough. This means model location must be FLASH rodata or FLASH partition, and param_copy must set to false.

## SDCard Directory

- CONFIG_COCO_DETECT_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the models to sdcard.