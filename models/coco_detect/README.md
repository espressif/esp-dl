# COCO Detection Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"
[no support]: https://img.shields.io/badge/-no_support-red "no support"

| Chip     | YOLO11N_S8_V1          | YOLO11N_S8_V2           |
|----------|------------------------|-------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][no support] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported]  |

- `yolo11n_s8_v1_s3` and `yolo11n_s8_v1_p4` uses [8bit default configuration quantization](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_yolo11n.html#bit-default-configuration-quantization).
- `yolo11n_s8_v2_p4` uses [Mixed-Precision + Horizontal Layer Split Pass Quantization](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_yolo11n.html#mixed-precision-horizontal-layer-split-pass-quantization).

## Model Latency

| name             | input(h*w*c)  | preprocess(us) | model(us) | postprocess(us) |
|------------------|---------------|----------------|-----------|-----------------|
| yolo11n_s8_v1_s3 | 640 * 640 * 3 | 207893         | 26919376  | 58994           |
| yolo11n_s8_v1_p4 | 640 * 640 * 3 | 105753         | 3109475   | 16610           |
| yolo11n_s8_v2_p4 | 640 * 640 * 3 | 105758         | 3627073   | 16644           |
## Model Usage

``COCODetect`` accepts a ``COCODetect::model_type_t`` parameter. It has a default value determined by [model type](#modeltype) option in menuconfig.

### How to New `HumanFaceFeat`

#### Only One Model

```cpp
COCODetect *detect = new COCODetect();
```

#### Mutiple Models

```cpp
// use YOLO11N_S8_V1
COCODetect *detect = new COCODetect(COCODetect::YOLO11N_S8_V1);
// use YOLO11N_S8_V2
// COCODetect *detect = new COCODetect(COCODetect::YOLO11N_S8_V2);
```
> **note:** If mutiple models is enabled in menuconfig, the default value is the first one. Pass in an explicit parameter to ``COCODetect`` to use one of them.

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model Type

- CONFIG_COCO_DETECT_YOLO11N_S8_V1
- CONFIG_COCO_DETECT_YOLO11N_S8_V2

These options determines which models will be enabled. 

> **note:** 
> - If model location is set to FLASH partition or FLASH rodata, only the selected model type will be flashed.
> - If model location is set to be in sdcard, all models will be selected automatically.

## Model Location

- CONFIG_COCO_DETECT_MODEL_IN_FLASH_RODATA
- CONFIG_COCO_DETECT_MODEL_IN_FLASH_PARTITION
- CONFIG_COCO_DETECT_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> **note:** 
> - If model location is set to FLASH partition, `partition.csv` must contain a partition named `coco_det`, and the partition should be big enough to hold the model file.
> - When using YOLO11n, do not set model location to sdcard if your PSRAM size is lower than 16MB. Because when loading YOLO11n, memory manager takes 6MB, and the model parameters is nearly 3MB. So we must ensure model parameters stay in FLASH or the PSRAM size is not enough. This means model location must be FLASH rodata or FLASH partition, and param_copy must set to false.

## SDCard Directory

- CONFIG_COCO_DETECT_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> **note:** Do not change the model name when copy the models to sdcard.