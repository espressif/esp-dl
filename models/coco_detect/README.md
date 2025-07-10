# COCO Detection Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"
[no support]: https://img.shields.io/badge/-no_support-red "no support"

| Chip     | YOLO11N_S8_V1          | YOLO11N_S8_V2           | YOLO11N_S8_V3           |  YOLO11N_320_S8_V3      |
|----------|------------------------|-------------------------|-------------------------|-------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported]  | ![alt text][supported]  | ![alt text][supported]  |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported]  | ![alt text][supported]  | ![alt text][supported]  |

- `yolo11n_s8_v1_s3` and `yolo11n_s8_v1_p4` use [8bit default configuration quantization](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_yolo11n.html#bit-default-configuration-quantization).
- `yolo11n_s8_v2_s3` and `yolo11n_s8_v2_p4` use [Mixed-Precision + Horizontal Layer Split Pass Quantization](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_yolo11n.html#mixed-precision-horizontal-layer-split-pass-quantization).
- `yolo11n_s8_v3_s3` , `yolo11n_s8_v3_p4` , `yolo11n_320_s8_v3_s3` and `yolo11n_320_s8_v3_p4` use [Quantization-Aware Training](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_yolo11n.html#quantization-aware-training).

## Model Latency

| name                 | input(h*w*c)  | Flash(MB) | PSRAM(MB) | preprocess(us) | model(us) | postprocess(us) | mAP50-95 on COCO val2017 |
|----------------------|---------------|-----------|-----------|----------------|-----------|-----------------|--------------------------|
| yolo11n_s8_v1_s3     | 640 * 640 * 3 | 8         | 8         | 202701         | 26162561  | 58571           | 0.307                    |
| yolo11n_s8_v2_s3     | 640 * 640 * 3 | 16        | 16        | 202662         | 15981078  | 59165           | 0.331                    |  
| yolo11n_s8_v3_s3     | 640 * 640 * 3 | 8         | 8         | 202804         | 26057061  | 58006           | 0.359                    |
| yolo11n_s8_v1_p4     | 640 * 640 * 3 | 16        | 32        | 105757         | 2769830   | 15955           | 0.307                    |
| yolo11n_s8_v2_p4     | 640 * 640 * 3 | 16        | 32        | 105766         | 3260384   | 16296           | 0.334                    |
| yolo11n_s8_v3_p4     | 640 * 640 * 3 | 16        | 32        | 105755         | 2764595   | 15860           | 0.360                    |
| yolo11n_320_s8_v3_s3 | 320 * 320 * 3 | 8         | 8         | 52255          | 6184218   | 17594           | 0.277                    |
| yolo11n_320_s8_v3_p4 | 320 * 320 * 3 | 16        | 32        | 27393          | 600033    | 5754            | 0.275                    |

Please note that the yolo11n_s8_v2_s3 model requires more than 8MB of PSRAM on ESP32-S3 when processing inputs of size 640 * 640 * 3.

## Model Usage

``COCODetect`` accepts a ``COCODetect::model_type_t`` parameter. It has a default value determined by [default model](#default-model) option in menuconfig.

### How to New `COCODetect`

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
// use YOLO11N_S8_V3
// COCODetect *detect = new COCODetect(COCODetect::YOLO11N_S8_V3);
// use YOLO11N_320_S8_V3
// COCODetect *detect = new COCODetect(COCODetect::YOLO11N_320_S8_V3);
```
> [!NOTE] 
> If mutiple models is flashed or stored in sdcard, in addition to the default model, you can pass an explicit parameter to ``COCODetect`` to use one of them.

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
- CONFIG_FLASH_COCO_DETECT_YOLO11N_S8_V2
- CONFIG_FLASH_COCO_DETECT_YOLO11N_S8_V3
- CONFIG_FLASH_COCO_DETECT_YOLO11N_320_S8_V3

Whether to flash the model when model location is set to FLASH rodata or FLASH partition.

## Default Model

- CONFIG_COCO_DETECT_YOLO11N_S8_V1
- CONFIG_COCO_DETECT_YOLO11N_S8_V2
- CONFIG_COCO_DETECT_YOLO11N_S8_V3
- CONFIG_COCO_DETECT_YOLO11N_320_S8_V3

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