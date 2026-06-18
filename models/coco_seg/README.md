# COCO Segmentation Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"
[no support]: https://img.shields.io/badge/-no_support-red "no support"

| Chip     |   YOLO11N_SEG_S8_V1   | 
|----------|------------------------|
| ESP32-S3 | ![alt text][supported] | 
| ESP32-P4 | ![alt text][supported] | 

- `yolo11n_seg_s8_v1_s3` and `yolo11n_seg_s8_v1_p4` use 8bit quantization.


## Model Latency

| name                  | input(h*w*c)   | Flash(MB) | PSRAM(MB) | preprocess(ms) | model(ms) | postprocess(ms) | Box mAP50-95 on COCO | Mask mAP50-95 on COCO |
| --------------------- | -------------- | ----------- | ----------- | ---------------- | ----------- | ----------------- | ------------------ | ------------------ |
|  yolo11n_seg_s8_v1_s3 |  640 * 640 * 3 |  8 |  8 | 51.5 | 27636.7 | 138.6 |  0.363 |  0.302 |
|  yolo11n_seg_s8_v1_p4 |  640 * 640 * 3 |  16 |  32 | 15.9 | 3054.5 | 39.6|  0.37 |  0.307 |

## Model Usage

``COCOSeg`` accepts a ``COCOSeg::model_type_t`` parameter. It has a default value determined by [default model](#default-model) option in menuconfig.

### How to New `COCOSeg`

```cpp
COCOSeg *seg = new COCOSeg();
```

> [!NOTE] 
> If multiple models is flashed or stored in sdcard, in addition to the default model, you can pass an explicit parameter to ``COCOSeg`` to use one of them.

### How to Segment

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
std::list<dl::detect::result_t> &res = seg->run(img);
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp) and [`dl::detect::result_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/detect/dl_detect_define.hpp).

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model to Flash

- CONFIG_FLASH_COCO_SEG_YOLO11N_SEG_S8_V1

Whether to flash the model when model location is set to FLASH rodata or FLASH partition.

## Default Model

- CONFIG_COCO_SEG_YOLO11N_SEG_S8_V1

Default model to use if no parameter is passed to ``COCOSeg``.

## Model Location

- CONFIG_COCO_SEG_MODEL_IN_FLASH_RODATA
- CONFIG_COCO_SEG_MODEL_IN_FLASH_PARTITION
- CONFIG_COCO_SEG_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE]
> - If model location is set to FLASH partition, `partition.csv` must contain a partition named `coco_seg`, and the partition should be big enough to hold the model file.
> - When using YOLO11n-seg, do not set model location to sdcard if your PSRAM size is lower than 16MB. 

## SDCard Directory

- CONFIG_COCO_SEG_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the models to sdcard.