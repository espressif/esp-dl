# COCO Pose Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"
[no support]: https://img.shields.io/badge/-no_support-red "no support"

| Chip     |   YOLO11N_POSE_S8_V1   |   YOLO11N_POSE_S8_V2    |
|----------|------------------------|-------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported]  |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported]  |

- `yolo11n_pose_s8_v1_s3` and `yolo11n_pose_s8_v1_p4` use 8bit default configuration quantization.
- `yolo11n_pose_s8_v2_s3` , `yolo11n_pose_s8_v2_p4` use Quantization-Aware Training.

## Model Latency

| name                      | input(h*w*c)  | Flash(MB) | PSRAM(MB) | preprocess(us) | model(us) | postprocess(us) | mAP50-95 on COCO |
|---------------------------|---------------|-----------|-----------|----------------|-----------|-----------------|------------------|
| yolo11n_pose_s8_v1_s3     | 640 * 640 * 3 | 8         | 8         | 202842         | 28687664  | 10427           | 0.431            |
| yolo11n_pose_s8_v2_s3     | 640 * 640 * 3 | 8         | 8         | 202837         | 28695904  | 10259           | 0.451            |  
| yolo11n_pose_s8_v1_p4     | 640 * 640 * 3 | 16        | 32        | 105768         | 3346157   | 6538            | 0.429            |
| yolo11n_pose_s8_v2_p4     | 640 * 640 * 3 | 16        | 32        | 105770         | 3344908   | 7258            | 0.454            |

## Model Usage

``COCOPose`` accepts a ``COCOPose::model_type_t`` parameter. It has a default value determined by [default model](#default-model) option in menuconfig.

### How to New `COCOPose`

#### Only One Model

```cpp
COCOPose *pose = new COCOPose();
```

#### Mutiple Models

```cpp
// use YOLO11N_POSE_S8_V1
COCOPose *pose = new COCOPose(COCOPose::YOLO11N_POSE_S8_V1);
// use YOLO11N_POSE_S8_V2
// COCOPose *pose = new COCOPose(COCOPose::YOLO11N_POSE_S8_V2);
```
> [!NOTE] 
> If mutiple models is flashed or stored in sdcard, in addition to the default model, you can pass an explicit parameter to ``COCOPose`` to use one of them.

### How to Detect

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
std::list<dl::detect::result_t> &res = pose->run(img);
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp) and [`dl::detect::result_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/detect/dl_detect_define.hpp).

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model to Flash

- CONFIG_FLASH_COCO_POSE_YOLO11N_POSE_S8_V1
- CONFIG_FLASH_COCO_POSE_YOLO11N_POSE_S8_V2

Whether to flash the model when model location is set to FLASH rodata or FLASH partition.

## Default Model

- CONFIG_COCO_POSE_YOLO11N_POSE_S8_V1
- CONFIG_COCO_POSE_YOLO11N_POSE_S8_V2

Default model to use if no parameter is passed to ``COCOPose``.

## Model Location

- CONFIG_COCO_POSE_MODEL_IN_FLASH_RODATA
- CONFIG_COCO_POSE_MODEL_IN_FLASH_PARTITION
- CONFIG_COCO_POSE_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE]
> - If model location is set to FLASH partition, `partition.csv` must contain a partition named `coco_pose`, and the partition should be big enough to hold the model file.
> - When using YOLO11n-pose, do not set model location to sdcard if your PSRAM size is lower than 16MB. 

## SDCard Directory

- CONFIG_COCO_POSE_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the models to sdcard.