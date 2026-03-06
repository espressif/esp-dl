# Human Face Detection Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | MSR_S8_V1 + <br>MNP_S8_V1 | ESPDET_PICO_224_224_FACE | ESPDET_PICO_416_416_FACE | 
|----------|---------------------------|--------------------------|--------------------------|
| ESP32-S3 | ![alt text][supported]    |  ![alt text][supported]  |  ![alt text][supported]  | 
| ESP32-P4 | ![alt text][supported]    |  ![alt text][supported]  |  ![alt text][supported]  | 

MSR_S8_V1 + MNP_S8_V1 is a two stage model. First stage model MSR_S8_V1 predicts some candidates, then every candidate go through the next stage model MNP_S8_V1.

Both ESPDET_PICO_224_224_FACE and ESPDET_PICO_416_416_FACE are one-stage models.

## Model Latency

| name                           | input(h*w*c)  | preprocess(us) | model(us) | postprocess(us) | mAP50-95 on custom val set  |
|--------------------------------|---------------|----------------|-----------|-----------------|-----------------------------|
| msr_s8_v1_s3                   | 120 * 160 * 3 | 4023           | 32403     | 222             | /                           |
| msr_s8_v1_p4                   | 120 * 160 * 3 | 1559           | 14611     | 150             | /                           |
| mnp_s8_v1_s3                   | 48 * 48 * 3   | 1110           | 5551      | 63              | 0.367(msr + mnp)            |
| mnp_s8_v1_p4                   | 48 * 48 * 3   | 637            | 2659      | 45              | 0.366(msr + mnp)            |
| espdet_pico_224_224_face_s8_s3 | 224 * 224 * 3 | 7227           | 122463    | 782             | 0.495                       |
| espdet_pico_416_416_face_s8_s3 | 416 * 416 * 3 | 21688          | 436376    | 1246            | 0.598                       |
| espdet_pico_224_224_face_s8_p4 | 224 * 224 * 3 | 2592           | 50594     | 468             | 0.496                       |
| espdet_pico_416_416_face_s8_p4 | 416 * 416 * 3 | 7177           | 200227    | 687             | 0.599                       |

## Model Usage

### How to New `HumanFaceDetect`

```cpp
HumanFaceDetect *detect = new HumanFaceDetect();
```
### How to Detect Human Faces

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
detect->run(img);
```

More detail about `dl::image::img_t`, see [dl_image_define.hpp](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp).

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model Location

- CONFIG_HUMAN_FACE_DETECT_MODEL_IN_FLASH_RODATA
- CONFIG_HUMAN_FACE_DETECT_MODEL_IN_FLASH_PARTITION
- CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> **note:** If model location is set to FLASH partition, `partition.csv` must contain a partition named `human_face_det`, and the partition should be big enough to hold the model file.

## SDCard Directory

- CONFIG_HUMAN_FACE_DETECT_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> **note:** Do not change the model name when copy the models to sdcard.