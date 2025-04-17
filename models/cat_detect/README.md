# Cat Detection Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"
[no support]: https://img.shields.io/badge/-no_support-red "no support"

| Chip     | ESPDET_PICO_224_224_CAT | ESPDET_PICO_416_416_CAT | 
|----------|-------------------------|-------------------------|
| ESP32-S3 | ![alt text][supported]  | ![alt text][supported]  | 
| ESP32-P4 | ![alt text][supported]  | ![alt text][supported]  | 

## Model Latency

| name                          | input(h*w*c)  | preprocess(us) | model(us) | postprocess(us) | mAP50-95 on COCO2017 cat |
|-------------------------------|---------------|----------------|-----------|-----------------|--------------------------|
| espdet_pico_224_224_cat_s8_s3 | 224 * 224 * 3 | 27139          | 126234    | 3350            | 0.666                    | 
| espdet_pico_416_416_cat_s8_s3 | 416 * 416 * 3 | 89373          | 449522    | 4790            | 0.758                    |  
| espdet_pico_224_224_cat_s8_p4 | 224 * 224 * 3 | 13975          | 51447     | 2686            | 0.667                    |
| espdet_pico_416_416_cat_s8_p4 | 416 * 416 * 3 | 45573          | 201696    | 2811            | 0.759                    |

## Model Usage

``CatDetect`` accepts a ``CatDetect::model_type_t`` parameter. It has a default value determined by [default model](#default-model) option in menuconfig.

### How to New `CatDetect`

#### Only One Model

```cpp
CatDetect *detect = new CatDetect();
```

#### Mutiple Models

```cpp
// use ESPDET_PICO_224_224_CAT
CatDetect *detect = new CatDetect(CatDetect::ESPDET_PICO_224_224_CAT);
// use ESPDET_PICO_416_416_CAT
// CatDetect *detect = new CatDetect(CatDetect::ESPDET_PICO_416_416_CAT);
```
> [!NOTE] 
> If mutiple models is flashed or stored in sdcard, in addition to the default model, you can pass an explicit parameter to ``CatDetect`` to use one of them.

### How to Detect

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
std::list<dl::detect::result_t> &res = detect->run(img);
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp) and [`dl::detect::result_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/detect/dl_detect_define.hpp).

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model to Flash

- CONFIG_FLASH_ESPDET_PICO_224_224_CAT
- CONFIG_FLASH_ESPDET_PICO_416_416_CAT

Whether to flash the model when model location is set to FLASH rodata or FLASH partition.

## Default Model

- CONFIG_ESPDET_PICO_224_224_CAT
- CONFIG_ESPDET_PICO_416_416_CAT

Default model to use if no parameter is passed to ``CatDetect``.

## Model Location

- CONFIG_CAT_DETECT_MODEL_IN_FLASH_RODATA
- CONFIG_CAT_DETECT_MODEL_IN_FLASH_PARTITION
- CONFIG_CAT_DETECT_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE]
> - If model location is set to FLASH partition, `partition.csv` must contain a partition named `cat_det`, and the partition should be big enough to hold the model file.

## SDCard Directory

- CONFIG_CAT_DETECT_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the models to sdcard.