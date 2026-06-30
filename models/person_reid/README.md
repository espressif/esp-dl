# Person Reid Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | OSN_S8_V1              |
|----------|------------------------|
| ESP32-S3 | ![alt text][supported] | 
| ESP32-P4 | ![alt text][supported] |

## Model Latency

| name         | input(h*w*c)  | preprocess(ms) | model(ms) | postprocess(ms) |
| -------------- | --------------- | ---------------- | ----------- | ----------------- |
|  osn_s8_v1_s3 |  192 * 96 * 3 | 5.2 | 110.0 | 0.1 |
|  osn_s8_v1_p4 |  192 * 96 * 3 | 2.1 | 40.2 | 0.0 |

## Model Usage

``PersonReidFeat`` accepts a ``PersonReidFeat::model_type_t`` parameter. It has a default value determined by [default model](#default-model) option in menuconfig.

### How to New `PersonReidFeat`

```cpp
PersonReidFeat *feat = new PersonReidFeat();
```


> [!NOTE] 
> If multiple models is flashed or stored in sdcard, in addition to the default model, you can pass an explicit parameter to ``PersonReidFeat`` to use one of them.

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model to Flash

- CONFIG_FLASH_PERSON_REID_FEAT_OSN_S8_V1

Whether to flash the model when model location is set to FLASH rodata or FLASH partition.

## Default Model

- CONFIG_PERSON_REID_FEAT_OSN_S8_V1

Default model to use if no parameter is passed to ``PersonReidFeat``.

## Model Location

- CONFIG_PERSON_REID_FEAT_MODEL_IN_FLASH_RODATA
- CONFIG_PERSON_REID_FEAT_MODEL_IN_FLASH_PARTITION
- CONFIG_PERSON_REID_FEAT_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE] 
> If model location is set to FLASH partition, `partition.csv` must contain a partition named `person_reid_feat`, and the partition should be big enough to hold the model file.

## SDCard Directory

- CONFIG_PERSON_REID_FEAT_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the models to sdcard.

# Person Reid Matcher

This component also contains `PesonReidMatcher`. It's a integration of `PesonReidFeat` and `dl::recognition::DataBase`. If you want to enroll/recognize a person, in addition to `PesonReidMatcher`, you also need `PedestrianDetect`. See [How to New PedestrianDetect](https://github.com/espressif/esp-dl/blob/master/models/pedestrian_detect/README.md#how-to-new-pedestriandetect).

## How to New Person Reid Matcher

```cpp
PesonReidMatcher *person_reid_matcher = new PesonReidMatcher("path/to/database");
```

## How to Enroll a Person

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
person_reid_matcher->enroll(img, pedestrian_detect->run(img));
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp).

## How to Recognize a Person

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
std::vector<dl::recognition::result_t> res = person_reid_matcher->recognize(img, pedestrian_detect->run(img));
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp) and [`dl::recognition::result_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/recognition/dl_recognition_define.hpp).

## How to Delete Feature in Database
### Delete all Features

```cpp
person_reid_matcher->clear_all_feats();
```
### Delete Last Feature

```cpp
person_reid_matcher->delete_last_feat();
```

### Delete Feature by Index

```cpp
person_reid_matcher->delete_feat(index);
```