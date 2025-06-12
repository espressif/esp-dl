# Human Face Recognition Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | MFN_S8_V1              | MBF_S8_V1              |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |

## Model Latency

| name         | input(h*w*c)  | preprocess(us) | model(us) | postprocess(us) |
|--------------|---------------|----------------|-----------|-----------------|
| mfn_s8_v1_s3 | 112 * 112 * 3 | 8380           | 248803    | 80              |
| mfn_s8_v1_p4 | 112 * 112 * 3 | 5198           | 93003     | 51              |
| mbf_s8_v1_s3 | 112 * 112 * 3 | 8386           | 1072427   | 81              |
| mbf_s8_v1_p4 | 112 * 112 * 3 | 5197           | 188221    | 52              |

## Model Metrics

| Model     | Params(M) | GFLOPs | TAR@FAR=1E-4 on IJB-C(%) |
|-----------|-----------|--------|--------------------------|
| mfn_s8_v1 | 1.2       | 0.46   | 90.03                    |
| mbf_s8_v1 | 3.4       | 0.90   | 93.94                    |

## Model Usage

``HumanFaceFeat`` accepts a ``HumanFaceFeat::model_type_t`` parameter. It has a default value determined by [default model](#default-model) option in menuconfig.

### How to New `HumanFaceFeat`

#### Only One Model

```cpp
HumanFaceFeat *feat = new HumanFaceFeat();
```

#### Mutiple Models

```cpp
// use MFN_S8_V1
HumanFaceFeat *feat = new HumanFaceFeat(HumanFaceFeat::MFN_S8_V1);
// use MBF_S8_V1
// HumanFaceFeat *feat = new HumanFaceFeat(HumanFaceFeat::MBF_S8_V1);
```
> [!NOTE] 
> If mutiple models is flashed or stored in sdcard, in addition to the default model, you can pass an explicit parameter to ``HumanFaceFeat`` to use one of them.

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model to Flash

- CONFIG_FLASH_HUMAN_FACE_FEAT_MFN_S8_V1
- CONFIG_FLASH_HUMAN_FACE_FEAT_MBF_S8_V1

Whether to flash the model when model location is set to FLASH rodata or FLASH partition.

## Default Model

- CONFIG_HUMAN_FACE_FEAT_MFN_S8_V1
- CONFIG_HUMAN_FACE_FEAT_MBF_S8_V1

Default model to use if no parameter is passed to ``HumanFaceFeat``.

## Model Location

- CONFIG_HUMAN_FACE_FEAT_MODEL_IN_FLASH_RODATA
- CONFIG_HUMAN_FACE_FEAT_MODEL_IN_FLASH_PARTITION
- CONFIG_HUMAN_FACE_FEAT_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE] 
> If model location is set to FLASH partition, `partition.csv` must contain a partition named `human_face_feat`, and the partition should be big enough to hold the model file.

## SDCard Directory

- CONFIG_HUMAN_FACE_FEAT_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the models to sdcard.

# Human Face Recognizer

This component also contains `HumanFaceRecognizer`. It's a integration of `HumanFaceFeat` and `dl::recognition::DataBase`. If you want to enroll/recognize a human face, in addition to `HumanFaceRecognizer`, you also need `HumanFaceDetect`. See [How to New HumanFaceDetect](https://github.com/espressif/esp-dl/blob/master/models/human_face_detect/README.md#how-to-new-humanfacedetect).

## How to New Human Face Recognizer

```cpp
HumanFaceRecognizer *human_face_recognizer = new HumanFaceRecognizer("path/to/database");
```

## How to Enroll a Human Face

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
human_face_recognizer->enroll(img, human_face_detect->run(img));
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp).

## How to Recognize a Human Face

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
std::vector<dl::recognition::result_t> res = human_face_recognizer->recognize(img, human_face_detect->run(img));
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp) and [`dl::recognition::result_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/recognition/dl_recognition_define.hpp).

## How to Delete Featrue in Database
### Delete all Features

```cpp
human_face_recognizer->clear_all_feats();
```
### Delete Last Feature

```cpp
human_face_recognizer->delete_last_feat();
```

### Delete Feature by Index

```cpp
human_face_recognizer->delete_feat(index);
```