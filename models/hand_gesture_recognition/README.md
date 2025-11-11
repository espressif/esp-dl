# Hand Gesture Recognition Models

## Model list
[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | MOBILENETV2_0_5_S8_V1  | 
|----------|------------------------|
| ESP32-S3 | ![alt text][supported] | 
| ESP32-P4 | ![alt text][supported] |


## Model Latency

| name                     | input(h*w*c)  | preprocess(us) | model(us) | postprocess(us) |
|--------------------------|---------------|----------------|-----------|-----------------|
| mobilenetv2_0_5_s8_v1_s3 | 128 * 128 * 3 | 2365           | 115572    |  405            |
| mobilenetv2_0_5_s8_v1_p4 | 128 * 128 * 3 | 891            | 28499     |  211            |

## Model Usage

``HandGestureCls`` accepts a ``HandGestureCls::model_type_t`` parameter. It has a default value determined by [default model](#default-model) option in menuconfig.

### How to New `HandGestureCls`

```cpp
HandGestureCls *m_cls = new HandGestureCls();
```

# Configurable Options in Menuconfig

See [Kconfig](Kconfig).

## Model to Flash

- CONFIG_FLASH_HAND_GESTURE_CLS_MOBILENETV2_0_5_S8_V1


Whether to flash the model when the model location is set to FLASH rodata or FLASH partition.

## Default Model

- CONFIG_HAND_GESTURE_CLS_MOBILENETV2_0_5_S8_V1

Default model to use if no parameter is passed to ``HandGestureCls``.

## Model Location

- CONFIG_HAND_GESTURE_CLS_MODEL_IN_FLASH_RODATA
- CONFIG_HAND_GESTURE_CLS_MODEL_IN_FLASH_PARTITION
- CONFIG_HAND_GESTURE_CLS_MODEL_IN_SDCARD

This component supports to [load model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html) from three different locations.

> [!NOTE] 
> If model location is set to FLASH partition, `partition.csv` must contain a partition named `hand_gesture_cls`, and the partition should be big enough to hold the model file.

## SDCard Directory

- CONFIG_HAND_GESTURE_CLS_MODEL_SDCARD_DIR

When model locates in sdcard, you can change the model directory relative to the sdcard mount point.   

The default value of this option is `models/s3` for ESP32S3 and `models/p4` for ESP32P4. 
When using default value, just copy [models](models) folder to sdcard root directory.

> [!NOTE] 
> Do not change the model name when copy the models to sdcard.

# Hand Gesture Recognizer

This component also contains `HandGestureRecognizer`. If you want to recognize a hand gesture, in addition to `HandGestureRecognizer`, you also need `HandDetect`. See [How to New HandDetect](https://github.com/espressif/esp-dl/blob/master/models/hand_detect/README.md#how-to-new-handdetect).

## How to New Hand Gesture Recognizer

```cpp
HandGestureRecognizer *hand_gesture_recognizer = new HandGestureRecognizer("????");
```

## How to Recognize a Hand Gesture

```cpp
dl::image::img_t img = {.data=DATA, .width=WIDTH, .height=HEIGHT, .pix_type=PIX_TYPE};
std::vector<dl::recognition::result_t> res = hand_gesture_recognizer->recognize(img, hand_detect->run(img));
```

More details, see [`dl::image::img_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/image/dl_image_define.hpp) and [`dl::recognition::result_t`](https://github.com/espressif/esp-dl/blob/master/esp-dl/vision/recognition/dl_recognition_define.hpp).


```