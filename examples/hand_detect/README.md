[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |

# Hand Detect Example

A simple image inference example. In this example, we use ``hand.jpg`` for test. With default setting(iou=0.5, conf=0.25), the detection result before quantization is shown below:

![](./img/hand_224_224_fp32.jpg)

And the detection result of espdet_pico_224_224_hand after int8 quantization on ESP32-P4 is as follows:

![](./img/hand_224_224_int8_p4.jpg)

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (1614) hand_detect: [category: 0, score: 0.622459, x1: 322, y1: 154, x2: 377, y2: 208]
I (1614) hand_detect: [category: 0, score: 0.622459, x1: 137, y1: 243, x2: 185, y2: 271]
I (1624) hand_detect: [category: 0, score: 0.377541, x1: 328, y1: 334, x2: 397, y2: 380]
I (1634) main_task: Returned from app_main()
```

## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options.

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`

