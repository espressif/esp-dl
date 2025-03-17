[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |

# Yolo11 Detect Example

A simple image inference example. In this example, we use ``bus.jpg`` for test. With default setting(iou=0.7, conf=0.25), the detection result before quantization is shown below:

![](./img/bus_fp32.jpg)

And the detection result after int8 quantization is as follows:

![](./img/bus_int8.jpg)

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (28477) yolo11n: [category: 0, score: 0.817575, x1: 24, y1: 196, x2: 111, y2: 453]
I (28477) yolo11n: [category: 5, score: 0.731059, x1: 81, y1: 115, x2: 400, y2: 372]
I (28477) yolo11n: [category: 0, score: 0.731059, x1: 112, y1: 203, x2: 171, y2: 429]
I (28487) yolo11n: [category: 0, score: 0.731059, x1: 336, y1: 196, x2: 404, y2: 436]
I (28497) yolo11n: [category: 0, score: 0.320821, x1: 0, y1: 276, x2: 29, y2: 434]
```

## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options. See [COCO Detect Model](https://github.com/espressif/esp-dl/blob/master/models/coco_detect/README.md)。

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`

