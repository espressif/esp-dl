[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |

# Cat Detect Example

A simple image inference example. In this example, we use ``cat.jpg`` for test. With default setting(iou=0.7, conf=0.25), the detection result before quantization is shown below:

![](./img/cat_fp32.jpg)

And the detection result of espdet_pico_224_224_cat after int8 quantization on ESP32-P4 is as follows:

![](./img/cat_n_p4.jpg)

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (1563) cat_detect: [category: 0, score: 0.939913, x1: 357, y1: 212, x2: 639, y2: 479]
I (1563) cat_detect: [category: 0, score: 0.851953, x1: 200, y1: 109, x2: 394, y2: 207]
I (1573) cat_detect: [category: 0, score: 0.817575, x1: 74, y1: 250, x2: 280, y2: 336]
I (1583) main_task: Returned from app_main()
```

## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options.

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`

