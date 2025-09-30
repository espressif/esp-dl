[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |

# Dog Detect Example

A simple image inference example. In this example, we use ``dog.jpg`` for test. With default setting(iou=0.7, conf=0.6), the detection result before quantization is shown below:

![](./img/dog_224_224_fp32.jpg)

And the detection result of espdet_pico_224_224_dog after int8 quantization on ESP32-P4 is as follows:

![](./img/dog_224_224_int8_p4.jpg)

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (1637) dog_detect: [category: 0, score: 0.880797, x1: 265, y1: 110, x2: 471, y2: 388]
I (1647) main_task: Returned from app_main()
```

## Configurable Options in Menuconfig

### Component configuration
We provide the models as components, each of them has some configurable options.

### Project configuration

- CONFIG_PARTITION_TABLE_CUSTOM_FILENAME

If model location is set to FLASH partition, please set this option to `partitions2.csv`

