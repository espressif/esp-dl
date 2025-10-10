[supported]: https://img.shields.io/badge/-supported-green "supported"

| Chip     | ESP-IDF v5.3           | ESP-IDF v5.4           |
|----------|------------------------|------------------------|
| ESP32-S3 | ![alt text][supported] | ![alt text][supported] |
| ESP32-P4 | ![alt text][supported] | ![alt text][supported] |


# Color Detect Example

A simple image inference example. 

## Quick start

Follow the [quick start](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html#quick-start) to flash the example, you will see the output in idf monitor:

```
I (886) color_detect: [name: red, [x1: 159, y1: 54, x2: 265, y2: 223]]
I (886) color_detect: [name: yellow, [x1: 139, y1: 295, x2: 299, y2: 402]]
I (896) color_detect: [name: green, [x1: 357, y1: 101, x2: 536, y2: 297]]
```