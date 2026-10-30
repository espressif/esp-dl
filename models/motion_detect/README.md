# Motion Detection Models

## Model Latency

| Target | Input | Stride | Latency (ms) | FPS |
|--------|-------|--------|--------------|-----|
| ESP32-S3 | rgb888 1280x720 | 1 | 197.01 | 5.08 |
| ESP32-S3 | rgb888 1280x720 | 2 | 60.68 | 16.48 |
| ESP32-S3 | rgb888 640x360 | 1 | 49.26 | 20.30 |
| ESP32-S3 | rgb888 640x360 | 2 | 15.18 | 65.88 |
| ESP32-S31 | rgb888 1280x720 | 1 | 117.88 | 8.48 |
| ESP32-S31 | rgb888 1280x720 | 2 | 38.41 | 26.03 |
| ESP32-S31 | rgb888 640x360 | 1 | 29.48 | 33.93 |
| ESP32-S31 | rgb888 640x360 | 2 | 9.61 | 104.11 |
| ESP32-P4 | rgb888 1280x720 | 1 | 86.75 | 11.53 |
| ESP32-P4 | rgb888 1280x720 | 2 | 26.51 | 37.72 |
| ESP32-P4 | rgb888 640x360 | 1 | 21.68 | 46.12 |
| ESP32-P4 | rgb888 640x360 | 2 | 6.63 | 150.81 |

> [!NOTE]
> - ESP32-S3: CPU 240 MHz, data cache 64 KB, octal PSRAM 80 MHz. ESP32-S31: CPU 320 MHz, PSRAM 250 MHz. ESP32-P4: CPU 400 MHz, L2 cache 256 KB, PSRAM 32 MB at 250 MHz, chips >= ECO5.
> - Stride controls the spacing between detection points. A smaller stride provides more reliable detection but increases latency, while a larger stride reduces latency but may miss smaller movements. For details, see [`get_moving_point_number`](motion_detect.hpp).
