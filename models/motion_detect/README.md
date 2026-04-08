# Motion Detection Models

## Model Latency

| Target | Input | Stride | Latency (ms) | FPS |
|--------|-------|--------|--------------|-----|
| ESP32-S3 | rgb888 1280x720 | 2 | 197.03 | 5.08 |
| ESP32-S3 | rgb888 1280x720 | 1 | 61.02 | 16.39 |
| ESP32-S3 | rgb888 640x360 | 2 | 49.37 | 20.26 |
| ESP32-S3 | rgb888 640x360 | 1 | 15.36 | 65.11 |
| ESP32-P4 | rgb888 1280x720 | 2 | 86.82 | 11.52 |
| ESP32-P4 | rgb888 1280x720 | 1 | 26.82 | 37.28 |
| ESP32-P4 | rgb888 640x360 | 2 | 21.92 | 45.62 |
| ESP32-P4 | rgb888 640x360 | 1 | 6.92 | 144.51 |

> [!NOTE]
> - The ESP32-P4 latency data above was tested on chips >= ECO5, with PSRAM frequency at 250MHz and CPU frequency at 400MHz.
> - Stride controls the spacing between detection points. A smaller stride provides more reliable detection but increases latency, while a larger stride reduces latency but may miss smaller movements. For details, see [`get_moving_point_number`](motion_detect.hpp).
