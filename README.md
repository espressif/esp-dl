# ESP-DL [[中文]](./README_cn.md)

 ESP-DL is a lightweight and efficient neural network inference framework optimized for ESP series chips. 

**Note:**
This release is not compatible with the master branch, and currently supports only the ESP32-P4.  
We will be adding more documentation and support ESP32-S3 in the next month.

## Overview
ESP-PPQ can quantize models fromm ONNX, pytorch and tensorflow to the ESP-DL standard model format. ESP-DL provides APIs to load, debug, and run those models. With ESP-DL, you can use Espressif's SoCs for AI applications easily and fast.

<p align="center">
    <img width="%" src="./docs/_static/architecture_en.drawio.svg">
</p>

## Support models

[Pedestrian Detection](./models/pedestrian_detect/)   
[Human Face Detection](./models/human_face_detect/)

## Getting Started

upcoming

> Please use [ESP-IDF](https://github.com/espressif/esp-idf) `release/v5.3` or above.