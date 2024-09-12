# ESP-DL [[English]](./README.md)

ESP-DL是一个为ESP系列芯片优化的轻量级且高效的神经网络推理框架。

**Note:**
该release与主分支不兼容，目前仅支持ESP32-P4。

## Overview
ESP-PPQ可以量化ONNX，pytorch和tensorflow模型到ESP-DL标准模型格式。ESP-DL提供了加载，调试和运行这些模型的API。使用ESP-DL，你可以轻松快速地使用Espressif的SoC进行AI应用开发。

<p align="center">
    <img width="600" src="./docs/_static/architecture_cn.drawio.svg">
</p>

## Support models

[行人检测](./models/pedestrian_detect/)  
[人脸检测](./models/human_face_detect/)

## Getting Started

upcoming

> Please use [ESP-IDF](https://github.com/espressif/esp-idf) `release/v5.3` or above.