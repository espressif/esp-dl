# ESP-DL [[English]](./README.md)

[![Documentation Status](./docs/_static/doc_latest.svg)](https://docs.espressif.com/projects/esp-dl/zh_CN/latest/index.html)    [![Component Registry](https://components.espressif.com/components/espressif/esp-dl/badge.svg)](https://components.espressif.com/components/espressif/esp-dl)

ESP-DL 是一个专为 ESP 系列芯片设计的轻量级且高效的神经网络推理框架。通过 ESP-DL，您可以轻松快速地使用乐鑫的系统级芯片 (SoC) 开发 AI 应用。

## Overview

ESP-DL 提供了加载、调试和运行 AI 模型的 API。该框架易于使用，并且可以与其他乐鑫 SDK 无缝集成。ESP-PPQ 作为 ESP-DL 的量化工具，能够量化来自 ONNX、Pytorch 和 TensorFlow 的模型，并将其导出为 ESP-DL 标准模型格式。

- **ESP-DL 标准模型格式：** 该格式类似于 ONNX，但使用 FlatBuffers 而不是 Protobuf，使其更轻量级并支持零拷贝反序列化，文件后缀为`.espdl`。

- **高效算子实现：** ESP-DL 高效地实现了常见的 AI 算子，如 Conv、Pool、Gemm、Add 和 Mul等。目前[算子支持状态](./operator_support_state.md)

- **静态内存规划器：** 内存规划器根据用户指定的内部 RAM 大小，自动将不同层分配到最佳内存位置，确保高效的整体运行速度同时最小化内存使用。

- **双核调度：** 自动双核调度允许计算密集型算子充分利用双核计算能力。目前，Conv2D 和 DepthwiseConv2D 支持双核调度。

- **8bit LUT Activation：** 除了Relu, PRelu(n>1)之外的所有激活函数，ESP-DL 默认使用 8bit LUT(Look Up Table)方式实现,以加速推理。


## 更新 

- [2025/04/30] 我们发布了全新的 [esp-detection](https://github.com/espressif/esp-detection) 项目和 `ESPDet-Pico` 模型，可轻松训练并部署目标检测模型。[espdet_pico_224_224_cat 模型](./models/cat_detect/) 和 [示例](./examples/cat_detect/) 是基于 esp-detection 训练的猫咪检测模型，欢迎试用并反馈意见！    
- [2025/02/18] 新增支持 YOLO11n 模型，提供 [espdl 模型](https://github.com/espressif/esp-dl/tree/master/models/coco_detect) 及 [示例](https://github.com/espressif/esp-dl/tree/master/examples/yolo11_detect)。    
- [2025/01/09] 更新了 espdl 模型架构，发布 ESP-DL v3.1.0。注意：旧版模型可被新架构加载，但新版模型不兼容旧版本。   
- [2024/12/20] 发布 ESP-DL v3.0.0。   


## Getting Started

### 软件要求

- **ESP-IDF**  

ESP-DL 基于 ESP-IDF 运行。有关如何获取 ESP-IDF 的详细说明，请参阅 [ESP-IDF 编程指南](https://idf.espressif.com)。

> 请使用 [ESP-IDF](https://github.com/espressif/esp-idf) `release/v5.3` 或更高版本。

- **ESP-PPQ**

ESP-PPQ 是基于 ppq 的量化工具。如果你想量化自己的模型，请使用以下命令安装 esp-ppq：
```
pip uninstall ppq
pip install git+https://github.com/espressif/esp-ppq.git
```
在 docker 中使用 esp-ppq：
```
docker build -t esp-ppq:your_tag https://github.com/espressif/esp-ppq.git
```

### Model Quantization

首先，请参考 [ESP-DL 算子支持状态](./operator_support_state.md)，确保您的模型中的算子已经得到支持。

ESP-PPQ 可以直接读取 ONNX 模型进行量化。Pytorch 和 TensorFlow 需要先转换为 ONNX 模型，因此请确保你的模型可以转换为 ONNX 模型。更多详细信息请参阅:  

[如何量化模型](https://docs.espressif.com/projects/esp-dl/zh_CN/latest/tutorials/how_to_quantize_model.html)  
[如何量化 MobileNetV2](https://docs.espressif.com/projects/esp-dl/zh_CN/latest/tutorials/how_to_deploy_mobilenetv2.html#how-to-quantize-mobilenetv2)   
[如何量化 YOLO11n](https://docs.espressif.com/projects/esp-dl/zh_CN/latest/tutorials/how_to_deploy_yolo11n.html#how-to-quantize-yolo11n)  


### Model Deployment
ESP-DL 提供了一系列 API 来快速加载和运行模型。更多详细信息，请参阅：

[如何加载和测试模型](https://docs.espressif.com/projects/esp-dl/zh_CN/latest/tutorials/how_to_load_test_profile_model.html)  
[如何进行模型推理](https://docs.espressif.com/projects/esp-dl/zh_CN/latest/tutorials/how_to_run_model.html)  


## Support models

[行人检测](./models/pedestrian_detect/)     
[人脸检测](./models/human_face_detect/)     
[人脸识别](./models/human_face_recognition/)     
[Imagenet 分类](./models/imagenet_cls/)    
[COCO 检测](./models/coco_detect/)    
[猫检测](./models/cat_detect/)  
[姿态估计](./models/coco_pose/)

## Suport Operators

如果你有遇到不支持的算子，请将问题在[issues](https://github.com/espressif/esp-dl/issues)中反馈给我们，我们会尽快支持。  
也欢迎大家贡献新的算子, 具体方法请参考[创建新模块（算子）](https://docs.espressif.com/projects/esp-dl/zh_CN/latest/tutorials/how_to_add_a_new_module%28operator%29.html)。

[算子支持状态](./operator_support_state.md)