# ESP-DL [[中文]](./README_cn.md)

ESP-DL is a lightweight and efficient neural network inference framework designed specifically for ESP series chips. With ESP-DL, you can easily and quickly develop AI applications using Espressif's System on Chips (SoCs).

## Overview

ESP-DL offers APIs to load, debug, and run AI models. The framework is easy to use and can be seamlessly integrated with other Espressif SDKs. ESP-PPQ serves as the quantization tool for ESP-DL, capable of quantizing models from ONNX, Pytorch, and TensorFlow, and exporting them into the ESP-DL standard model format.

- **ESP-DL Standard Model Format**: This format is similar to ONNX but uses FlatBuffers instead of Protobuf, making it more lightweight and supporting zero-copy deserialization, with a file extension of `.espdl`.
- **Efficient Operator Implementation**: ESP-DL efficiently implements common AI operators such as Conv, Gemm, Add, and Mul. [**The list of supported operators**](./operator_support_state.md).
- **Static Memory Planner**: The memory planner automatically allocates different layers to the optimal memory location based on the user-specified internal RAM size, ensuring efficient overall running speed while minimizing memory usage.
- **Dual Core Scheduling**: Automatic dual-core scheduling allows computationally intensive operators to fully utilize the dual-core computing power. Currently, Conv2D and DepthwiseConv2D support dual-core scheduling.
- **8bit LUT Activation**: All activation functions except for ReLU and PReLU are implemented using an 8-bit LUT (Look Up Table) method in ESP-DL to accelerate inference. You can use any activation function, and their computational complexity remains the same.

     

## Getting Started

### Software Requirements

- **ESP-IDF**  

ESP-DL runs based on ESP-IDF. For detailed instructions on how to get ESP-IDF, please see [ESP-IDF Programming Guide](https://idf.espressif.com).

> Please use [ESP-IDF](https://github.com/espressif/esp-idf) `release/v5.3` or above.


- **ESP-PPQ**

ESP-PPQ is a quantization tool based on ppq. If you want to quantize your own model, please install esp-ppq using the following command:
```
pip uninstall ppq
pip install git+https://github.com/espressif/esp-ppq.git
```

### Model Quantization

First, please refer to the [ESP-DL Operator Support State](./operator_support_state.md) to ensure that the operators in your model are already supported.  

ESP-PPQ can directly read ONNX models for quantization. Pytorch and TensorFlow need to be converted to ONNX models first, so make sure your model can be converted to ONNX models.  
We provide the following python script templates. Please select the appropriate template to quantize your models. For more details about quantization, please refer to [tutorial/how_to_quantize_model](./tutorial/how_to_quantize_model_en.md).  

[quantize_onnx_model.py](./tools/quantization/quantize_onnx_model.py): Quantize ONNX models

[quantize_torch_model.py](./tools/quantization/quantize_torch_model.py): Quantize Pytorch models

[quantize_tf_model.py](./tools/quantization/quantize_tf_model.py): Quantize TensorFlow models


### Model Deployment
ESP-DL provides a series of API to quickly load and run models.  A typical example is as follows:


```cpp
#include "dl_model_base.hpp"

extern const uint8_t espdl_model[] asm("_binary_model_name_espdl_start");
Model *model = new Model((const char *)espdl_model, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
model->run(inputs); // inputs is a tensor or a map of tensors
```

For more details, please refer to [tutorial/how_to_load_model](./tutorial/how_to_load_model_en.md) and [mobilenet_v2 examples](./examples/mobilenet_v2/)


## Support Models

[Pedestrian Detection](./models/pedestrian_detect/)     
[Human Face Detection](./models/human_face_detect/)     
[Human Face Recognition](./models/human_face_recognition/)     
[Imagenet classification](./models/imagenet_cls/)    


## Suport Operators

If you encounter unsupported operators, please point them out in the [issues](https://github.com/espressif/esp-dl/issues), and we will support them as soon as possible. Contributions to this ESPDL are also welcomed.

[ESP-DL Operator Support State](./operator_support_state.md)