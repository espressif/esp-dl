# ESP-DL 

The new ESP-DL offers the best AI performance on ESP series chips, and you can use the new ESP-DL models with the same convenience as ONNX.  Most of the code has been open-sourced, allowing you modify and add your own operators and models.

**Note:**
This release is not compatible with the master branch, and currently supports only the ESP32-P4.  
We will be adding more documentation and support ESP32-S3 in the next month.

## Overview
The new ESP-DL includes the following new features:


### ESP-DL standand model format

The ESP-DL standard model format is a binary format used to store the model graph, weights, and other information.The file extension is .espdl.


This format is similar to the ONNX model format, but replaces ONNX's [Protobuf](https://github.com/protocolbuffers/protobuf) with [FlatBuffers](https://github.com/google/flatbuffers), making models more lightweight and supporting zero-copy deserialization. This means that accessing the serialized data does not require first copying it into a separate part of memory. This makes data access much faster than formats that require more extensive processing, such as Protobuf.

### Model Quantization:[esp-ppq](https://github.com/espressif/esp-ppq)

ESP-PPQ is a model quantization tool developed based on the open-source project PPQ. Users can export ESP-DL standard model by selecting the ESPDL target platform. ESP-PPQ inherits all the functions and documents from the PPQ project, allowing users to try different quantization algorithms and analyze quantization errors.

For more details please refer to [esp-ppq](https://github.com/espressif/esp-ppq).

### Efficient Operator Implementation

Based on the AI instruction,Common AI operators is efficiently implemented , including Conv2d, Pool2D, Gemm, Add, Mul, etc.  At the same time. Operators precisely aligned with the PyTorch operator implementation to ensure that the results obtained from esp-ppq tool are consistent with the results running on ESP-DL.

### Static Memory Planner

A new static memory planner is designed for Internal RAM/PSRAM memory structure. Considering that internal RAM has faster access speed but limited capacity, we provide an API that allows users to customize the size of the internal RAM that the model can use. The memory planner will automatically allocate different layers to the optimal memory location based on the size of the internal RAM specified by the user, ensuring that the overall running speed is more efficient while occupying the minimum amount of memory.


### Dual Core Scheduling

The automatic dual-core scheduling enables computationally intensive operators to fully utilize the computing power of dual-cores. Currently, Conv2D and DepthwiseConv2D support dual-core scheduling. Below are some of our experimental results:

| |conv2d(input=224X224X3, kernel=3x3, output=112x112x16)|
|:---:|:---:|
|single core| 12.1.ms|
|dual core| 6.2 ms|


## Support models

[Pedestrian Detection](./models/pedestrian_detect/)

[Human Face Detection](./models/human_face_detect/)
