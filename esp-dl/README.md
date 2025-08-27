# ESP-DL

ESP-DL is a lightweight and efficient neural network inference framework specifically designed for ESP series chips (ESP32, ESP32-S3, ESP32-P4). It is built to maintain optimal performance while significantly reducing the workload in model deployment. Our project has achieved the following key features:

### ESP-DL Standard Model Format

The ESP-DL standard model format is a binary format used to store the model graph, weights, and other essential information, with a file extension of `.espdl`. This format is similar to the ONNX model format but replaces ONNX's Protobuf with FlatBuffers, making our models more lightweight and supporting zero-copy deserialization. This feature ensures faster data access by eliminating the need to copy serialized data into separate memory areas.

### [esp-ppq](https://github.com/espressif/esp-ppq)

ESP-PPQ is a model quantization tool developed based on the open-source project PPQ. Users can select the ESP-DL target platform and directly export ESP-DL standard model files. ESP-PPQ inherits all the functionalities and documentation from the PPQ project, allowing users to conveniently choose quantization algorithms and analyze quantization errors.

### Efficient Operator Implementation

We have efficiently implemented common AI operators, including Conv2d, Pool2D, Gemm, Add, Mul, etc., based on AI instructions. These operators are precisely aligned with the PyTorch operator implementation, ensuring that the results obtained from the esp-ppq tool are consistent with those running on ESP-DL.

### Static Memory Planner

A new static memory planner is designed for the Internal RAM/PSRAM memory structure. Considering that internal RAM has faster access speed but limited capacity, we provide an API that allows users to customize the size of the internal RAM that the model can use. The memory planner will automatically allocate different layers to the optimal memory location based on the size of the internal RAM specified by the user, ensuring that the overall running speed is more efficient while occupying the minimum amount of memory.

### Dual Core Scheduling

The automatic dual-core scheduling enables computationally intensive operators to fully utilize the computing power of dual-cores. Currently, Conv2D and DepthwiseConv2D support dual-core scheduling. Below are some of our experimental results:

| |conv2d(input=224X224X3, kernel=3x3, output=112x112x16)|
|:---:|:---:|
|single core| 12.1.ms|
|dual core| 6.2 ms|

---

## Project Structure

The ESP-DL project is organized to provide a clear separation of concerns for different functionalities. Here's a breakdown of the main directories and their purposes to help beginners get started quickly:

```
esp-dl/
├── dl/                  # Core deep learning library
│   ├── base/            # Fundamental data types and utilities
│   ├── tensor/          # TensorBase class for data handling
│   ├── model/           # Model class for loading, building, and running neural networks
│   ├── module/          # Base Module class for operators/layers
│   ├── math/            # Mathematical functions and operations
│   ├── tool/            # Utility tools for the framework
│   ├── dl_define.hpp    # Global definitions, quantization and activation types
│   └── dl_define_private.hpp # Private definitions
├── fbs_loader/          # FlatBuffers model loading functionality
│   ├── include/         # Header files for the loader
│   ├── src/             # Source files for the loader
│   ├── lib/             # Pre-compiled FlatBuffers model library
│   └── pack_espdl_models.py # Script to pack multiple models
├── audio/               # Audio processing module
│   ├── common/          # Common audio processing utilities (WAV decoding, etc.)
│   ├── speech_features/ # Speech feature extraction (Fbank, MFCC, Spectrogram)
│   └── README.md        # Detailed documentation for audio processing
├── vision/              # Vision processing module
│   ├── image/           # Image processing utilities (JPEG, BMP, drawing, preprocessing)
│   ├── detect/          # Object detection post-processors (YOLO, etc.)
│   ├── classification/  # Image classification post-processors (ImageNet, etc.)
│   └── recognition/     # Face recognition components
├── CMakeLists.txt       # CMake build configuration for the ESP-IDF component
├── idf_component.yml    # ESP-IDF component manifest
├── LICENSE              # Project license information
└── README.md            # This file
```

### Core Components (`dl/`)

This is the heart of the ESP-DL framework. It contains the fundamental classes and functions required for neural network inference.

- `base/`: Contains basic utilities and low-level operations.
- `tensor/`: Defines the `TensorBase` class, which is used throughout the framework to represent data.
- `model/`: Contains the `Model` class, which handles loading `.espdl` files, building an execution plan, and running inference.
- `module/`: Defines the `Module` base class, from which all neural network operators (like Conv2D, Pool2D) are derived.
- `math/`: Provides optimized mathematical functions used by operators.
- `tool/`: Offers various utility functions for the framework.
- `dl_define.hpp`: Central place for global definitions like quantization and activation types.

### FlatBuffers Loader (`fbs_loader/`)

This component is responsible for loading models stored in the `.espdl` format, which is based on FlatBuffers.

### Audio Processing (`audio/`)

This module provides functionalities for audio signal processing, particularly focused on speech feature extraction. It includes utilities for WAV decoding and extracting features like Fbank, MFCC, and Spectrogram, optimized for ESP platforms.

### Vision Processing (`vision/`)

This module provides functionalities for computer vision tasks.

- `image/`: Utilities for image loading (JPEG, BMP), preprocessing, color space conversion, and drawing.
- `detect/`: Post-processors for object detection models (e.g., YOLO variants).
- `classification/`: Post-processors for image classification models (e.g., ImageNet classifiers).
- `recognition/`: Components for face recognition tasks.

Explore ESP-DL to streamline your AI model deployment and achieve optimal performance with minimal resource usage.