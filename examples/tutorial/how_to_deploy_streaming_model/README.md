# How to Deploy Streaming Models

This tutorial demonstrates how to deploy streaming models on ESP chips using the ESP-DL framework. 

## Overview

Streaming models in ESP-DL are specialized neural networks that can process data in chunks over time rather than requiring the entire input at once. This capability is particularly useful for:

- Audio processing (continuous audio streams)
- Time series analysis
- Real-time sensor data processing
- Applications with memory constraints

### Key Concepts

- **Streaming Model**: A model optimized to process data in chunks, maintaining internal state between chunks
- **Non-streaming Model**: Traditional model that processes the entire input at once
- **Chunked Processing**: Breaking input data into smaller pieces processed sequentially
- **State Preservation**: Maintaining internal model state between chunks

## Directory Structure

```
how_to_deploy_streaming_model/
├── quantize_streaming_model/     # Model quantization and streaming conversion
│   ├── models.py                # Defines neural network models (TCN, TestModels)
│   └── quantize_torch_model.py  # Quantization and streaming conversion script
├── test_streaming_model/        # ESP-IDF project for testing on device
│   ├── main/
│   │   ├── app_main.cpp         # Main application code
│   │   └── CMakeLists.txt       # Component build configuration
│   ├── CMakeLists.txt           # Project build configuration
│   └── partitions.csv           # Flash partition table
└── README.md                   # This file
```

## What You'll Learn

1. How to create neural network models suitable for streaming
2. How to quantize models for ESP devices
3. How to convert regular models to streaming models
4. How to deploy and test streaming models on ESP devices
5. How to verify that streaming models produce the same results as non-streaming models

## Prerequisites

- ESP-IDF development environment
- Python with PyTorch and esp-ppq dependencies
- ESP32-P4 or ESP32-S3 development board
- Basic understanding of neural networks and quantization

## Step-by-Step Guide

### Step 1: Prepare the Model

The example uses a Temporal Convolutional Network (TCN) model defined in `models.py`. This model is designed for sequential data processing:

```python
class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expand_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        # Implementation with causal convolutions for streaming
```

### Step 2: Model Quantization and Streaming Conversion

In the `quantize_streaming_model/` directory, run the quantization script:

```bash
cd quantize_streaming_model
python quantize_torch_model.py
```

This script performs the following actions:

1. **Creates two model variants**:
   - `model.espdl`: Regular quantized model
   - `streaming_model.espdl`: Streaming-optimized model

2. **Uses different input shapes**:
   - Regular model: `[1, 16, 15]` (full sequence)
   - Streaming model: `[1, 16, 3]` (chunk size)

3. **Enables automatic streaming conversion** using the `auto_streaming=True` flag

Key parameters:
- `TARGET`: Target device ("esp32p4", "esp32s3", or "c")
- `NUM_OF_BITS`: Quantization bit width (8-bit)
- `INPUT_SHAPE`: Original model input shape
- `streaming_input_shape`: Chunk input shape for streaming

### Step 3: Deploy to ESP Device

Navigate to the test project directory:

```bash
cd ../test_streaming_model
```

Build and flash to your ESP device:

```bash
idf.py set-target esp32p4  # Or esp32s3
idf.py build
idf.py flash
```
