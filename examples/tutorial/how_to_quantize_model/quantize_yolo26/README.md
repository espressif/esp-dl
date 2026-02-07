# YOLOv26n: Quantization Aware Training (QAT) Tutorial

This tutorial guides you through the end-to-end process of preparing, training, and quantizing a **YOLOv26n** model for deployment on **ESP32-P4** and **ESP32-S3**.

## Overview

The workflow leverages the **YOLO26n** architecture, optimized for embedded deployment (NMS-Free, RegMax=1), and uses **ESP-PPQ** for hardware-aware quantization.

### Key Features
*   **Zero-Overhead Inference**: "NMS-Free" styling eliminates complex post-processing on the MCU.
*   **Hardware Optimized**: Generates `Int8` models specifically tuned for the ESP-DL accelerator.
*   **Flexible**: Supports any input resolution (e.g., 160x160 to 640x640) and any dataset.

## 1. Setup

### Install Dependencies
Ensure you have the correct versions of `esp-ppq`, `onnx`, and `ultralytics`.

```bash
pip install -r requirements.txt
```

*(Note: We use a specific patched workflow, so standard pip install ultralytics might not work as expected without our scripts.)*

## 2. Choose Your Workflow

We provide two Jupyter Notebooks to handle the entire pipeline:

### Option A: Custom Dataset (Roboflow / Lego)
**Recommended for most users.**
Use this notebook to train a model on your own data (e.g., "Lego Bricks", "Fruits", "Face Mask").

> **Flexibility**: This notebook allows you to:
> *   Download **any Roboflow dataset** with **any number of classes**.
> *   Train with **any image resolution** (e.g., 160, 320, 640...).
> *   Export optimized models for **both ESP32-P4 and ESP32-S3**.
> *   Control quantization parameters, calibration size, and fine-tuning epochs.

1.  Open **`quantize_yolo26_roboflow.ipynb`**.
2.  Paste your Roboflow API Key / Dataset URL.
3.  Run all cells to Train -> Export -> Quantize.

### Option B: COCO Dataset (Standard Benchmark)
Use this to reproduce our official benchmarks or train a generic 80-class object detector.

1.  Open **`quantize_yolo26_coco.ipynb`**.
2.  Run all cells.

## 3. Workflow Steps (Inside the Notebooks)

The notebooks automate the following complex steps:
1.  **Train/Fine-tune**: Trains YOLO26n on your dataset.
2.  **Export to ONNX**: Applies our custom `RegMax=1` patch to remove the DFL layer.
3.  **PPQ Quantization**:
    *   **Calibration**: Feeds calibration images to determine dynamic ranges.
    *   **Tuning**: Optimizes quantization errors (QAT).
4.  **Format Conversion**: Converts the quantized graph into `.espdl` format.

## 4. Output

After running the notebook, your optimized model will be saved in the `output/` directory:

```text
output/
├── coco_512_s8_p4/            # Example Output Folder
│   ├── yolo26n_512_s8_p4.espdl  <-- FINAL MODEL FOR FIRMWARE
│   ├── yolo26n_512_s8.info     <-- Model Metadata
│   └── ...
└── ...
```

## 5. Deployment

Once you have your `.espdl` file:

1.  **Copy** it to your firmware project:
    ```bash
    cp output/coco_512_s8_p4/yolo26n_512_s8_p4.espdl ../../../../examples/yolo26_detect/main/models/p4/
    ```

2.  **Update CMake**:
    Edit `examples/yolo26_detect/main/CMakeLists.txt` to select your new model file.

3.  **Build**:
    ```bash
    idf.py build flash monitor
    ```

## Advanced: Script Modules
The `scripts/` directory contains the Python logic used by the notebooks:
*   `scripts/exporter.py`: Handles the ONNX export with graph surgery.
*   `scripts/calibrator.py`: Manages the ESP-PPQ calibration loop.
*   `scripts/esp_ppq_patch.py`: Runtime patches for the quantization toolchain.
