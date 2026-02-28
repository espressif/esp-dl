| Supported Targets | ESP32-S3 | ESP32-P4 |
|-------------------|----------|----------|

# YOLOv26n Detect Example

This is a complete end-to-end example for running quantized YOLOv26n inference on Espressif SoCs. 
It features direct regression (No NMS), high performance via `esp-dl` optimizations, and a flexible model loading system.

> **Generic Framework:** This example is designed to run any YOLOv26 model trained on **any dataset**, with **any input size**, and with **any number of classes**. The application automatically adapts to your custom model's configuration.

## Quick Start

1.  **Set Target**
    ```bash
    idf.py set-target esp32p4  # or esp32s3
    ```

2.  **Build and Flash**
    ```bash
    idf.py build flash monitor
    ```

### Expected Output
```text
I (4250) image: bus.jpg
I (4256) YOLO26_DETECT: Pre: 4500 us | Inf: 17000 us | Post: 1200 us
I (4256) YOLO26: [category: bus, score: 0.88, x1: 15, y1: 120, x2: 550, y2: 600]
```

#### Custom Lego Model Output (Example)

```text
I (3991) image: bus.jpg
I (3991) YOLO26_DETECT: Pre: 20 ms | Inf: 1770 ms | Post: 0 ms

I (5921) image: person.jpg
I (5921) YOLO26_DETECT: Pre: 30 ms | Inf: 1760 ms | Post: 10 ms

I (7821) image: lego.jpg
I (7821) YOLO26_DETECT: Pre: 30 ms | Inf: 1760 ms | Post: 10 ms
I (7821) YOLO26: [category: 2x2_green, score: 0.56, x1: 228, y1: 136, x2: 328, y2: 300]
I (7821) YOLO26: [category: 2x4_green, score: 0.56, x1: 104, y1: 128, x2: 248, y2: 368]
I (7831) YOLO26: [category: 2x4_green, score: 0.44, x1: 288, y1: 152, x2: 424, y2: 408]
```

## Visualization

We provide a Python script to visualize detection results on the test images.

### 1. Requirements
```bash
pip install matplotlib pillow
```

### 2. Run Visualization
The script reads hardcoded logs (or you can modify it to read from a file) and saves the annotated images to the `results/` folder.

```bash
cd results
python visualize_logs.py
```

### 3. Result
The script generates the annotated images.

| **Model: `yolo26n_512_s8_p4.espdl` (Stock)** | **Model: `yolo26n_lego_512_s8_p4.espdl` (Custom)** |
| :---: | :---: |
| ![Bus Detection](results/result_bus.jpg) | ![Lego Detection](results/result_lego.jpg) |

## Configurable Options

### 1. Stock Models (Pre-Optimized)
The component includes several pre-quantized models in the model zoo. By default, the build system selects the **512x512** model for your target.

To switch to a different resolution (e.g., 640x640), edit `main/CMakeLists.txt`:

```cmake
# Select a model from the list below
set(MODEL_FILENAME "yolo26n_640_s8_p4.espdl")
```

**Available Models:**

| Resolution | ESP32-P4 Filename | ESP32-S3 Filename |
| :--- | :--- | :--- |
| **512x512** | `yolo26n_512_s8_p4.espdl` (Default) | `yolo26n_512_s8_s3.espdl` (Default) |
| **640x640** | `yolo26n_640_s8_p4.espdl` | `yolo26n_640_s8_s3.espdl` |


### 2. Using Your Own Custom Model

You can easily deploy your own custom-trained YOLOv26n model (e.g., for detecting specific objects like Lego bricks).

#### Step 1: Export & Quantize
Follow the [Quantization Tutorial](../tutorial/how_to_quantize_model/quantize_yolo26/README.md) to generate your quantized `.espdl` model file (e.g., `yolo26n_lego_512_s8_p4.espdl`, which has **28 classes**).

#### Step 2: Place the Model & Classes
Copy your `.espdl` file into the local platform-specific models directory:
```
main/models/[p4|s3]/
```
*(i.e., `main/models/p4/` for ESP32-P4 or `main/models/s3/` for ESP32-S3)*

You can also place your custom C++ header file (e.g., `lego_classes.hpp`) in the same folder. The build system will automatically include it.

#### Step 3: Update Build Configuration
Edit `main/CMakeLists.txt` to point to your new filename:
```cmake
set(MODEL_FILENAME "yolo26n_lego_512_s8_p4.espdl")
```

#### Step 4: Update Application Labels (C++)
If your model detects different classes than COCO (80 classes), you must update the label list in `main/app_main.cpp`.

1.  Create/Edit the header file with your class names (e.g., `lego_classes.hpp`):
    ```cpp
    const char* lego_classes[] = { "brick_2x4", "brick_1x2", ... };
    ```
2.  Include it in `app_main.cpp` and switch the pointer:
    ```cpp
    #include "lego_classes.hpp"  // <--- Found automatically if placed with the model
    // ...
    // Change from coco_classes to your custom list
    // const char **current_classes = coco_classes; 
    const char **current_classes = lego_classes;
    ```

3.  **Build and Flash.** The `YOLO26` component in the application will automatically detect the number of classes from the model file and map them to your new labels provided in `current_classes`.
