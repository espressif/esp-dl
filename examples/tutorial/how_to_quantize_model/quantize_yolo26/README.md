# YOLOv26n: PTQ + TQT + LUT <small>(INT16 Step-Interpolated Activation LUT)</small> Quantization Pipeline

This tutorial walks through the complete workflow for quantizing a **YOLOv26n** model for deployment on **ESP32-P4** and **ESP32-S3** using **ESP-PPQ**.

## Overview

The pipeline leverages the **YOLO26n** architecture, optimized for embedded deployment, and applies a three-stage hardware-aware quantization process.

### Key Features
*   **NMS-Free Inference**: One2One detection head  no post-processing NMS required on the MCU.
*   **RegMax=1**: Eliminates the DFL layer entirely, reducing compute by ~30%.
*   **INT8 + INT16 Hybrid**: Sensitive layers (box/class heads, neck exits) run in INT16; backbone in INT8.
*   **INT16 Step-Interpolated LUT for Swish**: Swish activations are replaced by compact INT16 Look-Up Tables with configurable step interpolation, enabling hardware-accurate emulation on ESP32-P4/S3.
*   **Generic**: Supports any input resolution (160–640) and any dataset (COCO, Roboflow, custom).

---

## 1. Setup

```bash
pip install -r requirements.txt
pip install roboflow  # only needed for the Roboflow notebook
```

---

## 2. Choose Your Workflow

Two notebooks cover the full pipeline:

### Option A  Custom Dataset via Roboflow *(recommended for most users)*
**`quantize_yolo26_roboflow.ipynb`**

1. Paste your Roboflow API key & dataset URL.
2. Runs fine-tuning (4 epochs by default) on your custom dataset.
3. Runs the full PTQ → TQT → LUT quantization pipeline.
4. Exports the optimized `.espdl` model.

Supports any Roboflow dataset with any number of classes and any image resolution.

### Option B  COCO 80-class Benchmark
**`quantize_yolo26_coco.ipynb`**

Reproduce our official mAP benchmarks or build a generic 80-class object detector.

---

## 3. Quantization Pipeline (Inside the Notebooks)

The pipeline runs **sequentially** across the following stages:

| Step | Name | Description |
|------|------|-------------|
| 1 | **ONNX Export** | Exports PyTorch weights with the `RegMax=1` patch applied (removes DFL layer). |
| 2 | **PTQ Calibration** | Feeds calibration images through the graph to determine per-layer dynamic ranges. |
| 3 | **TQT (Trained Quantization Threshold)** | Block-by-block scale optimization using reconstruction loss  fast, no real backprop needed. |
| 4 | **Passive + Alignment Passes** | Derives bias/passive scales; aligns elementwise ops (Add, Concat) to a common quantization scale. |
| 5 | **INT16 Step-Interpolated LUT Fusion** | Converts INT16 Swish activations into compact Look-Up Tables (step size configurable, default=32) for hardware-accurate emulation on the ESP-DL accelerator. |
| 6 | **Graph Surgery** | Splits Concat output nodes into 6 discrete tensors (`one2one_p3_box`, `one2one_p3_cls`, …). |
| 7 | **`.espdl` Export** | Writes the final deployment model with LUT tables embedded. |

### Optional: Inference Preview (Cell 10.1)
After graph surgery and before export, Cell 10.1 runs `eval_espdl_model` to visualize model predictions
on a test image using **bit-exact ESP-DL emulated preprocessing**. The annotated output is saved to `results/`.

---

## 4. Output Files

After running a notebook, the output is saved under:

```
output/
├── coco_512_s8_p4/
│   ├── yolo26n_512_s8_p4.espdl    ← FIRMWARE DEPLOYMENT MODEL
│   ├── yolo26n_512_s8_p4.info     ← Per-layer debug info (~15 MB)
│   ├── yolo26n_512_s8_p4.json     ← Quantization scales/config
│   └── yolo26n_export.onnx        ← Intermediate ONNX (pre-quantization)
└── lego_512_s8_p4/
    └── yolo26n_lego_512_s8_p4.espdl
```

Naming convention: `<model>_<img_sz>_s8_<platform>.*`

---

## 5. Deployment

Once you have your `.espdl` file:

1. **Copy** it to your firmware project:
   ```bash
   cp output/coco_512_s8_p4/yolo26n_512_s8_p4.espdl \
      ../../../../examples/yolo26_detect/main/models/p4/
   ```

2. **Update CMake**:
   Edit `examples/yolo26_detect/main/CMakeLists.txt` to select the new model file.

3. **Build & Flash**:
   ```bash
   idf.py build flash monitor
   ```

---

## 6. Script Modules (`scripts/`)

| File | Description |
|------|-------------|
| `export.py` | ONNX export with ESP-DL graph patches (`RegMax=1`, Attention static reshape, Detect head). |
| `dataset.py` | Calibration dataloader  works with COCO, Roboflow, or any `data.yaml` dataset. |
| `notebook_helpers.py` | Core helpers: `extract_model_meta`, `prepare_onnx`, `prune_graph_safely`, `espdl_preprocess`, `eval_espdl_model`. |
| `trainer.py` | `QATTrainer`  mAP evaluation using the quantized graph (emulates ESP-DL hardware). |
| `validator.py` | Validation loop utilities. |
| `utils.py` | `seed_everything`, `register_mod_op`, `get_exclusive_ancestors`. |
| `esp_ppq_patch.py` | Runtime patches for ESP-PPQ: `OnnxParser`, `Slice`, `Gather` backends. |
| `esp_ppq_patch_2.py` | `AddLUTPattern.export` patch for correct LUT step propagation. |

### `esp_ppq_lut/` Extension
Provides `EspdlLUTFusionPass`  converts INT16 Swish ops into compact step-interpolated Look-Up Tables (controllable step size via `INT16_LUT_STEP`, default=32)  and `HardwareAwareEspdlExporter` with LUT tables embedded in the `.espdl` output.
