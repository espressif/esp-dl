# esp_ppq_lut

> **Bit-Exact LUT Activation Deployment for ESP32**  
> A Python extension for [esp-ppq](https://github.com/espressif/esp-ppq) enabling hardware-accurate INT16 Look-Up Table activations on ESP32-P4 and ESP32-S3.

🔗 **Full documentation & source:** [github.com/BoumedineBillal/esp_ppq_lut](https://github.com/BoumedineBillal/esp_ppq_lut/tree/main)

---

## Why It Exists

When deploying YOLO26n with INT16 Swish activations, `esp-ppq`'s standard simulation diverges from what the ESP32 hardware actually computes. The hardware runs a **step-interpolated LUT** (4KB, 2049 entries, step=32)  not a float activation. Without this library, **88.4% of output values differ** between Python simulation and on-chip execution.

`esp_ppq_lut` closes this gap by emulating the ESP-DL LUT hardware in pure `torch.int32` arithmetic.

---

## What It Does

| Component | Description |
|---|---|
| `emulator.py` | Bit-exact `HardwareEmulator`  clones `dl_module_lut.hpp` integer math in PyTorch |
| `passes.py` | `EspdlLUTFusionPass`  rewrites activation ops to `LUT` nodes in the PPQ graph |
| `exporter.py` | `HardwareAwareEspdlExporter`  switches to Ideal Math during table generation, Simulation otherwise |
| `patches.py` | Registers Sigmoid/Tanh/Relu backends into esp-ppq's dispatch table |
| `verifier.py` | Exhaustive 65,536-point sweep validation |
| `utils.py` | LUT table generation, C-header export, comparison plots |

---

## Fencepost Bug Fix

The library also fixes a critical off-by-one bug in `esp-ppq`'s LUT exporter  the exporter generated 2048 table entries but ESP-DL's interpolation requires 2049 (the boundary of the last segment). The fix is adopted into `esp-ppq` upstream.

---

## Validation Tests (4-Test Protocol)

The `test_yolo/` pipeline exports a **dual-model firmware project** (LUT model + IDEAL model) and validates on real ESP32-P4 hardware:

| Test | HW Model | vs Reference | Result |
|------|----------|-------------|--------|
| **TEST 0** | LUT  Preprocess | Python `espdl_preprocess()` | ✅ 786,432 values  **0 errors** (pixel-exact) |
| **TEST 1** | LUT  Inference | Python SIMULATION vectors | ✅ 451,584 values  **0 errors** (100% bit-exact) |
| **TEST 2** | LUT  Inference | Python IDEAL_MATH vectors | ⚠️ 399,044 mismatches (proves simulation gap) |
| **TEST 3** | IDEAL  Inference | Python IDEAL_MATH vectors | ✅ 11,583 mismatches within ±5 (baseline sanity) |

> TEST 1 at 0 errors is the central claim: Python's integer emulator is a **digital twin** of the ESP32-P4 LUT hardware.

---

## Quick Usage

```python
import esp_ppq_lut as esp_lut

# 1. Initialize (once, before quantization)
esp_lut.initialize(step=32, verbose=True)

# 2. After calibration  fuse activation ops into LUT nodes
esp_lut.EspdlLUTFusionPass(target_ops=['Swish'], lut_step=32).optimize(
    graph=graph, dataloader=cali_loader, executor=executor,
    calib_steps=0, collate_fn=lambda x: x.to(device)
)

# 3. Export (mode switching is automatic)
PFL.Exporter(platform=TARGET_PLATFORM).export(
    "model.espdl", graph=graph, int16_lut_step=32
)
```

---

## Supported Activations

| Activation | Verified |
|---|---|
| **Swish / SiLU** | ✅ Bit-Exact on ESP32-P4 |
| **Sigmoid** | ✅ Bit-Exact |
| **Tanh** | ✅ Bit-Exact |

Adding a new activation requires only 2 lines  see the [full README](https://github.com/BoumedineBillal/esp_ppq_lut/tree/main#supported-activation-functions).
