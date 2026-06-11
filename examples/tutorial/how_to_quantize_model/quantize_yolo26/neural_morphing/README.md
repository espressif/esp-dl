# Neural Morphing Engine

A general-purpose framework for performing arbitrary structural graph transformations on
neural networks, recovering accuracy through block-wise knowledge distillation after each
modification. All transformations operate on the **FP32 graph** before quantization.

Part of the final graduation project (Master's degree in AI & Computer Vision, 2026) at
[USTHB](https://www.usthb.dz/) University, Algiers, Algeria.\
Author: [Boumedine Billal](https://github.com/BoumedineBillal)

---

## Architecture Overview

Inspired by ESP-DL's [TQT](https://github.com/espressif/esp-dl) (Trained Quantization
Thresholds) pipeline, which uses the block-wise reconstruction principle from
[BRECQ](https://arxiv.org/abs/2102.05426) to recover accuracy after quantization. This engine
generalises that same principle from quantization recovery to **arbitrary structural
transformations**: any modification that replaces a graph operation with a shape-compatible
subgraph capable of approximating the original function.

The framework operates on the [PPQ](https://github.com/openppl-public/ppq) intermediate
representation (`BaseGraph`) and is **platform-agnostic** by design:

```
PyTorch ──export──► ONNX ──parse──► PPQ Graph ──Neural Morphing──► Export
                                                                    ├── .native (ESP32-P4)
                                                                    ├── ONNX (TensorRT, OpenVINO)
                                                                    └── ...
```

Hardware-specific decisions are encapsulated entirely within **replacement strategies**.

### Core Mechanisms

1. **Op-to-subgraph splice**: A single operation is surgically removed and replaced by an
   entire subgraph. Input/output wiring, variable registration, and topological ordering are
   handled automatically. Genealogy metadata (`_parent_op`, `_entrance_wire`, `_exit_wire`)
   is stamped on every inserted operation to enable deterministic rollback.

2. **Block-wise teacher–student distillation**: The original graph is deep-copied before
   any transformation (frozen teacher). After each structural replacement, the modified block
   is trained to reproduce the teacher's output at the block boundaries using only unlabelled
   calibration data and a reconstruction loss.

### Replacement Paths

| Path | Method | Use Case |
|------|--------|----------|
| **PyTorch module** (default) | `build_replacement()` returns `nn.Module` → auto-traced via ONNX | Standard ops (Conv decomposition, pruning) |
| **PPQ-native** | `build_replacement_subgraph()` builds PPQ ops directly | Custom HW ops with no ONNX equivalent (HardSiluPie8) |

### Strategy Pattern

All transformation-specific logic is encapsulated in a `BaseReplacementStrategy` subclass:

```
BaseReplacementStrategy
├── Methods (abstract):
│   ├── select_target()          # What to replace
│   ├── build_replacement()      # How to build the replacement
│   ├── calculate_samples()      # Sample count for training
│   ├── evaluate_validation()    # Quality gate check
│   ├── get_criterion()          # Loss function
│   └── get_scheduler()          # LR scheduler
├── Properties (abstract):
│   ├── learning_rate, weight_decay
│   └── steps, patience
└── Hooks (optional):
    ├── requires_predecessor     # Extend block to include predecessor
    ├── compensate_on_reject     # Drift compensation after rollback
    ├── on_step_end              # Per-step intervention (e.g. mask enforcement)
    └── build_replacement_subgraph  # PPQ-native replacement path
```

A new transformation requires only a new strategy class; the engine code never changes.

### Five-Phase Pipeline

For each topological block in the graph:

| Phase | Name | Description |
|-------|------|-------------|
| **A** | Subgraph Replacement | Query `select_target()`, build replacement, splice into graph |
| **B** | Data Caching | Forward pass through teacher & student, cache activations at block boundaries |
| **C** | Distillation Training | Minimise reconstruction loss (Huber + Cosine) via AdamW |
| **D** | Quality Gate | Dual-threshold check (cosine similarity ≥ τ_cos ∧ scale ratio ∈ [τ_lo, τ_hi]). Rollback on failure |
| **E** | Logging | Per-block loss curves and accept/reject decisions |

### Distillation Loss

```
L(ŷ, y) = L_Huber(ŷ, y; δ=1.0) + λ · (1 − cos(ŷ, y))     λ = 2.0
```

- **Huber**: Gradient-robust (bounded at ±δ for large residuals, unlike MSE)
- **Cosine**: Directional fidelity, scale-invariant

### Quality Gate

```
accept(b) ⟺ cos(ŷ_b, y_b) ≥ τ_cos  ∧  ρ_b ∈ [τ_lo, τ_hi]
```

where `ρ_b = ‖ŷ_b‖₂ / ‖y_b‖₂`. Both thresholds are necessary: cosine is scale-invariant
(`cos(αŷ, y) = cos(ŷ, y)` for any `α > 0`), so a block could output twice the teacher's
magnitude while scoring a perfect 1.0. The scale ratio catches this.

### Rollback & Drift Compensation

- **Rollback**: Uses genealogy metadata to restore the original operation from the teacher
  clone. Topology-agnostic  only needs `_parent_op` and the entrance/exit wire names.
- **Drift compensation**: When a block is rejected after earlier blocks were accepted,
  upstream activations have shifted. The engine fine-tunes the restored block's original
  weights at 0.25× base LR to match the teacher under the drifted input distribution.

---

## Directory Structure

```
neural_morphing/
├── neural_morphing/                  # Engine core (model-agnostic)
│   ├── engine.py                     # AdaptiveSubgraphOptimizationPass
│   ├── interface.py                  # BaseReplacementStrategy ABC
│   └── __init__.py
├── integrations/
│   └── yolo26n/                      # YOLO26n-specific strategies
│       ├── Transformation1_SiLU_to_HardSiLU/
│       │   ├── run_transform.py      # Entry point
│       │   ├── silu_to_hardsilu_strategy.py
│       │   ├── plots/                # Loss curves + morph_log_*.txt
│       │   └── output/               # .native checkpoints
│       ├── Transformation2_Conv_to_DWPW/
│       │   ├── run_transform.py
│       │   ├── conv_to_dwpw_strategy.py
│       │   ├── plots/
│       │   └── output/
│       └── Transformation3_Conv1x1_Prune/
│           ├── run_transform.py
│           ├── conv1x1_prune_strategy.py
│           ├── plots/
│           └── output/
├── .gitignore
└── README.md                         # This file
```

---

## Transformations

### T1: SiLU → HardSiLU8

Replaces SiLU (Swish) activations with HardSiLU8, a piecewise-linear approximation using
division by 8 (single arithmetic right-shift) instead of 6:

```
g₈(x) = x · clamp(x/8 + 1/2, 0, 1)
```

**Key design elements**:

- **Learnable scale factor** `α ∈ [0, 1]`: Compensates for the systematic magnitude
  difference between SiLU and HardSiLU8. Quantized to 8-bit at export via STE:
  `α = Round(clamp(s, 0, 1) · 256) / 256`. Applied inside the PIE SIMD kernel with
  4 additional instructions (no separate Mul node in the graph).
- **PPQ-native replacement path**: HardSiluPie8 is a custom operation type with no
  ONNX equivalent  the module path cannot represent it.
- **`requires_predecessor = True`**: HardSiLU8 is parameterless. Without the predecessor
  Conv's weights as free parameters, distillation has zero degrees of freedom.
- **`compensate_on_reject = True`**: Drift compensation enabled for rejected blocks.

**Excluded layers**:
- INT16 layers (detection head `/model.23/`, neck exits `/model.16/cv2/`, `/model.19/cv2/`,
  `/model.22/cv2/`)  HardSiluPie8 kernel is INT8-only
- Activations after depthwise convolutions  only 9 weights per channel, insufficient
  degrees of freedom

**Quality gate**: `cos ≥ 0.9908`, `ρ ∈ [0.985, 1.015]`

### T2: Conv 3×3 → Residual DW+PW Decomposition

Decomposes standard 3×3 convolutions into a residual depthwise-separable form:

```
y = PW_main(DW(x)) + PW_skip(x)
```

**Key design elements**:

- **Center-pixel extraction**: The center position `W[:,:,1,1]` carries 40–60% of weight
  energy and acts as a channel mixer. It is extracted into `PW_skip`, leaving only the
  8 surrounding spatial positions for the DW+PW_main path.
- **Per-channel SVD initialization**: After center-pixel removal, the residual tensor is
  decomposed via truncated SVD at rank `r=3`, achieving 85–95% energy retention.
- **Compression ratio**: ~2× at 128 channels (`9rC + (r+1)C² vs. 9C²`).
- **PyTorch module path**: All sub-operations are standard ONNX Conv operators.
- **`requires_predecessor = False`**: The decomposed module has its own trainable weights.

**Excluded layers**:
- Depthwise convolutions (already efficient)
- Stem (`/model.0/`), detection head (`/model.23/`), PSA modules (`/model.10/`, `/model.22/`)

**Quality gate**: `cos ≥ 0.99`, `ρ ∈ [0.90, 1.10]`

### T3: Pointwise 1×1 Channel Pruning

Prunes the N least important input channels from 1×1 convolutions:

**Phase 1  Masked Distillation**:
1. Rank channels by L1 importance: `importance(c) = Σⱼ |W[j,c,0,0]|`
2. Zero the bottom-N channels via binary mask `M`
3. Train surviving weights via distillation, re-applying `W ← W ⊙ M` after every
   optimizer step (`on_step_end` hook) to prevent gradient drift

**Phase 2  Structural Removal** (planned):
- Physically shrink weight tensors: `[C_out, C_in, 1, 1] → [C_out, C_in−N, 1, 1]`
- Remove corresponding output filters from the upstream producer conv
- Real compute/memory savings, not just zero-valued weights

**Excluded layers**: Stem (`/model.0/`), detection head (`/model.23/`)

**Quality gate**: `cos ≥ 0.99`, `ρ ∈ [0.90, 1.10]`

---

## Usage

Each transformation is self-contained. Run from the transformation's directory:

```bash
# T1: SiLU → HardSiLU8
python integrations/yolo26n/Transformation1_SiLU_to_HardSiLU/run_transform.py

# T2: Conv decomposition
python integrations/yolo26n/Transformation2_Conv_to_DWPW/run_transform.py

# T3: Channel pruning
python integrations/yolo26n/Transformation3_Conv1x1_Prune/run_transform.py
```

### Chaining Transformations

Transformations can be chained by loading a previous transformation's `.native` checkpoint.
Set `INPUT_NATIVE` in `run_transform.py` to point to the previous output:

```python
# In T3's run_transform.py, chain after T1:
INPUT_NATIVE = os.path.join(
    QUANTIZE_ROOT, "neural_morphing", "integrations", "yolo26n",
    "Transformation1_SiLU_to_HardSiLU", "output",
    "T1_silu_hsilu_512_p4", "morphed_hsilu.native"
)
```

### Output Artifacts

Each transformation produces:

| File | Description |
|------|-------------|
| `output/<tag>/*.native` | PPQ graph checkpoint (input for next transformation or quantization) |
| `output/<tag>/baseline_mAP.txt` | Cached baseline mAP (avoids re-evaluation) |
| `plots/morph_log_*.txt` | Per-block accept/reject log with metrics |
| `plots/*.png` | Per-block loss curves (gitignored) |
| `finale_log.txt` | Summary of accepted/rejected/frozen operations |

---

## Adding a New Transformation

1. Create a new directory under `integrations/yolo26n/Transformation<N>_<Name>/`
2. Implement a strategy class extending `BaseReplacementStrategy` from
   `neural_morphing/interface.py`
3. Implement the 6 abstract methods + 4 properties
4. Create a `run_transform.py` that:
   - Configures `QATConfig` and injects into `sys.modules`
   - Loads the graph (ONNX or `.native`)
   - Instantiates your strategy
   - Calls `AdaptiveSubgraphOptimizationPass().optimize(...)`
   - Exports the result

The engine handles graph surgery, block partitioning, activation caching, training
orchestration, quality evaluation, rollback, and logging automatically.

---

## Adding a New Model

1. Create a new directory under `integrations/<model_name>/`
2. Implement transformation strategies specific to the model's architecture
3. Adjust `frozen_prefixes` / `skip_prefixes` for the model's sensitive layers
4. The engine core (`neural_morphing/engine.py`) requires no modification

---

## Dependencies

- [ESP-PPQ](https://github.com/espressif/esp-ppq) — PPQ fork with ESP32 target platforms
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO model export and mAP evaluation
- PyTorch ≥ 2.0
- CUDA (recommended for distillation training)

---

## License

MIT
