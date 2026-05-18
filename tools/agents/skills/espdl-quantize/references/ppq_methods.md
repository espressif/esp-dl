# esp-ppq Quantization Method Reference

This document covers every tunable knob exposed by `QuantizationSettingFactory.espdl_setting()`.
Each section follows the same template:

- **What it does** — one-paragraph principle.
- **Parameters** — name, type, valid values, what changing it does.
- **When to enable** — observable conditions in `layer_stats.json` / `layerwise_error.json`.
- **When to avoid** — anti-patterns and gotchas.
- **Cost** — split into two pieces: PC-time (one-shot quantization cost) and
  on-device-runtime (recurring inference cost on the ESP target). The accuracy-first
  ordering uses **on-device-runtime** as the soft penalty; PC-time is logged but does not
  affect priority.

The section ordering follows the **accuracy-first method ordering**: Tier A (high accuracy,
no on-device runtime cost) first, Tier B (high accuracy, with on-device runtime cost)
next, Tier C (last resort) last. LSQ is documented for completeness but tagged as
**skipped on POWER_OF_2 targets** (i.e. all esp-dl variants).

## Table of contents

**Tier A — high accuracy, no on-device runtime cost**

1. [Layer-wise equalization](#1-layer-wise-equalization-equalization)
2. [Bias correction](#2-bias-correction-bias_correct)
3. [TQT (trained quantization thresholds)](#3-tqt-trained-quantization-thresholds-tqt_optimization)
4. [Calibration algorithm](#4-calibration-algorithm-quantize_activation_settingcalib_algorithm)
5. [Quantization fusion alignment](#5-quantization-fusion-alignment-fusion_setting)

**Tier B — high accuracy, with on-device runtime cost (apply surgically)**

6. [Mixed precision via dispatching_table](#6-mixed-precision-via-dispatching_table)
7. [Horizontal weight split](#7-horizontal-weight-split-weight_split)

**Tier C — last resort**

8. [Blockwise reconstruction](#8-blockwise-reconstruction-blockwise_reconstruction)

**Tier D — skipped on POWER_OF_2 esp-dl targets**

9. [LSQ (learned step size)](#9-lsq-learned-step-size-lsq_optimization)

**Reference**

10. [Experimental: channel split, matrix factorization, SSD equalization](#10-experimental-passes)
11. [Compatibility matrix (method × method)](#11-compatibility-matrix-method--method)
12. [Target-policy compatibility (POWER_OF_2)](#12-target-policy-compatibility-power_of_2)

---

## 1. Layer-wise equalization (`equalization`)

> **Tier A — high accuracy, no on-device runtime cost.** Canonical use case: per-tensor
> weight targets (`esp32s3 / c`). On `target = esp32p4` every Conv/ConvTranspose/Gemm uses
> per-channel weight quantization, and esp-ppq officially flags the combination as
> "Not recommend" — but this is **warn-only** in the harness, not auto-disabled (changed
> in this revision). Some MobileNet-family / depthwise-separable networks empirically
> still benefit from equalization on esp32p4. Treat it as a Phase 3 lever to try **after**
> the calib×TQT cartesian product has settled; see §12 for the full reasoning.

### What it does

For two consecutive computing layers (e.g. `Conv1 → ReLU → Conv2`), per-channel weight
ranges can differ wildly between channels. Per-tensor quantization picks one scale for the
whole tensor, so a channel with weights in `[-128, 128]` next to a channel with weights in
`[-0.5, 0.5]` will quantize the smaller one to all zeros.

Markus Nagel's equalization rescales `W_1` (output channels) and `W_2` (input channels) by
a per-channel factor `s` so that `Y` is unchanged but per-channel ranges are flattened. It
is **data-free**: only weights move, calibration is not needed.

### Parameters

```python
setting.equalization = True
setting.equalization_setting.iterations         # int, default 10. Try 3, 6, 10, 100. More = flatter, but converges quickly.
setting.equalization_setting.value_threshold    # float, default 0.5. Weights below this absolute value are excluded from scale computation. Try 0.5 or 2.
setting.equalization_setting.opt_level          # 1 or 2. Level 2 also crosses Add/Sub branches.
setting.equalization_setting.including_bias     # bool, include bias in the scale calc. Default False.
setting.equalization_setting.including_act      # bool, include activation in the scale calc. Default False.
setting.equalization_setting.interested_layers  # list[str] or None. None/[] = all eligible layers.
```

### When to enable

- `target ∈ {esp32s3, c}` — these targets keep weights at per-tensor quantization, which
  is the textbook setting equalization was designed for.
- `target = esp32p4` is **allowed but warn-only**. Try it as a Phase 3 lever (after the
  Phase 2 calib×TQT cartesian product) when iter_0..3 plateau and the top-error layers
  match the structural trigger below. Do not lead with it on esp32p4.
- Top-error layers have **per-channel weight max/mean ratio > 5** (compute from
  `Float Max / Float Mean` of a parameter variable when `Is parameter == True`).
- Layer is part of a **`Conv → activation → Conv` chain**. Examples (not required):
  depthwise/pointwise pairs (MobileNet, EfficientNet), bottleneck residual blocks
  (ResNet), inverted residuals (MobileNet-V2/V3), or any back-to-back conv pattern. The
  rule is structural, not network-family-specific. ReLU6 should be replaced with ReLU
  pre-quantization for best results
  (see `examples/tutorial/how_to_quantize_model/quantize_mobilenetv2/quantize_torch_model.py`
  function `convert_relu6_to_relu`).
- Iter-0 SNR for those layers > 0.05 (5% noise:signal).

### When to avoid

- The graph has heavy branching with non-equalizable ops (concat with mismatched scales).
  Use `opt_level=1` if level 2 makes things worse.
- Already-equalized model (e.g. someone ran another tool first) — re-running can over-flatten.
- On `esp32p4`, **don't lead** with equalization. Run the Phase 2 calib×TQT cartesian
  product first; only try equalization after if there's a structural reason to
  (per-channel `max/mean > 5` on a Conv→activation→Conv chain) and you accept the
  per-tensor activation widening risk that esp-ppq's "Not recommend" warning is about.

### Cost

- **PC-time**: cheap. Pure weight transformation; seconds on CPU even for medium models.
- **On-device runtime**: zero. Only weight values change, the graph is unchanged.

---

## 2. Bias correction (`bias_correct`)

> **Tier A — high accuracy, no on-device runtime cost.**

### What it does

Quantization typically introduces a non-zero mean error in each layer's output (the
expected error is not 0, even though it should be after rounding). Bias correction measures
this expected error on calibration data and bakes it back into the layer's bias term:

    new_bias = old_bias + E[Y_fp - Y_quant]

It is a one-shot correction per layer, not iterative.

### Parameters

```python
setting.bias_correct = True
setting.bias_correct_setting.interested_layers  # list[str] or None. None = all layers.
setting.bias_correct_setting.block_size         # int, default 4. Smaller = more accurate, slower. Use 1 for best result.
setting.bias_correct_setting.steps              # int, default 32. Forward steps for collecting error. 8-32 is fine.
```

### When to enable

- Top-error layer has `Noise Mean > 0.1 × Noise Std` (sustained mean shift, not just noise).
- Layer is a Conv/Gemm with bias (skipped automatically if no bias).
- Already tried calibration tweaks and equalization; still seeing systematic offset.

### When to avoid

- Layers without bias (no-op).
- Very deep networks where shift accumulates non-linearly — bias correction fixes only the
  current layer, can be over-eager downstream.

### Cost

- **PC-time**: medium. Roughly `O(steps * num_layers)` forward passes on calibration data.
  Single-digit minutes for typical CV models on CPU.
- **On-device runtime**: zero. Only the bias values change, no new ops added.

---

## 3. TQT (trained quantization thresholds, `tqt_optimization`)

> **Tier A — high accuracy, no on-device runtime cost. POWER_OF_2-native — preferred over
> LSQ on every esp-dl target.**

### What it does

Treat each layer's quantization scale as a trainable parameter — specifically, esp-ppq's
TQT trains `log2_scale` so that `scale = 2 ^ log2_scale` always satisfies the POWER_OF_2
constraint. Run a few hundred forward passes on calibration data, backpropagate the
quantization error through a Straight-Through Estimator, update `log2_scale` (and
optionally weights) with Adam. This pushes scales closer to optimal for the actual data
distribution while staying within esp-dl's POWER_OF_2 hardware policy.

Different from LSQ (§9): TQT focuses on the threshold (scale) in log-domain and uses a
quantization-aware-style gradient that respects POWER_OF_2. LSQ trains a continuous scale
that is **disabled** under POWER_OF_2 — see §9 and §12. On esp-dl, **always pick TQT over
LSQ**.

### Parameters

```python
setting.tqt_optimization = True
setting.tqt_optimization_setting.interested_layers  # list[str] or None. None = all conv/gemm.
setting.tqt_optimization_setting.lr            # float, default 1e-5. Try 1e-5 ~ 1e-4.
setting.tqt_optimization_setting.steps         # int, default 500. Try 500, 1000, 2000.
setting.tqt_optimization_setting.block_size    # int, default 4. Bigger = faster, less stable. 1-4 typical.
setting.tqt_optimization_setting.is_scale_trainable  # bool, default True.
setting.tqt_optimization_setting.gamma         # regularization, default 0.0.
setting.tqt_optimization_setting.int_lambda    # encourage exponents close to int, default 0.0.
setting.tqt_optimization_setting.collecting_device   # 'cpu' or 'cuda'.
```

### When to enable

- Calibration tweaks + equalization + bias correction did NOT close the accuracy gap.
- The user has a CUDA GPU available, OR is willing to wait.
- The user provided enough calibration samples (256+ recommended).

### When to avoid

- CPU-only with > 50 layers — hours per iteration. Warn the user; this is PC-time, not
  device-time, so it is acceptable in the accuracy-first ordering, but they should know.
- Tiny calibration set (< 64 samples) — overfit to calibration, hurts test accuracy.
- The user wants determinism — gradient descent introduces noise across runs.

### Three-stage escalation in the harness

`scripts/compare_iterations.py` walks TQT in three discrete stages instead of jumping
from the conservative default to the most aggressive schedule in one shot. Each stage is
a separate Phase-3 lever; the state machine picks which one to suggest next.

| Stage | Knob change vs the previous best | Trigger to enter |
|-------|----------------------------------|------------------|
| Phase-2 default (lever 3a-0 implicit) | `lr=1e-5, steps=500, block_size=4` | Always part of the calib×TQT cartesian product. |
| 3a-1 | `steps: 500 → 1000` only (lr/block_size unchanged) | Phase-2 winner is TQT-based and gap to target is non-trivial. |
| 3a-2 | `lr: 1e-5 → 5e-5, steps: 1000 → 2000` (block_size unchanged) | 3a-1 already gave a positive net effect. |
| 3a-3 (CONDITIONAL) | `block_size: 4 → 2` only (lr/steps from best-so-far unchanged) | Path 1 — last iter was 3a-1 or 3a-2, regressed by < 0.5% relative AND introduced a new layer into the top-5 error list (joint training perturbed a quiet layer). Path 2 — 3a-1/3a-2 both improved on best AND none of R3/R5/R8 structural triggers match in `non_computing_hot_ops.json` / `layer_stats.json`. |

Disallowed parameter directions:
- `lr ≥ 1e-4` — overshoots regularly on representative reproducers (MobileNet-V2,
  YOLO-style detectors).
- `steps ≥ 4000` — diminishing returns + amplifies the lr=1e-4 instability.
- `block_size = 1` — equivalent to layerwise training; loses joint optimisation, no
  upside vs `block_size = 2`.
- `block_size ≥ 6` — unstable; overlaps the territory of `blockwise_reconstruction` (§8)
  but without its weight-only safety guard. If you reach for block_size ≥ 6, you should
  be considering lever 3g instead.

### Cost

- **PC-time**: expensive. CPU: 10-60+ minutes. GPU: 1-5 minutes for medium models.
- **On-device runtime**: zero. The graph topology and the per-op bit width are unchanged;
  only `log2_scale` exponents (and optionally weights) move.

---

## 4. Calibration algorithm (`quantize_activation_setting.calib_algorithm`)

> **Tier A — high accuracy, no on-device runtime cost.**

> **Important: calibration is not independently rankable in esp-dl quantization.** The
> "Strength / Weakness" descriptions below characterise the **calib-only** behaviour —
> what each algorithm does to scale picks **before** any further pass touches them. Do
> **not** use these to rank calibrations and pick a winner: a calibration that scores
> low standalone (e.g. `percentile` clipping a heavy tail and looking like it lost
> signal) can become the strongest base once paired with TQT, because the tighter scale
> gives the gradient pass more room to recover. Conversely, `kl` may win calib-only and
> then plateau under TQT. The Phase 2 procedure in `SKILL.md` evaluates calibrations as
> `calib × TQT(default)` cartesian product and ranks by the combined metric, never by
> calib-only — see Composition discipline #4 in `SKILL.md` and the Worked example.

### What it does

For each non-parameter tensor (activations), esp-ppq runs the calibration dataset through
the float graph and observes the value distribution to choose `(scale, zero_point)`. The
algorithm controls *how* that choice is made.

`espdl_setting()` defaults to `'kl'`. The available choices:

| Algorithm | How it chooses scale | Strength | Weakness |
|-----------|----------------------|----------|----------|
| `minmax`  | scale = max(\|x\|) / 127 (sym) | Captures full range, no clipping | One outlier ruins the scale; bad for long-tail distributions |
| `percentile` | scale = quantile(\|x\|, p) / 127 with `p ≈ 0.9999` | Robust against outliers | Clips real signal if `p` too low |
| `kl` | scale that minimises KL divergence between FP histogram and quantized histogram | Best for asymmetric / multi-modal distributions, default for esp-dl | Slow on huge histograms; can be unstable on bimodal data |
| `mse` | scale that minimises MSE between dequantized and FP values | Smooth, gradient-like; good when distributions are roughly Gaussian | Slightly more sensitive to skew than KL |
| `isotone` | monotonic-preserving variant; rare | Preserves ordering, useful for regression heads | Rarely needed for image classifiers/detectors |

### Parameters

```python
setting.quantize_activation_setting.calib_algorithm  # one of: minmax, kl, percentile, mse, isotone
```

`espdl_setting()` also sets `quantize_parameter_setting.calib_algorithm` indirectly via
the quantizer; it is left at the quantizer default and rarely needs changing.

The `percentile` algorithm uses `OBSERVER_PERCENTILE = 0.9999` by default
(see `esp-ppq/esp_ppq/core/common.py`). For
narrower or wider clipping you can override per quantization config via
`OBSERVER_PERCENTILE_MANUL_OVERRIDE`. The skill does **not** expose this knob in
`setting.json` — if you need to override it, use the `extra` escape hatch in
[setting_json_schema.md](setting_json_schema.md) to set the key on the underlying
quantization config; only do so as a last resort when the default 0.9999 has been
demonstrated to be the cause of accuracy loss on a specific layer's activation
histogram.

### When to enable / change

- Phase 2 sweeps `{kl, mse, percentile} × TQT(default)` automatically — agents do **not**
  pick a calibration heuristically. The list below is for *understanding* what each leg
  is likely to do, not for skipping legs.
  - Activation max/std ratio > 6, OR `Float Kurtosis > 6` ⇒ percentile leg likely wins.
  - Multi-modal float histogram (visible bimodality in `layer_stats.json` `Float Hist`)
    ⇒ mse leg likely wins.
  - Activation distribution close to Gaussian, low kurtosis ⇒ kl leg likely wins.
- All three legs must run unless `target_metric` is hit early (Phase 2 short-circuits).

### When to avoid

- Don't change `calib_algorithm` outside Phase 2 unless you have a very specific reason
  (e.g. extra escape hatch). Phase 3 levers operate on top of the Phase 2 winner's
  calibration — changing it again confounds the experiment.

### Cost

- **PC-time**: negligible — calibration runs in seconds either way.
- **On-device runtime**: zero. Only the per-tensor scale value changes.

---

## 5. Quantization fusion alignment (`fusion_setting`)

> **Tier A — surgical, typically no on-device runtime cost.**

### What it does

Controls how multi-input ops (Add, Concat, Resize, AveragePool) align their input
quantization configs. esp-ppq must enforce hardware constraints that all inputs share the
same scale/zero-point for these ops.

`espdl_setting()` defaults:

```python
fusion = True
fusion_setting.align_avgpooling_to = 'None'
fusion_setting.align_elementwise_to = 'Align to Output'   # esp-dl default; auto_quant uses 'Align to Large'
fusion_setting.align_concat_to = 'Align to Output'
fusion_setting.align_resize_to = 'Align to Output'
fusion_setting.force_alignment_overlap = False
```

### Options

- `'None'` — no alignment (will violate hardware constraint; use with care).
- `'Align to Large'` — pick the input with the largest range, scale others to match.
- `'Align to Output'` — use the output's quantization config for all inputs (default for esp-dl).
- `force_alignment_overlap = True` — propagate alignment upstream past producers; can cause
  cascading scale changes through the graph.

### When to change

- Concat/Add layers show high error AND their inputs have very different ranges. Try
  `'Align to Large'` for that pass.
- Don't flip these knobs unless the layerwise error report points specifically to alignment.

### Cost

- **PC-time**: free.
- **On-device runtime**: zero in normal cases. `force_alignment_overlap = True` can cause
  cascading scale rewrites that *theoretically* affect downstream activation ranges, but
  no new ops are inserted; the device runtime cost is unchanged. Re-run analysis after.

---

## 6. Mixed precision via `dispatching_table`

> **Tier B — high accuracy, with on-device runtime cost. Apply surgically (1-3 ops).**

### What it does

Dispatch specific operations to a higher-precision platform (`int16` instead of the global
`int8`). Affects only the listed ops; downstream/upstream ops stay at their native bits.

```python
from esp_ppq.api import get_target_platform

setting.dispatching_table.append(
    "/features/features.1/conv/conv.0/conv.0.0/Conv",
    get_target_platform(target, 16),
)
```

This is the *real* knob behind "mixed precision quantization" in the auto_quant example.
There is no `setting.mixed_precision = True` flag — `mixed_precision` in
`examples/tutorial/how_to_quantize_model/auto_quant/config.py`
just selects which ops to put in `dispatching_table`.

### Parameters

The dispatching table is a list of `(op_name, platform_int)` pairs. Each entry overrides
the dispatcher for that single op.

### When to enable

- A layer's SNR is **dramatically** higher than its neighbours (e.g. > 2× the median SNR)
  and structural fixes (equalization, calibration alg, bias correction) didn't help.
- The layer is on a critical path (early conv, last classifier layer, attention output).
- The user has accepted that int16 ops cost more memory/cycles on hardware.

### When to avoid

- Promoting more than ~10% of total ops to int16 — usually means a different problem
  (broken calibration, unsupported op shape) is dragging accuracy down everywhere; consider
  TQT (§3) for broad coverage instead, since TQT has zero on-device runtime cost.
- Promoting ops that are downstream of an already-promoted op without checking — esp-ppq
  handles boundary requantization automatically, but each int16↔int8 transition has a
  measurable cost.

### Cost

- **PC-time**: free at calibration time.
- **On-device runtime**: **non-zero** — int16 Conv/Gemm ops use roughly 2× the multiplier
  cycles and 2× the activation memory of int8 on esp-dl hardware, plus a small
  requantization at int16↔int8 boundaries. This is the soft penalty that drops mixed
  precision into Tier B; keep `dispatching_table` to the worst 1-3 ops.

### Op name discovery

Use `outputs/iter_<N>/simplified_ops.json` (emitted by the harness) to verify the op name
exists in the simplified graph. esp-ppq simplifies before quantization, and post-simplify
names sometimes differ from the original ONNX.

---

## 7. Horizontal weight split (`weight_split`)

> **Tier B — high accuracy on outlier-weight layers, with on-device runtime cost.**

### What it does

Split a layer with extreme per-channel weight outliers into two layers whose outputs are
summed. If `W` has a few channels with 10× larger range than the rest, splitting halves the
range so a single per-tensor scale can represent both halves accurately.

Splitting introduces a new Add op, which costs runtime. Only worth it for layers where
quantization error is genuinely caused by weight outliers and equalization (§1) didn't
already fix them.

### Parameters

```python
setting.weight_split = True
setting.weight_split_setting.interested_layers  # REQUIRED. None/[] means NO layers split. Always specify.
setting.weight_split_setting.value_threshold    # float, default 2.0. Weights below this are not split. Try 1.5, 2.0.
setting.weight_split_setting.method             # 'balance' (recommended) or 'random'.
```

### When to enable

- A specific Conv/Gemm has weight kurtosis > 10 OR `max(|w|) / std(w) > 8`.
- Equalization alone didn't fix that layer (you tried it in iter-N, still high error).
- The user values accuracy over runtime cost.

### When to avoid

- Splitting many layers — each adds a runtime op and the layer count grows fast.
- Layers in latency-critical inner loops if the user has tight perf budget.
- Before TQT (§3) — TQT has zero on-device runtime cost and may already close the gap on
  outlier-weight layers; try TQT first, fall back to weight_split for the residual.

### Cost

- **PC-time**: cheap.
- **On-device runtime**: **non-zero** — one extra Add op per split layer. Keep
  `interested_layers` to 1-3 entries.

---

## 8. Blockwise reconstruction (`blockwise_reconstruction`)

> **Tier C — last resort. Scales frozen by default → POWER_OF_2-safe. No on-device cost.**

### What it does

Most aggressive training-based pass. Splits the graph into blocks (size = `block_size`),
trains each block's quantized weights to match the float block's output via gradient
descent. Scales are typically frozen (`is_scale_trainable = False` by default), which makes
blockwise compatible with POWER_OF_2 esp-dl targets — the pass only nudges weights, not
scales.

Use this only when nothing else works.

### Parameters

```python
setting.blockwise_reconstruction = True
setting.blockwise_reconstruction_setting.interested_layers  # list[str] or None.
setting.blockwise_reconstruction_setting.lr            # float, default 1e-3 (much higher than TQT).
setting.blockwise_reconstruction_setting.steps         # int, default 5000.
setting.blockwise_reconstruction_setting.gamma         # default 1.0.
setting.blockwise_reconstruction_setting.block_size    # default 4.
setting.blockwise_reconstruction_setting.is_scale_trainable  # default False.
setting.blockwise_reconstruction_setting.collecting_device   # 'cuda' strongly recommended.
```

### When to enable / avoid

- Enable when accuracy gap is > 5% absolute and TQT (§3) didn't close it.
- Avoid on CPU — each iteration takes hours.
- Don't combine with TQT; pick one gradient-based pass.
- Keep `is_scale_trainable = False` on esp-dl: enabling it would attempt to train
  continuous scales which the POWER_OF_2 hardware cannot use.

### Cost

- **PC-time**: very expensive. GPU recommended.
- **On-device runtime**: zero. Default settings only adjust weights.

---

## 9. LSQ (learned step size, `lsq_optimization`)

> **Tier D — skipped on POWER_OF_2 esp-dl targets. Use TQT (§3) instead.**

### What it does

Like TQT but optimises both weights and a **continuous** scale simultaneously by
minimising the block-output reconstruction loss. esp-ppq's `LSQDelegator` gates
scale-training behind `not policy.has_property(POWER_OF_2)`; every esp-dl quantizer
policy includes `POWER_OF_2`, so LSQ silently degenerates to weight-only tuning while
paying TQT-level PC-time cost. The harness disables LSQ and warns on every esp-dl target
— see §12 for the full mechanism. **Always pick TQT (§3) instead** — it trains
`log2_scale` and is POWER_OF_2-native.

### Parameters (for reference only — the harness skips LSQ on esp-dl)

```python
setting.lsq_optimization = True
setting.lsq_optimization_setting.interested_layers  # list[str] or None.
setting.lsq_optimization_setting.lr            # float, default 1e-5.
setting.lsq_optimization_setting.steps         # int, default 500.
setting.lsq_optimization_setting.block_size    # int, default 4.
setting.lsq_optimization_setting.gamma         # regularization, default 0.0.
setting.lsq_optimization_setting.is_scale_trainable  # bool, default True (forced false on POWER_OF_2).
setting.lsq_optimization_setting.collecting_device   # 'cuda' recommended.
```

### When you might still want LSQ

Only if the user retargets to a non-POWER_OF_2 platform (e.g. `PPL_CUDA_INT8`). That is
outside esp-dl's scope; the rest of this skill assumes POWER_OF_2.

### Cost

- **PC-time**: expensive (same as TQT). Wasted on POWER_OF_2 targets.
- **On-device runtime**: zero — but moot, since the pass is skipped on esp-dl.

---

## 10. Experimental passes

These are documented for completeness. Default to **NOT enabling** them in iteration:

- `setting.ssd_equalization` — multi-iteration variant of equalization with loss check;
  ~30 min per run. Considered legacy.
- `setting.channel_split` — alternative to weight_split for OCS-style channel splitting.
  Usually weight_split is preferred for esp-dl.
- `setting.matrix_factorization` — SVD-based weight decomposition; experimental.
- `setting.extension` — placeholder for custom passes.
- `setting.ssd_setting`, `setting.channel_split_setting`, `setting.matrix_factorization_setting` —
  the per-pass parameters.

If the user explicitly asks for one of these, look up parameters in
`esp-ppq/esp_ppq/api/setting.py`.

---

## 11. Compatibility matrix (method × method)

|                       | calib_alg | equalization⁴ | mixed_precision | bias_correct | weight_split | TQT | LSQ³ | blockwise |
|-----------------------|:---------:|:-------------:|:---------------:|:------------:|:------------:|:---:|:----:|:---------:|
| **calib_alg**         | —         | OK            | OK              | OK           | OK           | OK  | OK   | OK        |
| **equalization⁴**     | OK        | —             | OK              | OK           | OK           | OK  | OK   | OK        |
| **mixed_precision**   | OK        | OK            | —               | OK           | OK           | OK  | OK   | OK        |
| **bias_correct**      | OK        | OK            | OK              | —            | OK           | OK  | OK   | careful¹  |
| **weight_split**      | OK        | OK            | OK              | OK           | —            | OK  | OK   | careful¹  |
| **TQT**               | OK        | OK            | OK              | OK           | OK           | —   | NO²  | NO²       |
| **LSQ³**              | OK        | OK            | OK              | OK           | OK           | NO² | —    | NO²       |
| **blockwise**         | OK        | OK            | OK              | careful¹     | careful¹     | NO² | NO²  | —         |

Notes:
- ¹ `bias_correct` / `weight_split` modify graph topology before training-based passes —
  combining works but the training will see a different graph than what the analysis
  observed. Re-run analysis after.
- ² Pick exactly one of TQT, LSQ, blockwise per iteration. They are different gradient-based
  approaches that fight each other.
- ³ LSQ is also incompatible with the **target** policy on every esp-dl variant — see §12.
  Even where the method-vs-method matrix says "OK", the harness will skip LSQ on
  POWER_OF_2 targets.
- ⁴ `equalization` on `esp32p4` is **warn-only, not auto-disabled** — see §12. esp-ppq
  officially flags per-channel weights + equalization as "Not recommend", and the harness
  emits a strong warning, but the pass still runs (some MobileNet-family models
  empirically benefit). Treat it as a Phase 3 lever to try **after** the calib×TQT
  cartesian product, not as a Phase 2 baseline component.

### Recommended combos (accuracy-first, on-device cost in parens)

- **No-on-device-cost bundle** *(0 device cost)*: `equalization + bias_correct + (calib_algorithm tweak)`
  — Tier A passes only. Try this before any Tier B method.
- **Mid-range** *(small device cost)*: above + `mixed_precision(top 1-3 layers)`
  — adds Tier B mixed precision surgically.
- **All-out (GPU recommended)** *(small device cost)*: `equalization + bias_correct + TQT + mixed_precision(top 1-3)`
  — TQT on top of Tier A passes, surgical mixed precision for residual.
- **Last resort (GPU strongly recommended)** *(0 device cost)*: above with `blockwise_reconstruction` instead of TQT.
- **Speed-priority bundle (GPU recommended)** *(0 device cost)*:
  `calib_algorithm + TQT (3a-1 → 3a-2) + bias_correct + fusion_setting + equalization + blockwise_reconstruction`
  — selected when the user sets `QUANT_CONFIG["deploy_runtime_priority"]="speed"`. The
  state machine in `compare_iterations.py` reorders the linear Phase-3 lever order to
  3a-1 → 3a-2 → 3b → 3c → 3d → **3g** → 3e → 3f, paying extra PC-time on blockwise to
  avoid the permanent on-device cost of mixed precision (3e) and weight split (3f). Use
  3e/3f only after 3g failed to close the gap.

---

## 12. Target-policy compatibility (POWER_OF_2 and per-channel weights)

There are **two** target-policy dimensions that matter for the skill:

1. **POWER_OF_2** (all esp-dl targets). Every concrete `EspdlQuantizer` (
   `esp-ppq/esp_ppq/quantization/quantizer/EspdlQuantizer.py`)
   declares `quantize_policy = SYMMETRICAL + LINEAR + PER_TENSOR + POWER_OF_2`, so
   `scale = 2 ^ exponent` for every quantized tensor. Some esp-ppq optimization passes
   were designed for continuous-scale quantizers and either no-op or actively misbehave
   under POWER_OF_2.

2. **Per-channel weight quantization** (only `esp32p4`). `EspdlQuantizer.create_espdl_quant_config`
   gates per-channel weight quantization on `operation.platform in _P4_PLATFORMS`
   (`ESPDL_INT8 / ESPDL_INT16 / ESPDL_H_PRE_INT16` — i.e. all P4 variants). On
   `target = esp32s3` or `target = c`, weights stay per-tensor. esp-ppq officially
   marks layer-wise equalization as "Not recommend" for per-channel weight quantization
   (see `esp-ppq/md_doc/Passes/LayerwiseEqualization.md`, "Usage" section);
   the harness honours this with a warning but still runs the pass (warn-only) since
   empirical results on some networks beat the official recommendation.

| Target | POWER_OF_2 | Per-channel weights for Conv/ConvTranspose/Gemm |
|--------|:----------:|:-----------------------------------------------:|
| `esp32p4` | ✅ | ✅ (every Conv/ConvTranspose/Gemm) |
| `esp32s3` | ✅ | ❌ (per-tensor) |
| `c`       | ✅ | ❌ (per-tensor) |

The full pass × target-policy compatibility:

| Pass | POWER_OF_2 | Per-channel weights (`esp32p4`) | Why | Harness behaviour |
|------|:----------:|:-------------------------------:|-----|-------------------|
| `equalization`              | ✅ Native | ⚠️ **Warn-only** | esp-ppq officially flags per-channel + equalization as "Not recommend": per-channel weight scales already absorb most channel imbalance, so equalization adds little to the weight side, and the rescaled activations get a wider per-channel spread that the per-tensor activation quantizer can handle worse. **However**, empirical runs on MobileNet-family / depthwise-separable networks show some still benefit. The harness lets the pass run and emits a strong warning so the agent can compare. | On `esp32p4`: **run + warn**; agent should treat it as a Phase 3 lever after the calib×TQT cartesian product. On `esp32s3 / c`: apply normally as Tier A canonical use case. |
| `bias_correct`              | ✅ Native | ✅ OK | Adjusts bias only; scales unchanged. | apply normally |
| `tqt_optimization`          | ✅ Native — designed for POWER_OF_2 | ✅ OK | Trains `log2_scale`, so `scale = 2 ^ log2_scale` is always POWER_OF_2 by construction (esp-ppq `TQTDelegator` requires `policy.has_property(POWER_OF_2)`). Per-channel weight scales are trained per-channel; no conflict. | apply normally |
| `calib_algorithm`           | ✅ Native | ✅ OK | Calibration picks an arbitrary `scale`, then esp-ppq rounds to the nearest POWER_OF_2 during commit. Independent of per-channel vs per-tensor weights. | apply normally |
| `fusion_setting`            | ✅ Native | ✅ OK | Alignment is a graph-level constraint, not a scale-format constraint. | apply normally |
| `dispatching_table` (mixed) | ✅ Native | ✅ OK | Just routes ops to a different POWER_OF_2 platform (int16). On `esp32p4` the dispatched platform is also a P4 variant (`ESPDL_INT16` / `ESPDL_H_PRE_INT16`), so per-channel still applies — int16 is the recommended substitute for equalization on P4. | apply normally |
| `weight_split`              | ✅ Native | ✅ OK | Weight transform; scales recomputed by calibration. Splits along the weight outlier axis, orthogonal to per-channel scaling. | apply normally |
| `blockwise_reconstruction`  | ✅ Native (with default `is_scale_trainable=False`) | ✅ OK | Default trains weights only; safe under POWER_OF_2. ⚠️ Setting `is_scale_trainable=True` would try to train continuous scales — leave it `False` on esp-dl. | apply normally; warn if `is_scale_trainable=True` |
| `lsq_optimization`          | ❌ **Conflict** | (n/a — already skipped on POWER_OF_2) | `LSQDelegator` checks `not policy.has_property(POWER_OF_2)`, so scale training is silently disabled. The pass degenerates to weight-only tuning while still paying TQT-level PC time. | **skip + warn**; recommend TQT |

The harness in [scripts/apply_setting.py](../scripts/apply_setting.py)
(`_check_target_compatibility()`) inspects the `target` field passed to
`apply_setting.apply()` and the enabled passes in `payload`. On esp-dl targets it:

1. **Per-channel-weight targets (`esp32p4`)**: whenever `equalization.enabled=true`,
   the harness lets the pass run unchanged but emits a **warning** noting esp-ppq's
   "Not recommend" stance. The agent should:
   - Run the Phase 2 calib×TQT cartesian product first.
   - Only enable equalization in Phase 3 if there's a structural reason (per-channel
     `max/mean > 5` on a Conv→activation→Conv chain) and accept the per-tensor
     activation widening risk.
   - Compare the result against best-so-far; if it regresses, drop it next iteration.
2. **POWER_OF_2 targets (all esp-dl)**: disables `lsq_optimization` and adds an
   explanatory warning recommending TQT. (LSQ would silently degenerate to weight-only
   tuning under POWER_OF_2 — `LSQDelegator` checks `not policy.has_property(POWER_OF_2)`.)
3. **POWER_OF_2 targets (all esp-dl)**: forces `blockwise_reconstruction.is_scale_trainable`
   back to `False` if the setting requested `True`, and warns — the pass still runs but
   only weights move, keeping the result POWER_OF_2-compliant.

All warnings appear in `apply_result.warnings`, which the harness writes to
`applied_setting_warnings` in `iteration_index.json` and prints to `console.log`.

The agent should read these warnings before writing the next iteration's setting. For
LSQ and `blockwise.is_scale_trainable=True` the harness has hard-disabled / forced the
flag — re-enabling them next round is a contract violation. For esp32p4 equalization the
warning is informational; the agent decides whether to keep / drop based on whether the
pass actually helped vs best-so-far.
