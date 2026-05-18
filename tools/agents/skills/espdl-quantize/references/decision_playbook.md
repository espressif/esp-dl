# Decision Playbook: distribution patterns → candidate methods

Read this together with [ppq_methods.md](ppq_methods.md). The playbook tells you *which*
method to try given an observation; `ppq_methods.md` tells you *how* the method works and
what its parameters do.

The playbook reads **three** harness artifacts, each closing a different blind spot of
the others (see SKILL.md "esp-ppq three-function coverage table" for the underlying
reason):

| Artifact | Source | Primary use |
|----------|--------|-------------|
| `outputs/iter_<N>/layerwise_error.json` | `layerwise_error_analyse` | Pick the top-5 *computing-op* layers driving final error in isolation. |
| `outputs/iter_<N>/layer_stats.json` (and full `layer_stats_full.json`) | `statistical_analyse` | Inspect distribution rows (input / weight / output) for the layers shortlisted by layerwise. Every Rule below uses these rows as evidence. |
| `outputs/iter_<N>/non_computing_hot_ops.json` | `statistical_analyse` aggregated | Surface non-COMPUTING_OP culprits (Concat, Add, Resize, Pool, Sigmoid, Softmax, GRU, LayerNorm, …) that `layerwise_error.json` cannot see. Drives R8 / R9. |
| `outputs/iter_<N>/graphwise_jumps.json` | `graphwise_error_analyse` − `layerwise_error_analyse` | Detect intervening non-computing ops between two computing ops where the cumulative SNR jump is bigger than the downstream computing op's isolated contribution. Confirms R8 candidates. |

Each entry in `layer_stats.json` / `layer_stats_full.json` is one tensor
(input / weight / output of one op) and looks like:

```json
{
  "Op name": "/features/features.1/conv/conv.0/conv.0.0/Conv",
  "Op type": "Conv",
  "Is parameter": false,
  "Is input": true,
  "Is output": false,
  "Variable name": "...",
  "Noise:Signal Power Ratio": 0.182,
  "Noise Mean": 0.013, "Noise Std": 0.041,
  "Quantized Mean": ..., "Quantized Std": ..., "Quantized Max": ..., "Quantized Min": ...,
  "Float Mean": 0.21, "Float Std": 0.94, "Float Max": 11.7, "Float Min": -0.83,
  "Float Skewness": 2.31, "Float Kurtosis": 14.8,
  "Float Hist": [...]
}
```

Use SNR > 0.05 (5% noise:signal) as a rough "needs work" threshold. Below that, the layer
is usually fine and tweaking it can hurt other layers.

## How to use this playbook

Step A — Computing-op signal (covers R1-R7, R10):

1. Take the top-5 ops from `layerwise_error.json`.
2. For each, read its three rows (input, weight, output) from `layer_stats.json`.
3. Walk rules R1-R7 / R10 in order — first matching rule wins. Collect candidate methods.

Step B — Non-computing-op signal (covers R8, R9):

4. Open `non_computing_hot_ops.json`. The list is sorted by max per-variable SNR.
5. For each entry with `max_snr > 0.05`, walk R8 / R9 (Concat/Add/Resize/Pool/Sigmoid/
   Softmax/GRU/LayerNorm). The aggregated `inputs_float_std_ratio` field is the direct
   R8 trigger — no need to recompute it from per-row data.
6. Cross-check with `graphwise_jumps.json`: any entry whose `intervening_non_computing_ops`
   includes a non_computing_hot_ops candidate is a high-confidence R8/R9 target. If the
   non-computing op only shows up in graphwise_jumps but not in non_computing_hot_ops, it
   still merits investigation (cumulative-cum-jump signal beats per-variable SNR for
   identity-like ops with low individual SNR but bad propagation).

Step C — Pick & rank:

7. If multiple top-5 layers (Step A) point to the same method, that's the strongest
   signal — apply that method first.
8. If only one layer is problematic, prefer surgical methods (mixed precision,
   weight_split on that layer alone). If 5/5 layers are problematic, prefer global
   methods (calib_algorithm change, equalization, TQT).
9. Step B candidates always take priority over Step A's R6 (mixed precision) — fixing a
   Concat scale mismatch with `fusion_alignment` (zero on-device cost) beats promoting
   the downstream Conv to int16 (Tier B cost) for the same root cause.

Order candidate methods by **expected accuracy gain on this model**, not by how long
the quantization step takes on the PC/server. Apply a soft penalty to passes that slow
on-device inference (mixed precision int16, weight_split) — they should be used surgically
on the worst 1-3 layers, not blanket-on. The "Tier" tag on each rule maps directly to the
"Accuracy-first method ordering" section in `SKILL.md` and to `ppq_methods.md` §11/§12.

---

## Rules

### R1 — Activation has heavy outliers (long tail)

**Trigger** (look at the layer's *input* row, where `Is input == True` and `Is parameter == False`):
- `Float Kurtosis > 6`, OR
- `Float Max / (Float Std + 1e-9) > 6`, OR
- The right-tail bins of `Float Hist` have non-trivial mass beyond `Float Mean + 5*Float Std`.

**Action**:
- Set `calib_algorithm = "percentile"` (Tier A, no on-device cost).
- If the issue is concentrated in 1-2 layers, also dispatch those layers to int16 via
  `dispatching_table` (Tier B, surgical: pays a small on-device cost on those ops only).

**Anti-pattern**: don't switch to `minmax` — that makes outliers *worse* by widening the scale.

### R2 — Activation is bimodal / multi-modal

**Trigger**:
- `Float Hist` shows two or more visibly separated peaks, OR
- `|Float Skewness| < 0.3` AND the histogram has a deep valley near the mean.

**Action**:
- Try `calib_algorithm = "mse"` (Tier A).
- If MSE doesn't help, dispatch the layer to int16 (Tier B, surgical).

**Why**: KL divergence behaves poorly on multi-modal distributions when one mode is much
narrower than the other. MSE smooths this out.

### R3 — Weight has per-channel imbalance (data-free fix candidate)

**Trigger** (weight row, `Is parameter == True`):
- Op type is `Conv`, `ConvTranspose`, `Gemm`, or `MatMul`.
- The weight per-channel `max(|w|) / mean(|w|) > 5`. (You may need to compute this from
  `Float Max / abs(Float Mean)` as a coarse proxy if true per-channel data isn't directly
  exposed; absolute kurtosis > 8 is also a reliable indicator.)
- Layer is part of a `Conv → activation → Conv` chain. Examples (not required):
  depthwise/pointwise pairs (MobileNet, EfficientNet), bottleneck residual blocks
  (ResNet), inverted residuals (MobileNet-V2/V3), or any back-to-back conv pattern. **The
  rule is structural, not network-family-specific.**

**Action — depends on the target's weight quantization policy** (see
[ppq_methods.md §12](ppq_methods.md) for the full target ↔ policy table):

- **`target ∈ {esp32s3, c}` (per-tensor weights)**: enable `equalization = True` with
  default `iterations=10, value_threshold=0.5, opt_level=2`
  (Tier A — large gains, no on-device cost, data-free). When applicable, replace `ReLU6`
  with `ReLU` *before* quantization (see the example in
      `examples/tutorial/how_to_quantize_model/quantize_mobilenetv2/quantize_torch_model.py`
      function `convert_relu6_to_relu`).
- **`target = esp32p4` (per-channel weights)**: equalization is **warn-only**. esp-ppq
  officially marks the combination as "Per-channel: Not recommend"
      (see `esp-ppq/md_doc/Passes/LayerwiseEqualization.md`, "Usage" section),
  but some MobileNet-family / depthwise-separable networks still benefit empirically. The
  recommended order on `esp32p4` is:
  1. **First** — finish the Phase 2 calib×TQT cartesian product (kl/mse/percentile each
     paired with TQT(default)). TQT is the strongest accuracy lever on POWER_OF_2 and
     usually closes most of the gap.
  2. **Then** — if iter_0..3 plateau and the dominant error layer matches the structural
     trigger above (per-channel `max/mean > 5` on a Conv→activation→Conv chain), try
     equalization as a Phase 3 lever (`3d`).
  3. **Sanity check** — equalization on per-channel widens the per-tensor activation
     range; if the iteration regresses, drop it next round.
- Other Phase 3 lever alternatives if equalization regresses:
  - **TQT escalation (3a-1 → 3a-2 → conditionally 3a-3)** — Tier A, POWER_OF_2-native,
    no on-device cost. The state machine picks the next sub-step automatically.
  - **Mixed precision int16 (3e)** — surgical fix for the worst 1-3 layers.
  - **Weight split (3f)** — only if a specific Conv has true outlier weights *within* an
    output channel (kurtosis > 10) that per-channel quantization still can't absorb.

**Anti-pattern**: don't enable equalization on graphs with heavy concat/branch structures
(YOLO necks, transformer attention) — try `opt_level=1` first; if still bad, fall back to
mixed precision. On `esp32p4`, do not lead with equalization in Phase 2 — always run the
calib×TQT cartesian product first.

### R4 — Weight has individual outliers

**Trigger** (weight row):
- `Float Kurtosis > 10`, AND
- `Float Max / Float Std > 8`, AND
- equalization (R3) didn't help in a previous iteration.

**Action**:
- Enable `weight_split = True` with `interested_layers` = [this layer's name],
  `value_threshold = 1.5`, `method = 'balance'`
  (Tier B — high-accuracy fix for outlier weights but inserts an extra `Add` op per split
  layer at runtime; cap at ≤3 layers).

**Anti-pattern**: don't split more than 3 layers; cumulative on-device runtime cost grows
quickly. Try TQT (R7) on the same layer first if you want a no-runtime-cost alternative.

### R5 — Output has systematic mean shift (bias error)

**Trigger** (output row):
- `|Noise Mean| > 0.1 * Noise Std`, AND
- The op type has a bias (Conv with bias, Gemm with C, etc.).

**Action**:
- Enable `bias_correct = True` (Tier A — no on-device cost; PC time ~5-15 min on CPU).
- Set `bias_correct_setting.block_size = 1` for best result if you can afford the time;
  default `4` is fine for a first try.

**Why**: bias error is a fixed shift, perfectly correctable by adjusting the bias term.

### R6 — Single layer dominates the error budget

**Trigger**:
- One layer's SNR is > 2× the median SNR of the top-5.
- Equalization, calibration tweaks, and bias_correct have all failed for it.
- The layer is on a critical path (early conv, classifier head, attention output).

**Action**:
- Dispatch this single op (and optionally its immediate successor) to int16 via
  `dispatching_table` (Tier B — usually closes the per-layer gap entirely; pays an
  on-device cost on the promoted op only).
- Strongly consider TQT (R7) as an alternative or a follow-up — TQT trains the
  log2-scale on every conv/gemm with no runtime cost. Choose mixed precision when only
  1-2 layers dominate; choose TQT when several layers share the error.

**How to find successors**: read `outputs/iter_<N>/simplified_ops.json` — it contains the
op graph structure. Or use the helper in `auto_quant/quantize.py`'s `get_next_node`.

### R7 — Many layers have moderate error, none dominates

**Trigger**:
- Top-5 SNRs are all roughly similar (within 1.5× of each other).
- Average SNR across them is > 0.05.
- Earlier iterations with calibration / equalization / bias_correct plateaued.

**Action**:
- Enable `tqt_optimization = True` with `interested_layers = None` (all conv/gemm).
- Use `lr=1e-5, steps=500, block_size=4`. On esp-dl POWER_OF_2 targets this trains
  `log2_scale` directly, which is the highest-accuracy training-based pass available
  here and **costs nothing extra at inference time** on device.
- **Warn the user** that this iteration is PC-time-expensive (10-60+ min on CPU; minutes
  on GPU). The PC-time cost is acceptable in the accuracy-first ordering: device inference
  speed and final accuracy are unchanged.

**Anti-pattern**: don't enable LSQ on POWER_OF_2 esp-dl targets — esp-ppq's `LSQDelegator`
disables scale training under POWER_OF_2, so LSQ degenerates to weight-only tuning. The
harness in [scripts/apply_setting.py](../scripts/apply_setting.py) detects this and skips
LSQ with a warning. Use TQT instead. Also: don't combine TQT with blockwise reconstruction
(R10) in the same iteration; pick one.

### R8 — Concat / Add / Resize layer has high error and inputs have very different ranges

**Trigger** (read from `non_computing_hot_ops.json` first; legacy `layer_stats.json` row
inspection is the fallback for the rare case where the aggregated artifact is missing):
- Op type ∈ {`Concat`, `Add`, `Sub`, `Mul`, `Resize`, `AveragePool`}.
- `inputs_float_std_ratio > 5` in `non_computing_hot_ops.json` (this is exactly
  max(Float Std) / min(Float Std) across non-parameter input variables).
- (Optional reinforcement) the same op name appears under
  `intervening_non_computing_ops` in `graphwise_jumps.json` — this confirms the op is
  contributing more cumulative error than its downstream computing op's isolated share
  would predict.

**Action**:
- Set `fusion_setting.align_elementwise_to = 'Align to Large'` (or `align_concat_to` for
  Concat, `align_resize_to` for Resize, `align_avgpooling_to` for AveragePool)
  (Tier A — no on-device cost).
- Don't enable `force_alignment_overlap = True` unless the layer is inside a critical path
  AND the change still doesn't help — overlap propagates scale changes upstream.

### R9 — Nothing in the playbook matched (rare structures: GRU, LSTM, attention)

**Trigger** (preferred source: `non_computing_hot_ops.json`; secondary: `layer_stats_full.json`):
- Op type ∈ {`GRU`, `LSTM`, `Softmax`, `LayerNorm`, custom op} with `max_snr > 0.05` on
  any per-variable row.
- Or the rules above didn't fire and SNR is still high on a Conv/Gemm.

**Action**:
- Default to mixed precision (R6 method): dispatch the worst 1-3 ops to int16
  (Tier B, surgical).
- If broad coverage is needed, try TQT (R7) — Tier A and POWER_OF_2-native.
- If both fail, ask the user — these structures sometimes need esp-ppq quantizer-level
  fixes that this skill cannot perform alone.

### R10 — Everything plateaued and the gap is still large

**Trigger**:
- Tier A passes (equalization, bias_correct, calib alg, TQT) and Tier B passes
  (mixed precision, weight_split) have all been tried.
- Accuracy gap from float baseline is still > 5% absolute.
- The user has GPU available (or is willing to wait hours on CPU).

**Action**:
- Enable `blockwise_reconstruction = True` with `lr=1e-3, steps=5000, block_size=4`,
  `is_scale_trainable=False` (Tier C — biggest hammer; scales frozen so POWER_OF_2-safe;
  no on-device cost).
- Don't combine with TQT (also gradient-based) — pick blockwise if TQT plateaued.

---

## Multi-iteration strategy

The playbook above tells you *which* method to consider given a distribution observation.
This section tells you *in which order to try them across iterations*. The ordering is
prescriptive on purpose — the most common search failure on esp-dl targets is judging a
calibration on its standalone score (which doesn't predict its TQT-paired score) and
then jumping to training-based passes (TQT, blockwise) on the wrong base.

The plan below mirrors the Phases in `SKILL.md`. Plan iter N+1 *only after* reading
`outputs/iter_<N>/metrics.json` and the
[scripts/compare_iterations.py](../scripts/compare_iterations.py) hint —
**never queue several iterations upfront**. Each iteration's `setting.json` is
literally the embedded template from `comparison.json["next_step_hint"]` plus a
filled-in `rationale`.

> **Run iterations strictly sequentially — never in parallel.** A single GPU cannot host
> two quantizations at once, `user_quant.py` calibration data downloads race on the
> cache, and any of the three Phase-2 legs is allowed to short-circuit the rest if
> `target_metric` is hit early. See `SKILL.md`'s *Phase 2* section for why.

The TQT default schedule used in Phase 2 is **strict** (compare_iterations.py enforces
this when deciding whether a calib leg has been "covered"):

```json
{"lr": 1e-5, "steps": 500, "block_size": 4, "is_scale_trainable": true,
 "gamma": 0.0, "int_lambda": 0.0, "collecting_device": "cuda"}
```

If the agent uses any other TQT schedule in Phase 2, that iteration does **not** count as
covering the calib leg — the state machine will require an additional iteration with the
exact default schedule.

**Iter sequence (Phase 1 + Phase 2):**

| iter_id | Phase | What to try | Notes |
|---------|:-----:|-------------|-------|
| 0       | 1 — baseline | `--baseline` (default `espdl_setting()`, `kl` calibration, no other passes) | Reference point. Source of truth for top-5 high-error layers and `layer_stats.json`. |
| 1       | 2 — calib×TQT sweep | `{"calib_algorithm": "kl", "tqt_optimization": <default>}` | Pair iter_0(kl-only) and iter_1(kl+TQT) → reads off the TQT-on-kl delta. If iter_1 hits `target_metric`, stop. |
| 2       | 2 — calib×TQT sweep | `{"calib_algorithm": "mse", "tqt_optimization": <default>}` | If iter_2 hits `target_metric`, stop. |
| 3       | 2 — calib×TQT sweep | `{"calib_algorithm": "percentile", "tqt_optimization": <default>}` | Often the hidden winner on heavy-tailed activations even when standalone percentile would regress (see Worked example in SKILL.md). |

**Phase 3 levers (iter_4, iter_5, ...):** pick **one** lever per iteration, in priority
order, on top of the current best-so-far. Each iteration starts fresh by copying
`comparison.json["best_iteration"]["dir"]/setting.json`, changing exactly one thing, and
writing a new `iteration_id`.

| Lever | Tier | On-dev cost | What changes | Notes |
|-------|:----:|:-----------:|--------------|-------|
| 3a-1 | A | 0 | TQT escalation step 1: `steps: 500 → 1000` only (lr/block_size unchanged). | One knob; the gentlest TQT escalation. |
| 3a-2 | A | 0 | TQT escalation step 2: `lr: 1e-5 → 5e-5, steps: 1000 → 2000` (block_size unchanged). | Only enter after 3a-1 gave a positive net effect. Do not push beyond — `lr=1e-4` / `steps=4000` regularly overshoots in representative reproducers. |
| 3a-3 | A | 0 | TQT escalation step 3 (CONDITIONAL): `block_size: 4 → 2` (lr/steps from best-so-far). | Two trigger paths: **Path 1 (unstable fallback)** — last iter was 3a-1 or 3a-2 AND regressed by < 0.5% relative AND introduced a new layer into the top-5 error list. **Path 2 (gap-shrink after convergence)** — 3a-1/3a-2 both improved on best AND none of R3/R5/R8 structural triggers match. block_size=2 trades joint-training scope for stability. Do not try block_size=1 (full layerwise, no upside) or block_size≥6 (overlaps lever 3g, unstable). |
| 3b | A | 0 | Bias correction (R5) when an output row shows `|Noise Mean| > 0.1 × Noise Std`. | Surgical. Skip if the prior iteration already enabled it without help. |
| 3c | A | 0 | Fusion alignment (R8) when `non_computing_hot_ops.json` has a Concat/Add/Resize entry with `inputs_float_std_ratio > 5`. | Surgical. Read non_computing_hot_ops.json first; do not derive the trigger from layer_stats.json by hand if non_computing_hot_ops is non-empty. |
| 3d | A | 0 | Equalization (R3) — `esp32s3 / c` is the canonical use case; **`esp32p4` is allowed but warn-only**, treat as best-effort lever after Phase 2. | Large gains on Conv→Conv chains. ReLU6 → ReLU pre-quant is recommended when applicable. |
| 3e | B | + | Mixed precision (R6) — promote 1–3 worst layers to int16 via `dispatching_table`. | Permanent on-device runtime cost (~2× cycles + ~2× activation memory on promoted ops). Cap at ≤3 ops; promoting >10% usually means a different bug. |
| 3f | B | + | Weight split (R4) — exactly one Conv/Gemm with weight kurtosis > 10. | Permanent on-device runtime cost (extra Add op per split layer). Only after equalization (where applicable) didn't fix it. |
| 3g | C | 0 | Blockwise reconstruction (R10) — last resort. | `is_scale_trainable=False` so POWER_OF_2-safe and zero on-device cost. PC time: hours on CPU, minutes on GPU. **Mutually exclusive with TQT**: pick exactly one gradient-based pass. |

Notes:

- **LSQ (`lsq_optimization`) is intentionally absent** — under POWER_OF_2 (all esp-dl
  targets) `LSQDelegator` disables scale training, so it degenerates to weight-only
  tuning and wastes a slot. The harness skips it with a warning. Use TQT.
- **Don't compress steps**. Skipping the Phase 2 cartesian product is the #1 cause of
  search failure on esp-dl targets — it is the only way to discover that
  `percentile + TQT(default)` can outperform `kl + TQT(default)` (or any aggressive
  TQT-on-kl) even though `percentile` standalone regresses below baseline. The cost is
  three cheap iterations (with short-circuit on `target_metric`); the upside can be
  0.3%+ on the primary metric.

### On-device cost reordering

`compare_iterations.py` reorders the linear-order Phase-3 levers based on
`QUANT_CONFIG["deploy_runtime_priority"]` (default `"balanced"`). 3a-3 is conditional and
not in either order.

| Priority | Linear-order Phase-3 sequence |
|----------|------------------------------|
| `balanced` (default) | 3a-1 → 3a-2 → 3b → 3c → 3d → 3e → 3f → 3g |
| `speed` | 3a-1 → 3a-2 → 3b → 3c → 3d → 3g → 3e → 3f |
| `pc_time` | (reserved — currently behaves like `balanced`) |

Why the speed reorder works: 3g (blockwise reconstruction) has **zero on-device cost**
but the largest PC-time. 3e (mixed precision int16) and 3f (weight split) are the only
two `+`-cost levers. Under `speed`, we pay the extra PC-time of 3g first to *avoid*
permanent on-device latency. Under `balanced`, we let the cheaper-PC levers (3e, 3f)
run first because their +cost is bounded (≤3 promoted/split layers per
"Composition discipline #2" cap) and a lot of users prefer to finish faster.

Setting `deploy_runtime_priority="pc_time"` is a placeholder: today it is treated as
`"balanced"`. Future revision could deprioritise 3g and the longest-running TQT step
(3a-2) when the user explicitly wants the iteration loop done in minutes.

## Composition discipline (why this ordering exists)

These rules apply between *every* iteration, not just inside Phase 3. They are the
search-side counterpart of the per-method playbook above.

1. **Mutate from best-so-far, not the last iteration.** Always start the next
   `setting.json` from `comparison.json["best_iteration"]["dir"]/setting.json`, then
   change exactly one knob. The latest run can be a regression you should not inherit.
2. **One new method (or one parameter change) per iteration.** If an iteration enables
   two new things at once and regresses, you cannot tell which one hurt. The Phase 2
   `calib_algorithm + tqt_optimization(default)` pair is treated as the **conjoined Phase
   2 base** — both fields enter and leave together inside Phase 2. Outside Phase 2,
   change exactly one knob.
3. **Stop escalating after one regression.** If iter_N raises a TQT hyper-parameter (or
   tightens a lever) and the metric drops, do not push further on that axis; pivot to a
   different lever from the Phase 3 table.
4. **Never retire a calibration algorithm based on its calib-only score.** Calibration is
   the input distribution shaper for the downstream training pass (TQT in esp-dl, since
   POWER_OF_2 disables LSQ). **Calib-only accuracy does not predict combined accuracy:**
   percentile may regress standalone but become the strongest TQT base, because
   tail-clipping leaves more "training space" for TQT to recover. The classic confounded
   failure mode (e.g. enabling `percentile + bias_correct` simultaneously and blaming
   percentile when the iteration regresses) is *not* fixed by re-testing percentile alone
   — clean percentile-only would also regress in that scenario, but `percentile + TQT(default)`
   could still be the overall winner. The only valid evaluation is the
   **`calib × TQT(default)` cartesian product** that Phase 2 mandates. Never rank or
   eliminate calibrations by their standalone iteration score.

## What if iteration N is *worse* than iter-0?

Possible causes:
- A method amplified an issue (equalization on a transformer-shaped graph; equalization
  on `esp32p4` widening per-tensor activation range past what the activation quantizer
  can handle).
- Calibration algorithm clipped legitimate signal (`percentile` standalone often does
  this — that's why it must be paired with TQT in Phase 2, not run alone).
- Mixed precision dispatched the wrong layer or too many layers.
- LSQ was enabled on a POWER_OF_2 target and silently degenerated to weight-only tuning
  (the harness warns and skips — check `applied_setting_warnings`).
- Two methods were turned on at once and the new one fights with the existing one — see
  *Composition discipline* #2.

Action:

- Roll back to the previous best iteration's `setting.json` as the next iteration's
  starting point.
- Drop only the method that caused the regression — and only if the experiment was clean
  (single change). If the experiment was confounded, re-test the suspect alone.
- Don't compound regressions — `compare_iterations.py` shows the trajectory.

## Sanity checks before declaring "done"

- The `model.espdl` file from the chosen iteration loads cleanly with esp-dl
  (the harness checks parse-ability after export — see `console.log` for any errors).
- The full `evaluate()` (not `evaluate_fast`) confirms the metric on the chosen iteration.
- The chosen `setting.json` rationale reads as a coherent story you could explain to the
  user — every Phase-3 step says *which distribution observation* triggered it.
