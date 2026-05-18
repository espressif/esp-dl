# Setting JSON schema

Each iteration N has a `setting.json` that the agent writes (or the harness writes for the
baseline). [scripts/apply_setting.py](../scripts/apply_setting.py) reads it and produces a
real `QuantizationSetting` object.

The schema is intentionally lossy — only the fields the skill commonly tunes are exposed.
For anything outside this schema (e.g. `ssd_equalization`), use the `extra` escape hatch.

## Top-level structure

```json
{
  "iteration_id": 1,
  "rationale": "Explain in plain text why this setting was chosen, referencing iter-0 numbers.",
  "calib_algorithm": "kl",
  "equalization": {...},
  "bias_correct": {...},
  "weight_split": {...},
  "tqt_optimization": {...},
  "lsq_optimization": {...},
  "blockwise_reconstruction": {...},
  "fusion_alignment": {...},
  "dispatching_table": [...],
  "extra": {...}
}
```

## Field reference

### `iteration_id` (int, required)

Iteration number. The harness uses this to name the output directory (`outputs/iter_<N>/`).
The baseline run uses `iteration_id = 0`.

### `rationale` (string, required for agent-written settings)

Free-text explanation of why this setting was chosen, citing iter-0 (or previous iteration)
numbers. The harness logs it; the final report renders it. Skipping rationale on
agent-written settings is a contract violation — the audit trail matters more than terse JSON.

The baseline iter-0 written by `--baseline` has `rationale: "default espdl_setting()"`.

### `calib_algorithm` (string, optional)

Sets `setting.quantize_activation_setting.calib_algorithm`. Allowed:
`kl` (default), `percentile`, `mse`, `minmax`, `isotone`. Case-insensitive.

If omitted, defaults to `"kl"` (the value `espdl_setting()` already uses).

### `equalization` (object, optional)

Wraps the equalization pass.

```json
"equalization": {
  "enabled": true,
  "iterations": 6,
  "value_threshold": 0.5,
  "opt_level": 2,
  "including_bias": false,
  "including_act": false,
  "interested_layers": null
}
```

- `enabled` (bool) — defaults to false. When false, all other keys are ignored.
- `iterations` (int) — esp-ppq default 10. Higher = flatter, but converges fast.
- `value_threshold` (float) — esp-ppq default 0.5.
- `opt_level` (int, 1 or 2) — **esp-ppq default is 1; the skill's lever-3d
  template recommends 2.** The two values are *not* interchangeable: level 2
  also crosses Add/Sub branches, which is what makes equalization useful on
  residual / inverted-residual architectures (ResNet, MobileNet-V2/V3 inverted
  residuals, etc.). Writing only `{"enabled": true}` resolves to `opt_level=1`
  and silently skips every Add/Sub-bounded chain — i.e. exactly the layers
  lever 3d was designed to fix. Always copy the full lever-3d snippet from
  `comparison.json["next_step_hint"]["advice"]` instead of abbreviating; the
  harness now warns when `opt_level` is unset (see
  [apply_setting.py](../scripts/apply_setting.py) `_apply_equalization`).
- `including_bias`, `including_act` (bool) — esp-ppq defaults false.
- `interested_layers` (list[str] or null) — null means "all eligible layers".

> **Target-policy note.** On `target = esp32p4` (per-channel weight quantization),
> esp-ppq officially flags equalization as "Not recommend"
> (see `esp-ppq/md_doc/Passes/LayerwiseEqualization.md`, "Usage" section).
> The harness behaviour is **warn-only**: the pass still runs when `enabled=true`, but
> `apply_result.warnings` carries an explanatory note (visible in
> `iteration_index.json["applied_setting_warnings"]` and `console.log`). Empirically some
> MobileNet-family / depthwise-separable networks still benefit on esp32p4, so the
> agent should treat this as a Phase 3 lever (`3d`) to try after the Phase 2 calib×TQT
> cartesian product.

### `bias_correct` (object, optional)

```json
"bias_correct": {
  "enabled": true,
  "block_size": 4,
  "steps": 32,
  "interested_layers": null
}
```

- `block_size` (int) — esp-ppq default 4. Use 1 for best accuracy, slower.
- `steps` (int) — esp-ppq default 32.
- `interested_layers` (list[str] or null).

### `weight_split` (object, optional)

```json
"weight_split": {
  "enabled": true,
  "interested_layers": ["/some/conv/Conv"],
  "value_threshold": 1.5,
  "method": "balance"
}
```

- `interested_layers` is REQUIRED when `enabled=true`. esp-ppq treats null/[] as "no
  layers" for this pass — explicitly listing layers is the only mode.
- `value_threshold` (float) — esp-ppq default 2.0.
- `method` (str) — `"balance"` (recommended) or `"random"`.

### `tqt_optimization` (object, optional)

```json
"tqt_optimization": {
  "enabled": true,
  "lr": 1e-5,
  "steps": 500,
  "block_size": 4,
  "is_scale_trainable": true,
  "gamma": 0.0,
  "int_lambda": 0.0,
  "collecting_device": "cpu",
  "interested_layers": null
}
```

- See [ppq_methods.md §6](ppq_methods.md#6-tqt-trained-quantization-thresholds-tqt_optimization).

### `lsq_optimization` (object, optional)

Same shape as `tqt_optimization` minus `int_lambda`. Don't enable both at once.

### `blockwise_reconstruction` (object, optional)

```json
"blockwise_reconstruction": {
  "enabled": true,
  "lr": 1e-3,
  "steps": 5000,
  "block_size": 4,
  "is_scale_trainable": false,
  "gamma": 1.0,
  "collecting_device": "cuda",
  "interested_layers": null
}
```

### `fusion_alignment` (object, optional)

```json
"fusion_alignment": {
  "align_avgpooling_to": "None",
  "align_elementwise_to": "Align to Output",
  "align_concat_to": "Align to Output",
  "align_resize_to": "Align to Output",
  "force_alignment_overlap": false
}
```

These map directly to fields under `setting.fusion_setting`. Only specify keys you want to
change; missing keys keep `espdl_setting()` defaults.

### `dispatching_table` (array, optional)

Mixed precision dispatch entries. Each entry:

```json
{"op": "/features/features.1/conv/conv.0/conv.0.0/Conv", "bits": 16}
```

- `op` (str) — must match an op name in `simplified_ops.json` (case-sensitive).
- `bits` (int) — `8` or `16`.

The harness validates each `op` against `simplified_ops.json` from the previous iteration
and emits a warning for any unknown name (without failing the run).

### `extra` (object, optional)

Escape hatch for fields outside the schema. The harness merges this dict into
`QuantizationSetting`'s `__dict__` after applying the typed fields. Use sparingly:

```json
"extra": {
  "ssd_equalization": false,
  "channel_split": false
}
```

## Validation

`apply_setting.py` rejects malformed JSON early and prints a clear error. Specifically it
checks:

- `iteration_id` is a non-negative int.
- Each enabled pass has the required sub-keys (e.g. `weight_split.interested_layers` when
  `weight_split.enabled=true`).
- `calib_algorithm` is one of the supported strings.
- `dispatching_table` ops exist in `simplified_ops.json` from the most recent prior run
  (warns; doesn't fail).
- No mutually-exclusive combinations (TQT + LSQ + blockwise_reconstruction more than one
  at a time).

## Recommended iteration templates

These are the canonical payloads for the prescriptive search procedure in `SKILL.md`
Phases 2–3. Copy them, fill in the `rationale`, and write to
`outputs/iter_<N>/setting.json`. The fastest way is to run
[scripts/compare_iterations.py](../scripts/compare_iterations.py) first — its
`comparison.json["next_step_hint"]` embeds the exact template for the next iteration.

### Phase 2 — calibration × TQT(default) cartesian product (iter_1 → iter_2 → iter_3)

Each Phase 2 iteration enables **exactly two** fields: `calib_algorithm` and
`tqt_optimization` (with the strict default schedule). No other pass is enabled. The
cartesian product is mandatory because **calibration is not separable from TQT** in
esp-dl quantization — see SKILL.md's Composition discipline #4 and the Worked example.

> **Run each iteration to completion before writing the next.** Read its
> `metrics.json` first. If any leg meets `QUANT_CONFIG["target_metric"]`, skip the rest
> of Phase 2. Iterations must be strictly sequential — see `SKILL.md`'s Phase 2 section
> for why parallel runs break the search (single GPU, calibration-data race, lost
> short-circuit signal).

The TQT default schedule is **strict**:
`lr=1e-5, steps=500, block_size=4, is_scale_trainable=true, gamma=0.0, int_lambda=0.0`.
[compare_iterations.py](../scripts/compare_iterations.py) enforces this fingerprint when
deciding whether a calib leg has been "covered". Using any other TQT schedule in Phase 2
means that iteration does **not** count as covering the calib leg, and the state machine
will require an additional iteration with the exact default schedule.

iter_1: kl + TQT(default).

```json
{
  "iteration_id": 1,
  "rationale": "Phase-2 calib×TQT cartesian product, leg 1 (kl). iter_0 (kl-only) gave <metric>=X; this iteration adds TQT(default) on top to read off the TQT-on-kl delta. Pure conjoined Phase-2 base — no other pass enabled.",
  "calib_algorithm": "kl",
  "tqt_optimization": {
    "enabled": true,
    "lr": 1e-5,
    "steps": 500,
    "block_size": 4,
    "is_scale_trainable": true,
    "gamma": 0.0,
    "int_lambda": 0.0,
    "collecting_device": "cuda",
    "interested_layers": null
  }
}
```

iter_2: mse + TQT(default) — *only after iter_1 lands and only if iter_1 didn't hit
`target_metric`*.

```json
{
  "iteration_id": 2,
  "rationale": "Phase-2 calib×TQT cartesian product, leg 2 (mse). iter_1 (kl+TQT) did not hit target_metric. Test mse paired with TQT(default) — no other pass enabled.",
  "calib_algorithm": "mse",
  "tqt_optimization": {
    "enabled": true,
    "lr": 1e-5,
    "steps": 500,
    "block_size": 4,
    "is_scale_trainable": true,
    "gamma": 0.0,
    "int_lambda": 0.0,
    "collecting_device": "cuda",
    "interested_layers": null
  }
}
```

iter_3: percentile + TQT(default) — *only after iter_2 lands and only if iter_2 didn't
hit `target_metric`*.

```json
{
  "iteration_id": 3,
  "rationale": "Phase-2 calib×TQT cartesian product, leg 3 (percentile). iter_2 (mse+TQT) did not hit target_metric. Test percentile paired with TQT(default) — often the hidden winner on heavy-tailed activations even when standalone percentile would regress.",
  "calib_algorithm": "percentile",
  "tqt_optimization": {
    "enabled": true,
    "lr": 1e-5,
    "steps": 500,
    "block_size": 4,
    "is_scale_trainable": true,
    "gamma": 0.0,
    "int_lambda": 0.0,
    "collecting_device": "cuda",
    "interested_layers": null
  }
}
```

### Phase 3 — surgical residual fix (example: equalization on `esp32s3`)

Distribution-driven, single-knob mutation on top of best-so-far (read from
`comparison.json["best_iteration"]["dir"]/setting.json`). Lever 3d — equalization — is
the canonical Tier A pass for per-tensor weight targets.

```json
{
  "iteration_id": 4,
  "rationale": "Phase-3 lever 3d (equalization). Best-so-far is iter_3 (percentile + TQT(default)) -> top1=Y. Top error layer /features/features.1/conv/conv.0/conv.0.0/Conv has weight max/mean ratio 12 → R3 trigger fires (target=esp32s3, per-tensor weights). Single new knob vs iter_3.",
  "calib_algorithm": "percentile",
  "tqt_optimization": {"enabled": true, "lr": 1e-5, "steps": 500, "block_size": 4, "is_scale_trainable": true, "gamma": 0.0, "int_lambda": 0.0, "collecting_device": "cuda"},
  "equalization": {"enabled": true, "iterations": 6, "value_threshold": 0.5, "opt_level": 2}
}
```

### Phase 3 — surgical residual fix (example: mixed precision on `esp32p4`)

Lever 3e — mixed precision — is the typical Tier B fallback when Phase 2 plateaued and
1-3 layers dominate the error budget.

```json
{
  "iteration_id": 5,
  "rationale": "Phase-3 lever 3e (mixed precision). Best-so-far is iter_3 (percentile + TQT(default)) -> top1=Y; gap to target is 0.4. Two layers' SNR > 2× median of top-5 (R6 trigger). Promote them to int16 via dispatching_table. Tier-B, ≤3 ops capped.",
  "calib_algorithm": "percentile",
  "tqt_optimization": {"enabled": true, "lr": 1e-5, "steps": 500, "block_size": 4, "is_scale_trainable": true, "gamma": 0.0, "int_lambda": 0.0, "collecting_device": "cuda"},
  "dispatching_table": [
    {"op": "/features/features.1/conv/conv.0/conv.0.0/Conv", "bits": 16},
    {"op": "/features/features.0/features.0.0/Conv", "bits": 16}
  ]
}
```

### Phase 3 — surgical residual fix (example: equalization on `esp32p4`, warn-only)

Lever 3d on `esp32p4` is **warn-only** — esp-ppq officially flags it as "Not recommend"
for per-channel weights, but some MobileNet-family / depthwise-separable networks still
benefit. Try it only after the Phase 2 cartesian product has settled and you have a
distribution reason (per-channel `max/mean > 5` on a Conv→activation→Conv chain).

```json
{
  "iteration_id": 4,
  "rationale": "Phase-3 lever 3d (equalization, warn-only on esp32p4). Best-so-far is iter_3 (percentile + TQT(default)) -> top1=Y. Top-error layer is a depthwise-separable Conv with per-channel weight max/mean=11 → R3 trigger fires structurally. esp-ppq officially flags equalization as Not recommend on per-channel weights; harness emits a warning but lets the pass run. Will compare against iter_3 best-so-far; if regression, drop this lever next round per Composition discipline #3.",
  "calib_algorithm": "percentile",
  "tqt_optimization": {"enabled": true, "lr": 1e-5, "steps": 500, "block_size": 4, "is_scale_trainable": true, "gamma": 0.0, "int_lambda": 0.0, "collecting_device": "cuda"},
  "equalization": {"enabled": true, "iterations": 6, "value_threshold": 0.5, "opt_level": 2}
}
```
