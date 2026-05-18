---
name: espdl-quantize
description: >
  Iteratively tune esp-ppq QuantizationSetting to recover post-quantization accuracy on
  ESP-DL targets. Drives a closed loop of "baseline -> calibration × TQT(default)
  cartesian product -> distribution-aware residual fixes -> re-evaluate" in the current
  Python environment, using a minimal user contract (calib dataloader + evaluate
  function). Generic across architectures (ResNet / EfficientNet / ViT / DETR / YOLO /
  LSTM and any esp-ppq-supported graph) — the search procedure is distribution-driven
  and does not depend on a specific network family. Method ordering is accuracy-first
  with a soft penalty for passes that slow down on-device inference; LSQ on POWER_OF_2
  targets is auto-disabled (silent degenerate; use TQT instead) and esp32p4 layer-wise
  equalization is warn-only (esp-ppq officially "Not recommend" for per-channel weights
  but empirically can still help on some models).
  Use this skill whenever the user wants to improve a quantized esp-dl/.espdl model's
  accuracy, debug high quantization error, choose between calibration algorithms,
  decide on equalization / bias correction / weight split / mixed precision / TQT /
  blockwise reconstruction settings, or run an automated search over esp-ppq
  quantization options. Also triggers for "esp-ppq 量化调参", "降低量化误差",
  "Layerwise quantization error 分析", "QuantizationSettingFactory.espdl_setting 怎么调",
  "混合精度量化", "量化精度恢复", "espdl 量化优化".
---

# ESP-DL Quantization Tuning Skill

This skill turns the human "stare at error report, guess setting, rerun" loop into a
structured, distribution-aware search. The user owns data loading and evaluation; the
skill owns `QuantizationSettingFactory.espdl_setting()` and the iteration loop.

> **About `<SKILL_DIR>` in shell snippets below.** This skill is agent-directory
> agnostic — it may be installed as `.cursor/skills/espdl-quantize/`,
> `.opencode/skills/espdl-quantize/`, or under any other agent's skills folder.
> Whenever you see `<SKILL_DIR>` in a shell command, substitute the **absolute path of
> the directory containing this `SKILL.md`** (the agent runtime gives you that path when
> it loads the skill). Setting it once at the start of a session makes the rest copy-pasteable:
>
> ```bash
> SKILL_DIR=/abs/path/to/espdl-quantize   # the directory holding this SKILL.md
> ```
>
> All in-skill markdown links (e.g. `[scripts/run_iteration.py](scripts/run_iteration.py)`)
> are already relative to `<SKILL_DIR>` and need no substitution.

## Generality boundary

This skill is a **general framework** for any esp-ppq-supported graph: ResNet,
EfficientNet, ViT, DETR, YOLO family, LSTM, custom CV/NLP backbones — anything
`espdl_quantize_torch` / `espdl_quantize_onnx` can ingest. The Phase 2 calib×TQT
cartesian product, the Phase 3 lever ordering, and the four Composition discipline
rules are all **distribution-driven**; none of them depends on a specific network
structure or family. The Worked example at the end uses MobileNet-V2 on ESP32-P4 as an
empirical demonstration — its concrete numbers are illustrative, not a model-selection
threshold.

## Why this skill exists

esp-ppq exposes a dozen tunable passes (calibration algorithm, layerwise equalization,
bias correction, weight split, mixed precision via dispatching table, TQT, LSQ, blockwise
reconstruction, ...). Each has 2-6 parameters. Trying them by hand is slow and error-prone.

What this skill brings to the table:

1. **Knowledge** — every esp-ppq method's principle, parameters, applicable scenarios, and
   anti-patterns are codified in [references/ppq_methods.md](references/ppq_methods.md).
2. **A decision rulebook** — given the top-K worst layers' input/weight/output distributions,
   [references/decision_playbook.md](references/decision_playbook.md) maps observed patterns to
   candidate methods.
3. **A fixed harness** — [scripts/run_iteration.py](scripts/run_iteration.py) takes the user's
   contract module plus a JSON setting and emits structured artifacts (metrics, layerwise error,
   per-layer stats) so the agent only has to read JSON to make the next decision.
4. **A search state machine** — [scripts/compare_iterations.py](scripts/compare_iterations.py)
   inspects what's already been tried and tells the agent which iteration to run next via
   `comparison.json["next_step_hint"]`. The hint embeds a complete `setting.json` template so
   the agent only has to fill in the rationale.
5. **Target-aware safety net** — the harness detects passes that conflict with the target's
   quantization policy:
   - **LSQ on POWER_OF_2 targets** (`esp32p4 / esp32s3 / c`) — auto-disabled. esp-ppq's
     `LSQDelegator` silently disables continuous-scale training under POWER_OF_2, so the
     pass would degenerate to weight-only tuning while paying full TQT-level PC time.
     Use TQT instead — it trains `log2_scale` and is POWER_OF_2-native.
   - **Layer-wise equalization on `esp32p4`** — **warn-only** (changed in this revision).
     esp-ppq officially marks the combination as "Not recommend"
      (see `esp-ppq/md_doc/Passes/LayerwiseEqualization.md`, "Usage" section),
     but empirical runs show some MobileNet-family / depthwise-separable networks still
     benefit. The harness now lets the pass run when `equalization.enabled=true` and
     emits a strong warning; the agent should treat it as a Phase 3 lever to try only
     after the calib×TQT cartesian product has settled.

## What the user has to provide

A single Python module (typically named `user_quant.py`) that exports:

- `QUANT_CONFIG` dict — model path, input shape, target chip, bits, primary_metric, etc.
- `create_calib_dataloader()` — returns the calibration `DataLoader`.
- `evaluate(quant_graph)` — returns a dict whose keys include `QUANT_CONFIG["primary_metric"]`.
- For torch flow only: `get_torch_model()` — returns the `nn.Module`.
- Optional: `collate_fn(batch)` and `evaluate_fast(quant_graph)`.

The full contract spec is in [references/contract.md](references/contract.md). Two ready-to-copy
examples live in [assets/user_quant_torch_example.py](assets/user_quant_torch_example.py) and
[assets/user_quant_onnx_example.py](assets/user_quant_onnx_example.py).

The skill never edits the contract module. All iteration state lives under `outputs/`.

---

## High-level flow

```mermaid
flowchart TD
    contract[user_quant.py] --> harness[run_iteration.py]
    setjson[outputs/iter_N/setting.json<br/>written by agent] --> harness
    harness --> ppqapi["esp_ppq.api.espdl_quantize_torch / _onnx"]
    ppqapi --> graph["esp-ppq BaseGraph<br/>(esp_ppq.IR.BaseGraph)"]
    graph --> lwerr["layerwise_error_analyse"]
    graph --> stat["statistical_analyse"]
    graph --> evalfn[user.evaluate]
    lwerr --> art[outputs/iter_N/]
    stat --> art
    evalfn --> art
    art --> compare[compare_iterations.py]
    compare --> hint["comparison.json<br/>next_step_hint"]
    hint --> agent[agent reads hint, fills in rationale]
    agent -.writes next setting.-> setjson
```

The agent's job each round shrinks to: read `comparison.json["next_step_hint"]`, copy the
embedded `setting.json` template, fill in the `rationale` field, run the harness.

## Phases

### Phase 0 — Validate contract and environment

> **Important — ignore Docker / image / `/work` mentions you may see elsewhere.**
> Some user projects (and `user_quant.py` itself) still carry comments left over
> from an older Docker-based workflow — phrases like "build the image",
> "Phase 0 — docker 准备", `/work inside Docker`, or `docker run --gpus all`. Those
> are **legacy text only**, not steps to execute. The skill now runs entirely in
> the current Python interpreter.

Before any quantization, do these once per session:

1. Make sure the **current Python environment** has `esp_ppq` (with the `[cpu]` extra),
   `torch`, plus the small set of helpers the harness needs:

   ```bash
   pip install -e <path/to/esp-ppq>[cpu]
   pip install -r "$SKILL_DIR/assets/extra_requirements.txt"
   ```

   The skill is **environment-agnostic** — it does not require Docker. As long as
   `python -c "import esp_ppq, torch, onnx, onnxsim, pandas, scipy, tqdm"` succeeds, you are
   ready to go.

2. Validate the user's contract module imports cleanly and exposes the required functions/keys:

   ```bash
   python "$SKILL_DIR/scripts/run_iteration.py" \
     --user-quant <path/to/user_quant.py> \
     --output-dir <path/to/user_project>/outputs/contract_check \
     --check-contract
   ```

3. Make sure the iteration workdir exists (default: `<contract_dir>/outputs/`). The harness
   creates it on first run.

> The working directory for `python` should be the directory containing `user_quant.py` (or
> any directory — the harness resolves relative paths in `QUANT_CONFIG` against the
> contract module's directory).

### Phase 1 — Baseline (iter_0)

Run the default `QuantizationSettingFactory.espdl_setting()` once. The agent does NOT propose
any settings here — the harness uses a built-in baseline JSON when `--baseline` is passed.

```bash
python "$SKILL_DIR/scripts/run_iteration.py" \
  --user-quant <path/to/user_quant.py> \
  --output-dir <path/to/user_project>/outputs/iter_0 \
  --baseline
```

After it finishes, read these files:

- `outputs/iter_0/metrics.json` — what `evaluate()` returned, plus `_primary` shortcut.
- `outputs/iter_0/layerwise_error.json` — `{op_name: snr}` sorted descending by error.
  Covers **only `is_computing_op`** (Conv / Gemm / ConvTranspose / MatMul / Attention /
  PPQBiasFusedMatMul / LSTM); the SNR is the *isolated* contribution of that op when
  it alone is quantized.
- `outputs/iter_0/layer_stats.json` — `statistical_analyse` filtered by the layerwise
  top-K (legacy artifact; same coverage as layerwise).
- `outputs/iter_0/layer_stats_full.json` — **(new)** the full `statistical_analyse`
  output: every non-passive op's per-input/per-output distribution + cumulative SNR.
  This is the only artifact that includes Add / Concat / Resize / AveragePool /
  Sigmoid / Softmax / GRU / LayerNorm.
- `outputs/iter_0/non_computing_hot_ops.json` — **(new)** the top-K non-COMPUTING_OP
  layers ranked by max per-variable SNR, plus `inputs_float_std_ratio` (max/min
  Float Std of input variables, used by playbook rule R8).
- `outputs/iter_0/graphwise_jumps.json` — **(new)** adjacent computing-op pairs whose
  cumulative SNR gap is *not* explained by the downstream op's isolated contribution.
  Lists the intervening non-computing ops as suspected culprits.
- `outputs/iter_0/console.log` — full stdout/stderr.

Tell the user the baseline numbers, the top-5 error layer names from layerwise, and (if
non-empty) the top-3 entries from non_computing_hot_ops.json. The state machine in
`compare_iterations.py` decides when to finalize — do not stop here on your own even if
iter_0 looks like it hit `target_metric`; run the comparison once and let
`next_step_hint["phase"] == "phase-4-final-report"` confirm.

### Phase 2 — Calibration × TQT(default) cartesian product (mandatory)

This phase **must** run three iterations in strict sequence, each enabling exactly two
fields: `calib_algorithm` and `tqt_optimization` (with the esp-ppq default schedule). No
other pass is enabled. The cartesian product is what makes the search robust — calibration
in esp-dl quantization is **not separable** from the training pass: a calibration that
regresses standalone may become the strongest base when paired with TQT, and vice versa.
See the Worked example below for the empirical case that motivated this design.

The TQT default schedule is **strict**:

```json
{
  "lr": 1e-5,
  "steps": 500,
  "block_size": 4,
  "is_scale_trainable": true,
  "gamma": 0.0,
  "int_lambda": 0.0,
  "collecting_device": "cuda"
}
```

Iteration sequence:

| Iter | calib_algorithm | other passes | Purpose |
|------|-----------------|--------------|---------|
| `iter_1` | `kl` | TQT(default) | Pair iter_0(kl-only) and iter_1(kl+TQT) to read off the TQT-on-kl delta. If iter_1 hits target, stop. |
| `iter_2` | `mse` | TQT(default) | Same with mse. If hits target, stop. |
| `iter_3` | `percentile` | TQT(default) | Same with percentile. Often the hidden winner on heavy-tailed activations even when standalone percentile would regress. |

The **way to drive this** is to run [scripts/compare_iterations.py](scripts/compare_iterations.py)
between iterations:

```bash
python "$SKILL_DIR/scripts/compare_iterations.py" \
  --output-dir <path/to/user_project>/outputs
```

`comparison.json["next_step_hint"]` will be `phase-2-calib-tqt-sweep` until all three
calibrations are covered with TQT(default), and the embedded `setting.json` template can
be copied verbatim into `outputs/iter_<N>/setting.json` (only fill in the `rationale`).

> **Critical: iterations are strictly sequential — never run two in parallel.** Single
> GPU, calibration-data download race, and any of the three legs can short-circuit the
> rest if it hits `target_metric`. If you spawn parallel subagents the search breaks.

### Phase 3 — Residual fixes from best-so-far

After Phase 2 the best-so-far iteration is the one with the highest `primary_value` among
iter_0..3. `comparison.json["next_step_hint"]` switches to `phase-3-residual` (or
`phase-3-pivot` if the last two iterations both regressed vs best).

Each Phase 3 iteration **must** mutate from `comparison.json["best_iteration"]`'s
`setting.json` and change exactly **one** thing. The lever order below is the linear
default for `deploy_runtime_priority="balanced"`; lever 3a-3 is **conditional** (entered
only when one of two specific signals fires); the speed-priority reorder is described
under `Accuracy-first method ordering` below.

| Lever | Tier | On-device cost | What changes | When to use |
|-------|:----:|:--------------:|--------------|-------------|
| 3a-1 | A | 0 | TQT `steps: 500 → 1000` (lr=1e-5, block_size=4 unchanged) | Phase-2 winner is TQT-based and gap to target is non-trivial. One knob only — Composition discipline #2. |
| 3a-2 | A | 0 | TQT `lr: 1e-5 → 5e-5, steps: 1000 → 2000` (block_size=4 unchanged) | 3a-1 already gave a positive net effect. Do NOT push beyond this on the lr/steps axis (lr=1e-4 / steps=4000 stably regress on representative reproducers). |
| 3a-3 | A | 0 | TQT `block_size: 4 → 2` (lr/steps from best-so-far unchanged) | **CONDITIONAL — enter only on one of these two signals**: (1) **unstable fallback** — last iter was 3a-1 or 3a-2, regressed by < 0.5% relative AND introduced a new layer into the top-5 error list (TQT joint training perturbed a previously-quiet layer); or (2) **gap-shrink after convergence** — 3a-1/3a-2 both improved on best AND none of R5/R8/R3 structural triggers match in best's `layer_stats.json` / `non_computing_hot_ops.json`. Smaller block_size = closer to layerwise = more stable. Do not try block_size=1 (full layerwise, no upside) or block_size≥6 (overlaps lever 3g, unstable). |
| 3b | A | 0 | `bias_correct.enabled=true` | A top-error op's *output* row shows `|Noise Mean| > 0.1 × Noise Std`. |
| 3c | A | 0 | `fusion_alignment.align_elementwise_to = 'Align to Large'` (and friends) | A row in `non_computing_hot_ops.json` with `op_type ∈ {Concat, Add, Resize, AveragePool}` AND `inputs_float_std_ratio > 5`. (Legacy fallback: same condition checked from layer_stats.json.) |
| 3d | A | 0 | enable `equalization` (full lever-3d template; do **not** abbreviate to `enabled=true` only — esp-ppq defaults `opt_level=1` while the template recommends `opt_level=2`, see Common pitfalls) | Conv→activation→Conv chain with weight per-channel `max/mean > 5`. **Per-tensor weight targets (`esp32s3 / c`) are the canonical use case; on `esp32p4` the pass is warn-only — esp-ppq officially "Not recommend" for per-channel weights but it can empirically help on some MobileNet-family / depthwise-separable nets.** |
| 3e | B | **+** | `dispatching_table` int16 on top 1-3 worst layers | One layer's SNR > 2× median of the top-5; structural fixes failed. ≤10% of total ops. **Permanent on-device runtime cost** (~2× cycles + ~2× activation memory on promoted ops). |
| 3f | B | **+** | `weight_split` on a single Conv with weight kurtosis > 10 | Equalization didn't fix it (or wasn't applicable on esp32p4). ≤3 split layers. **Permanent on-device runtime cost** (one extra Add op per split layer). |
| 3g | C | 0 | `blockwise_reconstruction` (last resort) | Tier A + Tier B all plateaued and gap > 5% absolute. GPU strongly recommended. **Mutually exclusive with TQT** — the lever template explicitly disables TQT before enabling blockwise; this is a bigger structural change than other levers. |

The state machine in `compare_iterations.py` automatically picks the next lever per the
table above and the `deploy_runtime_priority` knob. The agent's job each Phase-3
iteration shrinks to: read `comparison.json["next_step_hint"]["advice"]`, copy the
embedded change snippet onto best-so-far's `setting.json`, fill `rationale` with the
specific layer-stats observation that drove the choice, run the harness.

Stop conditions are no longer the agent's call — when **any** of the following hold,
the state machine emits `phase-4-final-report` and the agent finalizes:

- The user's `primary_metric` reached or exceeded `target_metric`.
- The most recent 2 iterations both sit within 0.1% relative of best-so-far (plateau).
- 5 Phase-3 iterations have been run after the cartesian product (default cap).
- All linear-order Phase-3 levers (3a-1, 3a-2, 3b, 3c, 3d, 3e, 3f, 3g) have been tried.

### Phase 4 — Final report

**Two ways to enter Phase 4:**

* **State-machine trigger** (machine view): `comparison.json["next_step_hint"]["phase"] == "phase-4-final-report"`. The state machine emits this when target reached, plateau, Phase-3 cap, or all linear-order levers tried (see Phase 3's stop-condition list).
* **User-budget trigger** (human view): the user gave the agent a specific iteration budget — phrasings like "iterate 3 times", "迭代 3 轮", "只跑 N 轮", "iterate N times", "最多 N 轮", "only N iterations". When this budget is hit, **the user-budget trigger always wins** even if the state machine still wants to keep going.

**Why the auto-finalize is bullet-proof.** `compare_iterations.py` writes `outputs/best/` and `outputs/final_report.md` whenever **either** trigger fires:

* **Automatic** when `phase == "phase-4-final-report"` — every invocation of `compare_iterations.py --output-dir <outputs>` checks this and finalizes if true. Agents reading the script's stdout will see a `[compare] phase-4 detected; finalize results: ...` block.
* **On demand** via the `--finalize` flag at any time, regardless of phase. This is the escape hatch for the user-budget case — agents should copy the command from `comparison.json["early_finalize_command"]` (or the printed "Tip: how to wrap up at any time" block at the bottom of `compare_iterations.py`'s stdout) and run it after the last user-budgeted iteration completes.

The generated `final_report.md` carries an HTML marker comment on its first line. Subsequent finalize runs detect the marker → safely refresh the report (no data loss). If an agent has hand-edited the file (and removed the marker), subsequent finalize preserves it untouched unless `--force` is passed. Sections `## Key findings` and `## Remaining gap (if target not met)` are seeded with auto-bullets but agents are explicitly invited (via the marker comment) to expand them with concrete distribution interpretations from `layer_stats.json` / `non_computing_hot_ops.json` / `graphwise_jumps.json`.

**Iteration history table — new columns.** The auto-generated table now includes:

* **`rank`** — dense ranking by `primary_value` (1 = best). Recomputed from disk on every finalize, so adding more iterations later won't reverse the relative order of any pre-existing pair (this is asserted by the unit tests). Columns visible in the report and in `comparison.json["iteration_ranks"]`.
* **`affects inference speed`** — `"No"` for almost all settings; `"Yes (...)"` only when the iteration enables `dispatching_table` int16 promotion or `weight_split` (the only two passes with permanent on-device runtime cost; see "On-device runtime cost cheat-sheet"). When the **best** iteration has `affects inference speed = Yes`, **inspect the rank-2/3 rows** — if they trade < 0.1% accuracy for `affects inference speed = No`, the user may prefer the runner-up for production deployment.

**Recommended agent workflow after finalize:**

1. Read `outputs/final_report.md`. The Summary, Iteration history (with rank + speed columns), Best setting, Python snippet are auto-generated; expand `## Key findings` and `## Remaining gap` with concrete bullets if the model warrants.
2. **Run a single full-eval re-check** to replace the iteration loop's `evaluate_fast()` number with the user's real `evaluate()`:
   `python {SKILL_DIR}/scripts/run_iteration.py --user-quant <...> --setting outputs/best/setting.json --output-dir outputs/iter_<NEW> --use-full-eval`.
   If the resulting `<primary_metric>` differs from what's in the Summary, update the Summary line in `final_report.md`. The marker line keeps the file refreshable; once you remove the marker (or pass `--force` from the script), subsequent automated runs won't clobber edits.
3. If you ever need to regenerate the report from scratch (e.g. after fixing a bug in an iteration), run:
   `python {SKILL_DIR}/scripts/compare_iterations.py --output-dir <outputs> --finalize --force`.

> **Legacy fallback.** The pre-auto-finalize workflow (manually run `--write-best`, hand-write `outputs/final_report.md`) is still supported for completeness — if for any reason `compare_iterations.py` does not emit the artifacts (e.g. broken iteration data on disk), the agent can fall back to that flow. The `--write-best` flag now writes only `outputs/best/`; the report remains the agent's responsibility in that fallback.

**Final-report template (auto-emitted by the script, for reference / audit):**

```
<!-- auto-generated marker line — agents may edit Key findings / Remaining gap -->
# Final Report: <model> on <target>

## Summary
- Best iteration: iter_<N>
- <primary_metric>: <value> (target_metric=<target or "not set">)
- _Note: value comes from evaluate_fast(); run --use-full-eval to refresh._
- On-device speed cost vs baseline (best): <No | Yes (...)>
- Other metrics: <copy from outputs/best/metrics.json>

## Iteration history
| iter | method changed | <primary_metric> | delta | outcome | rank | affects inference speed |

## Best setting
<inline the FULL outputs/best/setting.json>

## Python snippet
<auto-translated QuantizationSettingFactory.espdl_setting() recipe>

## Key findings
<auto-bullets — agents extend with concrete distribution interpretations>

## Remaining gap (if target not met)
<auto-bullets — agents replace boilerplate with model-specific recommendations>
```

---

## Composition discipline (read before every iteration)

These four rules govern the iteration loop. Violating any of them = current iteration is
discarded, agent rolls back to best-so-far and re-runs.

1. **Mutate from best-so-far, not the last iteration.** Always start the next
   `setting.json` from `comparison.json["best_iteration"]["dir"]/setting.json`. The most
   recent run can be a regression you should not inherit.

2. **One new method (or one parameter change) per iteration.** Calibration algorithm and
   `tqt_optimization` with the default schedule are treated as the **conjoined Phase 2
   base** — they enter and leave together inside Phase 2. Outside Phase 2, change exactly
   one knob. If two changes are stacked and the iteration regresses, you can't tell which
   one hurt.

3. **Stop escalating after one regression.** If iter_N raises a TQT hyper-parameter (or
   tightens a lever) and the metric drops, do not push further on that axis; pivot to a
   different lever from the Phase 3 table.

4. **Never retire a calibration algorithm based on its calib-only score.** Calibration is
   the input distribution shaper for downstream passes (especially TQT in esp-dl, where
   POWER_OF_2 makes TQT the only available training-based pass). Calib-only accuracy does
   **not** predict combined accuracy: percentile may regress standalone but become the
   strongest TQT base, because tail-clipping leaves more "training space" for TQT to
   recover. Phase 2 must always evaluate calibration with `calib × TQT(default)` cartesian
   product, never with calib-only ranking. See Worked example below for the concrete
   reproducer.

---

## Operating principles

### Always look at distributions before changing settings

The iter-0 layerwise table tells you *which* layers hurt, but not *why*. The
`layer_stats.json` tells you why. Don't propose `equalization=True` because the doc says
it helps with depthwise convs — propose it because the layer's per-channel weight
`max/mean` ratio is > 5 **and** the layer is part of a Conv→activation→Conv chain. The
[references/decision_playbook.md](references/decision_playbook.md) formalises this; follow
it unless you have a strong reason not to.

#### esp-ppq three-function coverage table

esp-ppq exposes three error analysers; each has a distinct scope and they are
**not interchangeable**. The harness invokes all three every iteration; the playbook
combines them.

| Function | Scope | What is filtered out | SNR semantics | Output artifact |
|----------|-------|----------------------|---------------|-----------------|
| `layerwise_error_analyse` | `is_computing_op` only (Conv/Gemm/ConvTranspose/MatMul/Attention/PPQBiasFusedMatMul/LSTM) | Everything else, including all activations / Concat / Add / Resize / Pool / Sigmoid / Softmax | **Isolated** — quantize this op only, leave the rest in FP, measure the final-output SNR delta. | `layerwise_error.json` |
| `graphwise_error_analyse` | Same `is_computing_op` set | Same | **Cumulative** — quantize everything up to and including this op, leave downstream in FP, measure the final-output SNR. Larger than layerwise; difference reveals interactions. | `graphwise_error.json` |
| `statistical_analyse` | All `QuantableOperation` minus `PASSIVE_OPERATIONS` (`MaxPool / Slice / Reshape / Transpose / Identity / Squeeze / Unsqueeze / Cast / GatherND / Scatter / Pad / Tile / NonMaxSuppression / RoiAlign / TopK / Resize / Split / Flatten / DepthToSpace / SpaceToDepth`) | Passive ops only — **includes Add / Concat / AveragePool / Sigmoid / Softmax / GRU / LayerNorm / Resize and many other non-COMPUTING_OP types** | **Per-variable cumulative SNR** — distribution stats (Float Mean/Std/Skew/Kurt, Quant counterparts, Noise Mean/Std, Noise:Signal ratio) on every input/output tensor. SNR is computed against the FP reference at that variable, not at final output. | `layer_stats_full.json`, plus filtered `layer_stats.json` and aggregated `non_computing_hot_ops.json` |

**Coverage gap and how the playbook closes it.** Because layerwise filters to
COMPUTING_OP, a Concat with mismatched input scales or a Resize after a wide-range
sigmoid will *never* show up in `layerwise_error.json` — yet they can dominate the
final-output error. Two harness-side products bridge the gap:

- **`non_computing_hot_ops.json`** ranks non-COMPUTING_OP layers by their max
  per-variable SNR (from `statistical_analyse`) and adds `inputs_float_std_ratio`
  (max/min Float Std across input variables) for direct R8 trigger detection.
- **`graphwise_jumps.json`** computes `graphwise[op_next] − graphwise[op_prev] −
  layerwise[op_next]` for every adjacent pair of computing ops in the simplified graph.
  When this excess is > 0.02 (overridable via
  `QUANT_CONFIG["graphwise_intervening_excess_threshold"]`), the non-computing ops
  sitting between `op_prev` and `op_next` are flagged as suspected culprits.

The decision_playbook reads all three artifacts before proposing any Phase-3 lever.

### Accuracy-first method ordering (with a soft penalty for on-device runtime cost)

Pick which pass to try next by **expected accuracy gain on this model**, not by how long
the quantization step takes on the PC/server. PC time is a one-shot cost; on-device
latency and final accuracy are what the user lives with. Apply a *soft* penalty to passes
that slow down inference on the ESP target so they get used surgically, not blanket-on.

**Tier A — High accuracy, no on-device runtime cost** (the Phase 2 base + most Phase 3
levers):

1. **TQT (`tqt_optimization`)** — POWER_OF_2-native gradient training of `log2_scale`. The
   strongest single pass for esp-dl POWER_OF_2 targets when calibration alone plateaus.
   The Phase 2 cartesian product uses the conservative default schedule
   (lr=1e-5, steps=500, block_size=4); Phase 3 escalates in three stages: 3a-1 (steps
   500→1000), 3a-2 (steps 1000→2000, lr 1e-5→5e-5), 3a-3 (CONDITIONAL — block_size 4→2
   for stability when 3a-1/3a-2 perturb a quiet layer or all other levers don't apply).
2. **Calibration algorithm** (`calib_algorithm`) — `{kl, mse, percentile}` are the three
   options Phase 2 sweeps (each combined with TQT(default)). The interplay between
   calibration and TQT is non-monotonic, see Composition discipline #4.
3. **Bias correction** — fixes systematic mean-shift error baked into Conv/Gemm bias.
4. **Equalization** — large gains on Conv→Conv chains (depthwise/pointwise pairs in
   MobileNet, bottleneck residuals in ResNet, inverted residuals in MobileNet-V2/V3, etc.).
   On per-tensor weight targets (`esp32s3 / c`) the canonical Tier A pass; on `esp32p4`
   per-channel weight target it is **warn-only** (esp-ppq officially "Not recommend" for
   per-channel; empirically can still help on some networks — see
   [references/ppq_methods.md §1](references/ppq_methods.md) for the full rationale).
5. **Fusion alignment** (`fusion_setting`) — surgical fix for Concat/Add/Resize layers
   with mismatched input ranges.

**Tier B — High accuracy, but slows on-device inference** (apply surgically; gain must
justify the slowdown — typically only on the worst 1-3 layers):

6. **Mixed precision via `dispatching_table` (→ int16)** — promote the very worst layer(s)
   from int8 to int16. Cap at ≤10% of total ops, ideally 1-3.
7. **Weight split (`weight_split`)** — only on layers whose weight outliers couldn't be
   handled by equalization. Inserts an extra `Add` op per split layer at runtime.

**Tier C — Last resort** (when Tier A+B plateau and the user accepts long offline training):

8. **Blockwise reconstruction (`blockwise_reconstruction`)** — biggest hammer. Scales are
   frozen by default (POWER_OF_2-safe); only weights move. GPU strongly recommended; CPU
   runs are hours per iteration. Mutually exclusive with TQT.

**Tier D — Auto-disabled on POWER_OF_2 targets**:

- **LSQ (`lsq_optimization`)** — esp-ppq's `LSQDelegator` disables scale training under
  POWER_OF_2, so LSQ would silently degenerate to weight-only tuning while paying
  TQT-level PC time. The harness in [scripts/apply_setting.py](scripts/apply_setting.py)
  detects this conflict and disables LSQ with a clear warning. Use TQT instead.

See [references/ppq_methods.md §11 (compatibility matrix)](references/ppq_methods.md) and
§12 (target-policy compatibility) for the full method-vs-method and method-vs-target
compatibility rules.

#### On-device runtime cost cheat-sheet

The state machine reorders Phase-3 levers based on `QUANT_CONFIG["deploy_runtime_priority"]`
(default `"balanced"`; `"speed"` defers all `+`-cost levers behind every zero-cost lever;
`"pc_time"` is reserved for future use). The cost column on each lever:

| Pass / Lever | On-device cost | Why |
|--------------|:--------------:|-----|
| `calib_algorithm` (kl/mse/percentile/minmax/isotone) | **0** | Only changes scale values, not the runtime graph. |
| `tqt_optimization` (any 3a-1/3a-2/3a-3 schedule) | **0** | Trains `log2_scale` offline; output is plain POWER_OF_2 scales. |
| `bias_correct` | **0** | Adjusts the bias tensor in-place; same op count, same memory. |
| `fusion_alignment` | **0** | Forces input scales to align — emits identical kernels. |
| `equalization` | **0** | Folds a diagonal scale into adjacent weights; runtime graph unchanged. |
| `blockwise_reconstruction` | **0** | Same as TQT: output is just adjusted weights/scales. PC-time is huge though. |
| `lsq_optimization` | **0** if it ran | Auto-disabled on POWER_OF_2; not a real option here. |
| `dispatching_table` (→ int16 promotion) | **+** | Each promoted op pays ~2× cycles + ~2× activation memory permanently. |
| `weight_split` | **+** | Inserts an extra `Add` op per split layer (and the underlying Conv loses its bias-fusion). |
| Mixed-precision via per-op rules in `dispatching_table` (FP fallback) | **++** | Any op that can't run native int8/16 falls back to esp-dl FP path; multi-x latency hit. |

The reordering logic: under `deploy_runtime_priority="speed"`, lever 3g (blockwise
reconstruction, zero on-device cost but biggest PC-time) is moved ahead of 3e/3f (the two
`+`-cost levers); under `"balanced"`, 3e/3f come before 3g because their PC-time is
shorter and their gains are more typical. The full ordering tables live in
[references/decision_playbook.md §"On-device cost reordering"](references/decision_playbook.md).

### Use `evaluate_fast()` during iteration if available

If the contract exposes `evaluate_fast()`, the harness automatically calls it during
iteration. Run the full `evaluate()` only once at the end on the chosen best iteration.
Encourage the user to provide both if their full eval is slow (>10 min).

### Names in `dispatching_table` must match simplified ONNX

esp-ppq simplifies the ONNX graph before quantization. Op names in the simplified graph
sometimes differ from the original. The harness saves `outputs/iter_<N>/simplified_ops.json`
listing every op name in the simplified graph; cross-check candidate names against that
file before adding them to `dispatching_table`. If the name is missing, the dispatch will
silently no-op.

---

## Common pitfalls

- **Overwriting the user's ONNX**: `espdl_quantize_onnx` saves the simplified ONNX back to
  the input path. The harness works around this by copying the user's ONNX to a temporary
  file in `outputs/iter_<N>/_input.onnx` before calling the API. Don't bypass this.
- **Stale calibration**: the dataloader is rebuilt every iteration — calibration is fast
  enough that caching across iterations isn't worth the bug surface.
- **Setting `interested_layers=None` vs `[]`**: in esp-ppq `None` and `[]` both mean "all
  layers" for some passes but only one is accepted by others. The harness normalises this
  via [scripts/apply_setting.py](scripts/apply_setting.py) — agents writing JSON should use
  `null` or omit the field; never use empty arrays for `interested_layers`.
- **TQT/blockwise on CPU**: these passes are 10-100x faster on GPU. If `device=cpu` and the
  user asks for TQT/blockwise, warn them the iteration will take a long time before submitting.
- **LSQ on POWER_OF_2 targets**: the harness will skip LSQ and warn — don't re-enable it
  in the next iteration. The accuracy you wanted from LSQ comes from TQT here.
- **Skipping the Phase 2 cartesian product**: this is the #1 search failure on esp-dl
  targets. Calib-only sweep does not work — see Composition discipline #4 and the Worked
  example. Always run all three legs (or short-circuit on `target_metric`).
- **Two changes per iteration**: tempting when one of the changes "looks safe", but it
  produces confounded experiments. The classic failure mode: enabling `percentile` and
  `bias_correct` together, then blaming percentile when the iteration regresses (see the
  Worked example). One knob at a time.
- **Abbreviating Phase-3 lever templates** (especially equalization). Writing only
  `{"equalization": {"enabled": true}}` is **not** equivalent to enabling lever 3d —
  the resolved `opt_level` falls back to esp-ppq's default `1`, which does not cross
  Add/Sub branches and therefore silently skips every residual / inverted-residual
  chain (ResNet bottlenecks, MobileNet-V2/V3 inverted residuals, etc.) — exactly the
  layers lever 3d was designed to reach. The skill's lever-3d template recommends
  `opt_level=2`. Always copy `comparison.json["next_step_hint"]["advice"]`'s
  Template change snippet **verbatim** and only fill in `rationale`; the harness now
  emits a warning when `opt_level` is unset on an enabled equalization pass, and the
  warning is your signal that the iteration didn't actually test what its rationale
  claims.
- **Python environment drift**: since the skill runs in the current Python interpreter (no
  Docker isolation), make sure the user's `esp_ppq` install matches the esp-dl tag they
  plan to deploy with.
- **Skipping `--finalize` when the user gave an iteration budget**: the user says
  "iterate 3 times" / "迭代 3 轮", agent runs iter_0/1/2, sees `comparison.json` still
  pointing at `phase-2-calib-tqt-sweep` ("run mse next"), and accidentally runs a 4th
  iteration. **User-budget stop > state-machine stop.** As soon as the user-budgeted
  iteration count is reached, run the command in
  `comparison.json["early_finalize_command"]` (or copy from the "Tip: how to wrap up
  at any time" block in stdout) so `outputs/best/` and `outputs/final_report.md` are
  produced regardless of phase.
- **Confusing on-device-cost with pc-time-cost**: the new `affects inference speed`
  column refers strictly to **deployment-time** cost (does the quantized model run
  slower on the ESP target?). Long PC quantization time (e.g. blockwise reconstruction)
  does **not** show up here — see the "On-device runtime cost cheat-sheet" for the full
  per-pass mapping. When picking between `rank=1` (best accuracy) and `rank=2`
  (runner-up), the only reason to defer to runner-up is if rank=1's `affects inference
  speed` is `Yes` and the accuracy delta is small enough to live with.

---

## Worked example: MobileNet-V2 on ESP32-P4

> **This phenomenon is not specific to MobileNet and not specific to esp32p4.** Any
> "calibration tightly coupled with a training-based pass" scenario in esp-dl quantization
> can exhibit it: a calibration regresses standalone but rebounds to the strongest base
> once paired with TQT(default). The numbers below use MobileNet-V2 on ESP32-P4 as a
> concrete demonstration only.

Reproducer artifacts (from a real run): `example_quantize_mobilenetv2_esp32p4/outputs/`,
particularly `outputs/comparison.json` and `outputs/iter_10/setting.json`.

**Key empirical observations**

- iter_0 baseline (kl, no extras): top1 = **71.15%**.
- iter_2 (kl + TQT(default)): top1 = 71.15% — TQT on kl produced **no improvement**.
- iter_6 (kl + TQT *aggressive* lr=1e-4 steps=2000): top1 = 71.30% — the best the old
  skill version could find, after 6 rounds of escalation.
- iter_10 (**percentile + TQT(default)**): top1 = **71.475%** — the highest result, found
  only when the user manually forced this combination.

**Standalone percentile would have looked bad.** A clean `percentile-only` (no TQT)
iteration would have regressed below the kl baseline (heavy-tailed activations + tighter
scale = legitimate signal clipped). The old skill version inferred from a confounded
`iter_1 = percentile + bias_correct` regression that "percentile is bad", then spent
iter_2..9 on kl-only and never re-tested percentile. Even a Phase-2-style
*calib-only sweep* (mse alone, percentile alone) would have repeated the verdict and
condemned percentile a second time.

**Why percentile + TQT wins anyway.** Percentile clips the heavy tail at p=0.9999, giving
TQT a tighter, more uniform scale grid to optimise on. TQT then recovers the clipped
information through `log2_scale` adjustment per layer. kl-fitted scales already
"compromise" with the tail, leaving less room for TQT to move; mse sits in between.

**Lesson encoded into Phase 2.** The skill now mandates `calib × TQT(default)` cartesian
product — three iterations: `kl + TQT(default)`, `mse + TQT(default)`, `percentile +
TQT(default)`. No calibration is judged by its standalone score; the combined
top1/top5/whatever-metric is the ranking signal. The Composition discipline #4 codifies
this rule.

---

## Reference index

- [references/ppq_methods.md](references/ppq_methods.md) — every esp-ppq method, principle,
  parameters, when to use, what to avoid, plus the target-policy compatibility matrix.
- [references/decision_playbook.md](references/decision_playbook.md) — distribution pattern
  to candidate method mapping, plus the Multi-iteration strategy table aligned with the
  Phases above.
- [references/contract.md](references/contract.md) — what the user's `user_quant.py` must
  expose.
- [references/setting_json_schema.md](references/setting_json_schema.md) — JSON schema the
  agent writes each iteration, with Phase 2 / Phase 3 templates.
- [scripts/run_iteration.py](scripts/run_iteration.py) — the harness; one quantize +
  analyse + evaluate pass.
- [scripts/apply_setting.py](scripts/apply_setting.py) — pure JSON → `QuantizationSetting`
  mapping; performs target-policy compatibility checks (LSQ on POWER_OF_2 → auto-disable;
  esp32p4 equalization → warn-only).
- [scripts/compare_iterations.py](scripts/compare_iterations.py) — cross-iteration diff +
  next-step state machine driving the Phase 1 → 2 → 3 procedure. Auto-finalises
  on `phase-4` (writes `outputs/best/` + `outputs/final_report.md`); `--finalize`
  forces finalize at any time (escape hatch when the user stops early at a fixed
  iteration budget); `--force` overrides the marker-based preserve check on
  hand-edited reports.
- [assets/user_quant_torch_example.py](assets/user_quant_torch_example.py) — copy/edit for
  torch model flows.
- [assets/user_quant_onnx_example.py](assets/user_quant_onnx_example.py) — copy/edit for
  ONNX model flows.
- [assets/extra_requirements.txt](assets/extra_requirements.txt) — extra Python packages
  the harness needs.
