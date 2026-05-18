# User contract specification (`user_quant.py`)

The skill never edits this module. Anything that depends on the user's data layout, model
files, or evaluation logic lives here. The skill loads it dynamically each iteration.

## Required exports

A valid `user_quant.py` must export *all* of the following symbols:

### `QUANT_CONFIG: dict`

Top-level configuration consumed by the harness. All keys are required unless marked
optional.

| Key | Type | Required | Description |
|-----|------|:-:|-------------|
| `model_type` | `"onnx"` or `"torch"` | yes | Picks which esp-ppq entrypoint to use. |
| `onnx_path` | `str` | when `model_type=="onnx"` | Absolute path to the user's ONNX model. The harness copies this to `outputs/iter_<N>/_input.onnx` before quantizing so the original is never overwritten. |
| `input_shape` | `list[int]` | yes | Per-sample shape passed to `espdl_quantize_*` (no batch). E.g. `[3, 224, 224]`. |
| `batch_size` | `int` | yes | Reported, also used as a sanity hint for the harness. |
| `target` | `"esp32p4"` / `"esp32s3"` / `"c"` | yes | Target chip. |
| `num_of_bits` | `8` or `16` | yes | Default precision; mixed precision overrides per-op. |
| `device` | `"cpu"` or `"cuda"` | yes | Where calibration runs. |
| `calib_steps` | `int` | yes | Number of batches to use during calibration. 32 is typical. |
| `primary_metric` | `str` | yes | Key in the dict returned by `evaluate()` to optimise. |
| `metric_direction` | `"max"` or `"min"` | yes | Whether higher is better. |
| `target_metric` | `float` | optional | Stop iterating when this is reached. Propagated into every `iteration_index.json`; the state machine in `compare_iterations.py` checks it against the latest `primary_value` to emit `phase-4-final-report`. |
| `analyse_steps` | `int` | optional, default `8` | Steps used by `layerwise_error_analyse`, `graphwise_error_analyse` and `statistical_analyse`. |
| `top_k_layers` | `int` | optional, default `20` | How many high-error layers to keep in the filtered `layer_stats.json` (the legacy top-K view of `statistical_analyse`). The full `layer_stats_full.json` is unaffected. |
| `non_computing_top_k` | `int` | optional, default `10` | How many non-COMPUTING_OP layers (Concat/Add/Resize/Pool/Sigmoid/Softmax/GRU/LayerNorm/…) to keep in `non_computing_hot_ops.json`, ranked by max per-variable SNR. Increase to surface long-tail candidates on graphs with many side branches (DETR, U-Net). |
| `graphwise_intervening_excess_threshold` | `float` | optional, default `0.02` | Threshold for flagging a region in `graphwise_jumps.json`. An adjacent computing-op pair is flagged when `graphwise[op_next] − graphwise[op_prev] − layerwise[op_next] > threshold`. Lower it (e.g. 0.005) when looking for subtle scale-mismatch culprits between identical-looking residual branches; raise it when the report is noisy. |
| `deploy_runtime_priority` | `"balanced"` / `"speed"` / `"pc_time"` | optional, default `"balanced"` | Tells the state machine how to reorder Phase-3 levers based on the deployment cost model (`+`-cost levers like `dispatching_table` int16 promotion and `weight_split` are deferred under `"speed"`; `"pc_time"` is reserved). The cheat-sheet in `SKILL.md` lists the per-lever cost; the reordering tables live in `references/decision_playbook.md` ("On-device cost reordering"). |

### `create_calib_dataloader() -> torch.utils.data.DataLoader`

Returns the calibration dataloader. Called fresh every iteration. Keep this fast.

```python
def create_calib_dataloader():
    dataset = MyDataset(...)
    return DataLoader(dataset, batch_size=QUANT_CONFIG["batch_size"], shuffle=False,
                      num_workers=4, collate_fn=my_collate)
```

### `evaluate(quant_graph) -> dict`

Runs the user's evaluation suite on the quantized PPQ graph (`esp_ppq.IR.BaseGraph`).
**Must return a dict that contains the key named in `QUANT_CONFIG["primary_metric"]`.**
Other keys (top5, mAP, etc.) are kept for the report but ignored for ranking.

```python
def evaluate(quant_graph) -> dict:
    from esp_ppq.executor import TorchExecutor
    executor = TorchExecutor(graph=quant_graph, device=QUANT_CONFIG["device"])
    # ... run user's eval loop ...
    return {"top1": 60.32, "top5": 83.10}
```

The `quant_graph` argument is the same object returned by `espdl_quantize_torch` /
`espdl_quantize_onnx`. The user can use a `TorchExecutor` to run inference, or pass the
graph to a custom evaluator.

## Optional exports

### `collate_fn(batch)`

Passed straight to `espdl_quantize_*`. If absent, the harness uses the API default
(detached `.to(device)`).

### `get_torch_model() -> torch.nn.Module`

Required only when `QUANT_CONFIG["model_type"] == "torch"`. Returns the model in `eval()`
mode on the configured device. Called fresh each iteration so model state is reproducible.

```python
def get_torch_model():
    import torchvision
    m = torchvision.models.mobilenet.mobilenet_v2(weights="IMAGENET1K_V1")
    m.eval()
    return m.to(QUANT_CONFIG["device"])
```

### `evaluate_fast(quant_graph) -> dict`

A faster, less precise variant of `evaluate()` for use during iteration. Should return the
same key set but be at least 5× faster. The harness uses `evaluate_fast()` if defined,
falling back to `evaluate()`. The final report always re-runs the full `evaluate()` on the
chosen best iteration.

Without `evaluate_fast`, every iteration runs the full eval. If your full eval takes
> 10 minutes, definitely provide one.

## Module-level execution rules

- **Don't run heavy code at module import time.** Defining functions and `QUANT_CONFIG` is
  fine. Building dataloaders, downloading data, loading models inside top-level statements
  will fire on every harness invocation.
- **Side effects are forbidden** — no `print` floods, no global GPU allocations.
- **Relative paths**: the harness resolves any relative path in `QUANT_CONFIG`
  (e.g. `onnx_path`) against the directory containing `user_quant.py`, so relative paths
  inside the contract module continue to work regardless of the user's current working
  directory when invoking `run_iteration.py`. The harness runs in the **current Python
  interpreter** (no Docker isolation), so `user_quant.py` should assume the same Python
  environment used to run the harness.

## Validation checklist

Before running iter-0, the harness validates:

1. `user_quant.py` imports without exception.
2. `QUANT_CONFIG` is a dict with all required keys (and types).
3. `create_calib_dataloader()` returns an iterable yielding tensors of the expected dtype.
4. `evaluate` is callable. (We can't validate it returns the right keys until we have a
   quantized graph — that's checked after iter-0.)
5. If `model_type == "torch"`, `get_torch_model()` returns a `torch.nn.Module`.

Run `scripts/run_iteration.py --check-contract` first to surface validation errors before
spending time on a calibration run. The check runs in the current Python environment;
ensure `esp_ppq`, `torch`, `onnx`, `onnxsim`, `pandas`, `scipy`, `tqdm` are all importable
before invoking the harness.
