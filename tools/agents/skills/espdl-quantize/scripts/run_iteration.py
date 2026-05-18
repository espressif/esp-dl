"""One quantize + analyse + evaluate pass for the espdl-quantize skill.

Reads a user contract module (``user_quant.py``) and (optionally) a setting JSON, runs:

1. Pre-flight contract validation
2. ``espdl_quantize_torch`` or ``espdl_quantize_onnx`` with the iteration's setting
3. ``layerwise_error_analyse`` to score each computing op
4. ``statistical_analyse`` on the top-K worst layers (per-tensor distributions)
5. ``user_quant.evaluate`` (or ``evaluate_fast``) to score the quantized graph

All outputs land in ``--output-dir`` as JSON the agent can read.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# The skill's scripts directory must be on sys.path so apply_setting can import.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import apply_setting  # noqa: E402
import analysis_helpers  # noqa: E402

from esp_ppq.api import espdl_quantize_onnx, espdl_quantize_torch  # noqa: E402
from esp_ppq.IR import BaseGraph  # noqa: E402
from esp_ppq.quantization.analyse import (  # noqa: E402
    graphwise_error_analyse,
    layerwise_error_analyse,
    statistical_analyse,
)


REQUIRED_CONFIG_KEYS = {
    "model_type",
    "input_shape",
    "batch_size",
    "target",
    "num_of_bits",
    "device",
    "calib_steps",
    "primary_metric",
    "metric_direction",
}

# Defaults for the D2/D3 analysis aggregations the harness builds on top of
# esp-ppq's analysers. Both are overridable via QUANT_CONFIG; the values below
# reflect empirical SNR distributions on MobileNet-V2 / YOLO outputs.
_DEFAULT_NON_COMPUTING_TOP_K = 10
_DEFAULT_GRAPHWISE_INTERVENING_EXCESS = 0.02
_VALID_RUNTIME_PRIORITIES = {"balanced", "speed", "pc_time"}

DEFAULT_CONFIG_VALUES = {
    "analyse_steps": 8,
    "top_k_layers": 20,
    "non_computing_top_k": _DEFAULT_NON_COMPUTING_TOP_K,
    "graphwise_intervening_excess_threshold": _DEFAULT_GRAPHWISE_INTERVENING_EXCESS,
    "deploy_runtime_priority": "balanced",
}
SUPPORTED_MODEL_TYPES = {"onnx", "torch"}
_GLOBAL_SEED = 42


# ---------------------------------------------------------------------------
# Contract loading & validation
# ---------------------------------------------------------------------------


def _load_user_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"user_quant.py not found at {path}")
    spec = importlib.util.spec_from_file_location("user_quant_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    # Make the user's module dir importable so it can do relative imports.
    user_dir = str(path.parent.resolve())
    if user_dir not in sys.path:
        sys.path.insert(0, user_dir)
    spec.loader.exec_module(module)
    return module


def validate_contract(module) -> Dict[str, Any]:
    if not hasattr(module, "QUANT_CONFIG"):
        raise ValueError("user_quant.py must define QUANT_CONFIG dict")
    config = dict(module.QUANT_CONFIG)
    missing = REQUIRED_CONFIG_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"QUANT_CONFIG missing required keys: {sorted(missing)}")

    if config["model_type"] not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {SUPPORTED_MODEL_TYPES}, "
            f"got {config['model_type']!r}"
        )
    if config["model_type"] == "onnx" and not config.get("onnx_path"):
        raise ValueError("model_type='onnx' requires QUANT_CONFIG['onnx_path']")
    if config["metric_direction"] not in ("max", "min"):
        raise ValueError("metric_direction must be 'max' or 'min'")
    if not isinstance(config["input_shape"], list) or not all(
        isinstance(d, int) for d in config["input_shape"]
    ):
        raise ValueError("input_shape must be a list[int]")
    if config["num_of_bits"] not in (8, 16):
        raise ValueError("num_of_bits must be 8 or 16")

    for key, default in DEFAULT_CONFIG_VALUES.items():
        config.setdefault(key, default)

    if config["deploy_runtime_priority"] not in _VALID_RUNTIME_PRIORITIES:
        raise ValueError(
            "deploy_runtime_priority must be one of "
            f"{sorted(_VALID_RUNTIME_PRIORITIES)}, got "
            f"{config['deploy_runtime_priority']!r}"
        )

    if not callable(getattr(module, "create_calib_dataloader", None)):
        raise ValueError("user_quant.py must define create_calib_dataloader()")
    if not callable(getattr(module, "evaluate", None)):
        raise ValueError("user_quant.py must define evaluate(quant_graph)")

    if config["model_type"] == "torch":
        if not callable(getattr(module, "get_torch_model", None)):
            raise ValueError(
                "model_type='torch' requires get_torch_model() in user_quant.py"
            )

    return config


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return _to_jsonable(obj.tolist())
        except Exception:
            return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return str(obj)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)


def _set_deterministic_seed(seed: int) -> None:
    """Force deterministic RNG behavior for reproducible iterations."""
    random.seed(seed)
    try:
        import numpy as np  # Local import: numpy is optional in some environments.

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Backward compatibility for older torch versions without warn_only.
        torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# ONNX op name extraction (post-simplify)
# ---------------------------------------------------------------------------


def _simplify_onnx(input_path: Path, output_path: Path) -> Path:
    """Simplify input ONNX into output_path; falls back to a plain copy if onnxsim fails."""
    import onnx
    from onnxsim import simplify

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(input_path))
    try:
        sim_model, ok = simplify(model)
        if not ok:
            raise RuntimeError("onnxsim returned check=False")
        sim_model = onnx.shape_inference.infer_shapes(sim_model)
        onnx.save(sim_model, str(output_path))
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] onnxsim failed ({exc}); using original ONNX copy.", file=sys.stderr
        )
        shutil.copyfile(str(input_path), str(output_path))
    return output_path


def _dump_simplified_ops(onnx_path: Path, output_path: Path) -> List[str]:
    import onnx

    model = onnx.load(str(onnx_path))
    ops = []
    for node in model.graph.node:
        ops.append(
            {
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
        )
    _dump_json(output_path, {"count": len(ops), "ops": ops})
    return [op["name"] for op in ops if op["name"]]


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def _build_kwargs(
    user_module,
    config: dict,
    setting,
    espdl_path: Path,
    onnx_input_path: Optional[Path],
) -> dict:
    kwargs: Dict[str, Any] = dict(
        espdl_export_file=str(espdl_path),
        calib_dataloader=user_module.create_calib_dataloader(),
        calib_steps=int(config["calib_steps"]),
        input_shape=[1] + list(config["input_shape"]),
        target=config["target"],
        num_of_bits=int(config["num_of_bits"]),
        setting=setting,
        device=config["device"],
        # The script runs layerwise_error_analyse and graphwise_error_analyse
        # itself below (and persists the JSON). Letting esp-ppq run them again
        # internally would double the analysis cost for results we throw away.
        error_report=False,
        skip_export=False,
        export_test_values=False,
        verbose=1,
    )
    if hasattr(user_module, "collate_fn") and callable(user_module.collate_fn):
        kwargs["collate_fn"] = user_module.collate_fn

    if config["model_type"] == "onnx":
        kwargs["onnx_import_file"] = str(onnx_input_path)
    else:
        kwargs["model"] = user_module.get_torch_model()
    return kwargs


def _run_quantization(
    user_module, config: dict, setting, output_dir: Path
) -> Tuple[BaseGraph, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    espdl_path = output_dir / "model.espdl"

    onnx_input: Optional[Path] = None
    if config["model_type"] == "onnx":
        original_onnx = Path(config["onnx_path"])
        if not original_onnx.is_absolute():
            # Resolve relative to user module's directory so users can use relative paths.
            original_onnx = (
                Path(config.get("_user_dir", ".")) / original_onnx
            ).resolve()
        if not original_onnx.exists():
            raise FileNotFoundError(f"onnx_path does not exist: {original_onnx}")
        # Copy first so esp-ppq's in-place simplify never touches the user's file.
        onnx_input = output_dir / "_input.onnx"
        shutil.copyfile(str(original_onnx), str(onnx_input))

    kwargs = _build_kwargs(user_module, config, setting, espdl_path, onnx_input)

    if config["model_type"] == "onnx":
        graph = espdl_quantize_onnx(**kwargs)
    else:
        graph = espdl_quantize_torch(**kwargs)
    return graph, espdl_path


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _resolve_collate(user_module, device):
    if hasattr(user_module, "collate_fn") and callable(user_module.collate_fn):
        return user_module.collate_fn

    def _default(batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch

    return _default


def _layerwise_errors(
    graph: BaseGraph, dataloader, collate_fn, device, steps: int
) -> Dict[str, float]:
    raw = layerwise_error_analyse(
        graph=graph,
        dataloader=dataloader,
        collate_fn=collate_fn,
        running_device=device,
        method="snr",
        steps=steps,
        verbose=False,
    )
    if raw is None:
        return {}
    sorted_pairs = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
    return {name: float(score) for name, score in sorted_pairs}


def _statistical_analyse_records(
    graph: BaseGraph,
    dataloader,
    collate_fn,
    device,
    steps: int,
) -> List[dict]:
    """Run esp-ppq's statistical_analyse once and return ALL per-variable records.

    statistical_analyse covers `QuantableOperation` minus `PASSIVE_OPERATIONS` —
    i.e. it is the only esp-ppq analyser that emits stats for non-COMPUTING_OP
    layers like Add / Concat / Resize / AveragePool / Sigmoid / Softmax / GRU /
    LayerNorm. The harness saves the full list (D1) and downstream helpers
    derive a filtered top-K view (compatible with old playbook usage) and a
    non-COMPUTING_OP shortlist (D2) from it.
    """
    records = statistical_analyse(
        graph=graph,
        running_device=device,
        dataloader=dataloader,
        collate_fn=collate_fn,
        steps=steps,
    )
    return records or []


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _tee_stdout(log_path: Path):
    """Mirror stdout/stderr to a log file while still printing to terminal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")

    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
            return len(data)

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, log_handle)
    sys.stderr = _Tee(old_err, log_handle)
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_handle.close()


def _run_evaluate(user_module, graph, fast_first: bool) -> Tuple[Dict[str, Any], str]:
    """Returns (metrics_dict, which_eval_was_used)."""
    if (
        fast_first
        and hasattr(user_module, "evaluate_fast")
        and callable(user_module.evaluate_fast)
    ):
        try:
            return dict(user_module.evaluate_fast(graph)), "evaluate_fast"
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] evaluate_fast raised {type(exc).__name__}: {exc}; falling back to evaluate."
            )
    return dict(user_module.evaluate(graph)), "evaluate"


def _load_setting_payload(args) -> dict:
    if args.baseline:
        return apply_setting.baseline_payload()
    if not args.setting:
        raise ValueError("--setting <path> is required unless --baseline is given")
    setting_path = Path(args.setting)
    if not setting_path.exists():
        raise FileNotFoundError(f"--setting file not found: {setting_path}")
    with setting_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a single espdl-quantize iteration."
    )
    parser.add_argument("--user-quant", required=True, help="Path to user_quant.py")
    parser.add_argument(
        "--setting", default=None, help="Path to iteration setting JSON"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use built-in baseline payload (default espdl_setting()); ignores --setting.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Where to write iteration artifacts"
    )
    parser.add_argument(
        "--check-contract",
        action="store_true",
        help="Validate user_quant.py only; do not run quantization.",
    )
    parser.add_argument(
        "--use-full-eval",
        action="store_true",
        help="Skip evaluate_fast and call evaluate() directly (typically for the final report).",
    )
    args = parser.parse_args(argv)

    user_path = Path(args.user_quant).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_deterministic_seed(_GLOBAL_SEED)
    print(f"[run_iteration] deterministic seed fixed to {_GLOBAL_SEED}")

    user_module = _load_user_module(user_path)
    config = validate_contract(user_module)
    config["_user_dir"] = str(user_path.parent)

    if args.check_contract:
        print(
            json.dumps(
                {"ok": True, "config": _to_jsonable(config)},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    payload = _load_setting_payload(args)

    log_path = output_dir / "console.log"
    with _tee_stdout(log_path):
        t0 = time.time()
        print(f"[run_iteration] user_quant={user_path}")
        print(f"[run_iteration] output_dir={output_dir}")
        print(
            f"[run_iteration] iteration_id={payload.get('iteration_id')}  "
            f"baseline={args.baseline}"
        )
        print(f"[run_iteration] rationale={payload.get('rationale')}")

        # Dump simplified ops first, so dispatching_table validation has something to use.
        simplified_op_names: Optional[List[str]] = None
        if config["model_type"] == "onnx":
            original_onnx = Path(config["onnx_path"])
            if not original_onnx.is_absolute():
                original_onnx = (Path(config["_user_dir"]) / original_onnx).resolve()
            sim_path = output_dir / "_simplified.onnx"
            _simplify_onnx(original_onnx, sim_path)
            simplified_op_names = _dump_simplified_ops(
                sim_path, output_dir / "simplified_ops.json"
            )
            print(f"[run_iteration] simplified_ops={len(simplified_op_names)}")

        # Translate JSON setting -> QuantizationSetting.
        apply_result = apply_setting.apply(
            payload=payload,
            target=config["target"],
            known_op_names=set(simplified_op_names) if simplified_op_names else None,
        )
        for w in apply_result.warnings:
            print(f"[apply_setting][warn] {w}")
        _dump_json(output_dir / "setting.json", payload)
        _dump_json(output_dir / "applied_setting_summary.json", apply_result.summary)

        # Run quantization.
        print("[run_iteration] starting quantization")
        graph, espdl_path = _run_quantization(
            user_module, config, apply_result.setting, output_dir
        )
        print(f"[run_iteration] quantized model exported to {espdl_path}")

        # If we didn't dump simplified_ops earlier (torch flow), do it now from the espdl-side onnx.
        if simplified_op_names is None:
            torch_onnx = espdl_path.with_suffix(".onnx")
            if torch_onnx.exists():
                simplified_op_names = _dump_simplified_ops(
                    torch_onnx, output_dir / "simplified_ops.json"
                )

        device = config["device"]
        analyse_steps = int(config.get("analyse_steps", 8))
        top_k = int(config.get("top_k_layers", 20))

        # Layerwise error.
        print("[run_iteration] layerwise_error_analyse")
        layerwise = _layerwise_errors(
            graph=graph,
            dataloader=user_module.create_calib_dataloader(),
            collate_fn=_resolve_collate(user_module, device),
            device=device,
            steps=analyse_steps,
        )
        _dump_json(output_dir / "layerwise_error.json", layerwise)
        if layerwise:
            top_layers = list(layerwise.keys())[:top_k]
            preview = list(layerwise.items())[:5]
            print("[run_iteration] top-5 error layers:")
            for name, snr in preview:
                print(f"   {snr:.6f}  {name}")
        else:
            top_layers = []

        # Per-tensor distribution stats — run statistical_analyse once, then
        # produce two views: the legacy filtered top-K (`layer_stats.json`) and
        # the full record (`layer_stats_full.json`, D1) which is the only
        # source for non-COMPUTING_OP signal in the rest of the pipeline.
        print("[run_iteration] statistical_analyse on all non-passive ops")
        all_stat_records = _statistical_analyse_records(
            graph=graph,
            dataloader=user_module.create_calib_dataloader(),
            collate_fn=_resolve_collate(user_module, device),
            device=device,
            steps=analyse_steps,
        )
        _dump_json(output_dir / "layer_stats_full.json", all_stat_records)
        filtered_stats = analysis_helpers.filter_layer_stats_for(
            all_stat_records, top_layers
        )
        _dump_json(output_dir / "layer_stats.json", filtered_stats)

        # D2 — non-COMPUTING_OP shortlist by max per-variable SNR.
        non_computing_top_k = int(
            config.get("non_computing_top_k", _DEFAULT_NON_COMPUTING_TOP_K)
        )
        non_computing_hot = analysis_helpers.aggregate_non_computing_hot_ops(
            all_stat_records, non_computing_top_k
        )
        _dump_json(output_dir / "non_computing_hot_ops.json", non_computing_hot)
        if non_computing_hot:
            preview = non_computing_hot[:5]
            print("[run_iteration] non-COMPUTING_OP top-5 (max per-var SNR):")
            for entry in preview:
                ratio = entry.get("inputs_float_std_ratio")
                ratio_str = (
                    f"  std-ratio={ratio:.2f}"
                    if isinstance(ratio, (int, float))
                    else ""
                )
                print(
                    f"   snr={entry['max_snr']:.6f}  {entry['op_type']:<14}  {entry['op_name']}{ratio_str}"
                )

        # Graphwise error (cumulative). Captured for D3 differential analysis.
        gw: Dict[str, float] = {}
        try:
            print("[run_iteration] graphwise_error_analyse")
            gw_raw = graphwise_error_analyse(
                graph=graph,
                running_device=device,
                dataloader=user_module.create_calib_dataloader(),
                collate_fn=_resolve_collate(user_module, device),
                method="snr",
                steps=analyse_steps,
                verbose=False,
            )
            gw = {
                k: float(v)
                for k, v in (gw_raw or {}).items()
                if isinstance(v, (int, float))
            }
            _dump_json(output_dir / "graphwise_error.json", gw)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] graphwise_error_analyse failed ({exc}); skipping.")
            _dump_json(output_dir / "graphwise_error.json", {})

        # D3 — graphwise vs layerwise differential. Surfaces pairs of adjacent
        # computing ops whose accumulated SNR gap is bigger than the downstream
        # op's isolated layerwise contribution; the intervening non-computing
        # ops in that region are likely the actual culprits.
        graphwise_jumps: List[dict] = []
        try:
            excess_threshold = float(
                config.get(
                    "graphwise_intervening_excess_threshold",
                    _DEFAULT_GRAPHWISE_INTERVENING_EXCESS,
                )
            )
            simplified_payload: Optional[dict] = None
            sim_ops_path = output_dir / "simplified_ops.json"
            if sim_ops_path.exists():
                with sim_ops_path.open("r", encoding="utf-8") as f:
                    simplified_payload = json.load(f)
            graphwise_jumps = analysis_helpers.compute_graphwise_jumps(
                layerwise=layerwise,
                graphwise=gw,
                simplified_ops_payload=simplified_payload,
                excess_threshold=excess_threshold,
            )
            _dump_json(output_dir / "graphwise_jumps.json", graphwise_jumps)
            if graphwise_jumps:
                print(
                    f"[run_iteration] graphwise_jumps: {len(graphwise_jumps)} suspicious "
                    f"intervening region(s) (excess > {excess_threshold:.3f})"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] graphwise_jumps computation failed ({exc}); skipping.")
            _dump_json(output_dir / "graphwise_jumps.json", [])

        # Evaluate.
        print("[run_iteration] evaluate")
        try:
            metrics, eval_used = _run_evaluate(
                user_module, graph, fast_first=not args.use_full_eval
            )
        except Exception as exc:
            traceback.print_exc()
            metrics = {"_error": f"{type(exc).__name__}: {exc}"}
            eval_used = "failed"

        primary_key = config["primary_metric"]
        primary_value = metrics.get(primary_key)
        out_metrics = {
            "_used": eval_used,
            "_primary_key": primary_key,
            "_primary_value": primary_value,
            "metric_direction": config["metric_direction"],
            **metrics,
        }
        _dump_json(output_dir / "metrics.json", out_metrics)

        # Index file the agent reads first. Newly added fields support the
        # state-machine in compare_iterations.py: target_metric drives the
        # phase-4 finalize trigger; deploy_runtime_priority drives the
        # speed/balanced/pc_time lever ordering; non_computing_*_count and
        # graphwise_jumps_count let the agent decide whether D-side artifacts
        # are worth opening for this iteration without re-reading the JSONs.
        # Persist the model identifier and target chip so compare_iterations.py
        # can render `final_report.md` headers without re-importing user_quant.py.
        # Order of preference: explicit QUANT_CONFIG["model_name"] -> derived
        # from onnx_path stem -> fallback "<model>".
        model_name = config.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            onnx_p = config.get("onnx_path") or ""
            if isinstance(onnx_p, str) and onnx_p:
                model_name = Path(onnx_p).stem
            else:
                model_name = "<model>"

        index = {
            "iteration_id": payload.get("iteration_id"),
            "rationale": payload.get("rationale"),
            "model_path": str(espdl_path),
            "model_name": model_name,
            "target_chip": config.get("target", "<target>"),
            "elapsed_seconds": round(time.time() - t0, 2),
            "primary_metric": primary_key,
            "primary_value": primary_value,
            "metric_direction": config["metric_direction"],
            "target_metric": config.get("target_metric"),
            "deploy_runtime_priority": config.get(
                "deploy_runtime_priority", "balanced"
            ),
            "non_computing_top_k": non_computing_top_k,
            "non_computing_hot_ops_count": len(non_computing_hot),
            "graphwise_jumps_count": len(graphwise_jumps),
            "top_5_error_layers": [
                {"op": name, "snr": score}
                for name, score in list(layerwise.items())[:5]
            ],
            "applied_setting_warnings": apply_result.warnings,
        }
        _dump_json(output_dir / "iteration_index.json", index)

        print(
            f"[run_iteration] done in {index['elapsed_seconds']:.2f}s. "
            f"{primary_key}={primary_value!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
