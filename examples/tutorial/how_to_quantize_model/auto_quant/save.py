"""Run-directory layout + experiment artifact bookkeeping.

Layout:
    outputs/run/                    # default; pick a different name for resume
        summary.json                # list of finished experiment records
        candidates.json             # Top-K, written by main, augmented by latency
        topk_report.md              # human-readable Top-K report (after latency)
        0000/
            config.json
            model.espdl
            ...
"""

import json
import os
import shutil
from typing import Any, Dict, List


def create_run_dir(base: str = "outputs", name: str = "run") -> str:
    """Create / reuse `<base>/<name>/`. Does NOT delete contents — that's the
    caller's job (e.g. main.py decides whether to reset based on --resume).
    """
    run_dir = os.path.join(base, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def reset_run_dir(run_dir: str) -> None:
    """Truncate summary + candidates for a fresh run."""
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump([], f)
    with open(os.path.join(run_dir, "candidates.json"), "w") as f:
        json.dump([], f)


# --- JSON safety ---


def _make_json_safe(obj: Any) -> Any:
    """Convert object into JSON-serializable format.
    Skips `calib_dataloader` (DataLoader is not JSON-friendly) and falls back
    to str() for unknown types.
    """
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {
            k: _make_json_safe(v) for k, v in obj.items() if k != "calib_dataloader"
        }
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    return str(obj)


# --- Summary (read-modify-write JSON list) ---


def update_summary(run_dir: str, record: Dict[str, Any]) -> None:
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = []
    summary.append(_make_json_safe(record))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)


def load_summary(run_dir: str) -> List[Dict[str, Any]]:
    """Read summary.json back into a list. Used by --resume."""
    path = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return json.load(f)


# --- Per-experiment artifact saving ---


def save_experiment(
    run_dir: str,
    index: int,
    strategy: Dict[str, dict],
    sampled_param: Dict[str, dict],
    export_path: str,
    runtime_config: dict,
) -> dict:
    """Move artifacts produced by `quant_model` into `run_dir/<index>/` and
    write a `config.json` snapshot. Returns a half-filled record dict
    (accuracy metrics and latency_ms are added by the caller).
    """
    folder = os.path.join(run_dir, f"{index:04d}")
    os.makedirs(folder, exist_ok=True)

    safe_runtime = _make_json_safe(runtime_config)
    config = {
        "strategy": {k: v["value"] for k, v in strategy.items()},
        "params": _make_json_safe(sampled_param),
        "runtime": safe_runtime,
    }
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    dir_name = os.path.dirname(export_path)
    file_name = os.path.basename(export_path)
    base_name = os.path.splitext(file_name)[0]
    base_path = os.path.join(dir_name, base_name) if dir_name else base_name

    suffixes = [".espdl", ".info", ".json", ".native"]
    moved_files = []
    for s in suffixes:
        src = base_path + s
        if not os.path.isfile(src):
            continue
        dst = os.path.join(folder, os.path.basename(src))
        shutil.move(src, dst)
        moved_files.append(os.path.basename(src))

    return {
        "index": index,
        "folder": folder,
        "files": moved_files,
        "strategy": config["strategy"],
        "params": config["params"],
    }
