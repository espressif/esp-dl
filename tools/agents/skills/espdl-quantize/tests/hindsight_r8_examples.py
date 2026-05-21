"""Hindsight verification: the new R8 rule must agree with historical outcomes.

This module walks every ``example_quantize_*/outputs/`` directory under the
repo root and asks the question: *at the moment 3c (fusion_alignment) was
about to be tried in this project, did our new R8 trigger classification
agree with what actually happened?*

The check is cheap — it reads the on-disk ``non_computing_hot_ops.json``
from the iteration that was best-so-far at the time, runs
``compare_iterations._r8_trigger_fires``, then compares against the empirical
delta (3c iteration's primary_value vs prior-best's primary_value). No
re-quantization needed.

Run as part of pytest::

    python -m pytest .cursor/skills/espdl-quantize/tests/hindsight_r8_examples.py -v

Or standalone, to get one-line per-example summaries::

    python .cursor/skills/espdl-quantize/tests/hindsight_r8_examples.py

This script is the regression guard for any future change to the R8 module
constants (``_R8_MAX_SNR_PRIMARY`` / ``_R8_STD_RATIO_REINFORCE`` /
``_R8_ACTIVATION_VETO_RATIO``). If you re-tune them, re-run this script and
ensure all known cases still pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

import compare_iterations as ci  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _iter_dirs(outputs_dir: Path) -> List[Tuple[int, Path]]:
    """Return ``[(iteration_id, iter_dir), ...]`` sorted by iteration_id."""
    items: List[Tuple[int, Path]] = []
    if not outputs_dir.is_dir():
        return items
    for child in outputs_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        try:
            idx = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        items.append((idx, child))
    items.sort(key=lambda t: t[0])
    return items


def _read_json(path: Path) -> Optional[dict | list]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _primary_value_of(iter_dir: Path) -> Optional[float]:
    """Read the iteration's primary metric value from iteration_index.json,
    falling back to metrics.json. Returns None if unavailable or non-finite."""
    index = _read_json(iter_dir / "iteration_index.json")
    if isinstance(index, dict):
        v = index.get("primary_value")
        if isinstance(v, (int, float)):
            return float(v)
    metrics = _read_json(iter_dir / "metrics.json")
    if isinstance(metrics, dict):
        v = metrics.get("_primary_value")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _metric_direction_of(iter_dir: Path) -> str:
    index = _read_json(iter_dir / "iteration_index.json")
    if isinstance(index, dict):
        d = index.get("metric_direction")
        if d in ("max", "min"):
            return str(d)
    metrics = _read_json(iter_dir / "metrics.json")
    if isinstance(metrics, dict):
        d = metrics.get("metric_direction")
        if d in ("max", "min"):
            return str(d)
    return "max"


def _setting_enables_fusion_alignment(setting: Optional[dict]) -> bool:
    if not isinstance(setting, dict):
        return False
    fa = setting.get("fusion_alignment")
    return bool(fa) and isinstance(fa, dict) and len(fa) > 0


def _find_first_3c_iter(
    iters: List[Tuple[int, Path]],
) -> Optional[Tuple[int, Path]]:
    for idx, iter_dir in iters:
        setting = _read_json(iter_dir / "setting.json")
        if _setting_enables_fusion_alignment(setting):
            return idx, iter_dir
    return None


def _prior_best(
    iters: List[Tuple[int, Path]],
    cutoff_id: int,
    direction: str,
) -> Optional[Tuple[int, Path, float]]:
    """Best (id, dir, primary_value) among iterations with id < cutoff_id."""
    candidates = [(i, p) for i, p in iters if i < cutoff_id]
    scored: List[Tuple[int, Path, float]] = []
    for i, p in candidates:
        v = _primary_value_of(p)
        if v is None:
            continue
        scored.append((i, p, v))
    if not scored:
        return None
    if direction == "min":
        scored.sort(key=lambda t: (t[2], t[0]))
    else:
        scored.sort(key=lambda t: (-t[2], t[0]))
    return scored[0]


def _classify_one_example(outputs_dir: Path) -> Dict[str, object]:
    """Run the hindsight classification on a single example_quantize_*/outputs/.

    Returns a record describing what happened, suitable for both pytest
    assertions and the stdout summary table.
    """
    iters = _iter_dirs(outputs_dir)
    if not iters:
        return {
            "ok": False,
            "skip_reason": "no iter_* directories found",
            "outputs_dir": str(outputs_dir),
        }

    first_3c = _find_first_3c_iter(iters)
    if first_3c is None:
        return {
            "ok": False,
            "skip_reason": "no iteration enabled fusion_alignment (3c never tried)",
            "outputs_dir": str(outputs_dir),
        }
    fc_id, fc_dir = first_3c

    direction = _metric_direction_of(fc_dir)
    prior = _prior_best(iters, fc_id, direction)
    if prior is None:
        return {
            "ok": False,
            "skip_reason": "no prior iteration with a finite primary metric",
            "outputs_dir": str(outputs_dir),
        }
    pb_id, pb_dir, pb_value = prior

    fc_value = _primary_value_of(fc_dir)
    if fc_value is None:
        return {
            "ok": False,
            "skip_reason": "3c iteration metrics.json missing or non-finite",
            "outputs_dir": str(outputs_dir),
        }

    helped = (direction == "max" and fc_value > pb_value) or (
        direction == "min" and fc_value < pb_value
    )

    hot_ops = ci._load_non_computing_hot_ops(pb_dir)
    fires, reason = ci._r8_trigger_fires(hot_ops)

    return {
        "ok": True,
        "outputs_dir": str(outputs_dir),
        "prior_best_iter": pb_id,
        "prior_best_value": pb_value,
        "3c_iter": fc_id,
        "3c_value": fc_value,
        "delta": fc_value - pb_value,
        "direction": direction,
        "helped": helped,
        "r8_fires": fires,
        "r8_reason": reason,
    }


# ---------------------------------------------------------------------------
# pytest entry points — one parametrised test per example_quantize_*/outputs/
# discovered under the repo root. Each project that's missing on disk skips
# instead of failing, so CI without checkouts of the example data still runs.
# ---------------------------------------------------------------------------


def _example_outputs_dirs() -> List[Path]:
    return sorted(
        d / "outputs"
        for d in _REPO_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("example_quantize_")
    )


@pytest.mark.parametrize(
    "outputs_dir",
    _example_outputs_dirs(),
    ids=lambda p: p.parent.name,
)
def test_r8_hindsight_matches_historical_outcome(outputs_dir: Path):
    """The new R8 trigger's verdict on prior-best's non_computing_hot_ops
    must match what actually happened in the 3c iteration on this project.
    If this fails after tuning the module constants, you're regressing on a
    known case — re-examine the thresholds."""
    record = _classify_one_example(outputs_dir)
    if not record.get("ok"):
        pytest.skip(str(record.get("skip_reason", "unknown")))
    assert record["r8_fires"] == record["helped"], (
        f"R8 classification disagrees with history on "
        f"{outputs_dir.parent.name}: prior_best=iter_{record['prior_best_iter']}"
        f"({record['prior_best_value']:.4f}) → 3c=iter_{record['3c_iter']}"
        f"({record['3c_value']:.4f}), delta={record['delta']:+.4f}, "
        f"empirically_helped={record['helped']}, r8_fires={record['r8_fires']}, "
        f"r8_reason={record['r8_reason']!r}"
    )


def main() -> int:
    """Standalone runner — print one summary line per discovered example."""
    examples = _example_outputs_dirs()
    if not examples:
        print("[hindsight] no example_quantize_*/outputs/ directories found")
        return 0
    fail = 0
    for outputs_dir in examples:
        record = _classify_one_example(outputs_dir)
        proj = outputs_dir.parent.name
        if not record.get("ok"):
            print(f"[hindsight] SKIP {proj}: {record.get('skip_reason')}")
            continue
        marker = "OK  " if record["r8_fires"] == record["helped"] else "FAIL"
        if marker == "FAIL":
            fail += 1
        print(
            f"[hindsight] {marker} {proj}: prior_best=iter_{record['prior_best_iter']}"
            f"({record['prior_best_value']:.4f}) -> 3c=iter_{record['3c_iter']}"
            f"({record['3c_value']:.4f}), delta={record['delta']:+.4f}, "
            f"helped={record['helped']}, fires={record['r8_fires']}, "
            f"reason={record['r8_reason']}"
        )
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
