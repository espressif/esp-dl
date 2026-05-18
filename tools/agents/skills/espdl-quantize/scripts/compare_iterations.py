"""Cross-iteration comparison.

Reads every ``iter_<N>/iteration_index.json`` under ``--output-dir`` and emits:

* ``outputs/comparison.json`` — structured summary the agent reads. Includes the best
  iteration's directory so the agent can mutate from it (the SKILL.md "mutate from
  best-so-far" rule) and a ``next_step_hint`` field that names the recommended Phase
  for the next iteration based on what's already been tried.
* A printed table to stdout for the user, plus a "Suggested next step" block aligned
  with the SKILL.md phased procedure (Phase 1 baseline -> Phase 2 calibration ×
  TQT(default) cartesian product -> Phase 3 surgical residual fixes from
  best-so-far -> Phase 4 final report).

Used between iterations to decide whether the latest run helped, regressed, or plateaued.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")


# ---------------------------------------------------------------------------
# Constants — kept in one place so the SKILL.md / playbook can quote them.
# ---------------------------------------------------------------------------

# TQT default schedule that defines a Phase-2 leg. See SKILL.md Phase 2.
_TQT_DEFAULT_LR = 1e-5
_TQT_DEFAULT_STEPS = 500
_TQT_DEFAULT_BLOCK_SIZE = 4

# Calibration algorithms swept in Phase 2.
_CALIB_SWEEP_VALUES = ("kl", "mse", "percentile")

# Phase-3 budget (cap on iterations run after Phase-2 sweep). Acts as a safety
# net; the primary stop conditions are target_metric and plateau detection.
_PHASE3_CAP = 5

# Plateau detection: if the most recent N iterations all sit within this
# relative window vs best-so-far, declare plateau and finalize.
_PLATEAU_WINDOW = 2
_PLATEAU_REL_TOL = 0.001  # 0.1%

# 3a-3 unstable-fallback path: max relative regression that still counts as a
# stability signal (rather than a confounded experiment).
_3A3_PATH1_REL_TOL = 0.005  # 0.5%

# Phase-3 lever order, indexed by deploy_runtime_priority. 3a-3 is intentionally
# absent — it is a conditional lever entered via _tqt_3a3_*_trigger, not a
# linear-order slot. 3g vs 3e/3f order flips between balanced and speed.
_PHASE3_ORDER_BALANCED = ("3a-1", "3a-2", "3b", "3c", "3d", "3e", "3f", "3g")
_PHASE3_ORDER_SPEED = ("3a-1", "3a-2", "3b", "3c", "3d", "3g", "3e", "3f")
_PHASE3_LEVER_IDS_ALL = ("3a-1", "3a-2", "3b", "3c", "3d", "3e", "3f", "3g")


# ---------------------------------------------------------------------------
# I/O & gather
# ---------------------------------------------------------------------------


def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _gather(output_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _ITER_DIR_RE.match(child.name)
        if not m:
            continue
        idx = int(m.group(1))
        index = _load(child / "iteration_index.json")
        metrics = _load(child / "metrics.json")
        layerwise = _load(child / "layerwise_error.json")
        setting = _load(child / "setting.json")
        if index is None and metrics is None:
            continue
        rows.append(
            {
                "iteration_id": idx,
                "dir": str(child),
                "primary_metric": (index or {}).get("primary_metric")
                or (metrics or {}).get("_primary_key"),
                "primary_value": (
                    (index or {}).get("primary_value")
                    if index
                    else (metrics or {}).get("_primary_value")
                ),
                "metric_direction": (index or {}).get("metric_direction")
                or (metrics or {}).get("metric_direction", "max"),
                "elapsed_seconds": (index or {}).get("elapsed_seconds"),
                "rationale": (setting or {}).get("rationale"),
                "setting": setting or {},
                "top_5_error_layers": (index or {}).get("top_5_error_layers")
                or _top5_from_layerwise(layerwise),
                "warnings": (index or {}).get("applied_setting_warnings", []),
                "extra_metrics": _strip_internal(metrics or {}),
                # A1 — propagated from QUANT_CONFIG by run_iteration.py so the
                # state machine no longer needs to import user_quant.py.
                "target_metric": (index or {}).get("target_metric"),
                "deploy_runtime_priority": (index or {}).get(
                    "deploy_runtime_priority", "balanced"
                ),
                "non_computing_hot_ops_count": (index or {}).get(
                    "non_computing_hot_ops_count", 0
                ),
                "graphwise_jumps_count": (index or {}).get("graphwise_jumps_count", 0),
                # A2 (this revision) — propagated by run_iteration.py so the
                # final-report renderer can fill in the title and Python-snippet
                # `target` argument without re-importing user_quant.py. Older
                # iteration_index.json files (pre-A2) won't have these; default
                # to placeholder strings that the renderer can still print.
                "model_name": (index or {}).get("model_name"),
                "target_chip": (index or {}).get("target_chip"),
            }
        )
    rows.sort(key=lambda r: r["iteration_id"])
    return rows


def _top5_from_layerwise(layerwise: Optional[dict]) -> List[dict]:
    if not layerwise:
        return []
    items = list(layerwise.items())[:5]
    return [{"op": name, "snr": score} for name, score in items]


def _strip_internal(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def _is_finite_metric(v: Any) -> bool:
    """True iff `v` is a real, finite metric value usable for ranking / best
    selection. Excludes ``bool`` explicitly (it's an ``int`` subclass in Python
    and would otherwise score True/False as 1.0/0.0 and pollute rankings).
    Excludes NaN/inf via :func:`math.isfinite`.
    """
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        try:
            return math.isfinite(float(v))
        except (TypeError, ValueError):
            return False
    return False


def _best(rows: List[dict]) -> Optional[dict]:
    finite = [r for r in rows if _is_finite_metric(r.get("primary_value"))]
    if not finite:
        return None
    direction = finite[0].get("metric_direction", "max")
    reverse = direction == "max"
    # Stable tie-breaker: iteration_id ascending (smaller id wins on ties). Same
    # rule used by `_compute_ranks` so "best" and "rank == 1" never disagree.
    finite.sort(key=lambda r: (r["primary_value"], -r["iteration_id"]), reverse=reverse)
    return finite[0]


# ---------------------------------------------------------------------------
# Per-iteration classifications: runtime cost + rank
#
# These are pure functions over `rows` (no I/O, no globals). They are the
# data backbone of the new Iteration-history table columns and of the
# `comparison.json` `runtime_cost_classification` / `iteration_ranks` /
# `ranking_warnings` fields.
#
# Determinism contract (mirror in the docstring of `_compute_ranks` because
# users will read it from there too):
#   1. compare_iterations.py is stateless — every CLI invocation re-walks
#      `outputs/iter_*/iteration_index.json` from disk via `_gather`. Ranks
#      are NEVER cached across runs; the `iteration_ranks` written into
#      `comparison.json` is just a read-only snapshot of the current run.
#   2. Every `iter_N` directory participates in ranking, regardless of phase
#      or run order. There is no "Phase-3 only" or "completed only" filter.
#   3. Direction is taken from the FIRST finite row's `metric_direction`. If
#      different rows declare different directions (rare; usually means the
#      user edited QUANT_CONFIG mid-session), a warning is emitted and the
#      first-finite-row direction is used uniformly.
#   4. If different rows declare different `primary_metric` strings, a
#      warning is emitted but ranks are still produced from the numeric
#      `primary_value` field (since metrics may be on similar scales, e.g.
#      top1 vs top5 both percentages).
#   5. Tie-breaker: iteration_id ascending. Combined with dense ranking this
#      preserves the relative order of any pair of pre-existing iterations
#      across re-invocations — appending new iterations may insert new ranks
#      but cannot reverse the relative order of two old iterations.
#   6. Iterations whose `primary_value` is non-finite (None, NaN, inf,
#      `"_error"` strings, bools, etc.) get rank `"-"` and do not occupy a
#      slot in the dense ranking sequence.
# ---------------------------------------------------------------------------


def _classify_runtime_cost(setting: dict) -> Dict[str, Any]:
    """Classify on-device inference cost incurred by ``setting``.

    Returns a dict::

        {"affects_speed": bool, "label": "No" | "Yes (...)", "reasons": [str, ...]}

    Rules — see SKILL.md "On-device runtime cost cheat-sheet". Two passes are
    the only ones that incur permanent on-device runtime cost:

    * ``dispatching_table`` entries with ``bits == 16`` — each promoted op
      pays ~2x cycles + ~2x activation memory.
    * ``weight_split`` enabled — inserts an extra ``Add`` op per split layer.

    All other tunable passes (calib_algorithm, tqt_optimization, bias_correct,
    fusion_alignment, equalization, blockwise_reconstruction, lsq_optimization)
    only adjust scales/weights — they leave the runtime graph unchanged.
    """
    if not isinstance(setting, dict):
        return {"affects_speed": False, "label": "No", "reasons": []}

    reasons: List[str] = []

    dispatch = setting.get("dispatching_table") or []
    if isinstance(dispatch, list):
        int16_ops: List[str] = []
        for entry in dispatch:
            if not isinstance(entry, dict):
                continue
            if entry.get("bits") == 16:
                # Schema sometimes uses "op", sometimes "op_name" historically.
                op = entry.get("op") or entry.get("op_name")
                if op:
                    int16_ops.append(str(op))
        if int16_ops:
            preview = ", ".join(int16_ops[:3])
            if len(int16_ops) > 3:
                preview += f", +{len(int16_ops) - 3} more"
            reasons.append(f"int16 promotion on {len(int16_ops)} op(s): {preview}")

    ws = setting.get("weight_split") or {}
    if isinstance(ws, dict) and ws.get("enabled"):
        layers = ws.get("interested_layers")
        if isinstance(layers, list) and layers:
            reasons.append(f"weight_split on {len(layers)} layer(s)")
        else:
            reasons.append("weight_split enabled")

    if reasons:
        return {
            "affects_speed": True,
            "label": "Yes (" + "; ".join(reasons) + ")",
            "reasons": reasons,
        }
    return {"affects_speed": False, "label": "No", "reasons": []}


def _compute_ranks(
    rows: List[dict],
) -> Tuple[Dict[int, Any], List[str]]:
    """Dense-rank iterations by ``primary_value``; return ``(ranks, warnings)``.

    See the section comment above for the full determinism contract. Key
    rules: rank 1 is best; ties share a rank and the next slot is rank+1
    (dense, not standard); failed iterations get ``"-"``; the global
    direction is the first finite row's ``metric_direction``.

    Calling this function twice on the same ``rows`` always returns identical
    output. Appending more rows can introduce new ranks but never reverses
    the relative order of any two pre-existing finite iterations — this is
    the property the unit tests guard.
    """
    ranks: Dict[int, Any] = {}
    warnings: List[str] = []

    finite_rows = [r for r in rows if _is_finite_metric(r.get("primary_value"))]
    if not finite_rows:
        for r in rows:
            ranks[r["iteration_id"]] = "-"
        return ranks, warnings

    raw_direction = finite_rows[0].get("metric_direction") or "max"
    direction = raw_direction if raw_direction in ("max", "min") else "max"

    distinct_directions = {
        r.get("metric_direction")
        for r in finite_rows
        if r.get("metric_direction") in ("max", "min")
    }
    if len(distinct_directions) > 1:
        warnings.append(
            "Inconsistent metric_direction across iterations: "
            f"{sorted(distinct_directions)}. Ranks computed using "
            f"'{direction}' (taken from the first finite "
            f"iter_{finite_rows[0]['iteration_id']}). This usually means "
            "QUANT_CONFIG['metric_direction'] was changed mid-session."
        )

    distinct_metrics = {
        r.get("primary_metric") for r in finite_rows if r.get("primary_metric")
    }
    if len(distinct_metrics) > 1:
        warnings.append(
            "Inconsistent primary_metric across iterations: "
            f"{sorted(distinct_metrics)}. Ranks compare across different "
            "metric names — interpret with caution. Did you change "
            "QUANT_CONFIG['primary_metric'] between runs?"
        )

    reverse = direction == "max"
    sorted_rows = sorted(
        finite_rows,
        key=lambda r: (
            -float(r["primary_value"]) if reverse else float(r["primary_value"]),
            r["iteration_id"],
        ),
    )

    current_rank = 0
    last_value: Optional[float] = None
    for r in sorted_rows:
        v = float(r["primary_value"])
        if last_value is None or v != last_value:
            current_rank += 1
            last_value = v
        ranks[r["iteration_id"]] = current_rank

    for r in rows:
        ranks.setdefault(r["iteration_id"], "-")

    return ranks, warnings


# ---------------------------------------------------------------------------
# Phase-2 cartesian product detection (unchanged from previous revision)
# ---------------------------------------------------------------------------


def _is_tqt_default_schedule(tqt: Any) -> bool:
    """True iff TQT is enabled and lr/steps/block_size all match the default
    schedule. Any explicit override of lr/steps/block_size disqualifies the leg."""
    if not (isinstance(tqt, dict) and tqt.get("enabled")):
        return False
    lr = tqt.get("lr", _TQT_DEFAULT_LR)
    steps = tqt.get("steps", _TQT_DEFAULT_STEPS)
    block_size = tqt.get("block_size", _TQT_DEFAULT_BLOCK_SIZE)
    return (
        lr == _TQT_DEFAULT_LR
        and steps == _TQT_DEFAULT_STEPS
        and block_size == _TQT_DEFAULT_BLOCK_SIZE
    )


def _calib_with_default_tqt_tried(rows: List[dict]) -> set:
    """Calibrations covered by the Phase-2 calib×TQT(default) cartesian product."""
    seen = set()
    for r in rows:
        setting = r.get("setting") or {}
        if not setting:
            continue
        algo = (setting.get("calib_algorithm") or "kl").lower()
        if algo not in _CALIB_SWEEP_VALUES:
            continue
        if not _is_tqt_default_schedule(setting.get("tqt_optimization") or {}):
            continue
        other_enabled = False
        for key in (
            "equalization",
            "bias_correct",
            "weight_split",
            "lsq_optimization",
            "blockwise_reconstruction",
        ):
            block = setting.get(key) or {}
            if isinstance(block, dict) and block.get("enabled"):
                other_enabled = True
                break
        if setting.get("dispatching_table"):
            other_enabled = True
        if setting.get("fusion_alignment"):
            other_enabled = True
        if not other_enabled:
            seen.add(algo)
    return seen


# ---------------------------------------------------------------------------
# Phase-4 finalize triggers
# ---------------------------------------------------------------------------


def _target_reached(rows: List[dict], target: Optional[float], direction: str) -> bool:
    if not isinstance(target, (int, float)):
        return False
    for r in rows:
        v = r.get("primary_value")
        if not isinstance(v, (int, float)):
            continue
        if direction == "max" and v >= target:
            return True
        if direction == "min" and v <= target:
            return True
    return False


def _phase3_iteration_count(rows: List[dict]) -> int:
    """Count iterations after the Phase-2 sweep (i.e. iter_id >= 4 by convention).

    SKILL.md mandates iter_0 = baseline; iter_1/2/3 = Phase-2 calib×TQT sweep;
    Phase-3 levers start at iter_4. We use that convention rather than the
    setting's `phase` field because not all settings carry it.
    """
    return sum(1 for r in rows if r.get("iteration_id", 0) >= 4)


def _plateau_detected(rows: List[dict], best: Optional[dict]) -> bool:
    """True iff the last N iterations all sit within `_PLATEAU_REL_TOL` of best."""
    if best is None or len(rows) < _PLATEAU_WINDOW + 1:
        return False
    p_best = best.get("primary_value")
    if not isinstance(p_best, (int, float)) or abs(p_best) < 1e-9:
        return False
    direction = best.get("metric_direction", "max")
    recent = rows[-_PLATEAU_WINDOW:]
    for r in recent:
        v = r.get("primary_value")
        if not isinstance(v, (int, float)):
            return False
        if direction == "max":
            if (p_best - v) / abs(p_best) > _PLATEAU_REL_TOL:
                return False
        else:
            if (v - p_best) / abs(p_best) > _PLATEAU_REL_TOL:
                return False
    return True


def _all_phase3_levers_tried(rows: List[dict], priority: str) -> bool:
    """True iff every linear-order Phase-3 lever has been used at least once.

    3a-3 is conditional and not part of the linear order; its absence does not
    by itself block finalization (otherwise the agent could never finish on a
    network where 3a-3's triggers never fire).
    """
    order = _phase3_order_for(priority)
    return all(_lever_tried(rows, lever) for lever in order)


# ---------------------------------------------------------------------------
# Phase-3 lever helpers
# ---------------------------------------------------------------------------


def _phase3_order_for(priority: str) -> Tuple[str, ...]:
    if priority == "speed":
        return _PHASE3_ORDER_SPEED
    return _PHASE3_ORDER_BALANCED


def _tqt_block(setting: dict) -> dict:
    block = setting.get("tqt_optimization") or {}
    if not isinstance(block, dict):
        return {}
    return block


def _lever_tried(rows: List[dict], lever_id: str) -> bool:
    """Inspect every iteration's setting.json for the discriminating change of
    the named Phase-3 lever. Conservative: any iteration that enables the pass
    counts as "tried", regardless of whether it improved or regressed."""
    for r in rows:
        s = r.get("setting") or {}
        tqt = _tqt_block(s)
        if lever_id == "3a-1":
            if (
                tqt.get("enabled")
                and tqt.get("steps") == 1000
                and tqt.get("lr", _TQT_DEFAULT_LR) == _TQT_DEFAULT_LR
                and tqt.get("block_size", _TQT_DEFAULT_BLOCK_SIZE)
                == _TQT_DEFAULT_BLOCK_SIZE
            ):
                return True
        elif lever_id == "3a-2":
            if (
                tqt.get("enabled")
                and tqt.get("steps") == 2000
                and tqt.get("lr") == 5e-5
                and tqt.get("block_size", _TQT_DEFAULT_BLOCK_SIZE)
                == _TQT_DEFAULT_BLOCK_SIZE
            ):
                return True
        elif lever_id == "3a-3":
            if tqt.get("enabled") and tqt.get("block_size") == 2:
                return True
        elif lever_id == "3b":
            if (s.get("bias_correct") or {}).get("enabled"):
                return True
        elif lever_id == "3c":
            if s.get("fusion_alignment"):
                return True
        elif lever_id == "3d":
            if (s.get("equalization") or {}).get("enabled"):
                return True
        elif lever_id == "3e":
            if s.get("dispatching_table"):
                return True
        elif lever_id == "3f":
            if (s.get("weight_split") or {}).get("enabled"):
                return True
        elif lever_id == "3g":
            if (s.get("blockwise_reconstruction") or {}).get("enabled"):
                return True
    return False


def _next_phase3_lever_in_order(rows: List[dict], priority: str) -> Optional[str]:
    for lever in _phase3_order_for(priority):
        if not _lever_tried(rows, lever):
            return lever
    return None


# ---------------------------------------------------------------------------
# 3a-3 conditional triggers
# ---------------------------------------------------------------------------


def _tqt_3a3_path1_trigger(rows: List[dict], best: Optional[dict]) -> bool:
    """Path 1 (unstable fallback). Fires iff:
    * the most recent iteration is a 3a-1 or 3a-2 escalation,
    * its primary_value regressed vs best-so-far by < 0.5% relative,
    * its top-5 error layers contain at least one op_name not in best's top-5
      (i.e. the TQT escalation perturbed a layer that used to be quiet).
    """
    if best is None or len(rows) < 1:
        return False
    latest = rows[-1]
    if latest["dir"] == best["dir"]:
        return False  # latest IS best, no regression
    setting = latest.get("setting") or {}
    tqt = _tqt_block(setting)
    is_3a1 = (
        tqt.get("enabled")
        and tqt.get("steps") == 1000
        and tqt.get("lr", _TQT_DEFAULT_LR) == _TQT_DEFAULT_LR
    )
    is_3a2 = tqt.get("enabled") and tqt.get("steps") == 2000 and tqt.get("lr") == 5e-5
    if not (is_3a1 or is_3a2):
        return False
    p_latest = latest.get("primary_value")
    p_best = best.get("primary_value")
    if not isinstance(p_latest, (int, float)) or not isinstance(p_best, (int, float)):
        return False
    if abs(p_best) < 1e-9:
        return False
    direction = best.get("metric_direction", "max")
    if direction == "max":
        rel = (p_best - p_latest) / abs(p_best)
    else:
        rel = (p_latest - p_best) / abs(p_best)
    if not (0 < rel < _3A3_PATH1_REL_TOL):
        return False
    best_top = {entry.get("op") for entry in (best.get("top_5_error_layers") or [])}
    latest_top = {entry.get("op") for entry in (latest.get("top_5_error_layers") or [])}
    return bool(latest_top - best_top - {None})


def _tqt_3a3_path2_eligible(rows: List[dict], best: Optional[dict]) -> bool:
    """Path 2 (gap-shrink after convergence). The state machine cannot directly
    confirm this — it requires reading layer_stats / non_computing_hot_ops to
    check whether R5/R8/R3 structural triggers match. This helper returns True
    when the *necessary preconditions* are met:
      * 3a-1 has been tried with positive net effect,
      * 3a-2 has been tried with positive net effect,
      * 3a-3, 3b, 3c, 3d are all not yet tried.
    The hint then asks the agent to confirm the structural-no-match condition
    before applying 3a-3 (otherwise prefer 3b/3c/3d as appropriate).
    """
    if best is None:
        return False
    if not _lever_tried(rows, "3a-1") or not _lever_tried(rows, "3a-2"):
        return False
    for lever in ("3a-3", "3b", "3c", "3d"):
        if _lever_tried(rows, lever):
            return False
    return True


# ---------------------------------------------------------------------------
# Templates (Phase-2 + Phase-3)
# ---------------------------------------------------------------------------


def _phase2_template_for(calib: str, next_id: int) -> str:
    """Phase-2 strict default-TQT schedule template, as inline JSON string."""
    return (
        f'`{{"iteration_id": {next_id}, "calib_algorithm": "{calib}", '
        '"tqt_optimization": {"enabled": true, "lr": 1e-05, "steps": 500, '
        '"block_size": 4, "is_scale_trainable": true, "gamma": 0.0, '
        '"int_lambda": 0.0, "collecting_device": "cuda"}}`'
    )


def _tqt_block_from_best(best: Optional[dict]) -> dict:
    """Extract best's TQT settings (or the Phase-2 default if absent)."""
    if best is None:
        return {
            "enabled": True,
            "lr": _TQT_DEFAULT_LR,
            "steps": _TQT_DEFAULT_STEPS,
            "block_size": _TQT_DEFAULT_BLOCK_SIZE,
            "is_scale_trainable": True,
            "gamma": 0.0,
            "int_lambda": 0.0,
            "collecting_device": "cuda",
        }
    base = _tqt_block(best.get("setting") or {})
    out = {
        "enabled": True,
        "lr": base.get("lr", _TQT_DEFAULT_LR),
        "steps": base.get("steps", _TQT_DEFAULT_STEPS),
        "block_size": base.get("block_size", _TQT_DEFAULT_BLOCK_SIZE),
        "is_scale_trainable": base.get("is_scale_trainable", True),
        "gamma": base.get("gamma", 0.0),
        "int_lambda": base.get("int_lambda", 0.0),
        "collecting_device": base.get("collecting_device", "cuda"),
    }
    return out


def _phase3_lever_template(
    lever_id: str, best: Optional[dict]
) -> Tuple[Dict[str, Any], str]:
    """Return (snippet, change_summary) for a Phase-3 lever.

    The snippet is the *minimal change* over best-so-far's setting.json. The
    agent is expected to deep-copy best's setting.json, apply this snippet,
    bump iteration_id, and fill rationale.
    """
    base_tqt = _tqt_block_from_best(best)
    if lever_id == "3a-1":
        new_tqt = dict(base_tqt)
        new_tqt.update(
            {
                "steps": 1000,
                "lr": _TQT_DEFAULT_LR,
                "block_size": _TQT_DEFAULT_BLOCK_SIZE,
            }
        )
        return (
            {"tqt_optimization": new_tqt},
            "Lever 3a-1: TQT `steps: 500→1000` only (lr/block_size unchanged). "
            "If this regresses with a small margin AND a new layer enters the "
            "top-5 error list, the next iteration may auto-route to 3a-3 instead "
            "of pivoting to 3b.",
        )
    if lever_id == "3a-2":
        new_tqt = dict(base_tqt)
        new_tqt.update(
            {"steps": 2000, "lr": 5e-5, "block_size": _TQT_DEFAULT_BLOCK_SIZE}
        )
        return (
            {"tqt_optimization": new_tqt},
            "Lever 3a-2: TQT `lr: 1e-5→5e-5, steps: 1000→2000`. If it regresses, "
            "do not push further (no lr=1e-4, no steps=4000); pivot per the lever "
            "table or fall through to 3a-3 if the unstable-fallback signal fires.",
        )
    if lever_id == "3a-3":
        new_tqt = dict(base_tqt)
        new_tqt["block_size"] = 2
        return (
            {"tqt_optimization": new_tqt},
            "Lever 3a-3 (CONDITIONAL): TQT `block_size: 4→2` only (lr/steps from "
            "best-so-far unchanged). Smaller block_size is closer to layerwise "
            "training — more stable but loses some cross-layer joint optimisation. "
            "Do NOT try block_size=1 (full layerwise, no upside) or block_size>=6 "
            "(unstable; overlaps with 3g blockwise_reconstruction).",
        )
    if lever_id == "3b":
        return (
            {"bias_correct": {"enabled": True, "block_size": 4, "steps": 32}},
            "Lever 3b: enable bias_correction. Trigger: top-error layer's *output* "
            "row in layer_stats.json shows |Noise Mean| > 0.1 × Noise Std. Cite "
            "the specific op + numbers in rationale.",
        )
    if lever_id == "3c":
        return (
            {"fusion_alignment": {"align_elementwise_to": "Align to Large"}},
            "Lever 3c: fusion alignment for Concat/Add/Resize. Trigger comes "
            "primarily from non_computing_hot_ops.json — pick a row where "
            "op_type ∈ {Concat, Add, Resize, AveragePool} AND "
            "inputs_float_std_ratio > 5. Add `align_concat_to`/`align_resize_to`/"
            "`align_avgpooling_to = 'Align to Large'` as appropriate.",
        )
    if lever_id == "3d":
        return (
            {
                "equalization": {
                    "enabled": True,
                    "iterations": 10,
                    "value_threshold": 0.5,
                    "opt_level": 2,
                    "including_bias": False,
                    "including_act": False,
                }
            },
            "Lever 3d: equalization. Trigger: a top-error Conv-chain layer with "
            "weight per-channel max/mean > 5. On esp32p4 this is warn-only "
            "(per-channel weights); cite the SNR + chain shape in rationale.",
        )
    if lever_id == "3e":
        return (
            {
                "dispatching_table": [
                    {
                        "op_name": "<FILL: top-1 error op_name from layerwise_error.json>",
                        "bits": 16,
                    },
                ]
            },
            "Lever 3e: mixed-precision int16 on the worst 1-3 ops. Verify the "
            "op name exists in simplified_ops.json before submitting. ⚠️ This is "
            "Tier B — every promoted op pays a permanent on-device runtime cost.",
        )
    if lever_id == "3f":
        return (
            {
                "weight_split": {
                    "enabled": True,
                    "value_threshold": 1.5,
                    "method": "balance",
                    "interested_layers": [
                        "<FILL: layer_stats.json op_name with " "weight kurtosis > 10>"
                    ],
                }
            },
            "Lever 3f: weight_split on a single Conv with weight outliers that "
            "equalization (or per-channel quantization on esp32p4) didn't absorb. "
            "⚠️ Tier B — adds an extra Add op per split layer at runtime.",
        )
    if lever_id == "3g":
        # blockwise_reconstruction is mutex with TQT.
        return (
            {
                "tqt_optimization": {"enabled": False},
                "blockwise_reconstruction": {
                    "enabled": True,
                    "lr": 1e-3,
                    "steps": 5000,
                    "gamma": 1.0,
                    "block_size": 4,
                    "is_scale_trainable": False,
                    "collecting_device": "cuda",
                },
            },
            "Lever 3g: blockwise_reconstruction. ⚠️ Mutually exclusive with TQT "
            "(apply_setting._check_mutex enforces). The template explicitly "
            "disables tqt_optimization. This is a structural change (TQT off → "
            "blockwise on), bigger than any other Phase-3 lever; if 3g regresses, "
            "next iteration must roll back to best-so-far (TQT on) before "
            "continuing.",
        )
    raise ValueError(f"Unknown lever_id: {lever_id!r}")


def _format_template(snippet: Dict[str, Any], next_id: int) -> str:
    full = {
        "iteration_id": next_id,
        "rationale": "<FILL: cite the specific layerwise / "
        "non_computing_hot_ops / graphwise_jumps observation "
        "driving this change>",
    }
    full.update(snippet)
    return "`" + json.dumps(full, separators=(", ", ": ")) + "`"


# ---------------------------------------------------------------------------
# Hint builders
# ---------------------------------------------------------------------------


def _build_phase3_hint(
    lever_id: str,
    next_id: int,
    best: Optional[dict],
    extra_note: str = "",
) -> str:
    snippet, change = _phase3_lever_template(lever_id, best)
    template_str = _format_template(snippet, next_id)
    best_dir = (best or {}).get("dir", "<best>")
    parts = [
        f"{change}",
        f"Mutate from {best_dir}/setting.json: copy it, set iteration_id={next_id}, "
        f"deep-merge the change snippet, and fill in `rationale` citing the "
        f"specific observation (layer name + numbers from layerwise_error.json / "
        f"non_computing_hot_ops.json / graphwise_jumps.json).",
        f"Template change snippet: {template_str}.",
    ]
    if extra_note:
        parts.append(extra_note)
    parts.append(
        "One knob per iteration: do not stack additional unrelated passes on "
        "top of this template. If this iteration regresses, drop this lever "
        "next round per Composition discipline #3."
    )
    return " ".join(parts)


def _build_phase4_hint(
    rows: List[dict],
    best: Optional[dict],
    reason: str,
) -> str:
    best_id = (best or {}).get("iteration_id")
    best_metric = (best or {}).get("primary_metric")
    best_value = (best or {}).get("primary_value")
    target = _target_metric(rows)
    target_str = f"{target}" if isinstance(target, (int, float)) else "not set"
    gap_msg = ""
    if isinstance(target, (int, float)) and isinstance(best_value, (int, float)):
        direction = (best or {}).get("metric_direction", "max")
        gap = (target - best_value) if direction == "max" else (best_value - target)
        if (direction == "max" and best_value < target) or (
            direction == "min" and best_value > target
        ):
            gap_msg = (
                f" Gap to target: {abs(gap):.4f}. Suggest one full-evaluation "
                f"re-check via `python {{SKILL_DIR}}/scripts/run_iteration.py "
                f"--user-quant <...> --setting {{best_dir}}/setting.json "
                f"--output-dir <outputs>/iter_<NEW> --use-full-eval` before "
                f"declaring done."
            )
    return (
        f"Stop iterating ({reason}). Best iteration: iter_{best_id} "
        f"({best_metric}={best_value}). target_metric={target_str}.{gap_msg}\n\n"
        "**This script auto-finalises on phase-4** — by the time you read this "
        "hint, `outputs/best/` and `outputs/final_report.md` have already been "
        "written (idempotent; safe to inspect). The Iteration history table in "
        "the report includes `rank` (dense ranking, 1 = best) and `affects "
        "inference speed` (Yes when the iteration enables `dispatching_table` "
        "int16 promotion or `weight_split`) — use these columns to surface "
        "rank-2/3 candidates that may be a better speed/accuracy trade-off "
        "than the rank-1 best.\n\n"
        "Recommended follow-up:\n"
        "1. Read `outputs/final_report.md`. The Summary, Iteration history, "
        "Best setting, and Python snippet sections are auto-generated; "
        "`## Key findings` and `## Remaining gap (if target not met)` are "
        "seeded with auto-bullets but agents should extend them with concrete "
        "interpretations from `layer_stats.json` / `non_computing_hot_ops.json` "
        "/ `graphwise_jumps.json` of the best iteration.\n"
        "2. Run a single full-eval re-check: `python {SKILL_DIR}/scripts/"
        "run_iteration.py --user-quant <...> --setting outputs/best/setting.json "
        "--use-full-eval --output-dir outputs/iter_<NEW>`. If the resulting "
        f"{best_metric} differs from the iteration's `evaluate_fast` value "
        "above, update the Summary in `final_report.md` accordingly (the marker "
        "comment at the top of the file lets the script overwrite it on "
        "subsequent runs; once you make manual edits, remove the marker line "
        "to lock the file from auto-refresh, or pass --force to override).\n"
        "3. If you need to regenerate the report from scratch, run "
        "`python {SKILL_DIR}/scripts/compare_iterations.py --output-dir "
        "<outputs> --finalize --force`."
    )


# ---------------------------------------------------------------------------
# QUANT_CONFIG-derived getters (read from any iteration_index.json)
# ---------------------------------------------------------------------------


def _target_metric(rows: List[dict]) -> Optional[float]:
    """Return the most-recently-set target_metric (latest QUANT_CONFIG wins)."""
    for r in reversed(rows):
        t = r.get("target_metric")
        if isinstance(t, (int, float)):
            return float(t)
    return None


def _runtime_priority(rows: List[dict]) -> str:
    """Return the most-recently-set runtime priority. In a normal run all rows
    agree (the value comes from QUANT_CONFIG and doesn't change), but if the
    user edits QUANT_CONFIG mid-experiment the latest iteration's value wins."""
    for r in reversed(rows):
        p = r.get("deploy_runtime_priority")
        if isinstance(p, str) and p in ("balanced", "speed", "pc_time"):
            return p
    return "balanced"


# ---------------------------------------------------------------------------
# Main state machine
# ---------------------------------------------------------------------------


def _suggest_next(rows: List[dict], best: Optional[dict]) -> Tuple[str, str]:
    """Pick the phase + write a hint for the next iteration.

    Phases (priority order):
      * phase-4-final-report — finalize: target hit, plateau, cap reached, or
        all linear-order Phase-3 levers exhausted.
      * phase-1-baseline — no iterations yet.
      * phase-2-calib-tqt-sweep — Phase-2 cartesian product still missing legs.
      * phase-3-residual / phase-3-pivot — Phase-3 levers from best-so-far.
        3a-3 is conditional inside this branch.
    """
    if not rows:
        return "phase-1-baseline", (
            "No iterations yet. Run iter_0 with --baseline first "
            "(default espdl_setting(), kl calibration, no other passes). If iter_0's "
            "_primary_value already meets QUANT_CONFIG['target_metric'], the next "
            "compare_iterations run will switch to phase-4-final-report."
        )

    direction = (best or {}).get("metric_direction", "max")
    target = _target_metric(rows)
    priority = _runtime_priority(rows)

    # ---------- Phase 4 finalize triggers ----------
    # target_metric and Phase-3 cap take priority over everything; plateau is
    # checked AFTER 3a-3 path-1 (below) because path-1 is a specific stability
    # signal that can fire inside the plateau band — block_size=2 then resolves
    # whether we really have plateaued or merely perturbed a quiet layer.
    if _target_reached(rows, target, direction):
        return "phase-4-final-report", _build_phase4_hint(
            rows, best, "target_metric reached"
        )
    if _phase3_iteration_count(rows) >= _PHASE3_CAP:
        return "phase-4-final-report", _build_phase4_hint(
            rows,
            best,
            f"Phase-3 budget reached ({_PHASE3_CAP} iterations after Phase-2)",
        )

    # ---------- Phase 2 cartesian product ----------
    tried = _calib_with_default_tqt_tried(rows)
    missing = [c for c in _CALIB_SWEEP_VALUES if c not in tried]
    if missing:
        nxt = missing[0]
        next_id = max(r["iteration_id"] for r in rows) + 1
        remaining = [c for c in missing if c != nxt]
        followup = (
            f" After this finishes, re-run this script — it will tell you which leg "
            f"to do next ({', '.join(remaining)} still pending)."
            if remaining
            else " This is the last leg of the cartesian product."
        )
        return "phase-2-calib-tqt-sweep", (
            "Phase-2 calibration × TQT(default) cartesian product. "
            "Run ONE iteration next, copy this template verbatim and only fill in the "
            f"rationale: {_phase2_template_for(nxt, next_id)}. "
            "Do NOT change the TQT schedule (lr/steps/block_size must stay at "
            "1e-5/500/4 — any change makes this a Phase-3 lever (3a-1 / 3a-2 / 3a-3) instead and "
            "leaves the leg uncovered). Do NOT enable any other pass (equalization / "
            "bias_correct / fusion_alignment / dispatching_table / weight_split / "
            "lsq_optimization / blockwise_reconstruction). "
            "The cartesian product is mandatory: calibration in esp-dl quantization is "
            "not separable from the training pass — calibration alone may regress "
            "while the same calibration with TQT becomes the best overall recipe. "
            "Do NOT rank calibrations by their standalone score (Composition discipline "
            "#4)."
            + followup
            + " Iterations must be strictly sequential (single GPU, calibration-data "
            "race, and any leg is allowed to short-circuit the rest if "
            "QUANT_CONFIG['target_metric'] is already hit — check this iteration's "
            "metrics.json before scheduling the next one)."
        )

    # ---------- Phase 3 ----------
    next_id = max(r["iteration_id"] for r in rows) + 1

    # 3a-3 unstable-fallback path: checked BEFORE plateau because the signal
    # (small regression + new top-5 layer) can sit entirely inside the plateau
    # band. block_size=2 either confirms plateau (no further improvement) or
    # explains it as a TQT joint-training perturbation.
    if _tqt_3a3_path1_trigger(rows, best) and not _lever_tried(rows, "3a-3"):
        latest = rows[-1]
        return "phase-3-residual", _build_phase3_hint(
            "3a-3",
            next_id,
            best,
            extra_note=(
                "3a-3 PATH 1 triggered: previous iter (iter_"
                f"{latest['iteration_id']}) was a TQT escalation that regressed "
                "slightly AND introduced a new layer into the top-5 error list. "
                "This is a stability signal; smaller block_size should reduce "
                "joint-training perturbation. If 3a-3 also regresses, pivot to "
                "3b/3c/3d on the next iteration."
            ),
        )

    # ---------- Plateau finalize trigger ----------
    # Checked after 3a-3 path-1 so a stability signal still has a chance to
    # fire inside the plateau band.
    if _plateau_detected(rows, best):
        return "phase-4-final-report", _build_phase4_hint(
            rows,
            best,
            f"plateau detected (last {_PLATEAU_WINDOW} iterations within "
            f"{_PLATEAU_REL_TOL * 100:.1f}% of best)",
        )

    # Linear-order Phase-3 levers, by priority preference.
    next_lever = _next_phase3_lever_in_order(rows, priority)
    if next_lever is None:
        return "phase-4-final-report", _build_phase4_hint(
            rows, best, "all linear-order Phase-3 levers tried"
        )

    # If the next linear lever is 3b/3c/3d AND 3a-3 path-2 preconditions hold,
    # offer the agent a conditional swap: read non_computing_hot_ops first.
    if next_lever in ("3b", "3c", "3d") and (
        not _lever_tried(rows, "3a-3") and _tqt_3a3_path2_eligible(rows, best)
    ):
        # Use the linear-order lever as the *default* template, but include a
        # path-2 escape clause in the hint.
        path2_note = (
            "3a-3 PATH 2 also eligible: 3a-1/3a-2 both succeeded; 3a-3, 3b, 3c, "
            "3d are all not yet tried. BEFORE applying this template, open "
            f"{(best or {}).get('dir')}/non_computing_hot_ops.json and "
            f"{(best or {}).get('dir')}/layer_stats.json: if NONE of R5 "
            "(|Noise Mean| > 0.1 × Noise Std on a top-error op output), R8 "
            "(non-computing op with inputs_float_std_ratio > 5), or R3 "
            "(weight per-channel max/mean > 5 on a Conv chain) match for the "
            f"top entries, prefer 3a-3 instead of {next_lever}. Use the 3a-3 "
            "template from `_phase3_lever_template('3a-3')` — to get it, "
            "re-run compare_iterations.py with --force-3a3 (pending) or copy "
            "the structure from SKILL.md's lever 3a-3 row."
        )
        return "phase-3-residual", _build_phase3_hint(
            next_lever, next_id, best, extra_note=path2_note
        )

    # Plain pivot detection: last two regressed vs best.
    recent = [r for r in rows[-2:] if isinstance(r.get("primary_value"), (int, float))]
    best_value = (best or {}).get("primary_value")
    pivoting = (
        len(recent) >= 2
        and isinstance(best_value, (int, float))
        and all(
            (
                (r["primary_value"] < best_value)
                if direction == "max"
                else (r["primary_value"] > best_value)
            )
            for r in recent
        )
    )
    phase = "phase-3-pivot" if pivoting else "phase-3-residual"
    return phase, _build_phase3_hint(next_lever, next_id, best)


# ---------------------------------------------------------------------------
# Outcome / "method changed" classification used by both stdout and final
# report rendering. Centralised here so the two views agree.
# ---------------------------------------------------------------------------


def _outcome_label(
    row: dict,
    best: Optional[dict],
    baseline_value: Optional[float],
    direction: str,
) -> str:
    """One of: ``baseline`` / ``best`` / ``regression`` / ``improvement`` / ``no change``.

    "improvement" is reserved for non-best rows whose primary_value still beats
    baseline (rare during Phase 3, common during Phase 2). "regression" is the
    inverse. "no change" means within 1e-6 of baseline.
    """
    if row["iteration_id"] == 0:
        return "baseline"
    primary = row.get("primary_value")
    if not _is_finite_metric(primary):
        return "failed"
    if best is not None and row.get("dir") == best.get("dir"):
        return "best"
    if not _is_finite_metric(baseline_value):
        return "no change"
    delta = float(primary) - float(baseline_value)
    if abs(delta) < 1e-6:
        return "no change"
    helpful = (delta > 0 and direction == "max") or (delta < 0 and direction == "min")
    return "improvement" if helpful else "regression"


def _short_method_change(rationale: Optional[str], max_len: int = 80) -> str:
    """Take the first non-empty line of ``rationale`` and truncate to ``max_len``.

    Falls back to ``-`` when rationale is empty so the table column always has
    a printable cell.
    """
    if not rationale:
        return "-"
    for line in rationale.splitlines():
        line = line.strip()
        if line:
            if len(line) > max_len:
                return line[: max_len - 1] + "…"
            return line
    return "-"


# ---------------------------------------------------------------------------
# Final-report rendering — pure functions over already-gathered rows.
# Keep these dependency-free (no torch/esp_ppq); the harness in
# run_iteration.py is the only place that needs the full quant stack.
# ---------------------------------------------------------------------------


# Marker line written at the top of script-generated final_report.md. We use
# it on subsequent finalize runs to decide whether the file was last written
# by us (safe to overwrite, even without --force) vs hand-edited by an agent
# (preserve unless --force).
_REPORT_MARKER_PREFIX = "<!-- auto-generated by compare_iterations.py at "
_REPORT_MARKER_SUFFIX = (
    "; sections '## Key findings' / '## Remaining gap (if target not met)' "
    "are safe to edit. Removing this comment marks the file as agent-edited; "
    "subsequent --finalize will preserve the file unless --force is passed. -->"
)


def _report_marker(now: Optional[datetime.datetime] = None) -> str:
    """Return the auto-generated marker line, parameterised on timestamp."""
    ts = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat(
        timespec="seconds"
    )
    return f"{_REPORT_MARKER_PREFIX}{ts}{_REPORT_MARKER_SUFFIX}"


def _marker_in_existing_report(path: Path) -> bool:
    """True iff the existing ``final_report.md`` still carries our marker.

    Reads only the first ~10 lines to avoid loading large files. Detection is
    prefix-based so older marker timestamps are still recognised.
    """
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            head: List[str] = []
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                head.append(line)
    except OSError:
        return False
    return any(_REPORT_MARKER_PREFIX in line for line in head)


def _baseline_value_from(rows: List[dict]) -> Optional[float]:
    """Extract iter_0's primary_value as the baseline anchor, or None."""
    for r in rows:
        if r.get("iteration_id") == 0 and _is_finite_metric(r.get("primary_value")):
            return float(r["primary_value"])
    return None


def _format_metric_value(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)) and _is_finite_metric(v):
        return f"{float(v):.4f}"
    return str(v)


def _render_iteration_history_md(
    rows: List[dict],
    ranks: Dict[int, Any],
    costs: Dict[int, Dict[str, Any]],
    best: Optional[dict],
    primary_metric: str,
) -> str:
    """Render the iteration history Markdown table.

    Columns: ``iter | method changed | <primary_metric> | delta | outcome | rank | affects inference speed``.

    The ``rank`` column is computed by :func:`_compute_ranks` (stable across
    re-invocations); the ``affects inference speed`` column is computed by
    :func:`_classify_runtime_cost` (deterministic from setting.json).

    Empty / single-row inputs are handled gracefully — the caller (the final
    report renderer) is allowed to call this with as few as one row.
    """
    if not rows:
        return "_No iterations to summarise._\n"

    baseline_value = _baseline_value_from(rows)
    direction = "max"
    for r in rows:
        if _is_finite_metric(r.get("primary_value")):
            direction = r.get("metric_direction", "max") or "max"
            break

    metric_header = primary_metric or "primary_metric"
    lines: List[str] = []
    lines.append(
        f"| iter | method changed | {metric_header} | delta | outcome "
        "| rank | affects inference speed |"
    )
    lines.append("|---|---|---|---|---|:-:|---|")

    for r in rows:
        primary = r.get("primary_value")
        primary_cell = _format_metric_value(primary)

        if _is_finite_metric(primary) and _is_finite_metric(baseline_value):
            delta = float(primary) - float(baseline_value)
            sign = "+" if delta >= 0 else ""
            helpful = (delta > 0 and direction == "max") or (
                delta < 0 and direction == "min"
            )
            arrow = "↑" if helpful else ("=" if delta == 0 else "↓")
            delta_cell = f"{sign}{delta:.4f} {arrow}"
        else:
            delta_cell = "-"

        outcome = _outcome_label(r, best, baseline_value, direction)
        if outcome == "best":
            outcome_cell = "**best**"
        else:
            outcome_cell = outcome

        rank_val = ranks.get(r["iteration_id"], "-")
        if outcome == "best" and isinstance(rank_val, int):
            rank_cell = f"**{rank_val}**"
        else:
            rank_cell = str(rank_val)

        cost = costs.get(r["iteration_id"]) or {"label": "No"}
        cost_cell = cost.get("label", "No")

        method_cell = _short_method_change(r.get("rationale"))
        # Markdown table cell escaping — pipes inside text break the table.
        method_cell = method_cell.replace("|", "\\|")
        cost_cell = cost_cell.replace("|", "\\|")

        lines.append(
            f"| {r['iteration_id']} | {method_cell} | {primary_cell} "
            f"| {delta_cell} | {outcome_cell} | {rank_cell} | {cost_cell} |"
        )

    return "\n".join(lines) + "\n"


def _render_python_snippet(setting: dict, target: str) -> str:
    """Best-effort `QuantizationSettingFactory.espdl_setting()` recipe matching
    ``setting``. The output is a faithful translation of the JSON schema —
    when in doubt the agent can refine it manually. Always emits a header
    + footer so the user can drop the block into their own quantize script.
    """
    lines: List[str] = []
    lines.append("from esp_ppq import QuantizationSettingFactory")
    needs_target = bool(setting.get("dispatching_table"))
    if needs_target:
        lines.append("from esp_ppq.api import get_target_platform")
    lines.append("")
    lines.append("setting = QuantizationSettingFactory.espdl_setting()")

    calib = setting.get("calib_algorithm")
    if calib:
        lines.append(f"setting.quantize_activation_setting.calib_algorithm = {calib!r}")

    eq = setting.get("equalization") or {}
    if isinstance(eq, dict) and eq.get("enabled"):
        lines.append("setting.equalization = True")
        for k, attr in (
            ("iterations", "iterations"),
            ("value_threshold", "value_threshold"),
            ("opt_level", "opt_level"),
            ("including_bias", "including_bias"),
            ("including_act", "including_act"),
        ):
            if k in eq:
                lines.append(f"setting.equalization_setting.{attr} = {eq[k]!r}")

    bc = setting.get("bias_correct") or {}
    if isinstance(bc, dict) and bc.get("enabled"):
        lines.append("setting.bias_correct = True")
        for k in ("block_size", "steps"):
            if k in bc:
                lines.append(f"setting.bias_correct_setting.{k} = {bc[k]!r}")

    tqt = setting.get("tqt_optimization") or {}
    if isinstance(tqt, dict) and tqt.get("enabled"):
        lines.append("setting.tqt_optimization = True")
        for k in (
            "lr",
            "steps",
            "block_size",
            "is_scale_trainable",
            "gamma",
            "int_lambda",
            "collecting_device",
        ):
            if k in tqt:
                lines.append(f"setting.tqt_optimization_setting.{k} = {tqt[k]!r}")

    blk = setting.get("blockwise_reconstruction") or {}
    if isinstance(blk, dict) and blk.get("enabled"):
        lines.append("setting.blockwise_reconstruction = True")
        for k in (
            "lr",
            "steps",
            "block_size",
            "is_scale_trainable",
            "gamma",
            "collecting_device",
        ):
            if k in blk:
                lines.append(
                    f"setting.blockwise_reconstruction_setting.{k} = {blk[k]!r}"
                )

    fa = setting.get("fusion_alignment") or {}
    if isinstance(fa, dict) and fa:
        for k in (
            "align_avgpooling_to",
            "align_elementwise_to",
            "align_concat_to",
            "align_resize_to",
        ):
            if k in fa:
                lines.append(f"setting.fusion_setting.{k} = {fa[k]!r}")
        if "force_alignment_overlap" in fa:
            lines.append(
                f"setting.fusion_setting.force_alignment_overlap = "
                f"{fa['force_alignment_overlap']!r}"
            )

    ws = setting.get("weight_split") or {}
    if isinstance(ws, dict) and ws.get("enabled"):
        lines.append("setting.weight_split = True")
        if "interested_layers" in ws and ws["interested_layers"]:
            lines.append(
                f"setting.weight_split_setting.interested_layers = "
                f"{ws['interested_layers']!r}"
            )
        for k in ("value_threshold", "method"):
            if k in ws:
                lines.append(f"setting.weight_split_setting.{k} = {ws[k]!r}")

    dispatch = setting.get("dispatching_table") or []
    if isinstance(dispatch, list) and dispatch:
        for entry in dispatch:
            if not isinstance(entry, dict):
                continue
            op = entry.get("op") or entry.get("op_name")
            bits = entry.get("bits")
            if not op or bits is None:
                continue
            lines.append(
                "setting.dispatching_table.append(\n"
                f"    operation={op!r},\n"
                f"    platform=get_target_platform({target!r}, {bits}),\n"
                ")"
            )

    return "\n".join(lines) + "\n"


def _gap_to_target(
    best_value: Optional[float],
    target: Optional[float],
    direction: str,
) -> Optional[float]:
    """Positive-magnitude gap (best is short of target) or None if met / N/A."""
    if not _is_finite_metric(best_value) or not _is_finite_metric(target):
        return None
    if direction == "max" and float(best_value) < float(target):
        return float(target) - float(best_value)
    if direction == "min" and float(best_value) > float(target):
        return float(best_value) - float(target)
    return None


def _render_final_report(
    rows: List[dict],
    best: Optional[dict],
    target: Optional[float],
    model_name: str,
    target_chip: str,
    primary_metric: str,
    metric_direction: str,
    ranks: Dict[int, Any],
    costs: Dict[int, Dict[str, Any]],
    ranking_warnings: List[str],
    *,
    now: Optional[datetime.datetime] = None,
) -> str:
    """Build the full ``final_report.md`` content as a single string.

    The marker line is the first line of the file; subsequent finalize runs
    use it to detect "this was last written by us, safe to refresh". Sections
    'Key findings' and 'Remaining gap (if target not met)' are deliberately
    seeded with auto-generated bullets but agents are encouraged to expand
    them — the marker comment explicitly tells them so.

    Robust to degenerate inputs: a single-iteration session, or one where
    no iteration produced a finite metric, still yields a valid report.
    """
    lines: List[str] = []
    lines.append(_report_marker(now))
    title_chip = target_chip or "<target>"
    title_model = model_name or "<model>"
    lines.append(f"# Final Report: {title_model} on {title_chip}")
    lines.append("")

    # Summary section
    lines.append("## Summary")
    if best is None:
        lines.append(
            "- **No successful iteration produced a finite metric.** "
            "Review console.log under outputs/iter_*/ for failures."
        )
        if rows:
            lines.append(f"- Iterations attempted: {len(rows)}")
        lines.append("")
    else:
        best_id = best.get("iteration_id")
        best_value = best.get("primary_value")
        target_str = (
            _format_metric_value(target) if _is_finite_metric(target) else "not set"
        )
        lines.append(f"- Best iteration: iter_{best_id}")
        lines.append(
            f"- {primary_metric}: {_format_metric_value(best_value)} "
            f"(target_metric={target_str})"
        )
        # Note the eval source — tells the agent / user this number came
        # from the iteration loop's evaluate_fast() if applicable.
        eval_used = (best.get("extra_metrics") or {}).get("_used")
        if eval_used == "evaluate_fast":
            lines.append(
                "- _Note_: value comes from `evaluate_fast()`. Run "
                "`run_iteration.py --setting outputs/best/setting.json "
                "--use-full-eval --output-dir outputs/iter_<NEW>` for the full "
                "eval and update this Summary if the number differs."
            )
        elif eval_used == "evaluate":
            lines.append("- Value source: full `evaluate()` (not evaluate_fast).")

        # Best iteration's runtime cost on-device
        best_cost = costs.get(best_id) or {"label": "No"}
        lines.append(
            f"- On-device speed cost vs baseline (best): {best_cost.get('label', 'No')}"
        )

        # Other extra metrics (non-internal keys). Skip the leaked
        # `metric_direction` configuration field — `_run_iteration.py`
        # writes it into metrics.json as a hint for downstream readers
        # but it's not actually a metric.
        extras: List[str] = []
        non_metric_keys = {"metric_direction"}
        for k, v in (best.get("extra_metrics") or {}).items():
            if (
                not k
                or k.startswith("_")
                or k == primary_metric
                or k in non_metric_keys
            ):
                continue
            extras.append(f"{k}={_format_metric_value(v)}")
        if extras:
            lines.append(f"- Other metrics: {', '.join(extras)}")

        if len(rows) == 1:
            lines.append("- _Single-iteration session_ (baseline only).")

        gap = _gap_to_target(best_value, target, metric_direction)
        if gap is not None:
            lines.append(f"- Gap to target: {gap:.4f} (absolute)")
        lines.append("")

    if ranking_warnings:
        lines.append("> **Ranking warnings**")
        for w in ranking_warnings:
            lines.append(f"> - {w}")
        lines.append("")

    # Iteration history table
    lines.append("## Iteration history")
    lines.append("")
    lines.append(
        "> Ranks are recomputed from disk every time this report is regenerated "
        "(dense ranking with iteration_id ascending tie-breaker — appending new "
        "iterations cannot reverse the relative order of previously-ranked "
        "iterations). The `affects inference speed` column flags settings whose "
        "`dispatching_table` (int16 promotion) or `weight_split` introduces "
        "permanent on-device runtime cost — use it to spot rank-2/3 candidates "
        "that trade < 0.1% accuracy for measurably faster inference vs rank-1."
    )
    lines.append("")
    lines.append(
        _render_iteration_history_md(rows, ranks, costs, best, primary_metric).rstrip()
    )
    lines.append("")

    # Best setting — inline the full JSON
    lines.append("## Best setting")
    lines.append("")
    if best is None:
        lines.append("_No best iteration available._")
    else:
        lines.append("```json")
        lines.append(
            json.dumps(best.get("setting") or {}, indent=2, ensure_ascii=False)
        )
        lines.append("```")
    lines.append("")

    # Python snippet
    lines.append("## Python snippet")
    lines.append("")
    if best is None:
        lines.append("_No best iteration available; nothing to translate._")
    else:
        lines.append("```python")
        lines.append(
            _render_python_snippet(best.get("setting") or {}, target_chip).rstrip()
        )
        lines.append("```")
    lines.append("")

    # Key findings — auto-seed with top-error layers and best-vs-baseline
    lines.append("## Key findings")
    lines.append("")
    if best is not None:
        baseline_value = _baseline_value_from(rows)
        best_value = best.get("primary_value")
        if _is_finite_metric(baseline_value) and _is_finite_metric(best_value):
            delta = float(best_value) - float(baseline_value)
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"- Best vs baseline (iter_0 = {_format_metric_value(baseline_value)}): "
                f"{sign}{delta:.4f}."
            )
        # Most helpful and most regressing iterations
        finite = [r for r in rows if _is_finite_metric(r.get("primary_value"))]
        if finite and _is_finite_metric(baseline_value):
            scored = [
                (
                    float(r["primary_value"]) - float(baseline_value),
                    r,
                )
                for r in finite
                if r.get("iteration_id") != 0
            ]
            if scored:
                if metric_direction == "max":
                    scored.sort(key=lambda kv: kv[0], reverse=True)
                else:
                    scored.sort(key=lambda kv: kv[0])
                top_helper = scored[0]
                worst = scored[-1]
                if top_helper is not worst and (
                    abs(top_helper[0]) > 1e-9 or abs(worst[0]) > 1e-9
                ):
                    lines.append(
                        f"- Most helpful iteration: iter_{top_helper[1]['iteration_id']} "
                        f"(delta {top_helper[0]:+.4f}; "
                        f"{_short_method_change(top_helper[1].get('rationale'), 60)})."
                    )
                    lines.append(
                        f"- Largest regression: iter_{worst[1]['iteration_id']} "
                        f"(delta {worst[0]:+.4f}; "
                        f"{_short_method_change(worst[1].get('rationale'), 60)})."
                    )
        # Top-5 error layers in best
        top_layers = best.get("top_5_error_layers") or []
        if top_layers:
            lines.append("- Top error layers in best iteration (layerwise SNR):")
            for entry in top_layers[:5]:
                op = entry.get("op", "?")
                snr = entry.get("snr", 0.0)
                lines.append(f"  - `{op}` — SNR {snr:.6f}")
        # Best on-device cost — only add if it has cost
        best_cost = costs.get(best.get("iteration_id")) or {}
        if best_cost.get("affects_speed"):
            lines.append(
                "- The best iteration's setting trades on-device speed for "
                "accuracy; consult the rank-2/3 candidates in the Iteration "
                "history table for faster alternatives."
            )
    if not best:
        lines.append("_(No best iteration; cannot summarise findings.)_")
    lines.append("")
    lines.append(
        "<!-- Agent: feel free to expand the bullets above with concrete "
        "interpretations of layer-stats, fusion-alignment hot spots, etc. -->"
    )
    lines.append("")

    # Remaining gap — only when target wasn't met
    gap = (
        _gap_to_target(
            best.get("primary_value") if best else None, target, metric_direction
        )
        if best is not None
        else None
    )
    if gap is not None:
        lines.append("## Remaining gap (if target not met)")
        lines.append("")
        lines.append(
            f"Target {_format_metric_value(target)} was not met. "
            f"Gap: {gap:.4f} (absolute)."
        )
        lines.append("")
        lines.append("Potential follow-up directions:")
        lines.append("")
        lines.append(
            "1. **Enlarge calibration set** — a larger / more diverse "
            "calibration loader often improves scale estimation for the "
            "remaining error layers."
        )
        lines.append(
            "2. **Mixed precision (`dispatching_table` int16) on the worst 1-3 "
            "layers** — pays a small on-device cost in exchange for closing "
            "the per-layer gap."
        )
        lines.append(
            "3. **Blockwise reconstruction (`blockwise_reconstruction`)** — "
            "mutually exclusive with TQT; GPU strongly recommended; can "
            "refine per-block scales beyond what TQT achieves."
        )
        lines.append("")
        lines.append(
            "<!-- Agent: replace the boilerplate above with the specific "
            "recommendations that match this model's residual error pattern. -->"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Best-dir + finalize orchestration. _write_best_dir is reused by both the
# legacy --write-best path and the new --finalize path.
# ---------------------------------------------------------------------------


# Files copied alongside the model when finalize / --write-best runs. Setting,
# metrics, and iteration_index together let the agent reconstruct the full
# best iteration without walking back into outputs/iter_<N>/.
_BEST_COPY_FILES = ("setting.json", "metrics.json", "iteration_index.json")


def _link_or_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copyfile(str(src), str(dst))


def _write_best_dir(output_dir: Path, best: Optional[dict]) -> Optional[Path]:
    """Mirror the best iteration into ``outputs/best/``. Returns the dir, or
    None if there was nothing to write (no finite-metric iteration yet).
    """
    if not best:
        return None
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    src_root = Path(best["dir"])
    _link_or_copy(src_root / "model.espdl", best_dir / "model.espdl")
    for fname in _BEST_COPY_FILES:
        src_path = src_root / fname
        if src_path.exists():
            shutil.copyfile(str(src_path), str(best_dir / fname))
    return best_dir


def _model_name_from(rows: List[dict]) -> str:
    """Pick the most recent iteration's `model_name`, falling back to a
    sensible default. The harness writes this field into iteration_index.json
    (see run_iteration.py)."""
    for r in reversed(rows):
        name = r.get("model_name")
        if isinstance(name, str) and name:
            return name
    return "<model>"


def _target_chip_from(rows: List[dict]) -> str:
    for r in reversed(rows):
        chip = r.get("target_chip")
        if isinstance(chip, str) and chip:
            return chip
    return "<target>"


def _finalize_outputs(
    output_dir: Path,
    rows: List[dict],
    best: Optional[dict],
    ranks: Dict[int, Any],
    costs: Dict[int, Dict[str, Any]],
    ranking_warnings: List[str],
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Write outputs/best/ and outputs/final_report.md. Idempotent.

    Behaviour summary (also documented in SKILL.md):

    * ``outputs/best/`` — always refreshed when there's a finite-metric best
      iteration. Skipped silently when no iteration has a finite metric.
    * ``outputs/final_report.md`` — three cases:
       1. file does not exist → write it (action ``written``).
       2. file exists with our marker (last write was ours) → safely refresh
          (action ``rewritten``), preserves agent-edited reports.
       3. file exists without our marker → preserve unchanged unless ``force``
          is True (actions ``preserved-by-agent-edit`` / ``rewritten-forced``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    best_dir_path = _write_best_dir(output_dir, best)
    best_action = "skipped (no finite-metric iteration)" if best is None else "written"

    report_path = output_dir / "final_report.md"
    primary_metric = (best or {}).get("primary_metric") or (
        rows[0].get("primary_metric") if rows else "primary_metric"
    )
    metric_direction = (
        (best or {}).get("metric_direction")
        or (rows[0].get("metric_direction") if rows else "max")
        or "max"
    )

    target = _target_metric(rows)
    model_name = _model_name_from(rows)
    target_chip = _target_chip_from(rows)

    new_content = _render_final_report(
        rows=rows,
        best=best,
        target=target,
        model_name=model_name,
        target_chip=target_chip,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
        ranks=ranks,
        costs=costs,
        ranking_warnings=ranking_warnings,
    )

    if report_path.exists():
        if _marker_in_existing_report(report_path):
            report_path.write_text(new_content, encoding="utf-8")
            report_action = "rewritten"
        elif force:
            report_path.write_text(new_content, encoding="utf-8")
            report_action = "rewritten-forced"
        else:
            report_action = "preserved-by-agent-edit"
    else:
        report_path.write_text(new_content, encoding="utf-8")
        report_action = "written"

    return {
        "best_dir": str(best_dir_path) if best_dir_path is not None else None,
        "best_action": best_action,
        "final_report": str(report_path),
        "report_action": report_action,
    }


def _format_delta(current: float, baseline: Optional[float], direction: str) -> str:
    if baseline is None:
        return ""
    delta = current - baseline
    sign = "+" if delta >= 0 else ""
    helpful = (delta > 0 and direction == "max") or (delta < 0 and direction == "min")
    arrow = "↑" if helpful else ("=" if delta == 0 else "↓")
    return f" ({sign}{delta:.4f} {arrow})"


def _print_table(
    rows: List[dict],
    next_hint: Optional[Tuple[str, str]] = None,
    early_finalize_command: Optional[str] = None,
) -> None:
    if not rows:
        print("No iterations found.")
        if early_finalize_command:
            print()
            print("=== Tip ===")
            print(
                "Even with no iterations yet, you can run --finalize once "
                "iter_0 lands to ensure outputs/best/ + final_report.md are "
                "produced. Copy-paste:"
            )
            print(f"  {early_finalize_command}")
        return
    baseline_value: Optional[float] = None
    for r in rows:
        if r["iteration_id"] == 0 and _is_finite_metric(r.get("primary_value")):
            baseline_value = r["primary_value"]
            break

    direction = rows[0].get("metric_direction", "max")
    print("\n=== Iteration comparison ===")
    print(f"primary_metric={rows[0].get('primary_metric')}  direction={direction}")
    print(f"baseline (iter_0) value: {baseline_value}")
    target = _target_metric(rows)
    if isinstance(target, (int, float)):
        print(f"target_metric: {target}")
    priority = _runtime_priority(rows)
    if priority != "balanced":
        print(f"deploy_runtime_priority: {priority}")

    # Re-rank from disk every time. See `_compute_ranks` determinism contract.
    ranks, ranking_warnings = _compute_ranks(rows)
    costs = {
        r["iteration_id"]: _classify_runtime_cost(r.get("setting") or {}) for r in rows
    }
    if ranking_warnings:
        print()
        print(
            "ranking warnings (see comparison.json[ranking_warnings] for the "
            "structured form):"
        )
        for w in ranking_warnings:
            print(f"  ! {w}")

    print()
    print(
        f"{'iter':>4}  {'rank':>4}  {'primary':>10}  {'delta':>14}  "
        f"{'speed':>5}  {'time(s)':>8}  rationale"
    )
    print("-" * 110)
    for r in rows:
        primary = r.get("primary_value")
        if _is_finite_metric(primary):
            primary_str = f"{primary:.4f}"
            delta_str = _format_delta(primary, baseline_value, direction)
        else:
            primary_str = str(primary)
            delta_str = ""
        elapsed = r.get("elapsed_seconds")
        elapsed_str = f"{elapsed:.1f}" if isinstance(elapsed, (int, float)) else "-"
        # Empty rationale → splitlines() returns [], so guard the index.
        rationale_lines = (r.get("rationale") or "").splitlines()
        rationale = rationale_lines[0][:50] if rationale_lines else ""
        rank_cell = str(ranks.get(r["iteration_id"], "-"))
        cost = costs.get(r["iteration_id"]) or {"affects_speed": False}
        speed_cell = "slow" if cost.get("affects_speed") else "ok"
        print(
            f"{r['iteration_id']:>4}  {rank_cell:>4}  {primary_str:>10}  "
            f"{delta_str:>14}  {speed_cell:>5}  {elapsed_str:>8}  {rationale}"
        )
    print()
    best = _best(rows)
    if best is not None:
        print(
            f"Best so far: iter_{best['iteration_id']} "
            f"({best['primary_metric']}={best['primary_value']:.4f})"
        )
        print(f"  setting.json (mutate from this): {best['dir']}/setting.json")
        best_cost = costs.get(best["iteration_id"]) or {"label": "No"}
        print(f"  affects inference speed: {best_cost.get('label', 'No')}")
        if best.get("top_5_error_layers"):
            print("  top-5 error layers in best iteration:")
            for entry in best["top_5_error_layers"]:
                print(f"    {entry.get('snr', 0.0):.6f}  {entry.get('op')}")
    if next_hint is not None:
        phase, hint = next_hint
        print()
        print(f"=== Suggested next step ({phase}) ===")
        print(hint)

    # Always print this — agents need to know there's a one-line escape hatch
    # regardless of phase. Especially relevant when the user gave an iteration
    # budget ("iterate 3 times") that doesn't align with the state machine's
    # internal phase-4 triggers.
    if early_finalize_command:
        print()
        print("=== Tip: how to wrap up at any time ===")
        print(
            "If the user gave a fixed iteration budget (e.g. 'iterate 3 times' "
            "/ '只跑 N 轮'), don't wait for phase-4 — run --finalize after the "
            "last iteration to write outputs/best/ + outputs/final_report.md. "
            "User-budget stop > state-machine stop."
        )
        print(f"  {early_finalize_command}")


def _early_finalize_command(output_dir: Path) -> str:
    """Build the copy-paste command an agent can run at any time to wrap up
    the session — even when the state machine is still mid-Phase-2/3.

    Surfaced both in stdout (printed by ``_print_table``) and in
    ``comparison.json["early_finalize_command"]``. The redundancy is
    deliberate — agents that read JSON pick it up from the field, agents
    that read stdout pick it up from the printed line.
    """
    script_path = os.path.abspath(__file__)
    return f"python {script_path} --output-dir {output_dir} --finalize"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare espdl-quantize iterations")
    parser.add_argument(
        "--output-dir", required=True, help="Directory containing iter_<N>/"
    )
    parser.add_argument(
        "--write-best",
        action="store_true",
        help="(Legacy) Mirror the best iteration into <output-dir>/best/. "
        "Implied by --finalize; kept for backward compatibility.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Force-write outputs/best/ AND outputs/final_report.md, regardless "
        "of phase. Use this at any time to wrap up a session, e.g. when "
        "the user gave a specific iteration budget ('iterate 3 times').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="When passed alongside --finalize, overwrite outputs/final_report.md "
        "even if a previous version was edited by an agent (i.e. lacks the "
        "auto-generation marker comment). Without --force, agent-edited "
        "reports are preserved.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    rows = _gather(output_dir)
    best = _best(rows)
    phase, hint = _suggest_next(rows, best)

    ranks, ranking_warnings = _compute_ranks(rows)
    costs: Dict[int, Dict[str, Any]] = {
        r["iteration_id"]: _classify_runtime_cost(r.get("setting") or {}) for r in rows
    }
    early_finalize_cmd = _early_finalize_command(output_dir)

    auto_finalize_target = phase == "phase-4-final-report"
    finalize_result: Optional[Dict[str, Any]] = None
    if auto_finalize_target or args.finalize:
        finalize_result = _finalize_outputs(
            output_dir,
            rows,
            best,
            ranks,
            costs,
            ranking_warnings,
            force=args.force,
        )

    summary: Dict[str, Any] = {
        "iterations": rows,
        "best_iteration": best,
        "iteration_ranks": {str(k): v for k, v in ranks.items()},
        "runtime_cost_classification": {str(k): v for k, v in costs.items()},
        "ranking_warnings": ranking_warnings,
        "early_finalize_command": early_finalize_cmd,
        "next_step_hint": {
            "phase": phase,
            "advice": hint,
            "auto_finalized": finalize_result is not None,
        },
    }
    out_path = output_dir / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    _print_table(
        rows, next_hint=(phase, hint), early_finalize_command=early_finalize_cmd
    )

    if finalize_result is not None:
        trigger = "phase-4 detected" if auto_finalize_target else "--finalize requested"
        print(f"\n[compare] {trigger}; finalize results:")
        print(f"  - outputs/best/         : {finalize_result['best_action']}")
        print(f"  - outputs/final_report.md: {finalize_result['report_action']}")
        if finalize_result["report_action"] == "preserved-by-agent-edit":
            print(
                "    (existing report lacks the auto-generation marker — "
                "treated as agent-edited. Pass --force to overwrite.)"
            )

    # Legacy --write-best path: when --finalize wasn't used but the user
    # explicitly asked for the best dir, honour that without writing the
    # report. Same I/O as before this revision; same printed messages.
    if args.write_best and finalize_result is None:
        if best:
            best_dir = output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            src_root = Path(best["dir"])
            _link_or_copy(src_root / "model.espdl", best_dir / "model.espdl")
            for fname in _BEST_COPY_FILES:
                src_path = src_root / fname
                if src_path.exists():
                    shutil.copyfile(str(src_path), str(best_dir / fname))
                    print(f"[compare] copied {fname} -> {best_dir / fname}")
            print(f"[compare] best iteration artifacts written to {best_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
