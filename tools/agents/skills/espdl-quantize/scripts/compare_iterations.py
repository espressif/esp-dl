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
_PLATEAU_WINDOW = 3
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
# R8 (fusion_alignment / lever 3c) trigger constants.
#
# The historical rule "op_type ∈ {Concat, Add, Resize, AveragePool} AND
# inputs_float_std_ratio > 5" almost never fires on real esp-dl quantization
# runs — the three example projects under example_quantize_*/outputs/ all
# show std_ratio < 2 yet 3c was empirically useful in exactly one of them.
# The new rule below was calibrated against those three runs (mobilenetv2 on
# esp32s3 / esp32p4, yolo11n on esp32s3) — see references/decision_playbook.md
# §R8 for the rationale and the hindsight check in
# tests/hindsight_r8_examples.py for the regression test.
#
# Surfaced as module constants so a future user collecting more data can
# re-tune the thresholds in one place. If you change any of these, re-run
# tests/hindsight_r8_examples.py before committing.
# ---------------------------------------------------------------------------

# Primary "Goldilocks band" for the candidate elementwise op's max_snr.
#
# The band emerged from the three example projects under
# example_quantize_*/outputs/. Below the lower bound there is not enough
# residual noise on the Add/Concat for fusion alignment to fix. Above the
# upper bound the residual is so severe (often co-existing with a tight
# TQT-trained scale grid the alignment would overwrite) that forcing
# alignment regresses — the right move is a deeper pass (3a-3 / 3d / 3e)
# instead.
#
# Anchors (best-so-far at the moment 3c was tried in each project):
#   * mobilenetv2-esp32s3 iter_1: Add max_snr=0.264 → IN BAND, helped (+0.275%)
#   * mobilenetv2-esp32p4 iter_5: Add max_snr=0.318 → ABOVE BAND, would
#     have regressed (-0.35%)
#   * yolo11n-esp32s3 iter_7:    Concat max_snr=0.094 → BELOW BAND, would
#     have regressed (-0.34%)
_R8_MAX_SNR_PRIMARY = 0.20
_R8_MAX_SNR_UPPER = 0.30

# Reinforcement (not required to fire — raises confidence if present).
_R8_STD_RATIO_REINFORCE = 5.0

# Activation veto: if a Relu/Swish/Sigmoid in the top-3 hot ops has max_snr
# strictly greater than this multiple of the candidate elementwise op's
# max_snr, skip 3c — the error is activation-dominated, fix that first.
_R8_ACTIVATION_VETO_RATIO = 1.2

_R8_ELEMWISE_OP_TYPES = ("Concat", "Add", "Sub", "Mul", "Resize", "AveragePool")
_R8_ACTIVATION_OP_TYPES = ("Relu", "Swish", "Sigmoid", "HardSwish")


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


def _all_phase3_levers_tried(
    rows: List[dict], priority: str, best: Optional[dict] = None
) -> bool:
    """True iff every linear-order Phase-3 lever has been used at least once
    OR has been correctly skipped by its trigger.

    3a-3 is conditional and not part of the linear order; its absence does not
    by itself block finalization (otherwise the agent could never finish on a
    network where 3a-3's triggers never fire).

    A lever counts as "covered" when:
      * it appears in some iteration's setting.json (``_lever_tried`` true), OR
      * its trigger does not fire on the current best (only 3c participates
        today via ``_r8_check_for_best``) — exhausting the linear list while
        respecting the structural-trigger rules.

    Without the trigger-skip path, a network whose hot ops never match R8
    would loop forever waiting for the agent to "try" 3c — which makes no
    sense; we'd be asking the agent to run an iteration the data says will
    regress. With the skip path, the state machine cleanly progresses to
    Phase 5 (or Phase 4) once every lever is either tried or skipped.
    """
    for lever in _phase3_order_for(priority):
        if _lever_tried(rows, lever):
            continue
        if lever == "3c":
            fires, _reason = _r8_check_for_best(best)
            if not fires:
                continue
        return False
    return True


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
# R8 trigger helpers (govern whether lever 3c fires or is skipped-by-trigger)
# ---------------------------------------------------------------------------


def _load_non_computing_hot_ops(iter_dir: Any) -> List[dict]:
    """Read ``<iter_dir>/non_computing_hot_ops.json`` if present, else []."""
    if iter_dir is None:
        return []
    path = Path(str(iter_dir)) / "non_computing_hot_ops.json"
    data = _load(path)
    if isinstance(data, list):
        return data
    return []


def _r8_trigger_fires(
    hot_ops: List[dict],
    *,
    primary_lower: float = _R8_MAX_SNR_PRIMARY,
    primary_upper: float = _R8_MAX_SNR_UPPER,
    std_ratio_threshold: float = _R8_STD_RATIO_REINFORCE,
    veto_ratio: float = _R8_ACTIVATION_VETO_RATIO,
) -> Tuple[bool, str]:
    """Return ``(fires, reason)`` for the R8 (fusion_alignment / lever 3c) check.

    ``hot_ops`` is the list parsed from ``non_computing_hot_ops.json`` —
    each entry has at least ``op_name`` / ``op_type`` / ``max_snr`` and may
    have ``inputs_float_std_ratio``.

    Logic (see references/decision_playbook.md §R8 for full rationale):

    * Pick the top elementwise candidate ``c`` (``op_type ∈ {Concat, Add,
      Sub, Mul, Resize, AveragePool}`` with the largest ``max_snr``).
    * **Primary Goldilocks band**: fire when
      ``primary_lower < c.max_snr ≤ primary_upper`` (default 0.20 < snr ≤
      0.30). Below the band there is not enough residual noise on the
      elementwise op for fusion alignment to fix; above the band the
      residual is too severe to be fixed by alignment alone — usually a
      sign that a deeper pass (3a-3 / 3d / 3e) is the right move and
      forcing alignment will regress by overwriting an already-trained
      tighter scale.
    * **Reinforcement gate** (sufficient on its own): a non-zero explicit
      ``inputs_float_std_ratio > std_ratio_threshold`` (default 5.0) keeps
      firing even outside the primary band — the legacy std-ratio escape
      hatch for the rare cases where wide input scales actually drive the
      residual error.
    * **Activation veto** (overrides primary and reinforcement): if any
      Relu/Swish/Sigmoid/HardSwish in the top-3 hot ops has ``max_snr >
      veto_ratio × c.max_snr`` (default 1.2×), skip 3c — the dominant
      error is activation-shaped, not fusion-alignment-shaped. Fix the
      activation via TQT escalation or int16 promotion first.

    The reason string is meant to be embedded in the next iteration's
    ``rationale`` so the agent (and the human reviewing later) can see
    *why* 3c fired or was skipped.
    """
    if not hot_ops:
        return False, "non_computing_hot_ops.json is empty (no hot ops to gate on)"

    candidates = [
        op
        for op in hot_ops
        if isinstance(op, dict) and op.get("op_type") in _R8_ELEMWISE_OP_TYPES
    ]
    if not candidates:
        return (
            False,
            "no Concat/Add/Sub/Mul/Resize/AveragePool entry in non_computing_hot_ops.json",
        )

    def _msnr(op: dict) -> float:
        v = op.get("max_snr")
        return float(v) if isinstance(v, (int, float)) else 0.0

    top = max(candidates, key=_msnr)
    top_snr = _msnr(top)
    top_name = top.get("op_name", "<unknown>")
    top_type = top.get("op_type", "<unknown>")
    std_ratio = top.get("inputs_float_std_ratio")
    std_ratio_v = float(std_ratio) if isinstance(std_ratio, (int, float)) else 0.0

    in_band = primary_lower < top_snr <= primary_upper
    above_band = top_snr > primary_upper
    below_band = top_snr <= primary_lower
    reinforcement_passed = std_ratio_v > std_ratio_threshold

    if not in_band and not reinforcement_passed:
        if above_band:
            reason = (
                f"R8 above-band skip: top elementwise op {top_name} ({top_type}) "
                f"max_snr={top_snr:.3f} > {primary_upper:.2f} upper bound — residual "
                "noise too severe for fusion alignment alone (forces overwriting "
                "an already-tight TQT scale); try lever 3a-3 / 3d / 3e instead. "
                f"Reinforcement std_ratio={std_ratio_v:.2f} <= "
                f"{std_ratio_threshold:.1f}, no escape hatch."
            )
        else:
            reason = (
                f"R8 below-band skip: top elementwise op {top_name} ({top_type}) "
                f"max_snr={top_snr:.3f} <= {primary_lower:.2f} lower bound — "
                "residual on this op is too small for fusion alignment to move "
                "the metric. "
                f"Reinforcement std_ratio={std_ratio_v:.2f} <= "
                f"{std_ratio_threshold:.1f}, no escape hatch."
            )
        return False, reason

    # Activation veto — check the top-3 hot ops (regardless of op type) for
    # an activation that out-shouts the candidate by veto_ratio×.
    top_three = hot_ops[:3] if isinstance(hot_ops, list) else []
    activations = [
        op
        for op in top_three
        if isinstance(op, dict) and op.get("op_type") in _R8_ACTIVATION_OP_TYPES
    ]
    if activations:
        worst_act = max(activations, key=_msnr)
        worst_act_snr = _msnr(worst_act)
        if top_snr > 0 and worst_act_snr > top_snr * veto_ratio:
            return (
                False,
                f"R8 activation veto: {worst_act.get('op_name', '<unknown>')} "
                f"({worst_act.get('op_type', '<unknown>')}) max_snr={worst_act_snr:.3f} "
                f"> {veto_ratio:.2f}× candidate {top_name} max_snr={top_snr:.3f} — "
                "activation-dominated error; fix via TQT or int16 instead of 3c",
            )

    if in_band:
        gate_label = (
            f"primary band {primary_lower:.2f} < max_snr={top_snr:.3f} "
            f"<= {primary_upper:.2f}"
        )
    elif below_band:
        gate_label = (
            f"reinforcement std_ratio={std_ratio_v:.2f} > "
            f"{std_ratio_threshold:.1f} (max_snr={top_snr:.3f} below band)"
        )
    else:
        gate_label = (
            f"reinforcement std_ratio={std_ratio_v:.2f} > "
            f"{std_ratio_threshold:.1f} (max_snr={top_snr:.3f} above band)"
        )
    return (
        True,
        f"R8 fired on {top_name} ({top_type}): {gate_label}",
    )


def _r8_check_for_best(best: Optional[dict]) -> Tuple[bool, str]:
    """Run :func:`_r8_trigger_fires` against ``best``'s
    ``non_computing_hot_ops.json``. Returns ``(False, reason)`` when ``best``
    is missing or its hot_ops file is unreadable — the conservative default,
    so 3c gets skipped on data starvation rather than firing blindly."""
    if best is None:
        return False, "no best iteration available yet; cannot run R8 check"
    hot_ops = _load_non_computing_hot_ops(best.get("dir"))
    if not hot_ops:
        return (
            False,
            "best iteration has no readable non_computing_hot_ops.json "
            "(too old? harness skipped the artifact?) — skipping 3c by default",
        )
    return _r8_trigger_fires(hot_ops)


def _next_phase3_lever_with_skips(
    rows: List[dict], priority: str, best: Optional[dict]
) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """Return ``(next_lever, skipped_levers)`` where each skipped entry is
    ``(lever_id, reason)``. A lever is skipped when its trigger does not
    fire on the current best-so-far's distribution data — at the moment,
    only 3c (R8) participates. Skipped levers count as "covered" for the
    all-linear-tried check, so the search progresses instead of getting
    stuck looking for a distribution match that the data will never produce.
    """
    skipped: List[Tuple[str, str]] = []
    for lever in _phase3_order_for(priority):
        if _lever_tried(rows, lever):
            continue
        if lever == "3c":
            fires, reason = _r8_check_for_best(best)
            if not fires:
                skipped.append(("3c", reason))
                continue
        return lever, skipped
    return None, skipped


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
        # Embed the actual R8 fire-reason from best's non_computing_hot_ops.json
        # so the agent's rationale can quote concrete numbers instead of
        # re-deriving them from the artifact.
        fires, reason = _r8_check_for_best(best)
        change = (
            "Lever 3c: fusion alignment for Concat/Add/Resize/AveragePool. "
            f"Suggested rationale prefix: {reason}. "
            "Switch `align_elementwise_to` (Add/Sub/Mul), `align_concat_to` "
            "(Concat), `align_resize_to` (Resize), or `align_avgpooling_to` "
            "(AveragePool) to 'Align to Large' depending on the triggering "
            "op's type."
        )
        if not fires:
            # Should not normally be reached — the state machine skips 3c
            # via _next_phase3_lever_with_skips when this returns False —
            # but keep a safe message in case an agent calls the template
            # directly. Carrying the reason still helps the human reviewer.
            change = (
                "Lever 3c: fusion alignment. ⚠ R8 trigger did NOT fire on "
                f"best-so-far ({reason}). The state machine normally skips "
                "this lever in that case; if you are calling the template "
                "manually, double-check the data before applying."
            )
        return (
            {"fusion_alignment": {"align_elementwise_to": "Align to Large"}},
            change,
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
        # blockwise_reconstruction is NOT mutex with TQT anymore — the engine
        # runs TrainedQuantizationThresholdPass and then AdaroundPass
        # sequentially (esp-ppq/esp_ppq/quantization/quantizer/base.py
        # L379-420). The 3g template adds blockwise on top of best's TQT
        # configuration instead of replacing it. PC quantization time roughly
        # doubles (two gradient-based passes back to back) but accuracy
        # attribution stays clean: blockwise is the only new variable.
        return (
            {
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
            "Lever 3g: blockwise_reconstruction added on top of best-so-far. "
            "Engine runs AdaroundPass after TrainedQuantizationThresholdPass — "
            "they coexist sequentially, so TQT (if on in best) stays on. "
            "PC quantization time roughly doubles vs the prior best; accuracy "
            "attribution stays clean because blockwise is the only new "
            "variable. If 3g regresses, the next iteration must roll back to "
            "best-so-far (blockwise off) before continuing.",
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


# ---------------------------------------------------------------------------
# Phase 5 — agent-driven open exploration
#
# Phase 5 is intentionally NOT a state machine. The script's job here is to
# stop prescribing settings and instead surface the data and history the
# agent needs to write the next setting.json on its own. The shape of the
# emitted hint mirrors Phase 2/3 (phase string + advice text) but the
# advice is meta-guidance + a structured `phase5_signals` block; there is
# NO setting.json template.
# ---------------------------------------------------------------------------


def _phase5_lever_label_for(setting_a: dict, setting_b: dict) -> Optional[str]:
    """Return a short lever label describing the single most distinctive
    change between two settings, or ``None`` when the diff is multi-knob
    (already a Phase-5 stack) or empty. This is heuristic — it underpins
    the ``improving_levers`` table only, where a one-line attribution is
    enough; agents always have the raw setting.json available to dig in.
    """
    tqt_a = _tqt_block(setting_a)
    tqt_b = _tqt_block(setting_b)
    if tqt_a.get("enabled") != tqt_b.get("enabled"):
        return "tqt_enabled" if tqt_b.get("enabled") else "tqt_disabled"
    # TQT schedule change.
    for k in ("steps", "lr", "block_size"):
        if tqt_a.get(k) != tqt_b.get(k):
            return f"tqt_{k}={tqt_b.get(k)}"
    # Calibration change.
    if (setting_a.get("calib_algorithm") or "kl") != (
        setting_b.get("calib_algorithm") or "kl"
    ):
        return f"calib={setting_b.get('calib_algorithm')}"
    # Pass-on toggle.
    for key in (
        "equalization",
        "bias_correct",
        "fusion_alignment",
        "weight_split",
        "blockwise_reconstruction",
        "lsq_optimization",
    ):
        a = (
            (setting_a.get(key) or {}).get("enabled")
            if isinstance(setting_a.get(key), dict)
            else bool(setting_a.get(key))
        )
        b = (
            (setting_b.get(key) or {}).get("enabled")
            if isinstance(setting_b.get(key), dict)
            else bool(setting_b.get(key))
        )
        # fusion_alignment is a dict without "enabled" — fall back to truthiness.
        if isinstance(setting_a.get(key), dict) and "enabled" not in setting_a[key]:
            a = bool(setting_a[key])
        if isinstance(setting_b.get(key), dict) and "enabled" not in setting_b[key]:
            b = bool(setting_b[key])
        if a != b:
            return f"{key}={'on' if b else 'off'}"
    # Dispatching table size change.
    da = setting_a.get("dispatching_table") or []
    db = setting_b.get("dispatching_table") or []
    if isinstance(da, list) and isinstance(db, list) and len(da) != len(db):
        return f"dispatching_table={len(db)}_ops"
    return None


def _phase5_improving_levers(rows: List[dict]) -> List[dict]:
    """Walk the iterations in id order; for each that beat its immediate-prior
    best, emit a record of ``(iter, delta, added_pass)``. The aggregate
    is what the agent uses to decide which passes are worth stacking in
    Phase 5.
    """
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: r.get("iteration_id", 0))
    direction = sorted_rows[0].get("metric_direction", "max") or "max"
    history: List[dict] = []
    best_so_far: Optional[dict] = None
    for r in sorted_rows:
        v = r.get("primary_value")
        if not _is_finite_metric(v):
            continue
        if best_so_far is None:
            best_so_far = r
            continue
        prior_best_val = best_so_far.get("primary_value")
        improved = (direction == "max" and float(v) > float(prior_best_val)) or (
            direction == "min" and float(v) < float(prior_best_val)
        )
        if improved:
            label = _phase5_lever_label_for(
                best_so_far.get("setting") or {}, r.get("setting") or {}
            )
            history.append(
                {
                    "iter": r.get("iteration_id"),
                    "delta_vs_prior_best": float(v) - float(prior_best_val),
                    "added_pass": label,
                    "primary_value": float(v),
                }
            )
            best_so_far = r
    return history


def _phase5_regressing_levers(rows: List[dict]) -> List[dict]:
    """Symmetric to :func:`_phase5_improving_levers` — iterations that fell
    behind the best-so-far at the time, with the single-knob attribution.
    Useful as negative evidence so the agent doesn't stack a regressing
    lever back onto the new winning recipe.
    """
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: r.get("iteration_id", 0))
    direction = sorted_rows[0].get("metric_direction", "max") or "max"
    history: List[dict] = []
    best_so_far: Optional[dict] = None
    for r in sorted_rows:
        v = r.get("primary_value")
        if not _is_finite_metric(v):
            continue
        if best_so_far is None:
            best_so_far = r
            continue
        prior_best_val = best_so_far.get("primary_value")
        worse = (direction == "max" and float(v) < float(prior_best_val)) or (
            direction == "min" and float(v) > float(prior_best_val)
        )
        if worse and r.get("iteration_id", 0) > 0:
            label = _phase5_lever_label_for(
                best_so_far.get("setting") or {}, r.get("setting") or {}
            )
            history.append(
                {
                    "iter": r.get("iteration_id"),
                    "delta_vs_prior_best": float(v) - float(prior_best_val),
                    "added_pass": label,
                }
            )
        else:
            best_so_far = r
    return history


def _phase5_untried_calib_swaps(rows: List[dict], best: Optional[dict]) -> List[str]:
    """Return calib algorithms from the Phase-2 sweep that are NOT the
    current best's calib_algorithm. Phase-5 cross-pollination candidates."""
    if best is None:
        return []
    best_calib = ((best.get("setting") or {}).get("calib_algorithm") or "kl").lower()
    tried_in_phase2 = _calib_with_default_tqt_tried(rows)
    return [c for c in _CALIB_SWEEP_VALUES if c != best_calib and c in tried_in_phase2]


# ---------------------------------------------------------------------------
# Phase 5 coverage tracking: classify each Phase-5 iteration into 5alpha /
# 5beta / 5gamma / 5delta patterns and report which Phase-3 linear-order
# levers were left uncovered by the Phase-3 cap.
#
# These signals are what the agent reads in Phase 5 to decide whether
# stop signal (4) ("ran out of data-driven ideas") can legitimately fire.
# Specifically: signal (4) is only allowed when BOTH
# ``untried_phase5_patterns`` and ``untried_phase3_levers`` are empty AND
# no user-given iteration budget remains. See SKILL.md "Phase 5 — Stop
# signals" for the full contract.
# ---------------------------------------------------------------------------


# Passes whose enable/disable toggle counts as a Phase-5 STACK / ABLATE.
# Excludes tqt_optimization because TQT toggling has a special semantics
# (mutex with blockwise_reconstruction = lever 3g; bare TQT toggle is the
# 3g lever entry path). For the coverage signal we treat tqt as part of
# "enabled passes" anyway via its toggle.
_PHASE5_TOGGLEABLE_PASSES = (
    "equalization",
    "bias_correct",
    "fusion_alignment",
    "weight_split",
    "blockwise_reconstruction",
    "tqt_optimization",
)


def _phase5_pass_enabled(setting: dict, pass_name: str) -> bool:
    """Robust on/off check for a pass entry in a setting.json.

    Each pass can show up as either ``{"enabled": True}`` (the canonical
    Phase-3 lever shape) or as a non-empty dict without ``enabled``
    (the fusion_alignment shape: ``{"align_elementwise_to": "..."}``).
    Missing key => disabled.
    """
    block = setting.get(pass_name)
    if not isinstance(block, dict):
        return bool(block)
    if "enabled" in block:
        return bool(block["enabled"])
    return len(block) > 0


def _dispatch_int16_ops(setting: dict) -> set:
    """Return the set of op names dispatched to int16 in this setting."""
    entries = setting.get("dispatching_table") or []
    if not isinstance(entries, list):
        return set()
    ops: set = set()
    for entry in entries:
        if isinstance(entry, dict) and entry.get("bits") == 16:
            name = entry.get("op") or entry.get("op_name")
            if name:
                ops.add(name)
    return ops


def _weight_split_layers(setting: dict) -> set:
    """Return the set of layer names targeted by weight_split, or empty
    if weight_split is off / has no interested_layers."""
    block = setting.get("weight_split")
    if not isinstance(block, dict) or not block.get("enabled"):
        return set()
    layers = block.get("interested_layers") or []
    if isinstance(layers, list):
        return {str(layer) for layer in layers}
    return set()


def _classify_phase5_patterns(setting_prev: dict, setting_new: dict) -> List[str]:
    """Structurally classify a Phase-5 iteration vs its prior-best setting.

    Returns a list (possibly empty, possibly multi) of pattern keys among
    ``5alpha`` / ``5beta`` / ``5gamma`` / ``5delta``. The list is multi
    because Composition discipline #5 allows a single Phase-5 iteration
    to combine multiple knob changes — and the agent's coverage tracker
    should credit each pattern that fired, not just the "primary" one.

    Patterns:
      * 5alpha STACK         — any pass turned ON that was OFF before
                              (equalization, bias_correct, fusion_alignment,
                              weight_split, blockwise_reconstruction,
                              tqt_optimization toggle). Also fires when
                              dispatching_table goes from empty to non-empty
                              (int16 promotion newly enabled).
      * 5beta CROSS-POLLINATE — calib_algorithm differs.
      * 5gamma ABLATE         — any pass turned OFF that was ON before, OR
                              dispatching_table going from non-empty to
                              empty (int16 dropped).
      * 5delta DIVE-INTO-ARTIFACTS — dispatching_table op set changed
                              while staying non-empty on both sides, OR
                              weight_split interested_layers set changed
                              while staying enabled on both sides. This
                              captures "targeted artifact-driven choice"
                              (different ops based on layerwise data).

    No-op iterations (settings identical) return ``[]``. Levers that
    correspond to canonical Phase-3 single-knob mutations (e.g. enabling
    equalization on top of best) are reported as 5alpha; this is by design
    — in Phase 5 they ARE Phase-5 STACK iterations even when they happen
    to be Phase-3-shaped.
    """
    patterns: List[str] = []

    for pass_name in _PHASE5_TOGGLEABLE_PASSES:
        prev_on = _phase5_pass_enabled(setting_prev, pass_name)
        new_on = _phase5_pass_enabled(setting_new, pass_name)
        if new_on and not prev_on:
            if "5alpha" not in patterns:
                patterns.append("5alpha")
        elif prev_on and not new_on:
            if "5gamma" not in patterns:
                patterns.append("5gamma")

    prev_int16 = _dispatch_int16_ops(setting_prev)
    new_int16 = _dispatch_int16_ops(setting_new)
    if not prev_int16 and new_int16:
        if "5alpha" not in patterns:
            patterns.append("5alpha")
    elif prev_int16 and not new_int16:
        if "5gamma" not in patterns:
            patterns.append("5gamma")
    elif prev_int16 and new_int16 and prev_int16 != new_int16:
        if "5delta" not in patterns:
            patterns.append("5delta")

    prev_ws = _weight_split_layers(setting_prev)
    new_ws = _weight_split_layers(setting_new)
    if prev_ws and new_ws and prev_ws != new_ws:
        if "5delta" not in patterns:
            patterns.append("5delta")

    prev_calib = (setting_prev.get("calib_algorithm") or "kl").lower()
    new_calib = (setting_new.get("calib_algorithm") or "kl").lower()
    if prev_calib != new_calib:
        if "5beta" not in patterns:
            patterns.append("5beta")

    return patterns


def _phase5_cutoff_iter_id(rows: List[dict]) -> int:
    """Find the iteration_id of the last iteration BEFORE Phase 5 began.

    Iterations with id > the returned cutoff are Phase-5 iterations.

    Returns -1 when the history never enters Phase 5 (e.g. target was hit
    in Phase 3, plateau routed to phase-4, or the agent stopped mid-Phase-3
    on user budget); coverage helpers using this should special-case the
    -1 return as "no Phase 5 history" and emit an empty coverage dict.

    Inlines the phase-5 trigger conditions directly (instead of calling
    ``_suggest_next``) to avoid a recursion cycle — ``_suggest_next``
    itself depends on this function via ``_phase5_signals_for``. The
    inlined conditions mirror ``_suggest_next`` order:

    * Phase-2 incomplete (some calib leg missing) → next iter would still
      be phase-2, not phase-5.
    * ``target_metric`` reached → next iter would be phase-4, not phase-5.
    * ``_phase3_iteration_count(partial) >= _PHASE3_CAP`` → next iter is
      phase-5.
    * ``_all_phase3_levers_tried(partial, priority, best)`` → next iter
      is phase-5.

    Cost is O(N * cheap_checks) where N is iteration count; on real-world
    N less than 30 this is microseconds.
    """
    if not rows:
        return -1
    sorted_rows = sorted(rows, key=lambda r: r.get("iteration_id", 0))
    direction = (sorted_rows[0].get("metric_direction") or "max").lower()

    for i in range(1, len(sorted_rows) + 1):
        partial = sorted_rows[:i]

        # Phase 2 still has missing calib legs? Next iter is phase-2.
        tried_calib = _calib_with_default_tqt_tried(partial)
        if any(c not in tried_calib for c in _CALIB_SWEEP_VALUES):
            continue

        target = _target_metric(partial)
        if _target_reached(partial, target, direction):
            # Target met after this prefix; next iter is phase-4, not phase-5.
            continue

        priority = _runtime_priority(partial)
        best = _best(partial)

        cap_hit = _phase3_iteration_count(partial) >= _PHASE3_CAP
        all_done = _all_phase3_levers_tried(partial, priority, best)
        if cap_hit or all_done:
            return int(sorted_rows[i - 1].get("iteration_id", -1))

    return -1


def _phase5_pattern_coverage(rows: List[dict], cutoff_id: int) -> Dict[str, List[int]]:
    """For each Phase-5 iteration (``iteration_id > cutoff_id``), classify
    it against the iteration that was best-at-the-time and record which
    pattern(s) it covered.

    Returns ``{"5alpha": [iter_ids], "5beta": [...], "5gamma": [...], "5delta": [...]}``.
    Iterations that classify to no pattern (no-op vs prior best) are
    silently dropped.
    """
    coverage: Dict[str, List[int]] = {
        "5alpha": [],
        "5beta": [],
        "5gamma": [],
        "5delta": [],
    }
    if cutoff_id < 0 or not rows:
        return coverage

    sorted_rows = sorted(rows, key=lambda r: r.get("iteration_id", 0))
    direction = (sorted_rows[0].get("metric_direction") or "max").lower()

    for it in sorted_rows:
        iid = it.get("iteration_id")
        if not isinstance(iid, int) or iid <= cutoff_id:
            continue
        # Find best-at-the-time over iterations strictly before iid.
        prior = [
            r
            for r in sorted_rows
            if isinstance(r.get("iteration_id"), int)
            and r["iteration_id"] < iid
            and _is_finite_metric(r.get("primary_value"))
        ]
        if not prior:
            continue
        if direction == "min":
            prior_best = min(prior, key=lambda r: float(r["primary_value"]))
        else:
            prior_best = max(prior, key=lambda r: float(r["primary_value"]))

        patterns = _classify_phase5_patterns(
            prior_best.get("setting") or {}, it.get("setting") or {}
        )
        for p in patterns:
            coverage[p].append(iid)

    return coverage


def _untried_phase3_levers(
    rows: List[dict], priority: str, best: Optional[dict]
) -> List[str]:
    """Return the list of linear-order Phase-3 levers (in order) that have
    never been tried in any iteration AND have not been correctly skipped
    by their trigger.

    "Correctly skipped" today applies only to 3c (R8 trigger): when R8
    does not fire on best's ``non_computing_hot_ops.json`` the lever is
    excluded from the untried list — the data says it would regress, so
    forcing a Phase-5 attempt would burn an iteration we already know
    will not help.

    Used in Phase 5 to remind the agent that ``_PHASE3_CAP=5`` may have
    left levers uncovered (3d / 3e / 3f / 3g are common candidates) and
    those still deserve a Phase-5 STACK attempt before stop signal (4)
    can fire. See SKILL.md "Why _PHASE3_CAP=5 leaves untried linear-order
    levers".
    """
    out: List[str] = []
    for lever in _phase3_order_for(priority):
        if _lever_tried(rows, lever):
            continue
        if lever == "3c":
            fires, _ = _r8_check_for_best(best)
            if not fires:
                continue
        out.append(lever)
    return out


# ---------------------------------------------------------------------------
# 5β-reapply: when the lever stack deepens after an early 5β CROSS-POLLINATE
# attempt, the prior verdict on the calibration may no longer hold. The
# canonical example is example_quantize_mobilenetv2_esp32p4 iter_13: an
# early calib swap (percentile) regressed on a shallow stack, but the same
# calib on top of the deepest lever stack delivered a +0.55% jump that
# closed the gap to target. Composition discipline #4 says calib ranking is
# unstable across stacks; this helper formalises "you tried percentile on
# stack S_old, but current best's stack S_now is a strict superset — go
# re-do percentile on S_now before declaring 5β covered".
#
# Coverage semantics: 5β is considered "fully covered" only when every
# unique calib in untried_calib_swaps has been tried on the CURRENT best's
# stack (or a deeper one). Otherwise the entry stays in
# untried_5beta_reapply and "5beta" is forced back into
# untried_phase5_patterns. See SKILL.md "Phase 5 — Stop signals" and the
# Composition discipline #4 callout in references/decision_playbook.md.
# ---------------------------------------------------------------------------


def _phase5_lever_stack(setting: dict) -> frozenset:
    """Return the stable, comparable set of "levers turned ON" in this setting.

    Used to compare two iterations' lever stacks for the 5β-reapply check.
    Each enabled pass contributes its name; the dispatching_table contributes
    a sentinel ``int16:<count>`` so growing the int16 op count counts as a
    deeper stack. Calibration is NOT part of the stack — that's the axis
    that 5β is swapping along.
    """
    out = set()
    for pass_name in _PHASE5_TOGGLEABLE_PASSES:
        if _phase5_pass_enabled(setting, pass_name):
            out.add(pass_name)
    int16_ops = _dispatch_int16_ops(setting)
    if int16_ops:
        out.add(f"int16:{len(int16_ops)}")
    ws_layers = _weight_split_layers(setting)
    if ws_layers:
        out.add(f"weight_split:{len(ws_layers)}")
    return frozenset(out)


def _phase5_5beta_attempts(rows: List[dict]) -> List[Tuple[str, frozenset]]:
    """Enumerate every iteration whose calib swap (5β classification) was
    fired vs its prior best, and record ``(target_calib, prior_best_stack)``.

    The prior best is the best-at-the-time, not the global best — same
    convention as ``_phase5_pattern_coverage``. We exclude iterations whose
    only "change" was the calib (no other lever change AND no dispatching_
    table change AND no weight_split change) because those carry the
    cleanest attribution — but include calib-plus-other-changes too, since
    if percentile + equalization both fired, percentile WAS attempted on a
    stack that includes equalization (a strictly-deeper context).
    """
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: r.get("iteration_id", 0))
    direction = (sorted_rows[0].get("metric_direction") or "max").lower()
    attempts: List[Tuple[str, frozenset]] = []
    for it in sorted_rows:
        iid = it.get("iteration_id")
        if not isinstance(iid, int):
            continue
        prior = [
            r
            for r in sorted_rows
            if isinstance(r.get("iteration_id"), int)
            and r["iteration_id"] < iid
            and _is_finite_metric(r.get("primary_value"))
        ]
        if not prior:
            continue
        if direction == "min":
            prior_best = min(prior, key=lambda r: float(r["primary_value"]))
        else:
            prior_best = max(prior, key=lambda r: float(r["primary_value"]))
        prev_calib = (
            (prior_best.get("setting") or {}).get("calib_algorithm") or "kl"
        ).lower()
        new_calib = ((it.get("setting") or {}).get("calib_algorithm") or "kl").lower()
        if prev_calib == new_calib:
            continue
        # Use this iteration's OWN stack (without the calib, which is the
        # axis being swapped). That's the stack on which `new_calib` was
        # tested.
        stack_at_test = _phase5_lever_stack(it.get("setting") or {})
        attempts.append((new_calib, stack_at_test))
    return attempts


def _untried_5beta_reapply(rows: List[dict], best: Optional[dict]) -> List[str]:
    """Return the list of calibrations that appeared as 5β targets earlier
    in history but have never been tested on a stack at least as deep as
    the current best's stack.

    An empty list means "5β-reapply fully covered" — every previously-
    tried calib has been re-confirmed on the deepest stack (or deeper).
    """
    if best is None:
        return []
    best_stack = _phase5_lever_stack(best.get("setting") or {})
    attempts = _phase5_5beta_attempts(rows)
    if not attempts:
        return []
    # For each unique calib, find the deepest stack on which it was tried.
    deepest_for: Dict[str, frozenset] = {}
    for calib, stack in attempts:
        cur = deepest_for.get(calib)
        # "Deeper" = strict superset; if neither stack is a superset of the
        # other, keep the existing (first-seen wins; the comparison below
        # against best_stack handles the actual decision).
        if cur is None:
            deepest_for[calib] = stack
            continue
        if cur < stack:  # frozenset < means proper subset
            deepest_for[calib] = stack
    untried: List[str] = []
    # Stable order: alphabetical (no canonical priority across calibs).
    for calib in sorted(deepest_for):
        deepest = deepest_for[calib]
        # The calib was tried on stack `deepest`. If `best_stack` is
        # strictly deeper, the calib has NOT been re-tested on the current
        # best stack — flag as untried_5beta_reapply.
        if deepest < best_stack:
            untried.append(calib)
        # If deepest == best_stack or deepest >= best_stack, the calib was
        # already tested at the current depth (or deeper); covered.
        # If the two stacks are incomparable (neither strict subset), we
        # don't require a re-run — that's an ablation/divergence, not a
        # deepening.
    return untried


def _untried_phase5_patterns(coverage: Dict[str, List[int]]) -> List[str]:
    """Return the list of Phase-5 patterns (in canonical order) that have
    no attempts recorded in ``coverage`` — the negative of the pattern
    coverage dict, surfaced for the agent's stop-signal-(4) check."""
    out: List[str] = []
    for key in ("5alpha", "5beta", "5gamma", "5delta"):
        if not coverage.get(key):
            out.append(key)
    return out


# ---------------------------------------------------------------------------
# Tunable-params soft advisory: lists the parameter knobs available inside
# each currently-enabled pass on best, with common value ranges drawn from
# references/ppq_methods.md. This is NOT prescriptive — the agent is meant
# to read the section, decide whether the data justifies a knob change,
# and propose the next iteration. The section is conditionally rendered:
# if best has no enabled gradient/structural passes, the section is empty.
# ---------------------------------------------------------------------------


def _phase5_tunable_params_lines(setting: dict) -> List[str]:
    """Return the bullet-style lines describing tunable parameters for each
    pass currently enabled in ``setting``. Used by ``_build_phase5_hint`` to
    surface a soft-advisory tunable-params section."""
    lines: List[str] = []
    if (setting.get("tqt_optimization") or {}).get("enabled"):
        tqt = setting.get("tqt_optimization") or {}
        cur_lr = tqt.get("lr", 1e-5)
        cur_steps = tqt.get("steps", 500)
        cur_bs = tqt.get("block_size", 4)
        lines.append(
            "tqt_optimization (current: "
            f"lr={cur_lr}, steps={cur_steps}, block_size={cur_bs}) — try "
            "lr in (1e-5 / 5e-5 / 1e-4), steps in (500 / 1000 / 2000 / "
            "5000), block_size in (2 / 4 / 6 / 8); larger block_size aligns "
            "with blockwise's reconstruction window"
        )
    if (setting.get("blockwise_reconstruction") or {}).get("enabled"):
        blk = setting.get("blockwise_reconstruction") or {}
        cur_lr = blk.get("lr", 1e-3)
        cur_steps = blk.get("steps", 5000)
        cur_bs = blk.get("block_size", 4)
        lines.append(
            "blockwise_reconstruction (current: "
            f"lr={cur_lr}, steps={cur_steps}, block_size={cur_bs}) — try "
            "lr in (5e-4 / 1e-3), steps in (5000 / 10000), block_size in "
            "(4 / 6 / 8)"
        )
    if (setting.get("equalization") or {}).get("enabled"):
        eq = setting.get("equalization") or {}
        cur_opt = eq.get("opt_level", 2)
        cur_iter = eq.get("iterations", 5)
        lines.append(
            "equalization (current: "
            f"opt_level={cur_opt}, iterations={cur_iter}) — try opt_level "
            "in (1 / 2), iterations in (5 / 10); higher opt_level explores "
            "more channel-balance solutions per layer"
        )
    fa = setting.get("fusion_alignment")
    if isinstance(fa, dict) and fa:
        direction = fa.get("align_elementwise_to") or "(unset)"
        lines.append(
            "fusion_alignment "
            f"(current: align_elementwise_to={direction!r}) — try other "
            "directions in (Align to Large / Align to Output / Align to "
            "Small); R8 hindsight shows the best direction is layer-"
            "dependent"
        )
    calib = (setting.get("calib_algorithm") or "kl").lower()
    if calib == "percentile":
        lines.append(
            "calibration percentile (default 99.99) — try 99.9 / 99.99 / "
            "99.999 in the calib_algorithm_setting block (percentile only)"
        )
    return lines


def _phase5_signals_for(rows: List[dict], best: Optional[dict]) -> Dict[str, Any]:
    """Build the structured signals dict embedded in
    ``comparison.json["next_step_hint"]["phase5_signals"]``. Pure data — the
    agent reads it and decides what to do; no template, no recipe."""
    if best is None:
        return {}
    best_setting = best.get("setting") or {}
    enabled_passes: List[str] = []
    if (best_setting.get("equalization") or {}).get("enabled"):
        enabled_passes.append("equalization")
    if (best_setting.get("bias_correct") or {}).get("enabled"):
        enabled_passes.append("bias_correct")
    if best_setting.get("fusion_alignment"):
        enabled_passes.append("fusion_alignment")
    if (best_setting.get("weight_split") or {}).get("enabled"):
        enabled_passes.append("weight_split")
    if (best_setting.get("blockwise_reconstruction") or {}).get("enabled"):
        enabled_passes.append("blockwise_reconstruction")
    if (best_setting.get("tqt_optimization") or {}).get("enabled"):
        enabled_passes.append("tqt_optimization")
    int16_ops = [
        entry.get("op") or entry.get("op_name")
        for entry in (best_setting.get("dispatching_table") or [])
        if isinstance(entry, dict) and entry.get("bits") == 16
    ]
    if int16_ops:
        enabled_passes.append(f"dispatching_table(int16x{len(int16_ops)})")

    cost = _classify_runtime_cost(best_setting)

    # Load the heavy artifacts as soft references — agents may want to read
    # them directly, but we surface the top entries inline for cheap glance.
    best_dir = best.get("dir")
    hot_ops = _load_non_computing_hot_ops(best_dir)[:3] if best_dir else []
    graphwise_jumps_path = (
        Path(str(best_dir)) / "graphwise_jumps.json" if best_dir else None
    )
    graphwise_jumps = (
        _load(graphwise_jumps_path) if graphwise_jumps_path is not None else None
    )
    if isinstance(graphwise_jumps, list):
        graphwise_jumps_top3 = graphwise_jumps[:3]
    else:
        graphwise_jumps_top3 = []

    # Coverage tracking — what Phase 5 patterns have been attempted and
    # which Phase-3 linear-order levers the cap=5 left uncovered. Agents
    # use these to decide whether stop signal (4) ("ran out of ideas") is
    # legitimate or whether more pattern + lever ground must be covered
    # before finalising. See SKILL.md "Phase 5 — Stop signals".
    cutoff = _phase5_cutoff_iter_id(rows)
    coverage = _phase5_pattern_coverage(rows, cutoff)
    # 5β-reapply must be computed BEFORE _untried_phase5_patterns so we can
    # force "5beta" back into the untried set when an earlier-tried calib
    # has not been re-confirmed on the current best's deeper stack. Without
    # this, the agent could treat shallow-stack calib swaps as "5β done"
    # and skip the high-leverage deep-stack reapply (see
    # example_quantize_mobilenetv2_esp32p4 iter_13 for the canonical
    # "+0.55% jump after stack-deep reapply" case).
    reapply_untried = _untried_5beta_reapply(rows, best)
    untried_patterns = _untried_phase5_patterns(coverage)
    if reapply_untried and "5beta" not in untried_patterns:
        untried_patterns.append("5beta")
    priority = _runtime_priority(rows)
    untried_levers = _untried_phase3_levers(rows, priority, best)

    iteration_count_total = len(
        [r for r in rows if isinstance(r.get("iteration_id"), int)]
    )
    if cutoff >= 0:
        phase5_iteration_count = len(
            [
                r
                for r in rows
                if isinstance(r.get("iteration_id"), int) and r["iteration_id"] > cutoff
            ]
        )
    else:
        phase5_iteration_count = 0

    return {
        "best_iteration_dir": str(best_dir) if best_dir else None,
        "best_iteration_id": best.get("iteration_id"),
        "best_primary_value": best.get("primary_value"),
        "best_setting_summary": {
            "calib_algorithm": best_setting.get("calib_algorithm") or "kl",
            "tqt": _tqt_block(best_setting) or None,
            "enabled_passes": enabled_passes,
            "affects_inference_speed": cost.get("affects_speed", False),
            "affects_inference_speed_label": cost.get("label", "No"),
        },
        "iteration_count_total": iteration_count_total,
        "phase5_iteration_count": phase5_iteration_count,
        "phase5_cutoff_iter_id": cutoff,
        "phase5_pattern_coverage": coverage,
        "untried_phase5_patterns": untried_patterns,
        "untried_phase3_levers": untried_levers,
        "untried_5beta_reapply": reapply_untried,
        "improving_levers": _phase5_improving_levers(rows),
        "regressing_levers": _phase5_regressing_levers(rows),
        "untried_calib_swaps": _phase5_untried_calib_swaps(rows, best),
        "top_error_layers": best.get("top_5_error_layers") or [],
        "non_computing_hot_ops_top3": hot_ops,
        "graphwise_jumps_top3": graphwise_jumps_top3,
        "artifacts_to_read": [
            "layerwise_error.json",
            "layer_stats.json",
            "layer_stats_full.json",
            "non_computing_hot_ops.json",
            "graphwise_jumps.json",
        ],
    }


def _build_phase5_hint(
    rows: List[dict],
    best: Optional[dict],
    signals: Dict[str, Any],
    skipped_levers: Optional[List[Tuple[str, str]]] = None,
    *,
    trigger_reason: str = "Phase-3 search exhausted without meeting target",
) -> str:
    """Compose the narrative meta-guidance text that goes into
    ``comparison.json["next_step_hint"]["advice"]``. Deliberately NO setting.json
    template — the agent writes the next setting from scratch based on
    ``phase5_signals`` and the on-disk artifacts. See SKILL.md "Phase 5 —
    Agent-driven exploration" for the full contract.
    """
    best_id = (best or {}).get("iteration_id")
    primary_metric = (best or {}).get("primary_metric") or "primary_metric"
    primary_value = (best or {}).get("primary_value")
    target = _target_metric(rows)
    target_str = f"{target}" if isinstance(target, (int, float)) else "not set"

    improving = signals.get("improving_levers", []) or []
    regressing = signals.get("regressing_levers", []) or []
    untried_calib = signals.get("untried_calib_swaps", []) or []
    cost_label = (signals.get("best_setting_summary") or {}).get(
        "affects_inference_speed_label", "No"
    )

    parts: List[str] = []
    parts.append(
        f"Phase 5 — agent-driven exploration ({trigger_reason}). "
        f"Best so far: iter_{best_id} ({primary_metric}={primary_value}; "
        f"target_metric={target_str}). The state machine is handing control "
        "back to YOU. Read the data, then write the next setting.json from "
        "scratch — no template here on purpose."
    )

    # Improving-levers attribution.
    if improving:
        bits: List[str] = []
        for entry in improving[:6]:
            it = entry.get("iter")
            d = entry.get("delta_vs_prior_best")
            lab = entry.get("added_pass") or "<multi-knob>"
            if isinstance(d, (int, float)):
                sign = "+" if d >= 0 else ""
                bits.append(f"iter_{it} via {lab} ({sign}{d:.4f})")
            else:
                bits.append(f"iter_{it} via {lab}")
        parts.append("Improving levers in history: " + "; ".join(bits) + ".")

    # Untried calib swaps.
    if untried_calib:
        parts.append(
            "Untried calibration swaps on top of current lever stack: "
            + ", ".join(untried_calib)
            + " (Composition discipline #4 — calib-only ranking does not "
            "predict the combined ranking; re-run them with the current "
            "lever stack on)."
        )

    # Regressing-lever caveat (negative evidence; don't re-stack).
    if regressing:
        bits = []
        for entry in regressing[:3]:
            it = entry.get("iter")
            d = entry.get("delta_vs_prior_best")
            lab = entry.get("added_pass") or "<multi-knob>"
            sign = "+" if isinstance(d, (int, float)) and d >= 0 else ""
            bits.append(
                f"iter_{it} via {lab} " f"({sign}{d:.4f})"
                if isinstance(d, (int, float))
                else f"iter_{it} via {lab}"
            )
        parts.append("Regressions to avoid re-stacking: " + "; ".join(bits) + ".")

    # Skipped levers from Phase 3 — included so the agent knows which trigger
    # rules were applied without re-reading the script.
    if skipped_levers:
        bits = [f"{lev} ({why})" for lev, why in skipped_levers]
        parts.append(
            "Levers skipped by trigger rules in Phase 3: " + "; ".join(bits) + "."
        )

    # Coverage state — pattern attempts so far + untried Phase-3 linear-order
    # levers + untried Phase-5 patterns. This is the data backing stop signal
    # (4): the agent cannot legitimately invoke "ran out of ideas" while
    # either list is non-empty AND user budget remains.
    coverage = signals.get("phase5_pattern_coverage") or {}
    untried_phase5 = signals.get("untried_phase5_patterns") or []
    untried_phase3 = signals.get("untried_phase3_levers") or []
    iter_total = signals.get("iteration_count_total", len(rows))
    iter_phase5 = signals.get("phase5_iteration_count", 0)
    pattern_label = {
        "5alpha": "5alpha STACK",
        "5beta": "5beta CROSS-POLLINATE",
        "5gamma": "5gamma ABLATE",
        "5delta": "5delta DIVE-INTO-ARTIFACTS",
    }
    coverage_bits: List[str] = []
    for key in ("5alpha", "5beta", "5gamma", "5delta"):
        iters = coverage.get(key) or []
        if iters:
            coverage_bits.append(
                f"{pattern_label[key]} x{len(iters)} "
                f"(iter_{', iter_'.join(str(i) for i in iters)})"
            )
    if coverage_bits:
        parts.append(
            f"Coverage so far: {iter_total} total iterations "
            f"({iter_phase5} in Phase 5). Patterns attempted: "
            + "; ".join(coverage_bits)
            + "."
        )
    else:
        parts.append(
            f"Coverage so far: {iter_total} total iterations "
            f"({iter_phase5} in Phase 5). Phase 5 patterns attempted: NONE yet."
        )
    if untried_phase5:
        parts.append(
            "Untried Phase 5 patterns: "
            + ", ".join(pattern_label[k] for k in untried_phase5)
            + "."
        )
    if untried_phase3:
        parts.append(
            "Untried Phase 3 linear-order levers (left uncovered by "
            f"`_PHASE3_CAP={_PHASE3_CAP}`): "
            + ", ".join(untried_phase3)
            + ". STACK each onto best-so-far as a Phase-5 iteration before "
            "invoking stop signal (4)."
        )

    # 5β-reapply: when an earlier calib swap was tried on a shallower stack
    # than the current best's stack, Composition discipline #4 says the
    # ranking may now be different. Surface this as a STRONG instruction —
    # the canonical example is example_quantize_mobilenetv2_esp32p4
    # iter_13, where re-running percentile on the deep stack delivered a
    # +0.55% jump that closed the gap to target. Stop signal (4) is
    # blocked while this list is non-empty.
    reapply_targets = signals.get("untried_5beta_reapply") or []
    if reapply_targets:
        parts.append(
            "5\u03b2-reapply coverage: the following calibrations were "
            "tried earlier but on a SHALLOWER stack than current best — "
            "re-run each on top of the current lever stack before invoking "
            "stop signal (4) (Composition discipline #4 — calib ranking "
            "across stacks is unstable; the canonical example is "
            "example_quantize_mobilenetv2_esp32p4 iter_13, where re-"
            "running percentile on the deepest stack delivered a +0.55% "
            "jump that closed the gap to target): " + ", ".join(reapply_targets) + "."
        )

    # Tunable parameters in best's enabled passes. Soft advisory — NOT in
    # coverage, the agent reads the section, decides if a parameter knob is
    # worth a Phase-5 iteration based on the layerwise / non_computing_hot_ops
    # data. References ppq_methods.md for the per-pass parameter rationale.
    best_setting = (best or {}).get("setting") or {}
    tunable_parts = _phase5_tunable_params_lines(best_setting)
    if tunable_parts:
        parts.append(
            "Tunable parameters in current best (NOT a script — pick at "
            "most one per iteration if the data motivates it; see "
            "ppq_methods.md for parameter rationale): "
            + "; ".join(tunable_parts)
            + ". Tune-within-pass is a valid Phase-5 move — not all "
            "variations require turning a pass on/off."
        )

    # Cost-trim hint when the best already costs.
    if cost_label and cost_label != "No":
        parts.append(
            f"Best iteration trades on-device speed for accuracy "
            f"({cost_label}). Once you find a higher accuracy in Phase 5, "
            "consider a single iteration that trims the highest-cost "
            "component (e.g. drop one int16 op, or disable weight_split) "
            "to verify the runner-up still beats target."
        )

    # Inspiration patterns. Phrased as starting points; NOT prescriptive.
    parts.append(
        "Starting points to consider (NOT a script — let the data decide): "
        "(a) STACK — pick 2+ improving levers and run a single iteration "
        "that turns them ALL on. Multi-knob is allowed in Phase 5 if your "
        "rationale cites the iter ids whose data motivates each pass "
        "(Composition discipline #5). "
        "(b) CROSS-POLLINATE CALIB — if untried_calib_swaps is non-empty, "
        "swap calib while keeping the current lever stack on. "
        "(c) ABLATE — drop one pass at a time from the new best to test "
        "minimality / cost-trim. "
        "(d) DIVE INTO ARTIFACTS — read best's layerwise_error.json + "
        "layer_stats_full.json + non_computing_hot_ops.json + "
        "graphwise_jumps.json and propose a change you can defend with a "
        "concrete number from one of those files."
    )

    # Stop conditions — rewritten contract. User-budget is the hard ceiling;
    # signal (4) "ran out of ideas" is gated on coverage + no-user-budget.
    # `compare_iterations.py` HARD REJECTS `--finalize` in phase-5 if the
    # state machine doesn't agree (target not met AND not plateau) unless
    # the agent passes `--force-finalize`; this enforces the contract
    # operationally and stops the agent from finalising prematurely.
    parts.append(
        "Stop signals (each → finalize): "
        "(1) target_metric reached — script AUTO-finalizes via phase-4. "
        "(2) plateau (last 3 iterations within 0.1% of best) — script "
        "AUTO-finalizes via phase-4. "
        "(3) USER-GIVEN ITERATION BUDGET REACHED — agent runs --finalize "
        "NOW, regardless of phase and regardless of remaining untried "
        "patterns/levers. User budget is the hard ceiling. Pass "
        "--force-finalize alongside --finalize so the script confirms the "
        "intentional early stop and records the gap in final_report.md. "
        "(4) ONLY when ALL of the following hold simultaneously: "
        "(a) the user did NOT give a specific iteration budget, AND "
        "(b) `untried_phase5_patterns` is empty (every pattern attempted "
        "at least once), AND "
        "(c) `untried_phase3_levers` is empty (every linear-order Phase-3 "
        "lever either tried or correctly skipped), AND "
        "(d) `untried_5beta_reapply` is empty (every calib swap re-tested "
        "on the deepest stack), AND "
        "(e) the most recent iterations did not produce a new best. "
        "If signal (3) is in play, signal (4) is DISABLED — keep iterating "
        "until the user budget is met, drawing fresh variations from the "
        "untried lists. Phase 5 has NO hard iteration cap from the state "
        "machine; the user is the cap. NOTE: --finalize without "
        "--force-finalize in phase-5 (target not met, no plateau) is HARD "
        "REJECTED by compare_iterations.py — best/ and final_report.md "
        "will NOT be written and the script exits with code 1."
    )
    parts.append(
        "Copy-paste finalize command when done: see "
        '`comparison.json["early_finalize_command"]` or the printed Tip '
        "block at the bottom of stdout."
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
      * phase-4-final-report — finalize when target hit or plateau detected.
        (Phase-3 cap and all-linear-levers-tried now route to Phase 5 instead
        of phase-4 when target is not yet met — the state machine stops
        prescribing but the agent loop continues.)
      * phase-1-baseline — no iterations yet.
      * phase-2-calib-tqt-sweep — Phase-2 cartesian product still missing legs.
      * phase-3-residual / phase-3-pivot — Phase-3 levers from best-so-far.
        3a-3 is conditional inside this branch.
      * phase-5-agent-driven — open exploration; emitted when Phase-3 stops
        without target. Hint carries phase5_signals (improving levers,
        untried calib swaps, top error layers) but NO setting.json template.
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
    # target_metric short-circuits everything — Phase 5 exists to close a
    # gap, not to keep poking a model that already hit target. Phase-3 cap
    # without target met now routes to Phase 5 instead of Phase 4: the
    # state machine yields, but the agent loop continues. Plateau is checked
    # AFTER 3a-3 path-1 (below) because path-1 is a specific stability
    # signal that can fire inside the plateau band — block_size=2 then
    # resolves whether we really have plateaued or merely perturbed a quiet
    # layer.
    if _target_reached(rows, target, direction):
        return "phase-4-final-report", _build_phase4_hint(
            rows, best, "target_metric reached"
        )
    if _phase3_iteration_count(rows) >= _PHASE3_CAP:
        return "phase-5-agent-driven", _build_phase5_hint(
            rows,
            best,
            _phase5_signals_for(rows, best),
            None,
            trigger_reason=(
                f"Phase-3 budget reached ({_PHASE3_CAP} iterations after "
                "Phase-2) without meeting target_metric"
            ),
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

    # Linear-order Phase-3 levers, by priority preference. The skip-aware
    # variant lets us bypass 3c when R8 doesn't fire on best-so-far's
    # non_computing_hot_ops.json — the data says fusion alignment would
    # not help, so we don't burn an iteration on it. Skipped levers are
    # surfaced in the next hint so the agent / user knows why.
    next_lever, skipped = _next_phase3_lever_with_skips(rows, priority, best)
    if next_lever is None:
        # All linear levers tried or correctly skipped — hand to Phase 5
        # so the agent can explore freely instead of finalising now. Plateau
        # / target-reached have already short-circuited above.
        return "phase-5-agent-driven", _build_phase5_hint(
            rows,
            best,
            _phase5_signals_for(rows, best),
            skipped,
            trigger_reason=(
                "all linear-order Phase-3 levers tried (or skipped by trigger)"
            ),
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


_STOP_REASON_PATTERN_LABEL = {
    "5alpha": "5alpha STACK",
    "5beta": "5beta CROSS-POLLINATE",
    "5gamma": "5gamma ABLATE",
    "5delta": "5delta DIVE-INTO-ARTIFACTS",
}


def _render_stop_reason(reason: Dict[str, Any], primary_metric: str) -> List[str]:
    """Render the ``## Stop reason`` section as a list of markdown lines.

    The section sits between ``## Summary`` and ``## Iteration history`` so
    a human scanning the report top-down sees "what was achieved" → "why we
    stopped here" → "the history" in that order. Categories follow the
    enum in ``_FINALIZE_CATEGORIES``; each gets its own narrative tuned to
    the data attached by ``_build_finalize_reason``.
    """
    lines: List[str] = ["## Stop reason", ""]
    category = reason.get("category", "unknown")
    best_id = reason.get("best_iteration_id")
    best_value = reason.get("best_primary_value")
    target = reason.get("target_metric")
    gap = reason.get("gap_to_target")
    lines.append(f"- **Category**: `{category}`")
    if best_id is not None and best_value is not None:
        lines.append(
            f"- **Best**: iter_{best_id} ({primary_metric}="
            f"{_format_metric_value(best_value)})"
        )
    if target is not None:
        if gap is not None:
            sign = "+" if gap >= 0 else ""
            lines.append(
                f"- **Target**: {_format_metric_value(target)} &nbsp; "
                f"**Gap**: {sign}{gap:.4f}"
            )
        else:
            lines.append(f"- **Target**: {_format_metric_value(target)}")
    lines.append("")

    if category == "target_reached":
        lines.append(
            "Target met at the best iteration; no remaining gap. The state "
            "machine auto-finalized via `phase-4-final-report`. No further "
            "iterations are needed."
        )
    elif category == "plateau":
        window = reason.get("plateau_window", _PLATEAU_WINDOW)
        recent = reason.get("plateau_recent_values") or []
        lines.append(
            f"The last {window} iterations were all within 0.1% (relative) "
            "of the best, indicating accuracy has stopped moving. The state "
            "machine auto-finalized via `phase-4-final-report`."
        )
        if recent:
            lines.append("")
            lines.append("Recent iterations triggering plateau:")
            for entry in recent:
                lines.append(
                    f"- iter_{entry.get('iteration_id')}: "
                    f"{primary_metric}={_format_metric_value(entry.get('primary_value'))}"
                )
    elif category == "force_finalize_phase5":
        total = reason.get("iteration_count_total")
        p5 = reason.get("phase5_iteration_count")
        lines.append(
            "`--force-finalize` was passed, overriding the premature-"
            "finalize guard while phase=phase-5-agent-driven and the target "
            "metric was NOT met (and not a plateau). The coverage gaps "
            "below were skipped — re-open this session and continue "
            "iterating if the user expects further improvement."
        )
        if total is not None and p5 is not None:
            lines.append("")
            lines.append(f"Iterations so far: {total} total ({p5} in Phase 5).")
        untried_p5 = reason.get("untried_phase5_patterns") or []
        untried_p3 = reason.get("untried_phase3_levers") or []
        untried_reapply = reason.get("untried_5beta_reapply") or []
        lines.append("")
        if untried_p5:
            lines.append(
                "- **Untried Phase 5 patterns**: "
                + ", ".join(_STOP_REASON_PATTERN_LABEL.get(k, k) for k in untried_p5)
            )
        else:
            lines.append("- **Untried Phase 5 patterns**: (none)")
        if untried_p3:
            lines.append(
                "- **Untried Phase 3 linear-order levers**: " + ", ".join(untried_p3)
            )
        else:
            lines.append("- **Untried Phase 3 linear-order levers**: (none)")
        if untried_reapply:
            lines.append(
                "- **Untried 5\u03b2-reapply targets** (re-run on the "
                "deepest stack): " + ", ".join(untried_reapply)
            )
        else:
            lines.append("- **Untried 5\u03b2-reapply targets**: (none)")
    elif category == "manual_finalize_phase4":
        lines.append(
            "`--finalize` was passed while the state machine had already "
            "routed to `phase-4-final-report` (target reached or plateau "
            "detected). The flag was redundant but harmless."
        )
    elif category == "manual_finalize_pre_phase5":
        lines.append(
            "`--finalize` was passed while the state machine was still in "
            "phase-1 / phase-2 / phase-3 — most often because the user "
            "specified an iteration budget that was reached before Phase-3 "
            "had a chance to run all linear-order levers. The skipped "
            "levers and Phase 5 patterns are inherent to this early stop; "
            "if the user expects more accuracy later, re-open the session "
            "and remove the budget."
        )
    else:
        lines.append(
            "Stop reason category `" + str(category) + "` is not recognised "
            "by this version of compare_iterations.py."
        )
    lines.append("")
    return lines


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
    finalize_reason: Optional[Dict[str, Any]] = None,
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

    if finalize_reason is not None:
        lines.extend(_render_stop_reason(finalize_reason, primary_metric))

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


# Categories embedded into final_report.md and comparison.json so the user
# (and any downstream tool) can answer "why did this session end here?". The
# script computes the category up-front in main() and threads it through
# both _finalize_outputs and _render_final_report.
_FINALIZE_CATEGORIES = (
    "target_reached",
    "plateau",
    "force_finalize_phase5",
    "manual_finalize_phase4",
    "manual_finalize_pre_phase5",
)


def _build_finalize_reason(
    category: str,
    rows: List[dict],
    best: Optional[dict],
    target: Any,
    direction: str,
    *,
    phase5_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct the structured ``finalize_reason`` dict that lands in
    ``comparison.json["next_step_hint"]["finalize_reason"]`` and is rendered
    into the ``## Stop reason`` section of ``final_report.md``.

    The ``category`` argument must come from ``_FINALIZE_CATEGORIES``; the
    helper does not validate (the call sites in ``main()`` are the only
    callers and they pick the category deterministically).

    Per-category fields:
      * ``target_reached``: ``gap_to_target`` = 0.
      * ``plateau``: ``plateau_window`` + ``plateau_recent_values``.
      * ``force_finalize_phase5``: ``untried_phase5_patterns``,
        ``untried_phase3_levers``, ``untried_5beta_reapply`` (from
        ``phase5_signals``).
      * ``manual_finalize_phase4`` / ``manual_finalize_pre_phase5``: only
        the common fields.
    """
    best_id = (best or {}).get("iteration_id")
    best_value = (best or {}).get("primary_value")
    target_val = target if isinstance(target, (int, float)) else None
    gap = None
    if isinstance(best_value, (int, float)) and target_val is not None:
        if direction == "max":
            gap = target_val - best_value
        else:
            gap = best_value - target_val
    reason: Dict[str, Any] = {
        "category": category,
        "best_iteration_id": best_id,
        "best_primary_value": best_value,
        "target_metric": target_val,
        "gap_to_target": gap,
    }
    if category == "plateau":
        sorted_rows = sorted(rows, key=lambda r: r.get("iteration_id", 0))
        recent = sorted_rows[-_PLATEAU_WINDOW:]
        reason["plateau_window"] = _PLATEAU_WINDOW
        reason["plateau_recent_values"] = [
            {
                "iteration_id": r.get("iteration_id"),
                "primary_value": r.get("primary_value"),
            }
            for r in recent
        ]
    if category == "force_finalize_phase5" and phase5_signals is not None:
        reason["untried_phase5_patterns"] = (
            phase5_signals.get("untried_phase5_patterns") or []
        )
        reason["untried_phase3_levers"] = (
            phase5_signals.get("untried_phase3_levers") or []
        )
        reason["untried_5beta_reapply"] = (
            phase5_signals.get("untried_5beta_reapply") or []
        )
        reason["phase5_iteration_count"] = phase5_signals.get(
            "phase5_iteration_count", 0
        )
        reason["iteration_count_total"] = phase5_signals.get(
            "iteration_count_total", len(rows)
        )
    return reason


def _finalize_outputs(
    output_dir: Path,
    rows: List[dict],
    best: Optional[dict],
    ranks: Dict[int, Any],
    costs: Dict[int, Dict[str, Any]],
    ranking_warnings: List[str],
    *,
    force: bool = False,
    finalize_reason: Optional[Dict[str, Any]] = None,
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
        finalize_reason=finalize_reason,
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
    parser.add_argument(
        "--force-finalize",
        action="store_true",
        help="Override the premature-finalize guard. By default, --finalize "
        "during phase-5-agent-driven (target NOT met, no plateau) is HARD "
        "REJECTED with exit code 1 — the script refuses to write "
        "outputs/best/ or outputs/final_report.md, because the most "
        "common cause is an agent giving up before the user's iteration "
        "budget has been reached. Pass --force-finalize to confirm that "
        "the early stop is intentional; the resulting final_report.md "
        "will document the coverage gaps that were skipped.",
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

    # Decide whether the requested finalize is premature BEFORE writing
    # anything. "Premature" means: --finalize was requested, but the state
    # machine itself hasn't put us in phase-4 (target reached / plateau);
    # the agent is voluntarily ending in phase-5 (or earlier) without
    # explicit --force-finalize opt-in. This is the failure mode that
    # produced example_quantize_mobilenetv2_bad_esp32s3 (12-of-20) and
    # example_quantize_mobilenetv2_esp32p4_tmp (18-of-30 + 21-of-30). The
    # previous attempt — printing a warning AFTER --finalize had already
    # written best/ + final_report.md — was a soft signal that the agent
    # ignored both times. The hard reject below makes the guardrail load-
    # bearing: no output is created, exit code is non-zero, and the agent
    # must either keep iterating or pass --force-finalize to confirm.
    target_val = _target_metric(rows)
    direction = (best or {}).get("metric_direction", "max") if best else "max"
    target_met = _target_reached(rows, target_val, direction)
    plateau_reached = _plateau_detected(rows, best)
    premature = (
        args.finalize
        and not auto_finalize_target
        and phase == "phase-5-agent-driven"
        and not target_met
        and not plateau_reached
        and not args.force_finalize
    )

    if premature:
        signals = _phase5_signals_for(rows, best)
        untried_p5 = signals.get("untried_phase5_patterns") or []
        untried_p3 = signals.get("untried_phase3_levers") or []
        p5_iters = signals.get("phase5_iteration_count", 0)
        total_iters = signals.get("iteration_count_total", len(rows))
        best_value = (best or {}).get("primary_value")
        gap_msg = ""
        if isinstance(target_val, (int, float)) and isinstance(
            best_value, (int, float)
        ):
            gap = (
                (target_val - best_value)
                if direction == "max"
                else (best_value - target_val)
            )
            gap_msg = f" (gap to target: {abs(gap):+.4f})"
        pattern_label = {
            "5alpha": "5alpha STACK",
            "5beta": "5beta CROSS-POLLINATE",
            "5gamma": "5gamma ABLATE",
            "5delta": "5delta DIVE-INTO-ARTIFACTS",
        }
        print()
        print("===============================================================")
        print(
            "[compare] PREMATURE --finalize REJECTED " "(phase=phase-5-agent-driven)."
        )
        print("===============================================================")
        print(f"  - target_metric NOT met{gap_msg}; NOT a plateau.")
        print(
            f"  - iterations so far: {total_iters} total " f"({p5_iters} in Phase 5)."
        )
        if untried_p5:
            print(
                "  - untried Phase 5 patterns: "
                + ", ".join(pattern_label.get(k, k) for k in untried_p5)
            )
        else:
            print("  - untried Phase 5 patterns: (none)")
        if untried_p3:
            print("  - untried Phase 3 linear-order levers: " + ", ".join(untried_p3))
        else:
            print("  - untried Phase 3 linear-order levers: (none)")
        reapply = signals.get("untried_5beta_reapply") or []
        if reapply:
            print(
                "  - untried 5\u03b2-reapply targets (re-run on current "
                "deepest stack): " + ", ".join(reapply)
            )
        print()
        print(
            "  outputs/best/ and outputs/final_report.md were NOT written. "
            "Pick ONE of:"
        )
        print(
            "    (a) Re-run WITHOUT --finalize and iterate again. The "
            "script auto-finalizes when target_metric is hit or plateau is "
            "detected; otherwise it emits a Phase 5 hint with the untried "
            "lists above as concrete targets."
        )
        print(
            "    (b) Pass --force-finalize alongside --finalize to confirm "
            "intentional early stop. The gap and untried lists above will "
            "be recorded in final_report.md so the user can see what was "
            "skipped."
        )
        print("===============================================================")

        # Still emit comparison.json so the agent / user can read the
        # phase-5 hint without re-running. But do NOT call
        # _finalize_outputs and do NOT touch outputs/best/.
        next_step_hint: Dict[str, Any] = {
            "phase": phase,
            "advice": hint,
            "auto_finalized": False,
            "premature_finalize_rejected": True,
            "phase5_signals": signals,
        }
        summary: Dict[str, Any] = {
            "iterations": rows,
            "best_iteration": best,
            "iteration_ranks": {str(k): v for k, v in ranks.items()},
            "runtime_cost_classification": {str(k): v for k, v in costs.items()},
            "ranking_warnings": ranking_warnings,
            "early_finalize_command": early_finalize_cmd,
            "next_step_hint": next_step_hint,
        }
        out_path = output_dir / "comparison.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        return 1

    # Compute the finalize reason BEFORE writing — it's persisted to
    # comparison.json and embedded into final_report.md (B3). The decision
    # order mirrors the routing precedence: auto-target > auto-plateau >
    # force-finalize-phase5 > manual-finalize-phase4 > manual-finalize-pre-
    # phase5. The "None" case (no finalize at all) leaves the field absent.
    finalize_reason: Optional[Dict[str, Any]] = None
    if auto_finalize_target and target_met:
        finalize_reason = _build_finalize_reason(
            "target_reached", rows, best, target_val, direction
        )
    elif auto_finalize_target and plateau_reached:
        finalize_reason = _build_finalize_reason(
            "plateau", rows, best, target_val, direction
        )
    elif args.finalize and args.force_finalize and phase == "phase-5-agent-driven":
        finalize_reason = _build_finalize_reason(
            "force_finalize_phase5",
            rows,
            best,
            target_val,
            direction,
            phase5_signals=_phase5_signals_for(rows, best),
        )
    elif args.finalize and phase == "phase-4-final-report":
        finalize_reason = _build_finalize_reason(
            "manual_finalize_phase4", rows, best, target_val, direction
        )
    elif args.finalize:
        finalize_reason = _build_finalize_reason(
            "manual_finalize_pre_phase5", rows, best, target_val, direction
        )

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
            finalize_reason=finalize_reason,
        )

    next_step_hint: Dict[str, Any] = {
        "phase": phase,
        "advice": hint,
        "auto_finalized": finalize_result is not None,
    }
    if finalize_reason is not None:
        next_step_hint["finalize_reason"] = finalize_reason
    # Phase 5 carries structured signals so agents (and humans inspecting
    # comparison.json) can see the same history summary the advice text
    # paraphrases. The field is omitted in other phases to keep the file
    # small.
    if phase == "phase-5-agent-driven":
        next_step_hint["phase5_signals"] = _phase5_signals_for(rows, best)

    summary = {
        "iterations": rows,
        "best_iteration": best,
        "iteration_ranks": {str(k): v for k, v in ranks.items()},
        "runtime_cost_classification": {str(k): v for k, v in costs.items()},
        "ranking_warnings": ranking_warnings,
        "early_finalize_command": early_finalize_cmd,
        "next_step_hint": next_step_hint,
    }
    out_path = output_dir / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    _print_table(
        rows, next_hint=(phase, hint), early_finalize_command=early_finalize_cmd
    )

    if finalize_result is not None:
        if auto_finalize_target:
            trigger = "phase-4 detected"
        elif args.finalize:
            trigger = "--finalize requested"
        else:
            trigger = "finalize"
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
