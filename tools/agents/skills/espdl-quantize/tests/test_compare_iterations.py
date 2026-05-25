"""Dry-run unit tests for the compare_iterations.py state machine.

No real quantization runs — every test fabricates the iteration row dicts that
``_gather`` would have produced and asserts on the (phase, hint) tuple coming
out of ``_suggest_next``. The tests cover:

* Phase-4 finalize triggers (target reached, plateau, Phase-3 cap, all linear
  levers tried)
* TQT three-stage escalation (3a-1 → 3a-2 → 3a-3 path 1)
* 3a-3 path-1 unstable-fallback detection
* 3a-3 path-2 gap-shrink-after-convergence eligibility check
* Speed-priority lever reorder (3g before 3e/3f)
* Phase-2 cartesian product completeness check (calib_algorithm coverage)

Run via ``python -m pytest "$SKILL_DIR/tests/test_compare_iterations.py" -v``;
no esp_ppq dependency required.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

import compare_iterations as ci  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — build the row dicts that compare_iterations.py works on.
# ---------------------------------------------------------------------------


def _row(
    iteration_id: int,
    primary_value: float,
    setting: dict,
    *,
    target_metric: float | None = None,
    deploy_runtime_priority: str = "balanced",
    metric_direction: str = "max",
    top_5_error_layers: list[dict] | None = None,
    rationale: str = "",
) -> dict:
    return {
        "iteration_id": iteration_id,
        "dir": f"/tmp/iter_{iteration_id}",
        "primary_metric": "top1",
        "primary_value": primary_value,
        "metric_direction": metric_direction,
        "elapsed_seconds": 100.0,
        "rationale": rationale,
        "setting": setting,
        "top_5_error_layers": top_5_error_layers or [],
        "warnings": [],
        "extra_metrics": {"top1": primary_value},
        "target_metric": target_metric,
        "deploy_runtime_priority": deploy_runtime_priority,
        "non_computing_hot_ops_count": 0,
        "graphwise_jumps_count": 0,
    }


def _baseline_setting() -> dict:
    """A Phase-1 baseline setting (no TQT, kl calibration)."""
    return {"calib_algorithm": "kl"}


def _phase2_setting(calib: str) -> dict:
    """A Phase-2 leg setting: calib + default-schedule TQT."""
    return {
        "calib_algorithm": calib,
        "tqt_optimization": {
            "enabled": True,
            "lr": 1e-5,
            "steps": 500,
            "block_size": 4,
            "is_scale_trainable": True,
            "gamma": 0.0,
            "int_lambda": 0.0,
            "collecting_device": "cuda",
        },
    }


def _tqt_3a1_setting(calib: str = "kl") -> dict:
    s = _phase2_setting(calib)
    s["tqt_optimization"]["steps"] = 1000
    return s


def _tqt_3a2_setting(calib: str = "kl") -> dict:
    s = _phase2_setting(calib)
    s["tqt_optimization"]["steps"] = 2000
    s["tqt_optimization"]["lr"] = 5e-5
    return s


def _tqt_3a3_setting(calib: str = "kl") -> dict:
    s = _phase2_setting(calib)
    s["tqt_optimization"]["block_size"] = 2
    return s


def _three_phase2_iters() -> list[dict]:
    """Iter_0 baseline + iter_1/2/3 covering kl/mse/percentile + TQT(default)."""
    return [
        _row(0, 70.00, _baseline_setting()),
        _row(1, 70.50, _phase2_setting("kl")),
        _row(2, 70.30, _phase2_setting("mse")),
        _row(3, 70.80, _phase2_setting("percentile")),
    ]


# ---------------------------------------------------------------------------
# Phase 1 / Phase 2
# ---------------------------------------------------------------------------


def test_phase1_when_no_iterations():
    rows = []
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase == "phase-1-baseline"
    assert "iter_0" in hint and "--baseline" in hint


def test_phase2_when_legs_missing():
    rows = [_row(0, 70.0, _baseline_setting()), _row(1, 70.5, _phase2_setting("kl"))]
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase == "phase-2-calib-tqt-sweep"
    # First missing leg in declared order is `mse`.
    assert '"calib_algorithm": "mse"' in hint
    assert '"steps": 500' in hint  # default schedule preserved


def test_phase2_strict_default_schedule_matters():
    """If the iteration changed the TQT schedule, the leg is NOT covered."""
    rows = [_row(0, 70.0, _baseline_setting())]
    s = _phase2_setting("kl")
    s["tqt_optimization"]["steps"] = 1000  # no longer default → not Phase-2
    rows.append(_row(1, 70.5, s))
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase == "phase-2-calib-tqt-sweep"
    assert (
        '"calib_algorithm": "kl"' in hint
    )  # state machine still wants the default-kl leg


# ---------------------------------------------------------------------------
# Phase 4 — finalize triggers
# ---------------------------------------------------------------------------


def test_phase4_target_reached():
    rows = _three_phase2_iters()
    # Pretend iter_3 hit the target metric (set on every iter via QUANT_CONFIG).
    for r in rows:
        r["target_metric"] = 70.5
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase == "phase-4-final-report"
    assert "target_metric reached" in hint


def test_phase4_target_not_reached_continues_phase3():
    rows = _three_phase2_iters()
    for r in rows:
        r["target_metric"] = 99.0  # unreachable
    phase, _ = ci._suggest_next(rows, ci._best(rows))
    assert phase in ("phase-3-residual", "phase-3-pivot")


def test_phase4_plateau_detected():
    rows = _three_phase2_iters()
    # Three consecutive iters within 0.1% of best → plateau (_PLATEAU_WINDOW=3).
    best_value = max(r["primary_value"] for r in rows)
    rows.append(
        _row(4, best_value - best_value * 0.0005, _tqt_3a1_setting("percentile"))
    )
    rows.append(
        _row(5, best_value - best_value * 0.0008, _tqt_3a2_setting("percentile"))
    )
    rows.append(_row(6, best_value - best_value * 0.0006, _tqt_3a2_setting("kl")))
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase == "phase-4-final-report"
    assert "plateau" in hint
    # Hint must cite the window size matching the constant.
    assert "3 iterations" in hint


def test_phase3_cap_routes_to_phase5_when_target_not_met():
    """Phase-3 cap reached without hitting target → hand control to Phase 5
    instead of finalising. This is the regression test for the old
    behaviour (which forced phase-4 here even when the user could still
    benefit from further exploration)."""
    rows = _three_phase2_iters()
    # 5 Phase-3 iterations after the cartesian product → cap.
    for i, lever in enumerate(["3a-1", "3a-2", "3b", "3c", "3d"], start=4):
        s = _phase2_setting("percentile")
        if lever == "3a-1":
            s["tqt_optimization"]["steps"] = 1000
        elif lever == "3a-2":
            s["tqt_optimization"]["steps"] = 2000
            s["tqt_optimization"]["lr"] = 5e-5
        elif lever == "3b":
            s["bias_correct"] = {"enabled": True}
        elif lever == "3c":
            s["fusion_alignment"] = {"align_elementwise_to": "Align to Large"}
        elif lever == "3d":
            s["equalization"] = {"enabled": True}
        # target_metric must be on every row; pick something unreachable so
        # the cap branch fires (target-reached short-circuits to phase-4).
        rows.append(_row(i, 70.7, s, target_metric=99.0))
    for r in rows:
        r["target_metric"] = 99.0
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase == "phase-5-agent-driven"
    assert "Phase-3 budget reached" in hint
    # Phase 5 hint is meta-guidance, not a template.
    assert '"iteration_id"' not in hint  # no setting.json template embedded
    assert "Starting points to consider" in hint


def test_phase3_cap_still_routes_to_phase4_when_target_reached():
    """If target_metric is met by the time the Phase-3 cap fires, target-
    reached short-circuits and we go straight to phase-4."""
    rows = _three_phase2_iters()
    for i, lever in enumerate(["3a-1", "3a-2", "3b", "3c", "3d"], start=4):
        s = _phase2_setting("percentile")
        if lever == "3a-1":
            s["tqt_optimization"]["steps"] = 1000
        elif lever == "3a-2":
            s["tqt_optimization"]["steps"] = 2000
            s["tqt_optimization"]["lr"] = 5e-5
        elif lever == "3b":
            s["bias_correct"] = {"enabled": True}
        elif lever == "3c":
            s["fusion_alignment"] = {"align_elementwise_to": "Align to Large"}
        elif lever == "3d":
            s["equalization"] = {"enabled": True}
        rows.append(_row(i, 70.7, s))
    for r in rows:
        r["target_metric"] = 70.5  # reachable
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase == "phase-4-final-report"
    assert "target_metric reached" in hint


# ---------------------------------------------------------------------------
# TQT three-stage escalation
# ---------------------------------------------------------------------------


def test_phase3_first_lever_is_3a1():
    rows = _three_phase2_iters()
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase in ("phase-3-residual", "phase-3-pivot")
    assert "3a-1" in hint
    assert '"steps": 1000' in hint
    assert '"lr": 1e-05' in hint


def test_phase3_3a2_after_3a1_succeeded():
    rows = _three_phase2_iters()
    # iter_4 (3a-1) improved → it becomes best.
    rows.append(_row(4, 71.20, _tqt_3a1_setting("percentile")))
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase in ("phase-3-residual", "phase-3-pivot")
    assert "3a-2" in hint
    assert '"steps": 2000' in hint
    assert '"lr": 5e-05' in hint


def test_phase3_3a3_path1_unstable_fallback():
    rows = _three_phase2_iters()
    # iter_4 (3a-1) at 71.20 (best). iter_5 (3a-2) regresses by ~0.1% AND
    # introduces a new layer into top-5 → path 1 fires.
    rows.append(
        _row(
            4,
            71.20,
            _tqt_3a1_setting("percentile"),
            top_5_error_layers=[
                {"op": "/conv1", "snr": 0.05},
                {"op": "/conv2", "snr": 0.04},
            ],
        )
    )
    rows.append(
        _row(
            5,
            71.13,
            _tqt_3a2_setting("percentile"),  # 71.20 → 71.13, ~0.1% drop
            top_5_error_layers=[
                {"op": "/conv1", "snr": 0.06},
                {"op": "/conv_NEW", "snr": 0.05},
            ],  # /conv_NEW is new
        )
    )
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase in ("phase-3-residual", "phase-3-pivot")
    assert "3a-3" in hint
    assert "PATH 1" in hint
    assert '"block_size": 2' in hint


def test_phase3_3a3_path1_does_not_fire_when_no_new_hot_layer():
    rows = _three_phase2_iters()
    rows.append(
        _row(
            4,
            71.20,
            _tqt_3a1_setting("percentile"),
            top_5_error_layers=[{"op": "/conv1", "snr": 0.05}],
        )
    )
    rows.append(
        _row(
            5,
            71.13,
            _tqt_3a2_setting("percentile"),
            top_5_error_layers=[{"op": "/conv1", "snr": 0.06}],  # same op
        )
    )
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    # 3a-1 + 3a-2 both tried; 3b is next in linear order.
    assert "3b" in hint or "3a-3" not in hint


def test_phase3_3a3_path1_does_not_fire_when_regression_too_large():
    rows = _three_phase2_iters()
    rows.append(
        _row(
            4,
            71.20,
            _tqt_3a1_setting("percentile"),
            top_5_error_layers=[{"op": "/conv1", "snr": 0.05}],
        )
    )
    # Drop > 0.5% relative → too noisy to be a stability signal, route per linear order.
    rows.append(
        _row(
            5,
            70.50,
            _tqt_3a2_setting("percentile"),
            top_5_error_layers=[{"op": "/conv_NEW", "snr": 0.05}],
        )
    )
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert "PATH 1" not in hint


def test_phase3_3a3_path2_eligible_when_no_structural_match():
    """Path 2 hint is *embedded as an escape clause* on the linear-order lever
    when 3a-1/3a-2 both succeeded and 3b/3c/3d are all not yet tried."""
    rows = _three_phase2_iters()
    # 3a-1 succeeded (pushed best), 3a-2 succeeded (further pushed best).
    rows.append(_row(4, 71.20, _tqt_3a1_setting("percentile")))
    rows.append(_row(5, 71.50, _tqt_3a2_setting("percentile")))
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    # Default linear order's next entry is 3b, but the path-2 escape clause
    # must be present in the hint.
    assert "3b" in hint
    assert "PATH 2" in hint
    assert "3a-3" in hint


# ---------------------------------------------------------------------------
# Speed-priority reorder
# ---------------------------------------------------------------------------


def test_speed_priority_order_constant():
    assert ci._phase3_order_for("speed") == ci._PHASE3_ORDER_SPEED
    assert ci._phase3_order_for("balanced") == ci._PHASE3_ORDER_BALANCED


def test_speed_priority_picks_3g_before_3e():
    """Under speed priority, after 3a-1/3a-2/3b/3c/3d, the next lever is 3g
    (zero on-device cost) rather than 3e (Tier B int16 promotion)."""
    rows = _three_phase2_iters()
    base = _phase2_setting("percentile")
    # Mark all 3a-1/3a-2/3b/3c/3d as tried (stay roughly flat, no plateau).
    s = _tqt_3a1_setting("percentile")
    rows.append(_row(4, 71.10, s, deploy_runtime_priority="speed"))
    s = _tqt_3a2_setting("percentile")
    rows.append(_row(5, 70.85, s, deploy_runtime_priority="speed"))
    s = dict(base)
    s["bias_correct"] = {"enabled": True}
    rows.append(_row(6, 71.05, s, deploy_runtime_priority="speed"))
    s = dict(base)
    s["fusion_alignment"] = {"align_elementwise_to": "Align to Large"}
    rows.append(_row(7, 70.95, s, deploy_runtime_priority="speed"))
    s = dict(base)
    s["equalization"] = {"enabled": True}
    rows.append(_row(8, 71.00, s, deploy_runtime_priority="speed"))
    # Override the cap for this test by stubbing — but the function checks
    # iter_id >= 4. We have iter_4..8 = 5 → cap reached. Bump cap temporarily.
    saved_cap = ci._PHASE3_CAP
    saved_window = ci._PLATEAU_WINDOW
    ci._PHASE3_CAP = 100  # disable cap to expose lever-order behaviour
    ci._PLATEAU_WINDOW = 100  # disable plateau too
    try:
        phase, hint = ci._suggest_next(rows, ci._best(rows))
    finally:
        ci._PHASE3_CAP = saved_cap
        ci._PLATEAU_WINDOW = saved_window
    assert "3g" in hint
    assert "blockwise_reconstruction" in hint
    # 3g no longer disables TQT — engine runs TQT + blockwise sequentially.
    assert "AdaroundPass after TrainedQuantizationThresholdPass" in hint


def test_balanced_priority_picks_3e_before_3g():
    """Mirror of test_speed_priority_picks_3g_before_3e for balanced default."""
    rows = _three_phase2_iters()
    base = _phase2_setting("percentile")
    s = _tqt_3a1_setting("percentile")
    rows.append(_row(4, 71.10, s))
    s = _tqt_3a2_setting("percentile")
    rows.append(_row(5, 70.85, s))
    s = dict(base)
    s["bias_correct"] = {"enabled": True}
    rows.append(_row(6, 71.05, s))
    s = dict(base)
    s["fusion_alignment"] = {"align_elementwise_to": "Align to Large"}
    rows.append(_row(7, 70.95, s))
    s = dict(base)
    s["equalization"] = {"enabled": True}
    rows.append(_row(8, 71.00, s))
    saved_cap = ci._PHASE3_CAP
    saved_window = ci._PLATEAU_WINDOW
    ci._PHASE3_CAP = 100
    ci._PLATEAU_WINDOW = 100
    try:
        phase, hint = ci._suggest_next(rows, ci._best(rows))
    finally:
        ci._PHASE3_CAP = saved_cap
        ci._PLATEAU_WINDOW = saved_window
    assert "3e" in hint
    assert "dispatching_table" in hint


# ---------------------------------------------------------------------------
# All-levers-tried finalize
# ---------------------------------------------------------------------------


def test_all_levers_tried_routes_to_phase5_when_target_not_met():
    """All eight linear levers tried without hitting target → phase-5 (the
    state machine yields, the agent continues). Mirrors
    test_phase3_cap_routes_to_phase5_when_target_not_met for the
    "exhausted lever list" trigger path."""
    rows = _three_phase2_iters()
    base = _phase2_setting("percentile")
    levers = {
        "3a-1": _tqt_3a1_setting("percentile"),
        "3a-2": _tqt_3a2_setting("percentile"),
        "3b": {**base, "bias_correct": {"enabled": True}},
        "3c": {**base, "fusion_alignment": {"align_elementwise_to": "Align to Large"}},
        "3d": {**base, "equalization": {"enabled": True}},
        "3e": {**base, "dispatching_table": [{"op_name": "/x", "bits": 16}]},
        "3f": {**base, "weight_split": {"enabled": True}},
        # 3g adds blockwise on top of best (mutex with TQT was lifted; engine
        # runs AdaroundPass after TrainedQuantizationThresholdPass).
        "3g": {**base, "blockwise_reconstruction": {"enabled": True}},
    }
    for i, lever in enumerate(levers):
        rows.append(_row(4 + i, 71.0, levers[lever], target_metric=99.0))
    for r in rows:
        r["target_metric"] = 99.0  # unreachable
    saved_cap = ci._PHASE3_CAP
    saved_window = ci._PLATEAU_WINDOW
    ci._PHASE3_CAP = 100
    ci._PLATEAU_WINDOW = 100
    try:
        phase, hint = ci._suggest_next(rows, ci._best(rows))
    finally:
        ci._PHASE3_CAP = saved_cap
        ci._PLATEAU_WINDOW = saved_window
    assert phase == "phase-5-agent-driven"
    assert "all linear-order Phase-3 levers tried" in hint


def test_all_levers_tried_still_routes_to_phase4_when_target_reached():
    """Symmetric to the cap path — when target is reached, phase-4 still wins
    even if the linear list is also exhausted."""
    rows = _three_phase2_iters()
    base = _phase2_setting("percentile")
    levers = {
        "3a-1": _tqt_3a1_setting("percentile"),
        "3a-2": _tqt_3a2_setting("percentile"),
        "3b": {**base, "bias_correct": {"enabled": True}},
        "3c": {**base, "fusion_alignment": {"align_elementwise_to": "Align to Large"}},
        "3d": {**base, "equalization": {"enabled": True}},
        "3e": {**base, "dispatching_table": [{"op_name": "/x", "bits": 16}]},
        "3f": {**base, "weight_split": {"enabled": True}},
        "3g": {**base, "blockwise_reconstruction": {"enabled": True}},
    }
    for i, lever in enumerate(levers):
        rows.append(_row(4 + i, 71.0, levers[lever]))
    for r in rows:
        r["target_metric"] = 70.5  # reachable
    saved_cap = ci._PHASE3_CAP
    saved_window = ci._PLATEAU_WINDOW
    ci._PHASE3_CAP = 100
    ci._PLATEAU_WINDOW = 100
    try:
        phase, hint = ci._suggest_next(rows, ci._best(rows))
    finally:
        ci._PHASE3_CAP = saved_cap
        ci._PLATEAU_WINDOW = saved_window
    assert phase == "phase-4-final-report"
    assert "target_metric reached" in hint


# ---------------------------------------------------------------------------
# Direct helper coverage
# ---------------------------------------------------------------------------


def test_lever_tried_distinguishes_3a_variants():
    rows = [
        _row(0, 70.0, _baseline_setting()),
        _row(1, 70.5, _phase2_setting("kl")),
        _row(2, 70.5, _tqt_3a1_setting("kl")),
    ]
    assert ci._lever_tried(rows, "3a-1")
    assert not ci._lever_tried(rows, "3a-2")
    assert not ci._lever_tried(rows, "3a-3")


def test_lever_tried_3a3_block_size_2():
    rows = [_row(0, 70.0, _tqt_3a3_setting("kl"))]
    assert ci._lever_tried(rows, "3a-3")
    assert not ci._lever_tried(rows, "3a-1")
    assert not ci._lever_tried(rows, "3a-2")


def test_target_reached_with_min_direction():
    rows = [
        _row(0, 0.30, _baseline_setting(), metric_direction="min", target_metric=0.20)
    ]
    rows[0]["primary_value"] = 0.15  # met (lower is better)
    assert ci._target_reached(rows, 0.20, "min")
    assert not ci._target_reached(rows, 0.10, "min")


# ---------------------------------------------------------------------------
# _classify_runtime_cost (new)
# ---------------------------------------------------------------------------


def test_classify_runtime_cost_empty_or_default():
    assert ci._classify_runtime_cost({})["affects_speed"] is False
    assert ci._classify_runtime_cost(_baseline_setting())["affects_speed"] is False
    assert ci._classify_runtime_cost(_phase2_setting("kl"))["affects_speed"] is False


def test_classify_runtime_cost_int16_promotion_flags_yes():
    setting = {
        **_phase2_setting("percentile"),
        "dispatching_table": [
            {"op": "/conv1", "bits": 16},
            {"op": "/conv2", "bits": 16},
        ],
    }
    cost = ci._classify_runtime_cost(setting)
    assert cost["affects_speed"] is True
    assert "int16 promotion on 2 op(s)" in cost["label"]
    assert "/conv1" in cost["label"]


def test_classify_runtime_cost_int8_only_dispatch_does_not_count():
    """Dispatching ops to int8 (or any non-16 bits) is just a routing change,
    no runtime cost. Only int16 promotion incurs ~2x cycles."""
    setting = {
        **_phase2_setting("kl"),
        "dispatching_table": [{"op": "/conv1", "bits": 8}],
    }
    cost = ci._classify_runtime_cost(setting)
    assert cost["affects_speed"] is False
    assert cost["label"] == "No"


def test_classify_runtime_cost_weight_split_flags_yes():
    setting = {
        **_phase2_setting("kl"),
        "weight_split": {
            "enabled": True,
            "interested_layers": ["/conv_outlier", "/conv_other"],
        },
    }
    cost = ci._classify_runtime_cost(setting)
    assert cost["affects_speed"] is True
    assert "weight_split on 2 layer(s)" in cost["label"]


def test_classify_runtime_cost_both_int16_and_weight_split():
    setting = {
        **_phase2_setting("kl"),
        "dispatching_table": [{"op": "/conv1", "bits": 16}],
        "weight_split": {
            "enabled": True,
            "interested_layers": ["/conv_outlier"],
        },
    }
    cost = ci._classify_runtime_cost(setting)
    assert cost["affects_speed"] is True
    # Both reasons appear in the label, joined with ";".
    assert "int16 promotion" in cost["label"]
    assert "weight_split" in cost["label"]
    assert len(cost["reasons"]) == 2


# ---------------------------------------------------------------------------
# _compute_ranks (new) — covers the determinism contract
# ---------------------------------------------------------------------------


def test_compute_ranks_idempotent_under_re_invocation():
    """Same rows in → same ranks out, every time. Rank-3 user pain point."""
    rows = [
        _row(0, 70.0, _baseline_setting()),
        _row(1, 71.5, _phase2_setting("kl")),
        _row(2, 70.5, _phase2_setting("mse")),
        _row(3, 72.0, _phase2_setting("percentile")),
    ]
    ranks_a, _ = ci._compute_ranks(rows)
    ranks_b, _ = ci._compute_ranks(rows)
    assert ranks_a == ranks_b


def test_compute_ranks_recompute_after_appending_iterations():
    """Appending new iters cannot reverse the relative order of pre-existing
    iters. This is the property the user's previous bad experience violated."""
    base_rows = [
        _row(0, 70.0, _baseline_setting()),
        _row(1, 71.5, _phase2_setting("kl")),
        _row(2, 70.5, _phase2_setting("mse")),
        _row(3, 71.0, _phase2_setting("percentile")),
    ]
    ranks_before, _ = ci._compute_ranks(base_rows)
    # Append new iters; iter_4 is the new rank-1.
    extended = base_rows + [
        _row(4, 73.0, _tqt_3a1_setting("percentile")),
        _row(5, 70.2, _tqt_3a2_setting("percentile")),
    ]
    ranks_after, _ = ci._compute_ranks(extended)
    # Relative order of all (i, j) pairs from base_rows must be preserved.
    base_ids = [r["iteration_id"] for r in base_rows]
    for i in base_ids:
        for j in base_ids:
            if i == j:
                continue
            before_lt = ranks_before[i] < ranks_before[j]
            after_lt = ranks_after[i] < ranks_after[j]
            before_eq = ranks_before[i] == ranks_before[j]
            after_eq = ranks_after[i] == ranks_after[j]
            assert before_lt == after_lt and before_eq == after_eq, (
                f"Rank order between iter_{i} and iter_{j} flipped when "
                f"appending new iterations! before: {ranks_before}; "
                f"after: {ranks_after}"
            )
    # iter_4 must take rank 1 (highest score 73.0).
    assert ranks_after[4] == 1
    # All values are distinct, so dense ranking goes 1..6 across all iters.
    finite_ids = [k for k, v in ranks_after.items() if isinstance(v, int)]
    assert sorted(ranks_after[k] for k in finite_ids) == [1, 2, 3, 4, 5, 6]


def test_compute_ranks_recompute_after_overwriting_iter():
    """User re-runs iter_3 with a different setting; old value gone, new value
    enters ranking."""
    rows_v1 = [
        _row(0, 70.0, _baseline_setting()),
        _row(3, 71.0, _phase2_setting("percentile")),
    ]
    ranks_v1, _ = ci._compute_ranks(rows_v1)
    assert ranks_v1[3] == 1

    rows_v2 = [
        _row(0, 70.0, _baseline_setting()),
        _row(3, 69.0, _phase2_setting("percentile")),  # re-ran, regressed
    ]
    ranks_v2, _ = ci._compute_ranks(rows_v2)
    # iter_0 should now be the best, iter_3 second.
    assert ranks_v2[0] == 1
    assert ranks_v2[3] == 2


def test_compute_ranks_dense_with_ties():
    """Three iters tie for top → all rank 1; next iter is rank 2 (dense, not
    standard ranking)."""
    rows = [
        _row(0, 70.0, _baseline_setting()),
        _row(1, 71.5, _phase2_setting("kl")),
        _row(2, 71.5, _phase2_setting("mse")),
        _row(3, 71.5, _phase2_setting("percentile")),
        _row(4, 71.0, _tqt_3a1_setting("percentile")),
    ]
    ranks, _ = ci._compute_ranks(rows)
    assert ranks[1] == 1
    assert ranks[2] == 1
    assert ranks[3] == 1
    assert ranks[4] == 2  # dense — rank advances by 1, not by 3
    assert ranks[0] == 3


def test_compute_ranks_excludes_failed_iterations():
    """None / NaN / inf / "_error" / bool → rank `-`, do not occupy slots."""
    rows = [
        _row(0, 70.0, _baseline_setting()),
        _row(1, float("nan"), _phase2_setting("kl")),
        _row(2, 71.0, _phase2_setting("mse")),
        _row(3, None, _phase2_setting("percentile")),  # type: ignore[arg-type]
        _row(4, "_error", _tqt_3a1_setting("percentile")),  # type: ignore[arg-type]
        _row(5, True, _tqt_3a2_setting("percentile")),  # type: ignore[arg-type]
    ]
    ranks, _ = ci._compute_ranks(rows)
    assert ranks[2] == 1
    assert ranks[0] == 2
    assert ranks[1] == "-"
    assert ranks[3] == "-"
    assert ranks[4] == "-"
    assert ranks[5] == "-"


def test_compute_ranks_min_direction():
    """metric_direction='min' (loss / error metrics) → smaller is better."""
    rows = [
        _row(0, 0.30, _baseline_setting(), metric_direction="min"),
        _row(1, 0.10, _phase2_setting("kl"), metric_direction="min"),
        _row(2, 0.50, _phase2_setting("mse"), metric_direction="min"),
    ]
    ranks, _ = ci._compute_ranks(rows)
    assert ranks[1] == 1
    assert ranks[0] == 2
    assert ranks[2] == 3


def test_compute_ranks_warns_on_metric_mismatch():
    rows = [
        _row(0, 70.0, _baseline_setting()),
        _row(1, 71.0, _phase2_setting("kl")),
    ]
    rows[1]["primary_metric"] = "top5"  # different metric mid-stream
    ranks, warns = ci._compute_ranks(rows)
    assert any("primary_metric" in w for w in warns)
    # Ranks still produced from the numeric values.
    assert ranks[1] == 1


def test_compute_ranks_warns_on_direction_mismatch():
    rows = [
        _row(0, 70.0, _baseline_setting(), metric_direction="max"),
        _row(1, 71.0, _phase2_setting("kl"), metric_direction="min"),
    ]
    ranks, warns = ci._compute_ranks(rows)
    assert any("metric_direction" in w for w in warns)
    # Ranks computed using the first finite row's direction (max here).
    assert ranks[1] == 1
    assert ranks[0] == 2


# ---------------------------------------------------------------------------
# Marker / final-report rendering (new)
# ---------------------------------------------------------------------------


def test_render_final_report_contains_marker_and_columns(tmp_path):
    rows = _three_phase2_iters()
    best = ci._best(rows)
    ranks, _ = ci._compute_ranks(rows)
    costs = {
        r["iteration_id"]: ci._classify_runtime_cost(r.get("setting") or {})
        for r in rows
    }
    md = ci._render_final_report(
        rows=rows,
        best=best,
        target=72.0,
        model_name="DummyNet",
        target_chip="esp32p4",
        primary_metric="top1",
        metric_direction="max",
        ranks=ranks,
        costs=costs,
        ranking_warnings=[],
    )
    assert md.startswith("<!-- auto-generated by compare_iterations.py at ")
    assert "# Final Report: DummyNet on esp32p4" in md
    # Iteration history must include the new columns.
    assert "rank" in md
    assert "affects inference speed" in md
    # Best section + Python snippet placeholders must exist.
    assert "## Best setting" in md
    assert "## Python snippet" in md


def test_render_iteration_history_table_columns():
    rows = _three_phase2_iters()
    best = ci._best(rows)
    ranks, _ = ci._compute_ranks(rows)
    costs = {
        r["iteration_id"]: ci._classify_runtime_cost(r.get("setting") or {})
        for r in rows
    }
    md = ci._render_iteration_history_md(rows, ranks, costs, best, "top1")
    header_line = md.splitlines()[0]
    for col in (
        "iter",
        "method changed",
        "top1",
        "delta",
        "outcome",
        "rank",
        "affects inference speed",
    ):
        assert col in header_line
    # Rank values should appear in cells.
    assert "| 1 |" in md or "| 2 |" in md or "| 3 |" in md


# ---------------------------------------------------------------------------
# _finalize_outputs marker semantics (new)
# ---------------------------------------------------------------------------


def _setup_iter_dirs(tmp_path: Path, rows: list[dict]) -> Path:
    """Recreate the on-disk shape that _gather expects: iter_<N>/{iteration_index, setting, metrics}.json."""
    import json

    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        iter_dir = out / f"iter_{r['iteration_id']}"
        iter_dir.mkdir()
        # Update r["dir"] to point at the just-created path (so _best returns
        # a row whose "dir" points at a real folder).
        r["dir"] = str(iter_dir)
        index = {
            "iteration_id": r["iteration_id"],
            "rationale": r.get("rationale", ""),
            "primary_metric": r.get("primary_metric", "top1"),
            "primary_value": r.get("primary_value"),
            "metric_direction": r.get("metric_direction", "max"),
            "target_metric": r.get("target_metric"),
            "deploy_runtime_priority": r.get("deploy_runtime_priority", "balanced"),
            "elapsed_seconds": r.get("elapsed_seconds", 1.0),
            "top_5_error_layers": r.get("top_5_error_layers", []),
            "applied_setting_warnings": [],
            "model_name": "TestNet",
            "target_chip": "esp32p4",
        }
        (iter_dir / "iteration_index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        (iter_dir / "setting.json").write_text(
            json.dumps(
                {
                    "iteration_id": r["iteration_id"],
                    "rationale": r.get("rationale", ""),
                    **(r.get("setting") or {}),
                }
            ),
            encoding="utf-8",
        )
        (iter_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "_primary_key": r.get("primary_metric", "top1"),
                    "_primary_value": r.get("primary_value"),
                    "_used": "evaluate_fast",
                }
            ),
            encoding="utf-8",
        )
        # Empty model file so `_write_best_dir`'s _link_or_copy has a source.
        (iter_dir / "model.espdl").write_bytes(b"\x00")
    return out


def test_finalize_writes_when_missing(tmp_path):
    rows = _three_phase2_iters()
    out = _setup_iter_dirs(tmp_path, rows)
    fresh_rows = ci._gather(out)
    best = ci._best(fresh_rows)
    ranks, warns = ci._compute_ranks(fresh_rows)
    costs = {
        r["iteration_id"]: ci._classify_runtime_cost(r.get("setting") or {})
        for r in fresh_rows
    }
    result = ci._finalize_outputs(out, fresh_rows, best, ranks, costs, warns)
    assert result["report_action"] == "written"
    assert (out / "best" / "model.espdl").exists()
    assert (out / "best" / "setting.json").exists()
    assert (out / "final_report.md").exists()
    body = (out / "final_report.md").read_text(encoding="utf-8")
    assert ci._REPORT_MARKER_PREFIX in body


def test_finalize_preserves_agent_edited_report(tmp_path):
    """When final_report.md exists but lacks our marker, do not overwrite
    unless --force."""
    rows = _three_phase2_iters()
    out = _setup_iter_dirs(tmp_path, rows)
    agent_text = "# Hand-written report by agent\n\nimportant analysis here\n"
    (out / "final_report.md").write_text(agent_text, encoding="utf-8")
    fresh_rows = ci._gather(out)
    best = ci._best(fresh_rows)
    ranks, warns = ci._compute_ranks(fresh_rows)
    costs = {
        r["iteration_id"]: ci._classify_runtime_cost(r.get("setting") or {})
        for r in fresh_rows
    }
    result = ci._finalize_outputs(
        out, fresh_rows, best, ranks, costs, warns, force=False
    )
    assert result["report_action"] == "preserved-by-agent-edit"
    assert (out / "final_report.md").read_text(encoding="utf-8") == agent_text
    # With --force, it gets overwritten.
    result_forced = ci._finalize_outputs(
        out, fresh_rows, best, ranks, costs, warns, force=True
    )
    assert result_forced["report_action"] == "rewritten-forced"
    new_body = (out / "final_report.md").read_text(encoding="utf-8")
    assert ci._REPORT_MARKER_PREFIX in new_body


def test_finalize_overwrites_marker_version(tmp_path):
    """If the existing report carries our marker, finalize refreshes it
    even without --force."""
    rows = _three_phase2_iters()
    out = _setup_iter_dirs(tmp_path, rows)
    fresh_rows = ci._gather(out)
    best = ci._best(fresh_rows)
    ranks, warns = ci._compute_ranks(fresh_rows)
    costs = {
        r["iteration_id"]: ci._classify_runtime_cost(r.get("setting") or {})
        for r in fresh_rows
    }
    # First write: action=written.
    ci._finalize_outputs(out, fresh_rows, best, ranks, costs, warns)
    # Second write: action=rewritten (marker still in place).
    result = ci._finalize_outputs(out, fresh_rows, best, ranks, costs, warns)
    assert result["report_action"] == "rewritten"


# ---------------------------------------------------------------------------
# main() — phase-4 auto-finalize and early-finalize on user budget
# ---------------------------------------------------------------------------


def test_phase4_main_auto_finalizes(tmp_path, capsys):
    """When the state machine declares phase-4, main() auto-finalizes without
    needing --finalize."""
    rows = _three_phase2_iters()
    # Force phase-4 via target_metric reached.
    for r in rows:
        r["target_metric"] = 70.5
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out)])
    assert rc == 0
    assert (out / "best" / "model.espdl").exists()
    assert (out / "final_report.md").exists()
    captured = capsys.readouterr().out
    assert "phase-4 detected" in captured


def test_finalize_with_only_baseline(tmp_path, capsys):
    """Single-iteration session — user might run iter_0 and stop. Files still
    produced via --finalize."""
    rows = [_row(0, 70.0, _baseline_setting())]
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out), "--finalize"])
    assert rc == 0
    assert (out / "best" / "model.espdl").exists()
    body = (out / "final_report.md").read_text(encoding="utf-8")
    assert "Single-iteration session" in body


def test_finalize_during_phase2(tmp_path, capsys):
    """Phase-2 mid-stream — only iter_0 + iter_1, user said 'iterate 2 times'.
    --finalize must still produce both artifacts and comparison.json must
    expose early_finalize_command."""
    rows = [
        _row(0, 70.0, _baseline_setting()),
        _row(1, 70.5, _phase2_setting("kl")),
    ]
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out), "--finalize"])
    assert rc == 0
    assert (out / "best" / "model.espdl").exists()
    assert (out / "final_report.md").exists()
    import json

    cmp = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert "early_finalize_command" in cmp
    assert "--finalize" in cmp["early_finalize_command"]
    assert cmp["next_step_hint"]["auto_finalized"] is True


def test_stdout_tip_always_printed_even_outside_phase4(tmp_path, capsys):
    """Even without --finalize and outside phase-4, stdout must mention the
    early-finalize escape hatch — agents need to know it's there."""
    rows = [
        _row(0, 70.0, _baseline_setting()),
        _row(1, 70.5, _phase2_setting("kl")),
    ]
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "--finalize" in captured
    assert "wrap up at any time" in captured.lower() or "wrap up" in captured.lower()


def test_comparison_json_contains_ranks_and_costs(tmp_path):
    rows = [
        _row(0, 70.0, _baseline_setting()),
        _row(1, 71.0, _phase2_setting("kl")),
        _row(
            2,
            71.5,
            {
                **_phase2_setting("percentile"),
                "dispatching_table": [{"op": "/x", "bits": 16}],
            },
        ),
    ]
    out = _setup_iter_dirs(tmp_path, rows)
    ci.main(["--output-dir", str(out)])
    import json

    cmp = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert "iteration_ranks" in cmp
    assert "runtime_cost_classification" in cmp
    assert cmp["iteration_ranks"]["2"] == 1  # best
    assert cmp["runtime_cost_classification"]["2"]["affects_speed"] is True


# ---------------------------------------------------------------------------
# R8 trigger helper — covers all four branches of `_r8_trigger_fires`.
# Hot-ops dicts are hand-crafted to match the empirical patterns from the
# three example projects under example_quantize_*/outputs/ — see the
# hindsight_r8_examples.py module for the full disk-anchored verification.
# ---------------------------------------------------------------------------


def _hot_op(op_name: str, op_type: str, max_snr: float, std_ratio=None) -> dict:
    return {
        "op_name": op_name,
        "op_type": op_type,
        "max_snr": max_snr,
        "inputs_float_std_ratio": std_ratio,
    }


def test_r8_fires_when_max_snr_is_in_goldilocks_band():
    """mobilenetv2-esp32s3 iter_1 pattern: Add max_snr=0.264 ∈ (0.20, 0.30],
    std_ratio=1.19, Relu top-1 at 0.300 (ratio 1.14 < veto 1.2). Fires
    via primary band."""
    hot_ops = [
        _hot_op("/relu_a", "Relu", 0.300),
        _hot_op("/features.16/Add", "Add", 0.264, 1.19),
        _hot_op("/features.15/Add", "Add", 0.172, 1.11),
    ]
    fires, reason = ci._r8_trigger_fires(hot_ops)
    assert fires is True
    assert "/features.16/Add" in reason
    assert "primary band" in reason


def test_r8_below_band_does_not_fire():
    """mobilenetv2-esp32p4 iter_3 pattern: Add max_snr=0.162 ≤ 0.20 lower,
    std_ratio=1.19. Below the band — too little residual for fusion
    alignment to move the metric. (yolo11n's Concat 0.094 sits here too.)"""
    hot_ops = [
        _hot_op("/features.16/Add", "Add", 0.162, 1.19),
        _hot_op("/features.15/Add", "Add", 0.140, 1.11),
    ]
    fires, reason = ci._r8_trigger_fires(hot_ops)
    assert fires is False
    assert "below-band" in reason
    assert "0.20 lower bound" in reason


def test_r8_above_band_does_not_fire():
    """mobilenetv2-esp32p4 iter_5 pattern (best when 3c was actually tried
    in that project): Add max_snr=0.318 > 0.30 upper, std_ratio=1.19. Above
    the band — residual too severe for fusion alignment alone; another
    pass (3a-3 / 3d / 3e) should run first. This is the hindsight case
    that proves the upper bound matters."""
    hot_ops = [
        _hot_op("/features.16/Add", "Add", 0.318, 1.19),
        _hot_op(
            "/relu_a", "Relu", 0.297
        ),  # below the 1.2× veto ratio (0.297/0.318 = 0.93)
        _hot_op("/features.15/Add", "Add", 0.254, 1.11),
    ]
    fires, reason = ci._r8_trigger_fires(hot_ops)
    assert fires is False
    assert "above-band" in reason
    assert "0.30 upper bound" in reason
    assert "3a-3 / 3d / 3e" in reason


def test_r8_activation_veto_blocks_fire():
    """Synthetic: Add max_snr=0.25 (in band, would fire) but Swish 0.40 in
    top-3 (ratio 0.40/0.25 = 1.6 > 1.2 veto). Activation-dominated → skip.
    Note: Swish goes first in the list so it appears in the top-3 window;
    veto cares about the worst activation in top-3, not the overall max."""
    hot_ops = [
        _hot_op("/swish_a", "Swish", 0.40),
        _hot_op("/add_a", "Add", 0.25, 1.0),
        _hot_op("/swish_b", "Swish", 0.30),
    ]
    fires, reason = ci._r8_trigger_fires(hot_ops)
    assert fires is False
    assert "activation veto" in reason
    assert "/swish_a" in reason


def test_r8_yolo11n_concat_does_not_fire():
    """yolo11n-esp32s3 iter_3/7 pattern: Concat max_snr=0.093 (below band),
    Swish 0.093 in top-3 (ratio 1.00 — tied, NOT > veto 1.2). The skip is
    driven by the band check, not the veto — and that's the correct
    outcome (empirically fusion_alignment regressed on this project)."""
    hot_ops = [
        _hot_op("/model.6/m.0/Concat", "Concat", 0.093, 1.96),
        _hot_op("/swish_a", "Swish", 0.093),
        _hot_op("/swish_b", "Swish", 0.072),
    ]
    fires, reason = ci._r8_trigger_fires(hot_ops)
    assert fires is False
    # Below-band check fires first — the veto check is short-circuited.
    assert "below-band" in reason


def test_r8_fires_via_reinforcement_below_band():
    """Synthetic: a Resize with low max_snr=0.10 (below band) but
    std_ratio=8 (>5). Reinforcement signal fires — the legacy R8 path is
    preserved for the rare cases where wide input scales actually drive
    the residual error."""
    hot_ops = [
        _hot_op("/resize_a", "Resize", 0.10, 8.0),
    ]
    fires, reason = ci._r8_trigger_fires(hot_ops)
    assert fires is True
    assert "reinforcement" in reason
    assert "/resize_a" in reason


def test_r8_fires_via_reinforcement_above_band():
    """Above-band candidates also fire when std_ratio is wide enough — the
    legacy semantics override the band's "too severe" interpretation when
    the std-ratio signal is unambiguous. Surfaced via the reason string so
    the agent can see it's the reinforcement path, not the band."""
    hot_ops = [
        _hot_op("/concat_a", "Concat", 0.45, 7.5),
    ]
    fires, reason = ci._r8_trigger_fires(hot_ops)
    assert fires is True
    assert "reinforcement" in reason
    assert "above band" in reason


def test_r8_no_candidate_elementwise():
    """Hot ops contain only Relu/Conv-like entries — no Concat/Add/Resize → skip."""
    hot_ops = [
        _hot_op("/relu_a", "Relu", 0.30),
        _hot_op("/relu_b", "Relu", 0.25),
    ]
    fires, reason = ci._r8_trigger_fires(hot_ops)
    assert fires is False
    assert "no Concat/Add/Sub/Mul/Resize/AveragePool" in reason


def test_r8_empty_hot_ops_is_skip():
    """Empty input is conservative — skip with a clear reason."""
    fires, reason = ci._r8_trigger_fires([])
    assert fires is False
    assert "empty" in reason.lower()


def test_r8_check_for_best_with_no_dir(tmp_path):
    """No best yet → conservative skip; the 3c lever is never picked
    before iter_0 lands anyway."""
    fires, reason = ci._r8_check_for_best(None)
    assert fires is False
    assert "no best iteration" in reason


def test_r8_check_for_best_loads_disk_file(tmp_path):
    """Write a non_computing_hot_ops.json that triggers, point a fake best
    at it, verify _r8_check_for_best fires and quotes the candidate."""
    import json as _json

    iter_dir = tmp_path / "iter_1"
    iter_dir.mkdir()
    (iter_dir / "non_computing_hot_ops.json").write_text(
        _json.dumps(
            [
                {
                    "op_name": "/some/Add",
                    "op_type": "Add",
                    "max_snr": 0.30,
                    "inputs_float_std_ratio": 1.2,
                }
            ]
        )
    )
    fake_best = {"dir": str(iter_dir)}
    fires, reason = ci._r8_check_for_best(fake_best)
    assert fires is True
    assert "/some/Add" in reason


# ---------------------------------------------------------------------------
# 3c lever skip behaviour in the linear order (uses _next_phase3_lever_with_skips)
# ---------------------------------------------------------------------------


def test_3c_skipped_advances_to_3d_when_r8_does_not_fire(tmp_path):
    """End-to-end: best's non_computing_hot_ops.json has only low-max_snr
    Add ops → R8 doesn't fire → 3c skipped → next lever is 3d."""
    import json as _json

    rows = _three_phase2_iters()
    rows.append(_row(4, 71.20, _tqt_3a1_setting("percentile")))  # 3a-1 (best)
    rows.append(_row(5, 71.15, _tqt_3a2_setting("percentile")))  # 3a-2
    rows.append(
        _row(  # 3b
            6,
            71.10,
            {**_phase2_setting("percentile"), "bias_correct": {"enabled": True}},
        )
    )
    # Wire up the best iteration's directory so _r8_check_for_best can find
    # the hot-ops file on disk.
    best_dir = tmp_path / "iter_4"
    best_dir.mkdir()
    (best_dir / "non_computing_hot_ops.json").write_text(
        _json.dumps(
            [
                {
                    "op_name": "/quiet_Add",
                    "op_type": "Add",
                    "max_snr": 0.10,
                    "inputs_float_std_ratio": 1.1,
                }
            ]
        )
    )
    rows[4]["dir"] = str(best_dir)

    next_lever, skipped = ci._next_phase3_lever_with_skips(
        rows, "balanced", ci._best(rows)
    )
    assert next_lever == "3d"
    assert any(lev == "3c" for lev, _ in skipped)
    skip_reason = next(r for lev, r in skipped if lev == "3c")
    assert "below-band" in skip_reason


def test_3c_fires_when_r8_trigger_matches(tmp_path):
    """Mirror of the above: same iteration history but the hot-ops file
    contains an Add with max_snr > 0.20 → R8 fires → next lever IS 3c."""
    import json as _json

    rows = _three_phase2_iters()
    rows.append(_row(4, 71.20, _tqt_3a1_setting("percentile")))
    rows.append(_row(5, 71.15, _tqt_3a2_setting("percentile")))
    rows.append(
        _row(
            6,
            71.10,
            {**_phase2_setting("percentile"), "bias_correct": {"enabled": True}},
        )
    )
    best_dir = tmp_path / "iter_4"
    best_dir.mkdir()
    (best_dir / "non_computing_hot_ops.json").write_text(
        _json.dumps(
            [
                {
                    "op_name": "/loud_Add",
                    "op_type": "Add",
                    "max_snr": 0.30,
                    "inputs_float_std_ratio": 1.1,
                }
            ]
        )
    )
    rows[4]["dir"] = str(best_dir)

    next_lever, skipped = ci._next_phase3_lever_with_skips(
        rows, "balanced", ci._best(rows)
    )
    assert next_lever == "3c"
    assert skipped == []


def test_all_levers_tried_treats_r8_skipped_3c_as_covered(tmp_path):
    """Closing the loop on the skip path: when 3c is the only lever not
    "tried" but R8 doesn't fire, _all_phase3_levers_tried treats it as
    covered so the state machine progresses to Phase 5 instead of looping
    on a lever the data says won't help."""
    import json as _json

    rows = _three_phase2_iters()
    base = _phase2_setting("percentile")
    levers_except_3c = {
        "3a-1": _tqt_3a1_setting("percentile"),
        "3a-2": _tqt_3a2_setting("percentile"),
        "3b": {**base, "bias_correct": {"enabled": True}},
        "3d": {**base, "equalization": {"enabled": True}},
        "3e": {**base, "dispatching_table": [{"op_name": "/x", "bits": 16}]},
        "3f": {**base, "weight_split": {"enabled": True}},
        "3g": {**base, "blockwise_reconstruction": {"enabled": True}},
    }
    for i, lever in enumerate(levers_except_3c):
        rows.append(_row(4 + i, 71.0, levers_except_3c[lever], target_metric=99.0))
    for r in rows:
        r["target_metric"] = 99.0
    # best is iter_3 (percentile + TQT default at 70.80 per _three_phase2_iters)
    # — but pick the actual best from rows so the test is robust.
    best = ci._best(rows)
    best_dir = tmp_path / "iter_best"
    best_dir.mkdir()
    (best_dir / "non_computing_hot_ops.json").write_text(
        _json.dumps(
            [
                {
                    "op_name": "/quiet",
                    "op_type": "Add",
                    "max_snr": 0.05,
                    "inputs_float_std_ratio": 1.0,
                }
            ]
        )
    )
    # Make sure compare_iterations finds it when it inspects best.
    best["dir"] = str(best_dir)
    # iterate over rows and set the *actual* best's dir too (in case _best
    # returns a different one across implementations).
    for r in rows:
        if r["iteration_id"] == best["iteration_id"]:
            r["dir"] = str(best_dir)

    assert ci._all_phase3_levers_tried(rows, "balanced", best) is True

    saved_cap = ci._PHASE3_CAP
    saved_window = ci._PLATEAU_WINDOW
    ci._PHASE3_CAP = 100
    ci._PLATEAU_WINDOW = 100
    try:
        phase, hint = ci._suggest_next(rows, best)
    finally:
        ci._PHASE3_CAP = saved_cap
        ci._PLATEAU_WINDOW = saved_window
    assert phase == "phase-5-agent-driven"
    # Hint mentions the skipped 3c so the agent sees the trigger reason.
    assert "3c" in hint
    assert "below-band" in hint


# ---------------------------------------------------------------------------
# Phase 5 signal builders
# ---------------------------------------------------------------------------


def test_phase5_signals_lists_improving_levers():
    rows = _three_phase2_iters()
    rows.append(_row(4, 71.50, _tqt_3a1_setting("percentile")))  # +0.70 → improvement
    rows.append(_row(5, 71.20, _tqt_3a2_setting("percentile")))  # regression
    # Mark target unreachable so improving_levers is the focus.
    for r in rows:
        r["target_metric"] = 99.0
    best = ci._best(rows)
    signals = ci._phase5_signals_for(rows, best)
    improving = signals.get("improving_levers", [])
    # iter_1 (kl + TQT improved over iter_0 baseline), iter_3 (percentile + TQT improved
    # over iter_1), iter_4 (3a-1 improved over iter_3).
    iter_ids_improved = [e["iter"] for e in improving]
    assert 4 in iter_ids_improved
    # Phase-2 swap from kl to percentile bumped best at iter_3.
    assert 3 in iter_ids_improved or 1 in iter_ids_improved


def test_phase5_signals_lists_untried_calib_swaps():
    """Best is on kl + TQT default; mse and percentile should appear under
    untried_calib_swaps (both were also tried in Phase 2 → eligible)."""
    rows = _three_phase2_iters()
    # Make kl the best by giving it the highest value.
    rows[1]["primary_value"] = 72.0  # kl + TQT
    for r in rows:
        r["target_metric"] = 99.0
    best = ci._best(rows)
    signals = ci._phase5_signals_for(rows, best)
    untried = signals.get("untried_calib_swaps", [])
    assert "mse" in untried
    assert "percentile" in untried
    assert "kl" not in untried  # current best's calib is excluded


def test_phase5_signals_carries_best_summary():
    rows = _three_phase2_iters()
    rows.append(
        _row(
            4,
            72.0,
            {
                **_phase2_setting("percentile"),
                "equalization": {"enabled": True},
                "dispatching_table": [{"op": "/x", "bits": 16}],
            },
        )
    )
    for r in rows:
        r["target_metric"] = 99.0
    best = ci._best(rows)
    signals = ci._phase5_signals_for(rows, best)
    summary = signals.get("best_setting_summary", {})
    assert summary.get("calib_algorithm") == "percentile"
    assert "equalization" in summary.get("enabled_passes", [])
    assert "tqt_optimization" in summary.get("enabled_passes", [])
    assert any(
        p.startswith("dispatching_table(int16x")
        for p in summary.get("enabled_passes", [])
    )
    assert summary.get("affects_inference_speed") is True


def test_phase5_hint_is_meta_guidance_not_template():
    """The hint must not embed a fillable setting.json template; that's
    Phase 3's contract. Phase 5 is meta-guidance + signals."""
    rows = _three_phase2_iters()
    rows.append(_row(4, 71.50, _tqt_3a1_setting("percentile")))
    for r in rows:
        r["target_metric"] = 99.0
    best = ci._best(rows)
    signals = ci._phase5_signals_for(rows, best)
    hint = ci._build_phase5_hint(rows, best, signals)
    assert '"iteration_id"' not in hint  # no setting.json skeleton
    assert "Starting points to consider" in hint
    assert "STACK" in hint and "CROSS-POLLINATE" in hint and "ABLATE" in hint
    assert "Composition discipline #5" in hint


def test_comparison_json_includes_phase5_signals_when_phase5(tmp_path):
    """When the script emits phase-5-agent-driven, comparison.json grows
    a phase5_signals block. Other phases omit the field."""
    rows = _three_phase2_iters()
    rows.append(_row(4, 71.0, _tqt_3a1_setting("percentile")))
    rows.append(_row(5, 71.0, _tqt_3a2_setting("percentile")))
    rows.append(
        _row(
            6,
            71.0,
            {**_phase2_setting("percentile"), "bias_correct": {"enabled": True}},
        )
    )
    rows.append(
        _row(
            7,
            71.0,
            {**_phase2_setting("percentile"), "equalization": {"enabled": True}},
        )
    )
    rows.append(
        _row(
            8,
            71.0,
            {
                **_phase2_setting("percentile"),
                "fusion_alignment": {"align_elementwise_to": "Align to Large"},
            },
        )
    )
    # 5 Phase-3 iterations → cap. Target unreachable → phase-5.
    for r in rows:
        r["target_metric"] = 99.0
    out = _setup_iter_dirs(tmp_path, rows)
    ci.main(["--output-dir", str(out)])
    import json as _json

    cmp = _json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert cmp["next_step_hint"]["phase"] == "phase-5-agent-driven"
    assert "phase5_signals" in cmp["next_step_hint"]
    signals = cmp["next_step_hint"]["phase5_signals"]
    assert "improving_levers" in signals
    assert "untried_calib_swaps" in signals
    assert "best_setting_summary" in signals


def test_comparison_json_omits_phase5_signals_in_other_phases(tmp_path):
    """Phase 2 / phase 3 emissions must NOT carry phase5_signals — keep the
    JSON small and the schema deterministic."""
    rows = [_row(0, 70.0, _baseline_setting()), _row(1, 70.5, _phase2_setting("kl"))]
    out = _setup_iter_dirs(tmp_path, rows)
    ci.main(["--output-dir", str(out)])
    import json as _json

    cmp = _json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert cmp["next_step_hint"]["phase"] == "phase-2-calib-tqt-sweep"
    assert "phase5_signals" not in cmp["next_step_hint"]


# ---------------------------------------------------------------------------
# Phase 5 — pattern classifier (_classify_phase5_patterns)
# ---------------------------------------------------------------------------


def test_classify_phase5_alpha_stack_pass_turned_on():
    """Turning a pass ON that was OFF before is 5alpha STACK."""
    prev = {"calib_algorithm": "percentile"}
    new = {
        "calib_algorithm": "percentile",
        "equalization": {"enabled": True},
    }
    patterns = ci._classify_phase5_patterns(prev, new)
    assert patterns == ["5alpha"]


def test_classify_phase5_gamma_ablate_pass_turned_off():
    """Turning a pass OFF that was ON before is 5gamma ABLATE."""
    prev = {
        "calib_algorithm": "percentile",
        "equalization": {"enabled": True},
        "bias_correct": {"enabled": True},
    }
    new = {
        "calib_algorithm": "percentile",
        "equalization": {"enabled": True},
    }
    patterns = ci._classify_phase5_patterns(prev, new)
    assert patterns == ["5gamma"]


def test_classify_phase5_beta_calib_swap():
    """Calib swap with no other changes is pure 5beta CROSS-POLLINATE."""
    prev = {"calib_algorithm": "kl", "equalization": {"enabled": True}}
    new = {"calib_algorithm": "percentile", "equalization": {"enabled": True}}
    patterns = ci._classify_phase5_patterns(prev, new)
    assert patterns == ["5beta"]


def test_classify_phase5_delta_dispatching_table_op_swap():
    """Dispatching table with different op set (both non-empty) is 5delta."""
    prev = {
        "calib_algorithm": "kl",
        "dispatching_table": [{"op": "/conv1/Conv", "bits": 16}],
    }
    new = {
        "calib_algorithm": "kl",
        "dispatching_table": [{"op": "/conv2/Conv", "bits": 16}],
    }
    patterns = ci._classify_phase5_patterns(prev, new)
    assert patterns == ["5delta"]


def test_classify_phase5_dispatching_table_first_enable_is_alpha():
    """Empty → non-empty dispatch is 5alpha STACK (turning the pass on)."""
    prev = {"calib_algorithm": "kl"}
    new = {
        "calib_algorithm": "kl",
        "dispatching_table": [{"op": "/conv1/Conv", "bits": 16}],
    }
    patterns = ci._classify_phase5_patterns(prev, new)
    assert patterns == ["5alpha"]


def test_classify_phase5_multi_pattern_stack_plus_calib_swap():
    """Composition discipline #5: a single Phase-5 iter may combine multiple
    patterns. The classifier returns the union."""
    prev = {"calib_algorithm": "kl"}
    new = {
        "calib_algorithm": "percentile",
        "equalization": {"enabled": True},
    }
    patterns = ci._classify_phase5_patterns(prev, new)
    assert set(patterns) == {"5alpha", "5beta"}


def test_classify_phase5_noop_returns_empty_list():
    """Identical settings classify to no patterns."""
    s = {"calib_algorithm": "kl", "equalization": {"enabled": True}}
    assert ci._classify_phase5_patterns(s, dict(s)) == []


def test_classify_phase5_fusion_alignment_toggle_is_alpha():
    """fusion_alignment uses a non-enabled shape; classifier must still
    recognise the truthy-from-falsy transition as a 5alpha STACK."""
    prev = {"calib_algorithm": "kl"}
    new = {
        "calib_algorithm": "kl",
        "fusion_alignment": {"align_elementwise_to": "Align to Large"},
    }
    patterns = ci._classify_phase5_patterns(prev, new)
    assert patterns == ["5alpha"]


# ---------------------------------------------------------------------------
# Phase 5 — coverage helpers
# ---------------------------------------------------------------------------


def _phase5_cap_rows() -> list[dict]:
    """Build rows that match the bad_esp32s3 history shape: baseline +
    3 Phase-2 legs + 5 Phase-3 levers + 3 Phase-5 iters that REALISTICALLY
    stack onto best-so-far (matching Composition discipline #5 semantics).

    Each Phase-5 iter is built on the iter_4 (TQT-1000) baseline so the
    classifier sees clean single-pattern transitions.
    """
    rows = _three_phase2_iters()
    # Phase 3: 5 single-knob iters that don't beat iter_4 (so iter_4 stays
    # best throughout Phase 3). Each lever applied independently.
    rows.append(_row(4, 70.95, _tqt_3a1_setting("percentile")))  # 3a-1
    rows.append(_row(5, 70.90, _tqt_3a2_setting("percentile")))  # 3a-2
    rows.append(
        _row(
            6,
            70.85,
            {**_phase2_setting("percentile"), "bias_correct": {"enabled": True}},
        )
    )  # 3b
    rows.append(
        _row(
            7,
            70.80,
            {
                **_phase2_setting("percentile"),
                "fusion_alignment": {"align_elementwise_to": "Align to Large"},
            },
        )
    )  # 3c
    rows.append(
        _row(
            8,
            70.75,
            {**_phase2_setting("percentile"), "equalization": {"enabled": True}},
        )
    )  # 3d
    # Best so far: iter_4 (70.95) with TQT(1000, kl-default lr).

    # Phase 5 iter_9: STACK equalization onto iter_4 → 5alpha.
    rows.append(
        _row(
            9,
            71.10,
            {**_tqt_3a1_setting("percentile"), "equalization": {"enabled": True}},
        )
    )
    # Best now: iter_9 (71.10) with TQT(1000)+equalization.

    # Phase 5 iter_10: STACK int16 dispatch onto iter_9 → 5alpha
    # (dispatching_table goes from empty to non-empty).
    rows.append(
        _row(
            10,
            71.05,
            {
                **_tqt_3a1_setting("percentile"),
                "equalization": {"enabled": True},
                "dispatching_table": [{"op": "/conv1/Conv", "bits": 16}],
            },
        )
    )
    # iter_10 did not beat iter_9 (71.05 < 71.10), so iter_9 still best.

    # Phase 5 iter_11: CROSS-POLLINATE calib percentile→mse on iter_9 → 5beta.
    rows.append(
        _row(
            11,
            71.02,
            {**_tqt_3a1_setting("mse"), "equalization": {"enabled": True}},
        )
    )

    for r in rows:
        r["target_metric"] = 99.0  # unreachable, so phase-5 fires
    return rows


def test_phase5_cutoff_iter_id_identifies_first_phase5_iter():
    rows = _phase5_cap_rows()
    cutoff = ci._phase5_cutoff_iter_id(rows)
    # After iter_8 (the 5th Phase-3 iter), the next phase routes to phase-5.
    assert cutoff == 8


def test_phase5_cutoff_iter_id_returns_neg1_when_no_phase5_history():
    """When Phase 3 has only 2 iters (no cap, no all-levers-done), cutoff
    is -1 — coverage helpers must special-case this."""
    rows = _three_phase2_iters()
    rows.append(_row(4, 71.0, _tqt_3a1_setting("percentile")))
    for r in rows:
        r["target_metric"] = 99.0
    assert ci._phase5_cutoff_iter_id(rows) == -1


def test_phase5_pattern_coverage_classifies_each_phase5_iter():
    rows = _phase5_cap_rows()
    cutoff = ci._phase5_cutoff_iter_id(rows)
    coverage = ci._phase5_pattern_coverage(rows, cutoff)
    # iter_9: equalization stacked on best (best is one of iter_4..8) — 5alpha.
    # iter_10: int16 dispatch turned on — 5alpha.
    # iter_11: calib swap → 5beta.
    assert 9 in coverage["5alpha"]
    assert 10 in coverage["5alpha"]
    assert 11 in coverage["5beta"]
    assert coverage["5gamma"] == []
    assert coverage["5delta"] == []


def test_untried_phase5_patterns_lists_unattempted_patterns():
    rows = _phase5_cap_rows()
    cutoff = ci._phase5_cutoff_iter_id(rows)
    coverage = ci._phase5_pattern_coverage(rows, cutoff)
    untried = ci._untried_phase5_patterns(coverage)
    # 5alpha + 5beta attempted; 5gamma + 5delta untried.
    assert set(untried) == {"5gamma", "5delta"}


def test_untried_phase3_levers_when_cap_hit_first():
    """In a bad_esp32s3-shape history (3a-1/3a-2/3b/3c/3d ran in Phase 3;
    cap=5 hits; 3e covered by a Phase-5 stack iter_10), 3f / 3g were
    never touched. Untried list should include only those two."""
    rows = _phase5_cap_rows()
    priority = ci._runtime_priority(rows)
    untried = ci._untried_phase3_levers(rows, priority, ci._best(rows))
    # 3a-1/3a-2/3b/3c/3d (Phase 3) + 3e (iter_10 dispatched int16)
    # all structurally tried. 3f / 3g never appeared.
    assert "3f" in untried
    assert "3g" in untried
    # 3e covered via Phase-5 STACK iter_10 — lever-tried check is structural.
    assert "3e" not in untried


def test_untried_phase3_levers_treats_correctly_skipped_3c_as_covered():
    """If R8 doesn't fire on best (no elementwise-add ops with snr in the
    Goldilocks band), 3c is correctly skipped — it should NOT appear in
    `untried_phase3_levers`."""
    rows = _three_phase2_iters()
    # 4 Phase-3 iters that don't include 3c (skip it altogether).
    rows.append(_row(4, 71.0, _tqt_3a1_setting("percentile")))
    rows.append(_row(5, 71.0, _tqt_3a2_setting("percentile")))
    rows.append(
        _row(
            6,
            71.0,
            {**_phase2_setting("percentile"), "bias_correct": {"enabled": True}},
        )
    )
    rows.append(
        _row(
            7,
            71.0,
            {**_phase2_setting("percentile"), "equalization": {"enabled": True}},
        )
    )
    for r in rows:
        r["target_metric"] = 99.0
    # Best has no non_computing_hot_ops.json on disk → R8 cannot fire.
    best = ci._best(rows)
    priority = ci._runtime_priority(rows)
    untried = ci._untried_phase3_levers(rows, priority, best)
    assert "3c" not in untried


# ---------------------------------------------------------------------------
# Phase 5 — signals + hint additions
# ---------------------------------------------------------------------------


def test_phase5_signals_carries_coverage_fields():
    rows = _phase5_cap_rows()
    best = ci._best(rows)
    signals = ci._phase5_signals_for(rows, best)
    assert signals["iteration_count_total"] == 12
    assert signals["phase5_iteration_count"] == 3
    assert signals["phase5_cutoff_iter_id"] == 8
    assert isinstance(signals["phase5_pattern_coverage"], dict)
    assert "5alpha" in signals["phase5_pattern_coverage"]
    assert isinstance(signals["untried_phase5_patterns"], list)
    assert isinstance(signals["untried_phase3_levers"], list)


def test_phase5_hint_surfaces_untried_patterns_and_levers():
    rows = _phase5_cap_rows()
    best = ci._best(rows)
    signals = ci._phase5_signals_for(rows, best)
    hint = ci._build_phase5_hint(rows, best, signals)
    assert "Untried Phase 5 patterns" in hint
    assert "5gamma" in hint or "5delta" in hint
    assert "Untried Phase 3 linear-order levers" in hint
    assert "Coverage so far" in hint
    # New strict stop signal language MUST be present.
    assert "ONLY when ALL of the following hold" in hint
    assert "USER-GIVEN ITERATION BUDGET REACHED" in hint
    assert "untried_phase5_patterns" in hint
    assert "untried_phase3_levers" in hint


def test_phase5_hint_no_misleading_pretending_line():
    """Old text said agents should not 'pretend there's a 5th idea' — this
    license to bail is what produced the bad_esp32s3 12-of-20 outcome. The
    rewrite must remove that phrasing."""
    rows = _phase5_cap_rows()
    best = ci._best(rows)
    signals = ci._phase5_signals_for(rows, best)
    hint = ci._build_phase5_hint(rows, best, signals)
    assert "Pretending" not in hint
    assert "pretending" not in hint


# ---------------------------------------------------------------------------
# Phase 5 — premature-finalize warning
# ---------------------------------------------------------------------------


def test_premature_finalize_rejects_in_phase5(tmp_path, capsys):
    """When --finalize fires while phase=phase-5 AND target not met AND
    not a plateau, main() must HARD REJECT (rc=1, no best/ written, no
    final_report.md written) and print the rejection block citing untried
    patterns + levers.

    The previous behaviour ("WARNING printed but finalize completes") was
    a soft signal that agents ignored in
    example_quantize_mobilenetv2_bad_esp32s3 (12-of-20) and
    example_quantize_mobilenetv2_esp32p4_tmp (18-of-30). This test
    regression-proofs the hard-reject contract.
    """
    rows = _phase5_cap_rows()
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out), "--finalize"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "PREMATURE --finalize REJECTED" in captured.out
    assert "untried Phase 5 patterns" in captured.out
    assert "untried Phase 3 linear-order levers" in captured.out
    # No outputs/best/ and no outputs/final_report.md must exist.
    assert not (out / "best").exists()
    assert not (out / "final_report.md").exists()
    # comparison.json IS still written so the agent can re-read the hint.
    import json as _json

    cmp = _json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert cmp["next_step_hint"]["premature_finalize_rejected"] is True


def test_force_finalize_overrides_premature_check(tmp_path, capsys):
    """--force-finalize alongside --finalize lets the agent (or user)
    confirm intentional early stop. Output files must be written and
    the finalize_reason category must be `force_finalize_phase5`."""
    rows = _phase5_cap_rows()
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out), "--finalize", "--force-finalize"])
    assert rc == 0
    assert (out / "best").exists()
    assert (out / "final_report.md").exists()
    import json as _json

    cmp = _json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    reason = cmp["next_step_hint"].get("finalize_reason") or {}
    assert reason.get("category") == "force_finalize_phase5"
    # final_report.md must include the Stop reason section + the untried
    # lists so the user can see what got skipped.
    report = (out / "final_report.md").read_text(encoding="utf-8")
    assert "## Stop reason" in report
    assert "force_finalize_phase5" in report
    assert "Untried Phase 5 patterns" in report
    assert "Untried Phase 3 linear-order levers" in report


def test_premature_finalize_warning_silent_when_target_met(tmp_path, capsys):
    """If target_metric is reached, phase-4 fires automatically and the
    warning must NOT print — the finalize is well-deserved."""
    rows = _phase5_cap_rows()
    rows[-1]["primary_value"] = 99.5  # exceed target
    for r in rows:
        r["target_metric"] = 99.0
    rows[-1]["target_metric"] = 99.0
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out), "--finalize"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "PREMATURE" not in captured.out


def test_final_report_writes_stop_reason_target_reached(tmp_path):
    """When target is reached the auto-finalize must record category
    `target_reached` in finalize_reason and the Stop reason section."""
    rows = _three_phase2_iters()
    for r in rows:
        r["target_metric"] = 70.5  # iter_3 = 70.8 already past
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out)])
    assert rc == 0
    import json as _json

    cmp = _json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    reason = cmp["next_step_hint"].get("finalize_reason") or {}
    assert reason.get("category") == "target_reached"
    report = (out / "final_report.md").read_text(encoding="utf-8")
    assert "## Stop reason" in report
    assert "`target_reached`" in report


def test_final_report_writes_stop_reason_plateau(tmp_path):
    """Plateau auto-finalize must record category `plateau` and embed the
    recent flat values."""
    rows = _three_phase2_iters()
    best_value = max(r["primary_value"] for r in rows)
    rows.append(
        _row(4, best_value - best_value * 0.0005, _tqt_3a1_setting("percentile"))
    )
    rows.append(
        _row(5, best_value - best_value * 0.0008, _tqt_3a2_setting("percentile"))
    )
    rows.append(_row(6, best_value - best_value * 0.0006, _tqt_3a2_setting("kl")))
    for r in rows:
        r["target_metric"] = 99.0  # unreachable
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out)])
    assert rc == 0
    import json as _json

    cmp = _json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    reason = cmp["next_step_hint"].get("finalize_reason") or {}
    assert reason.get("category") == "plateau"
    assert reason.get("plateau_window") == 3
    assert len(reason.get("plateau_recent_values") or []) == 3
    report = (out / "final_report.md").read_text(encoding="utf-8")
    assert "## Stop reason" in report
    assert "`plateau`" in report
    assert "within 0.1%" in report


def test_final_report_writes_stop_reason_manual_pre_phase5(tmp_path):
    """When the user --finalize-s during Phase 1/2/3 (e.g. budget=3 with
    only Phase 2 done), category must be `manual_finalize_pre_phase5`."""
    rows = [
        _row(0, 70.0, _baseline_setting(), target_metric=99.0),
        _row(1, 70.5, _phase2_setting("kl"), target_metric=99.0),
    ]
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out), "--finalize"])
    assert rc == 0
    import json as _json

    cmp = _json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    reason = cmp["next_step_hint"].get("finalize_reason") or {}
    assert reason.get("category") == "manual_finalize_pre_phase5"
    report = (out / "final_report.md").read_text(encoding="utf-8")
    assert "## Stop reason" in report
    assert "`manual_finalize_pre_phase5`" in report


def test_premature_finalize_warning_silent_in_phase4_plateau(tmp_path, capsys):
    """When --finalize fires in phase=phase-4 (plateau or target), the
    warning must NOT print."""
    rows = _three_phase2_iters()
    best_value = max(r["primary_value"] for r in rows)
    # Three consecutive flat iters within 0.1% (_PLATEAU_WINDOW=3).
    rows.append(
        _row(4, best_value - best_value * 0.0005, _tqt_3a1_setting("percentile"))
    )
    rows.append(
        _row(5, best_value - best_value * 0.0008, _tqt_3a2_setting("percentile"))
    )
    rows.append(_row(6, best_value - best_value * 0.0006, _tqt_3a2_setting("kl")))
    for r in rows:
        r["target_metric"] = 99.0
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out), "--finalize"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "PREMATURE" not in captured.out


# ---------------------------------------------------------------------------
# Phase C — 5β-reapply + tunable-params hint
# ---------------------------------------------------------------------------


def _deep_stack_setting(calib: str = "percentile") -> dict:
    """A setting representing a 'deep' lever stack: TQT + equalization +
    bias_correct + int16x2 dispatching_table. Used by 5β-reapply tests to
    simulate the state where the prior shallow-stack calib swap should be
    re-evaluated."""
    s = _tqt_3a1_setting(calib)
    s["equalization"] = {"enabled": True}
    s["bias_correct"] = {"enabled": True}
    s["dispatching_table"] = [
        {"op_name": "/features.0/Conv", "bits": 16},
        {"op_name": "/features.1/Conv", "bits": 16},
    ]
    return s


def test_5beta_reapply_detected_when_stack_deepened():
    """history: iter_9 swaps kl→percentile on a shallow Phase-2 stack;
    iter_15 is the new best on a much deeper stack. untried_5beta_reapply
    must list 'percentile'."""
    rows = _three_phase2_iters()
    # Phase 3 levers (some don't have to be tried — just need iter_15 to
    # land on a strictly deeper stack than iter_9's stack).
    rows.append(_row(4, 71.0, _tqt_3a1_setting("percentile")))
    # iter_9-ish 5β CROSS-POLLINATE on shallow stack (Phase-2 TQT + percentile).
    rows.append(_row(5, 70.5, _phase2_setting("mse")))  # shallow stack
    # ... fill to the deep best.
    rows.append(_row(6, 71.2, _deep_stack_setting("kl")))
    # iter at id=7 is the deep best (with calib=kl).
    for r in rows:
        r["target_metric"] = 99.0  # unreachable so we stay in phase-5
    best = ci._best(rows)
    untried = ci._untried_5beta_reapply(rows, best)
    # mse was tried on shallow stack; best now has deeper stack with kl.
    # untried_5beta_reapply must include 'mse' (not yet re-tested on deep stack).
    assert "mse" in untried


def test_5beta_reapply_covered_when_done_on_deepest_stack():
    """history: iter_5 swaps kl→mse on shallow; iter_10 re-runs mse on
    the deepest stack. untried_5beta_reapply must be empty for that calib."""
    rows = _three_phase2_iters()
    # Shallow 5β attempt.
    rows.append(_row(4, 70.5, _phase2_setting("mse")))
    # Deep stack best (calib=kl).
    rows.append(_row(5, 71.2, _deep_stack_setting("kl")))
    # Reapply mse on the deep stack — same lever set, just calib differs.
    deep_mse = _deep_stack_setting("mse")
    rows.append(_row(6, 71.0, deep_mse))
    for r in rows:
        r["target_metric"] = 99.0
    best = ci._best(rows)
    untried = ci._untried_5beta_reapply(rows, best)
    assert "mse" not in untried


def test_5beta_reapply_keeps_5beta_in_untried_phase5_patterns():
    """Even when 5β fired in history (phase5_pattern_coverage shows it),
    'untried_5beta_reapply' being non-empty must force '5beta' back into
    untried_phase5_patterns. Otherwise stop signal (4) could fire without
    closing the canonical iter_13 gap."""
    rows = _three_phase2_iters()
    rows.append(_row(4, 70.5, _phase2_setting("mse")))  # shallow 5β
    rows.append(_row(5, 71.2, _deep_stack_setting("kl")))  # deep best
    for r in rows:
        r["target_metric"] = 99.0
    best = ci._best(rows)
    signals = ci._phase5_signals_for(rows, best)
    assert signals[
        "untried_5beta_reapply"
    ], "expected untried_5beta_reapply to contain mse"
    assert "5beta" in signals["untried_phase5_patterns"]


def test_phase5_hint_lists_tunable_params_for_enabled_passes():
    """The hint must include the tunable-params section for each enabled
    pass on best."""
    rows = _three_phase2_iters()
    rows.append(_row(4, 71.5, _deep_stack_setting("percentile")))
    for r in rows:
        r["target_metric"] = 99.0
    best = ci._best(rows)
    signals = ci._phase5_signals_for(rows, best)
    hint = ci._build_phase5_hint(rows, best, signals)
    assert "Tunable parameters in current best" in hint
    assert "tqt_optimization" in hint
    assert "equalization" in hint
    # percentile calib -> percentile knob mentioned.
    assert "99.9" in hint and "99.999" in hint


def test_phase5_hint_omits_tunable_params_when_no_passes_enabled():
    """A best with only calib (no gradient/structural passes on) → no
    tunable-params section (empty body)."""
    rows = _three_phase2_iters()
    rows.append(_row(4, 71.0, _baseline_setting()))
    for r in rows:
        r["target_metric"] = 99.0
    best = ci._best(rows)
    # Best is iter_3 (percentile + TQT default) per fixture. Tunable section
    # should still appear (TQT enabled). To exercise the empty path, build
    # a degenerate phase5 entry directly.
    bare_setting = {"calib_algorithm": "kl"}
    lines = ci._phase5_tunable_params_lines(bare_setting)
    assert lines == []


def test_force_finalize_records_5beta_reapply_in_stop_reason(tmp_path):
    """When --force-finalize fires while untried_5beta_reapply is
    non-empty, the final_report.md Stop reason section must list those
    targets so the user sees what was skipped."""
    rows = _three_phase2_iters()
    rows.append(_row(4, 70.5, _phase2_setting("mse")))  # shallow 5β
    rows.append(_row(5, 71.2, _deep_stack_setting("kl")))  # deep best
    # Force phase-5 by adding enough Phase-3 iters past the cap.
    rows.append(_row(6, 71.0, _tqt_3a2_setting("kl")))
    rows.append(
        _row(
            7,
            70.9,
            {
                **_phase2_setting("kl"),
                "fusion_alignment": {"align_elementwise_to": "Align to Large"},
            },
        )
    )
    rows.append(_row(8, 70.8, _tqt_3a3_setting("kl")))
    for r in rows:
        r["target_metric"] = 99.0
    out = _setup_iter_dirs(tmp_path, rows)
    rc = ci.main(["--output-dir", str(out), "--finalize", "--force-finalize"])
    assert rc == 0
    report = (out / "final_report.md").read_text(encoding="utf-8")
    assert "Untried 5\u03b2-reapply targets" in report
    assert "mse" in report


if __name__ == "__main__":
    import inspect

    failed = 0
    funcs = [
        obj
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if name.startswith("test_") and callable(obj)
    ]
    for fn in funcs:
        try:
            fn()
            print(f"OK   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    sys.exit(1 if failed else 0)
