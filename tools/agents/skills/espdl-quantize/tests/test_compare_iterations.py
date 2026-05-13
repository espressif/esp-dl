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
    # Two consecutive Phase-3 iters within 0.1% of best → plateau.
    best_value = max(r["primary_value"] for r in rows)
    rows.append(
        _row(4, best_value - best_value * 0.0005, _tqt_3a1_setting("percentile"))
    )
    rows.append(
        _row(5, best_value - best_value * 0.0008, _tqt_3a2_setting("percentile"))
    )
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase == "phase-4-final-report"
    assert "plateau" in hint


def test_phase4_phase3_cap_reached():
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
        rows.append(_row(i, 70.7, s))
    phase, hint = ci._suggest_next(rows, ci._best(rows))
    assert phase == "phase-4-final-report"
    assert "Phase-3 budget reached" in hint


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
    assert '"enabled": false' in hint  # mutex with TQT — template disables TQT


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


def test_all_levers_tried_triggers_phase4():
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
        "3g": {
            **base,
            "tqt_optimization": {"enabled": False},
            "blockwise_reconstruction": {"enabled": True},
        },
    }
    for i, lever in enumerate(levers):
        rows.append(_row(4 + i, 71.0, levers[lever]))
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
    assert "all linear-order" in hint or "levers tried" in hint


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
