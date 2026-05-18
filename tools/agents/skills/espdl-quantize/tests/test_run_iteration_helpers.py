"""Dry-run tests for the D-side analysis helpers.

* aggregate_non_computing_hot_ops — D2 (non_computing_hot_ops.json)
* compute_graphwise_jumps         — D3 (graphwise_jumps.json)
* filter_layer_stats_for          — legacy top-K view (layer_stats.json)

These don't load any model — they feed mock `statistical_analyse` records and
mock simplified-ops payload directly to the helpers and assert on the shape /
ranking / threshold semantics of the outputs. The helpers live in
``scripts/analysis_helpers.py`` precisely so this file can run in any Python
env without esp_ppq / torch installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

import analysis_helpers as ah  # noqa: E402


def _stat_record(
    op_name: str,
    op_type: str,
    var: str,
    snr: float,
    *,
    is_input: bool = False,
    is_output: bool = False,
    is_parameter: bool = False,
    float_std: float | None = None,
) -> dict:
    rec = {
        "Op name": op_name,
        "Op type": op_type,
        "Variable name": var,
        "Is input": is_input,
        "Is output": is_output,
        "Is parameter": is_parameter,
        "Noise:Signal Power Ratio": snr,
    }
    if float_std is not None:
        rec["Float Std"] = float_std
    return rec


# ---------------------------------------------------------------------------
# aggregate_non_computing_hot_ops
# ---------------------------------------------------------------------------


def test_aggregate_filters_out_computing_ops():
    records = [
        _stat_record("/conv1", "Conv", "/conv1.in", 0.50, is_input=True),
        _stat_record("/concat1", "Concat", "/concat1.in0", 0.30, is_input=True),
    ]
    out = ah.aggregate_non_computing_hot_ops(records, top_k=10)
    assert len(out) == 1
    assert out[0]["op_name"] == "/concat1"
    assert out[0]["op_type"] == "Concat"


def test_aggregate_max_snr_per_op_and_ranking():
    records = [
        _stat_record("/concat_low", "Concat", "/v0", 0.10, is_input=True),
        _stat_record("/concat_low", "Concat", "/v1", 0.15, is_output=True),
        _stat_record("/add_high", "Add", "/v0", 0.40, is_input=True),
        _stat_record("/add_high", "Add", "/v1", 0.05, is_output=True),
    ]
    out = ah.aggregate_non_computing_hot_ops(records, top_k=10)
    assert [b["op_name"] for b in out] == ["/add_high", "/concat_low"]
    assert out[0]["max_snr"] == 0.40
    assert out[1]["max_snr"] == 0.15


def test_aggregate_inputs_float_std_ratio():
    records = [
        _stat_record("/concat", "Concat", "/v0", 0.10, is_input=True, float_std=0.10),
        _stat_record("/concat", "Concat", "/v1", 0.20, is_input=True, float_std=2.00),
        _stat_record("/concat", "Concat", "/vout", 0.05, is_output=True),
    ]
    out = ah.aggregate_non_computing_hot_ops(records, top_k=5)
    assert len(out) == 1
    assert out[0]["inputs_float_std_ratio"] is not None
    assert abs(out[0]["inputs_float_std_ratio"] - 20.0) < 1e-6


def test_aggregate_inputs_float_std_ratio_none_when_single_input():
    records = [
        _stat_record("/sigmoid", "Sigmoid", "/v0", 0.30, is_input=True, float_std=1.0),
        _stat_record("/sigmoid", "Sigmoid", "/v1", 0.20, is_output=True),
    ]
    out = ah.aggregate_non_computing_hot_ops(records, top_k=5)
    assert out[0]["inputs_float_std_ratio"] is None


def test_aggregate_top_k_truncation():
    records = [
        _stat_record(f"/op_{i}", "Concat", "/v0", 0.10 * (i + 1), is_input=True)
        for i in range(5)
    ]
    out = ah.aggregate_non_computing_hot_ops(records, top_k=2)
    assert len(out) == 2
    assert out[0]["op_name"] == "/op_4"  # SNR 0.50, the highest
    assert out[1]["op_name"] == "/op_3"  # SNR 0.40


def test_aggregate_excludes_records_without_snr():
    rec = _stat_record("/concat", "Concat", "/v0", 0.0, is_input=True)
    rec["Noise:Signal Power Ratio"] = "n/a"  # bad type
    out = ah.aggregate_non_computing_hot_ops([rec], top_k=5)
    assert out == []


def test_aggregate_top_k_zero_returns_empty():
    records = [_stat_record("/concat", "Concat", "/v0", 0.30, is_input=True)]
    assert ah.aggregate_non_computing_hot_ops(records, top_k=0) == []


# ---------------------------------------------------------------------------
# compute_graphwise_jumps
# ---------------------------------------------------------------------------


def _ops_payload(seq: list[tuple[str, str]]) -> dict:
    return {"ops": [{"name": n, "op_type": t} for n, t in seq]}


def test_graphwise_jumps_flags_intervening_concat():
    layerwise = {"/conv_a": 0.05, "/conv_b": 0.06}
    graphwise = {"/conv_a": 0.10, "/conv_b": 0.30}  # gap = 0.20
    payload = _ops_payload(
        [
            ("/conv_a", "Conv"),
            ("/concat_x", "Concat"),
            ("/relu_x", "Relu"),
            ("/conv_b", "Conv"),
        ]
    )
    jumps = ah.compute_graphwise_jumps(
        layerwise=layerwise,
        graphwise=graphwise,
        simplified_ops_payload=payload,
        excess_threshold=0.02,
    )
    assert len(jumps) == 1
    j = jumps[0]
    assert j["op_prev"] == "/conv_a"
    assert j["op_next"] == "/conv_b"
    assert abs(j["delta_graphwise"] - 0.20) < 1e-9
    assert abs(j["isolated_layerwise_op_next"] - 0.06) < 1e-9
    assert abs(j["intervening_excess"] - 0.14) < 1e-9
    intervening_types = {x["type"] for x in j["intervening_non_computing_ops"]}
    assert intervening_types == {"Concat", "Relu"}


def test_graphwise_jumps_silent_when_excess_below_threshold():
    layerwise = {"/conv_a": 0.05, "/conv_b": 0.05}
    graphwise = {"/conv_a": 0.10, "/conv_b": 0.16}  # gap=0.06; excess=0.01 < 0.02
    payload = _ops_payload(
        [
            ("/conv_a", "Conv"),
            ("/concat_x", "Concat"),
            ("/conv_b", "Conv"),
        ]
    )
    jumps = ah.compute_graphwise_jumps(
        layerwise=layerwise,
        graphwise=graphwise,
        simplified_ops_payload=payload,
        excess_threshold=0.02,
    )
    assert jumps == []


def test_graphwise_jumps_sorted_descending_by_excess():
    layerwise = {"/c1": 0.01, "/c2": 0.01, "/c3": 0.01}
    graphwise = {"/c1": 0.00, "/c2": 0.10, "/c3": 0.50}
    payload = _ops_payload(
        [
            ("/c1", "Conv"),
            ("/concat_a", "Concat"),
            ("/c2", "Conv"),
            ("/resize_b", "Resize"),
            ("/c3", "Conv"),
        ]
    )
    jumps = ah.compute_graphwise_jumps(
        layerwise=layerwise,
        graphwise=graphwise,
        simplified_ops_payload=payload,
        excess_threshold=0.0,
    )
    assert len(jumps) == 2
    assert jumps[0]["op_next"] == "/c3"  # bigger excess
    assert jumps[1]["op_next"] == "/c2"
    assert jumps[0]["intervening_excess"] > jumps[1]["intervening_excess"]


def test_graphwise_jumps_handles_missing_payload():
    assert ah.compute_graphwise_jumps({}, {}, None, 0.02) == []
    assert ah.compute_graphwise_jumps({}, {}, {"ops": []}, 0.02) == []


def test_graphwise_jumps_skips_first_computing_op():
    """No prev_computing yet → no jump emitted for the very first op."""
    layerwise = {"/c1": 0.50}
    graphwise = {"/c1": 0.50}
    payload = _ops_payload([("/c1", "Conv")])
    assert ah.compute_graphwise_jumps(layerwise, graphwise, payload, 0.0) == []


# ---------------------------------------------------------------------------
# filter_layer_stats_for
# ---------------------------------------------------------------------------


def test_filter_layer_stats_preserves_order():
    records = [
        _stat_record("/c1", "Conv", "/c1.in", 0.10, is_input=True),
        _stat_record("/c2", "Conv", "/c2.in", 0.20, is_input=True),
        _stat_record("/c3", "Conv", "/c3.in", 0.05, is_input=True),
    ]
    # Request /c2 first, then /c1; /c3 is excluded.
    out = ah.filter_layer_stats_for(records, ["/c2", "/c1"])
    assert [r["Op name"] for r in out] == ["/c2", "/c1"]


def test_filter_layer_stats_returns_empty_for_empty_inputs():
    assert ah.filter_layer_stats_for([], ["/c1"]) == []
    assert ah.filter_layer_stats_for([{"Op name": "/c1"}], []) == []


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
