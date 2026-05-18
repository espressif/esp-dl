"""Pure-Python aggregation helpers for the D-side analysis artifacts.

These helpers operate on dicts/lists already produced by esp-ppq's analysers,
so they do not import esp-ppq themselves. Keeping them in a standalone module
lets the test suite cover them without an esp-ppq install.

Two artifacts are produced from a single `statistical_analyse` call plus the
existing `layerwise_error_analyse` / `graphwise_error_analyse` outputs:

* ``non_computing_hot_ops.json`` (D2) — top-K non-COMPUTING_OP layers ranked by
  worst per-variable SNR; includes ``inputs_float_std_ratio`` so playbook rule
  R8 (Concat/Add scale mismatch → fusion_alignment) can fire from a single
  pre-aggregated record instead of re-walking ``layer_stats_full.json``.
* ``graphwise_jumps.json`` (D3) — adjacent COMPUTING_OP pairs whose cumulative
  SNR gap is larger than the downstream op's isolated layerwise contribution.
  The intervening non-COMPUTING_OP between them are listed as suspected culprits.

See SKILL.md "esp-ppq three-function coverage table" for the underlying motivation.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# COMPUTING_OP — esp-ppq's notion of "core compute layers" (Conv/Gemm/MatMul/...).
# Inlined from esp-ppq/esp_ppq/core/common.py (`COMPUTING_OP`) to avoid coupling on
# its private import path. The set is what `is_computing_op` compares against and
# what `layerwise_error_analyse` / `graphwise_error_analyse` filter their work to.
COMPUTING_OP_TYPES = frozenset(
    {
        "Conv",
        "Gemm",
        "ConvTranspose",
        "MatMul",
        "Attention",
        "PPQBiasFusedMatMul",
        "LSTM",
    }
)


def aggregate_non_computing_hot_ops(
    records: List[dict],
    top_k: int,
) -> List[dict]:
    """Build the D2 short-list of non-COMPUTING_OP layers ranked by max per-var SNR.

    For each non-COMPUTING_OP, take the worst SNR across its input/output
    variables and additionally compute ``inputs_float_std_ratio``
    (max/min of Float Std across non-parameter input variables). The latter is
    the trigger for playbook rule R8.

    Args:
        records: ``statistical_analyse`` output (a flat list of per-variable
            dicts with keys "Op name", "Op type", "Variable name",
            "Is input"/"Is output"/"Is parameter", "Noise:Signal Power Ratio",
            and "Float Std").
        top_k: Maximum number of non-COMPUTING_OP entries to return. Items are
            sorted descending by ``max_snr``.

    Returns:
        List of dicts with keys ``op_name``, ``op_type``, ``max_snr``,
        ``per_var_snrs`` (list of {var, is_input, is_output, is_parameter, snr}),
        and ``inputs_float_std_ratio`` (float or None).
    """
    if top_k <= 0 or not records:
        return []

    by_op: Dict[str, dict] = {}
    for r in records:
        op_name = r.get("Op name")
        op_type = r.get("Op type")
        if not op_name or op_type in COMPUTING_OP_TYPES:
            continue
        snr = r.get("Noise:Signal Power Ratio")
        if not isinstance(snr, (int, float)):
            continue
        bucket = by_op.setdefault(
            op_name,
            {
                "op_name": op_name,
                "op_type": op_type,
                "max_snr": float("-inf"),
                "per_var_snrs": [],
                "_input_float_stds": [],
            },
        )
        bucket["max_snr"] = max(bucket["max_snr"], float(snr))
        bucket["per_var_snrs"].append(
            {
                "var": r.get("Variable name"),
                "is_input": bool(r.get("Is input")),
                "is_output": bool(r.get("Is output")),
                "is_parameter": bool(r.get("Is parameter")),
                "snr": float(snr),
            }
        )
        if r.get("Is input") and not r.get("Is parameter"):
            fstd = r.get("Float Std")
            if isinstance(fstd, (int, float)):
                bucket["_input_float_stds"].append(float(fstd))

    for bucket in by_op.values():
        stds = bucket.pop("_input_float_stds")
        if len(stds) >= 2:
            denom = max(min(stds), 1e-9)
            bucket["inputs_float_std_ratio"] = max(stds) / denom
        else:
            bucket["inputs_float_std_ratio"] = None

    return sorted(by_op.values(), key=lambda b: b["max_snr"], reverse=True)[:top_k]


def compute_graphwise_jumps(
    layerwise: Dict[str, float],
    graphwise: Dict[str, float],
    simplified_ops_payload: Optional[dict],
    excess_threshold: float,
) -> List[dict]:
    """D3 differential analysis.

    Find adjacent COMPUTING_OP pairs whose graphwise SNR gap
    (``gw[op_next] - gw[op_prev]``) exceeds the downstream op's isolated
    layerwise contribution by more than ``excess_threshold``. The non-COMPUTING_OP
    sitting between them in the simplified graph are likely contributing the
    extra noise (e.g. a Concat with mismatched input scales between two Convs).

    Args:
        layerwise: ``{op_name: snr}`` from ``layerwise_error_analyse``. Only
            COMPUTING_OP keys are present.
        graphwise: ``{op_name: snr}`` from ``graphwise_error_analyse``. Same
            COMPUTING_OP scope.
        simplified_ops_payload: The ``outputs/iter_<N>/simplified_ops.json`` body,
            i.e. ``{"ops": [{"name": ..., "op_type": ...}, ...]}``. Used as the
            ground-truth topological order.
        excess_threshold: Minimum positive ``intervening_excess`` required to
            list a region. Default in the harness is 0.02.

    Returns:
        List of dicts (sorted descending by ``intervening_excess``) with keys
        ``op_prev``, ``op_next``, ``delta_graphwise``,
        ``isolated_layerwise_op_next``, ``intervening_excess``,
        ``intervening_non_computing_ops``.
    """
    if not simplified_ops_payload or not isinstance(simplified_ops_payload, dict):
        return []
    ops = simplified_ops_payload.get("ops") or []
    if not ops:
        return []

    jumps: List[dict] = []
    prev_computing: Optional[dict] = None
    intervening: List[dict] = []
    for node in ops:
        op_type = node.get("op_type")
        op_name = node.get("name")
        if op_type in COMPUTING_OP_TYPES:
            if prev_computing is not None and op_name:
                gw_prev = graphwise.get(prev_computing["name"])
                gw_next = graphwise.get(op_name)
                lw_next = layerwise.get(op_name)
                if (
                    isinstance(gw_prev, (int, float))
                    and isinstance(gw_next, (int, float))
                    and isinstance(lw_next, (int, float))
                ):
                    delta = float(gw_next) - float(gw_prev)
                    excess = delta - float(lw_next)
                    if excess > excess_threshold:
                        jumps.append(
                            {
                                "op_prev": prev_computing["name"],
                                "op_next": op_name,
                                "delta_graphwise": delta,
                                "isolated_layerwise_op_next": float(lw_next),
                                "intervening_excess": excess,
                                "intervening_non_computing_ops": [
                                    {"name": n["name"], "type": n["op_type"]}
                                    for n in intervening
                                    if n.get("name")
                                ],
                            }
                        )
            prev_computing = {"name": op_name, "type": op_type}
            intervening = []
        elif op_type and op_name and prev_computing is not None:
            intervening.append(node)

    return sorted(jumps, key=lambda j: j["intervening_excess"], reverse=True)


def filter_layer_stats_for(
    records: List[dict],
    interested_op_names: List[str],
) -> List[dict]:
    """Filter statistical records by op-name and keep the requested order.

    Used to derive the legacy ``layer_stats.json`` (top-K filtered) view from
    the full ``layer_stats_full.json`` artifact.
    """
    if not records or not interested_op_names:
        return []
    interest_set = set(interested_op_names)
    filtered = [r for r in records if r.get("Op name") in interest_set]
    order = {name: idx for idx, name in enumerate(interested_op_names)}
    filtered.sort(key=lambda r: order.get(r.get("Op name"), 1 << 30))
    return filtered
