"""Pure mapping from setting JSON -> esp_ppq.QuantizationSetting.

This module is intentionally side-effect free. It reads a dict (the iteration's
``setting.json`` content) and returns a ``QuantizationSetting`` plus a list of warnings.
The harness handles I/O; tests can call ``apply()`` directly with synthetic dicts.

The schema is documented in ``references/setting_json_schema.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import get_target_platform


_VALID_CALIB_ALGORITHMS = {"kl", "percentile", "mse", "minmax", "isotone"}
_VALID_FUSION_ALIGNMENTS = {"None", "Align to Large", "Align to Output"}

# Targets whose esp-ppq quantizer policy includes POWER_OF_2. See
# esp-ppq/esp_ppq/quantization/quantizer/EspdlQuantizer.py — every concrete EspdlQuantizer
# subclass declares quantize_policy = SYMMETRICAL + LINEAR + PER_TENSOR + POWER_OF_2,
# which means scale = 2 ^ exponent for every quantized tensor.
_POWER_OF_2_TARGETS = {"esp32p4", "esp32s3", "c"}

# Targets whose esp-ppq quantizer enables PER_CHANNEL weight quantization for
# Conv / ConvTranspose / Gemm. See EspdlQuantizer.create_espdl_quant_config():
# only operations dispatched to a P4 platform (ESPDL_INT8 / ESPDL_INT16 /
# ESPDL_H_PRE_INT16) get _PER_CHANNEL_POLICY on their weight tensor. The S3 and
# C variants stay per-tensor. Layer-wise equalization is "Not recommended" for
# per-channel quantization (see esp-ppq/md_doc/Passes/LayerwiseEqualization.md
# §Usage and equalization.py LayerwiseEqualizationPass docstring): the per-channel
# weight scales already absorb the equalization scale, so the pass is *expected*
# to be a no-op or mildly hurt. Empirically, however, some networks still gain
# accuracy from equalization on esp32p4 — the activation side stays per-tensor and
# can benefit when equalization narrows its range. The harness therefore treats
# this as **warn-only**: equalization runs as configured, but a warning is emitted
# explaining the official guidance and pointing at safer Phase-3 levers (TQT,
# better calib_algorithm, mixed-precision int16 via dispatching_table).
_PER_CHANNEL_WEIGHT_TARGETS = {"esp32p4"}


@dataclass
class ApplyResult:
    setting: Any
    warnings: List[str]
    summary: Dict[str, Any]


def _ensure_subdict(payload: dict, key: str) -> dict:
    val = payload.get(key)
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise ValueError(f"Field '{key}' must be an object, got {type(val).__name__}")
    return val


def _normalise_interested_layers(value: Any) -> Optional[List[str]]:
    """Map JSON value to esp-ppq's expected interested_layers shape.

    None or [] in JSON -> None in Python (means "all"). Some passes treat empty list as
    "no layers", which is rarely what the agent wants; the harness consistently maps
    null/missing to None and only a populated list to that list.
    """
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) == 0:
            return None
        if not all(isinstance(item, str) for item in value):
            raise ValueError("interested_layers must be a list of strings")
        return list(value)
    raise ValueError(
        "interested_layers must be a list of strings or null, got "
        + type(value).__name__
    )


def _check_mutex(payload: dict) -> None:
    """Reject combinations that fight each other at training time."""
    on = []
    for key in ("tqt_optimization", "lsq_optimization", "blockwise_reconstruction"):
        block = payload.get(key) or {}
        if isinstance(block, dict) and block.get("enabled"):
            on.append(key)
    if len(on) > 1:
        raise ValueError(
            f"Mutually exclusive passes enabled simultaneously: {on}. "
            "Pick exactly one of TQT / LSQ / blockwise_reconstruction."
        )


def _check_target_compatibility(
    payload: Dict[str, Any], target: str, warnings: List[str]
) -> None:
    """Reconcile the requested passes with the target's quantization policy.

    All esp-dl targets (``esp32p4``, ``esp32s3``, ``c``) use a POWER_OF_2 quantization
    policy (every concrete ``EspdlQuantizer`` subclass declares
    ``quantize_policy = SYMMETRICAL + LINEAR + PER_TENSOR + POWER_OF_2``). A few esp-ppq
    optimization passes were designed for continuous-scale quantizers and either no-op
    or actively misbehave under POWER_OF_2:

    * ``lsq_optimization`` — the ``LSQDelegator`` gates scale training behind
      ``not policy.has_property(POWER_OF_2)``, so on esp-dl targets it silently
      degenerates to weight-only tuning while still paying TQT-level PC time. The harness
      disables LSQ here and recommends TQT (which trains ``log2_scale`` and is
      POWER_OF_2-native).

    * ``blockwise_reconstruction`` with ``is_scale_trainable=True`` — the default is
      already ``False`` for this exact reason. If the agent overrode it to ``True``, we
      revert to ``False`` and warn; weights are still trained but scales stay POWER_OF_2.

    Per-channel weight quantization is a separate target-level distinction
    (`_PER_CHANNEL_WEIGHT_TARGETS`):

    * ``equalization`` — esp-ppq's ``LayerwiseEqualizationPass`` is officially
      "Not recommended" for per-channel quantization (see
      esp-ppq/md_doc/Passes/LayerwiseEqualization.md §Usage). On esp32p4, every
      ``Conv / ConvTranspose / Gemm`` has its weight quantized per-channel, so the
      pass is *expected* to be a no-op (the per-channel weight scales already
      absorb the equalization scale) or mildly hurt. **Empirically, this is not
      always the case** — for some networks, narrowing the per-tensor activation
      range still buys accuracy on esp32p4, even though the weight side is
      per-channel. The harness therefore treats this as **warn-only**:
      ``equalization`` runs as configured, but a warning is emitted pointing at
      safer Phase-3 levers (TQT, better ``calib_algorithm``, mixed-precision int16
      via ``dispatching_table``) so the agent can fall back if it regresses.

    The function may mutate ``payload`` in place (e.g. forcing
    ``blockwise_reconstruction.is_scale_trainable=False`` on POWER_OF_2 targets) so
    subsequent ``_apply_*`` helpers see the cleaned-up flags. For ``equalization`` on
    per-channel targets, ``payload`` is **not** modified — only ``warnings`` is
    appended to. The harness surfaces the warning list in ``console.log`` and
    ``iteration_index.json``.
    """
    if not isinstance(target, str):
        return
    target_norm = target.strip().lower()

    if target_norm in _PER_CHANNEL_WEIGHT_TARGETS:
        eq_block = payload.get("equalization") or {}
        if isinstance(eq_block, dict) and eq_block.get("enabled"):
            warnings.append(
                "equalization is officially 'Not recommended' for target '"
                f"{target}': esp-ppq's EspdlQuantizer enables per-channel weight "
                "quantization for every Conv/ConvTranspose/Gemm on P4 platforms "
                "(_P4_PLATFORMS in EspdlQuantizer.py), and "
                "esp-ppq/md_doc/Passes/LayerwiseEqualization.md §Usage flags the "
                "pass as not recommended for per-channel quantization. "
                "Letting equalization run anyway because some networks empirically "
                "still gain accuracy from it on esp32p4 (the activation side stays "
                "per-tensor and can benefit when its range is narrowed). If this "
                "iteration regresses vs the baseline, drop equalization next round "
                "and reach for tqt_optimization (POWER_OF_2-native, no device cost), "
                "a different calib_algorithm, or surgical mixed precision via "
                "dispatching_table on the worst layers instead."
            )

    if target_norm not in _POWER_OF_2_TARGETS:
        return

    lsq_block = payload.get("lsq_optimization") or {}
    if isinstance(lsq_block, dict) and lsq_block.get("enabled"):
        warnings.append(
            "lsq_optimization is incompatible with target '"
            f"{target}': all esp-dl targets use POWER_OF_2 quantization, which makes "
            "LSQ's continuous-scale training a no-op (esp-ppq's LSQDelegator disables "
            "scale training when POWER_OF_2 is set). Disabling LSQ for this iteration; "
            "use tqt_optimization instead — it trains log2_scale and is POWER_OF_2-native."
        )
        new_block = dict(lsq_block)
        new_block["enabled"] = False
        payload["lsq_optimization"] = new_block

    blk_block = payload.get("blockwise_reconstruction") or {}
    if (
        isinstance(blk_block, dict)
        and blk_block.get("enabled")
        and bool(blk_block.get("is_scale_trainable", False))
    ):
        warnings.append(
            "blockwise_reconstruction.is_scale_trainable=True is incompatible with "
            f"target '{target}' (POWER_OF_2). Forcing it back to False so only weights "
            "are trained; the scale-training path would attempt continuous-scale updates "
            "that the device cannot execute."
        )
        new_block = dict(blk_block)
        new_block["is_scale_trainable"] = False
        payload["blockwise_reconstruction"] = new_block


def _apply_calib_algorithm(
    setting, payload: dict, warnings: List[str]
) -> Optional[str]:
    algo = payload.get("calib_algorithm")
    if algo is None:
        return None
    if not isinstance(algo, str):
        raise ValueError("calib_algorithm must be a string")
    algo_norm = algo.lower()
    if algo_norm not in _VALID_CALIB_ALGORITHMS:
        raise ValueError(
            f"calib_algorithm '{algo}' not in {sorted(_VALID_CALIB_ALGORITHMS)}"
        )
    setting.quantize_activation_setting.calib_algorithm = algo_norm
    return algo_norm


def _apply_equalization(setting, payload: dict, warnings: List[str]) -> Optional[dict]:
    block = _ensure_subdict(payload, "equalization")
    if not block.get("enabled"):
        return None
    # Silent-default-mismatch guard. esp-ppq's EqualizationSetting defaults
    # opt_level=1 (does NOT cross Add/Sub branches), but the skill's lever-3d
    # template in compare_iterations.py recommends opt_level=2 (crosses branches).
    # The two values are not interchangeable: on residual / inverted-residual
    # architectures (ResNet, MobileNet-V2/V3, etc.), opt_level=1 silently skips
    # every Add/Sub-bounded chain — which is precisely the structure lever 3d
    # was designed to reach. Writing only {"enabled": true} therefore quietly
    # disables most of the pass on these networks. Surface this explicitly so
    # the agent either copies the full lever-3d snippet or makes opt_level=1 an
    # explicit, deliberate choice.
    if "opt_level" not in block:
        warnings.append(
            "equalization.opt_level was not specified. esp-ppq defaults "
            "opt_level=1 (does NOT cross Add/Sub branches), but the skill's "
            "lever-3d template recommends opt_level=2 (crosses Add/Sub "
            "branches — required for residual / inverted-residual networks "
            "such as ResNet, MobileNet-V2/V3 inverted residuals, etc.). "
            "Resolved opt_level=1 will silently skip every Add/Sub-bounded "
            "chain in this iteration. If you intended to test lever 3d, copy "
            "the full snippet from comparison.json['next_step_hint']['advice'] "
            "verbatim. If you really wanted opt_level=1 (e.g. for a heavily "
            "branched graph like YOLO/transformer), set it explicitly to "
            "silence this warning."
        )
    setting.equalization = True
    eq = setting.equalization_setting
    if "iterations" in block:
        eq.iterations = int(block["iterations"])
    if "value_threshold" in block:
        eq.value_threshold = float(block["value_threshold"])
    if "opt_level" in block:
        eq.opt_level = int(block["opt_level"])
    if "including_bias" in block:
        eq.including_bias = bool(block["including_bias"])
    if "including_act" in block:
        eq.including_act = bool(block["including_act"])
    if "interested_layers" in block:
        eq.interested_layers = _normalise_interested_layers(block["interested_layers"])
    return {
        "iterations": eq.iterations,
        "value_threshold": eq.value_threshold,
        "opt_level": eq.opt_level,
        "interested_layers": eq.interested_layers,
    }


def _apply_bias_correct(setting, payload: dict) -> Optional[dict]:
    block = _ensure_subdict(payload, "bias_correct")
    if not block.get("enabled"):
        return None
    setting.bias_correct = True
    bc = setting.bias_correct_setting
    if "block_size" in block:
        bc.block_size = int(block["block_size"])
    if "steps" in block:
        bc.steps = int(block["steps"])
    if "interested_layers" in block:
        bc.interested_layers = _normalise_interested_layers(block["interested_layers"])
    return {
        "block_size": bc.block_size,
        "steps": bc.steps,
        "interested_layers": bc.interested_layers,
    }


def _apply_weight_split(setting, payload: dict) -> Optional[dict]:
    block = _ensure_subdict(payload, "weight_split")
    if not block.get("enabled"):
        return None
    interested = _normalise_interested_layers(block.get("interested_layers"))
    if not interested:
        raise ValueError(
            "weight_split.interested_layers must be a non-empty list when enabled "
            "(esp-ppq treats null/[] as 'no layers' for this pass)"
        )
    setting.weight_split = True
    ws = setting.weight_split_setting
    ws.interested_layers = interested
    if "value_threshold" in block:
        ws.value_threshold = float(block["value_threshold"])
    if "method" in block:
        method = str(block["method"]).lower()
        if method not in {"balance", "random"}:
            raise ValueError(
                f"weight_split.method must be 'balance' or 'random', got '{method}'"
            )
        ws.method = method
    return {
        "interested_layers": ws.interested_layers,
        "value_threshold": ws.value_threshold,
        "method": ws.method,
    }


def _apply_training_pass(
    setting, payload: dict, key: str, attr: str, sub_attr: str
) -> Optional[dict]:
    block = _ensure_subdict(payload, key)
    if not block.get("enabled"):
        return None
    setattr(setting, attr, True)
    sub = getattr(setting, sub_attr)
    for jkey, attr_name in [
        ("lr", "lr"),
        ("steps", "steps"),
        ("block_size", "block_size"),
        ("is_scale_trainable", "is_scale_trainable"),
        ("gamma", "gamma"),
        ("collecting_device", "collecting_device"),
    ]:
        if jkey in block:
            setattr(sub, attr_name, block[jkey])
    if key == "tqt_optimization" and "int_lambda" in block:
        sub.int_lambda = float(block["int_lambda"])
    if "interested_layers" in block:
        sub.interested_layers = _normalise_interested_layers(block["interested_layers"])
    summary = {
        "lr": sub.lr,
        "steps": sub.steps,
        "block_size": sub.block_size,
        "is_scale_trainable": sub.is_scale_trainable,
        "gamma": sub.gamma,
        "collecting_device": sub.collecting_device,
        "interested_layers": sub.interested_layers,
    }
    if key == "tqt_optimization":
        summary["int_lambda"] = sub.int_lambda
    return summary


def _apply_fusion_alignment(setting, payload: dict) -> Optional[dict]:
    block = _ensure_subdict(payload, "fusion_alignment")
    if not block:
        return None
    fs = setting.fusion_setting
    summary: Dict[str, Any] = {}
    for key in (
        "align_avgpooling_to",
        "align_elementwise_to",
        "align_concat_to",
        "align_resize_to",
    ):
        if key in block:
            value = block[key]
            if value not in _VALID_FUSION_ALIGNMENTS:
                raise ValueError(
                    f"fusion_alignment.{key} must be one of "
                    f"{sorted(_VALID_FUSION_ALIGNMENTS)}, got '{value}'"
                )
            setattr(fs, key, value)
            summary[key] = value
    if "force_alignment_overlap" in block:
        fs.force_alignment_overlap = bool(block["force_alignment_overlap"])
        summary["force_alignment_overlap"] = fs.force_alignment_overlap
    return summary or None


def _apply_dispatching_table(
    setting,
    payload: dict,
    target: str,
    known_op_names: Optional[set],
    warnings: List[str],
) -> Optional[List[dict]]:
    entries = payload.get("dispatching_table")
    if entries is None:
        return None
    if not isinstance(entries, list):
        raise ValueError("dispatching_table must be a list of {op, bits} objects")
    applied: List[dict] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"dispatching_table[{idx}] must be an object")
        op = entry.get("op")
        bits = entry.get("bits")
        if not isinstance(op, str) or not isinstance(bits, int):
            raise ValueError(
                f"dispatching_table[{idx}] requires string 'op' and int 'bits', got {entry!r}"
            )
        if bits not in (8, 16):
            raise ValueError(
                f"dispatching_table[{idx}].bits must be 8 or 16, got {bits}"
            )
        if known_op_names is not None and op not in known_op_names:
            warnings.append(
                f"dispatching_table op '{op}' not found in simplified graph; "
                "the dispatch will silently no-op."
            )
        platform = get_target_platform(target, bits)
        setting.dispatching_table.append(op, platform)
        applied.append({"op": op, "bits": bits})
    return applied


def _apply_extra(setting, payload: dict, warnings: List[str]) -> Optional[dict]:
    extra = payload.get("extra")
    if not extra:
        return None
    if not isinstance(extra, dict):
        raise ValueError("'extra' must be an object")
    applied: Dict[str, Any] = {}
    for key, value in extra.items():
        if not hasattr(setting, key):
            warnings.append(f"extra.{key} not present on QuantizationSetting; skipped.")
            continue
        setattr(setting, key, value)
        applied[key] = value
    return applied or None


def apply(
    payload: Dict[str, Any],
    target: str,
    known_op_names: Optional[set] = None,
) -> ApplyResult:
    """Translate a setting payload into a fresh QuantizationSetting.

    Args:
        payload: parsed setting JSON.
        target: chip target string ("esp32p4", "esp32s3", "c"). Needed for mixed precision.
        known_op_names: if provided, dispatching_table entries pointing to unknown ops emit
            warnings (without failing).
    """
    if not isinstance(payload, dict):
        raise ValueError("setting payload must be a JSON object")

    iteration_id = payload.get("iteration_id")
    if not isinstance(iteration_id, int) or iteration_id < 0:
        raise ValueError("iteration_id must be a non-negative integer")

    _check_mutex(payload)

    setting = QuantizationSettingFactory.espdl_setting()
    warnings: List[str] = []
    summary: Dict[str, Any] = {
        "iteration_id": iteration_id,
        "rationale": payload.get("rationale"),
    }

    # Reconcile the requested passes with the target's quantization policy. This may
    # mutate ``payload`` (e.g. disable lsq_optimization on POWER_OF_2 targets) so the
    # subsequent _apply_* helpers see the cleaned flags. Any mutations are logged to
    # ``warnings`` for the harness to expose.
    _check_target_compatibility(payload, target, warnings)

    summary["calib_algorithm"] = (
        _apply_calib_algorithm(setting, payload, warnings)
        or setting.quantize_activation_setting.calib_algorithm
    )
    summary["equalization"] = _apply_equalization(setting, payload, warnings)
    summary["bias_correct"] = _apply_bias_correct(setting, payload)
    summary["weight_split"] = _apply_weight_split(setting, payload)
    summary["tqt_optimization"] = _apply_training_pass(
        setting,
        payload,
        "tqt_optimization",
        "tqt_optimization",
        "tqt_optimization_setting",
    )
    summary["lsq_optimization"] = _apply_training_pass(
        setting,
        payload,
        "lsq_optimization",
        "lsq_optimization",
        "lsq_optimization_setting",
    )
    summary["blockwise_reconstruction"] = _apply_training_pass(
        setting,
        payload,
        "blockwise_reconstruction",
        "blockwise_reconstruction",
        "blockwise_reconstruction_setting",
    )
    summary["fusion_alignment"] = _apply_fusion_alignment(setting, payload)
    summary["dispatching_table"] = _apply_dispatching_table(
        setting, payload, target, known_op_names, warnings
    )
    summary["extra"] = _apply_extra(setting, payload, warnings)

    return ApplyResult(setting=setting, warnings=warnings, summary=summary)


def baseline_payload() -> dict:
    """Iter-0 baseline = pure espdl_setting() defaults."""
    return {
        "iteration_id": 0,
        "rationale": "default espdl_setting() baseline",
    }


# Allow `python apply_setting.py path/to/setting.json target` for quick smoke testing.
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: apply_setting.py <setting.json> <target>", file=sys.stderr)
        sys.exit(2)
    path, target = sys.argv[1], sys.argv[2]
    with open(path) as f:
        payload = json.load(f)
    result = apply(payload, target=target)
    print(
        json.dumps(
            {"summary": result.summary, "warnings": result.warnings},
            indent=2,
            default=str,
        )
    )
