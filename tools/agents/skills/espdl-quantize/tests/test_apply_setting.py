"""Lightweight unit tests for apply_setting.apply().

These tests don't run any quantization — they only verify the JSON → QuantizationSetting
mapping plus the POWER_OF_2 target compatibility checks. Run them in any Python env that
has ``esp_ppq`` importable. ``$SKILL_DIR`` is the absolute path of the espdl-quantize
skill directory (the directory holding ``SKILL.md``); the skill is agent-directory
agnostic and may live under ``.cursor/skills/``, ``.opencode/skills/``, or any other
agent's skills folder:

    python -m pytest "$SKILL_DIR/tests/test_apply_setting.py" -v

If pytest isn't available you can run the file directly: it has a ``__main__`` block that
runs every ``test_*`` function.
"""

from __future__ import annotations

import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

import apply_setting  # noqa: E402


def _baseline_payload(extra: dict | None = None) -> dict:
    payload = {"iteration_id": 1, "rationale": "test"}
    if extra:
        payload.update(extra)
    return payload


def test_baseline_is_default_espdl_setting():
    result = apply_setting.apply(apply_setting.baseline_payload(), target="esp32p4")
    setting = result.setting
    # espdl_setting() defaults to kl
    assert setting.quantize_activation_setting.calib_algorithm == "kl"
    assert setting.equalization is False
    assert setting.bias_correct is False
    assert setting.weight_split is False
    assert setting.tqt_optimization is False
    assert result.warnings == []


def test_calib_algorithm_normalised_lowercase():
    result = apply_setting.apply(
        _baseline_payload({"calib_algorithm": "Percentile"}), target="esp32p4"
    )
    assert result.setting.quantize_activation_setting.calib_algorithm == "percentile"


def test_bad_calib_algorithm_rejected():
    try:
        apply_setting.apply(
            _baseline_payload({"calib_algorithm": "bogus"}), target="esp32p4"
        )
    except ValueError as exc:
        assert "calib_algorithm" in str(exc)
    else:
        raise AssertionError("expected ValueError for bad calib_algorithm")


def test_equalization_fields_applied():
    payload = _baseline_payload(
        {
            "equalization": {
                "enabled": True,
                "iterations": 6,
                "value_threshold": 1.0,
                "opt_level": 2,
            }
        }
    )
    result = apply_setting.apply(payload, target="esp32s3")
    setting = result.setting
    assert setting.equalization is True
    assert setting.equalization_setting.iterations == 6
    assert setting.equalization_setting.value_threshold == 1.0
    assert setting.equalization_setting.opt_level == 2


def test_equalization_warns_when_opt_level_missing():
    # Silent-default-mismatch guard. esp-ppq's EqualizationSetting defaults
    # opt_level=1 (does NOT cross Add/Sub branches), but the skill's lever-3d
    # template recommends opt_level=2 (crosses branches — required for
    # residual / inverted-residual networks). Writing only {"enabled": true}
    # used to silently inherit opt_level=1 with no signal to the agent. Now
    # apply_setting emits a warning so the agent can either copy the full
    # snippet or make opt_level=1 an explicit choice. We test on esp32s3 to
    # avoid the per-channel warning that fires on esp32p4 (which lives on the
    # same payload but is a separate concern).
    payload = _baseline_payload({"equalization": {"enabled": True}})
    result = apply_setting.apply(payload, target="esp32s3")
    assert result.setting.equalization is True
    # esp-ppq default: opt_level should resolve to 1 because the agent did
    # not override it.
    assert result.setting.equalization_setting.opt_level == 1
    assert any(
        "opt_level" in w and "Add/Sub" in w for w in result.warnings
    ), f"expected opt_level/Add-Sub warning, got {result.warnings!r}"


def test_equalization_silent_when_opt_level_set_explicitly():
    # The warning is about the *silent* default mismatch. When the agent sets
    # opt_level explicitly (either 1 or 2), the choice is deliberate and we
    # must not nag — otherwise every iteration that uses opt_level=1 for a
    # legitimate reason (heavy concat/branch graph) would emit a stale
    # warning. We test with opt_level=1 because that's the value most likely
    # to look "wrong" without context.
    payload = _baseline_payload({"equalization": {"enabled": True, "opt_level": 1}})
    result = apply_setting.apply(payload, target="esp32s3")
    assert result.setting.equalization_setting.opt_level == 1
    assert not any(
        "opt_level" in w and "Add/Sub" in w for w in result.warnings
    ), f"unexpected opt_level warning, got {result.warnings!r}"


def test_weight_split_requires_interested_layers():
    payload = _baseline_payload(
        {"weight_split": {"enabled": True, "value_threshold": 1.5}}
    )
    try:
        apply_setting.apply(payload, target="esp32p4")
    except ValueError as exc:
        assert "interested_layers" in str(exc)
    else:
        raise AssertionError(
            "expected ValueError when weight_split.enabled but interested_layers empty"
        )


def test_weight_split_with_layers():
    payload = _baseline_payload(
        {
            "weight_split": {
                "enabled": True,
                "interested_layers": ["/op/Conv"],
                "value_threshold": 1.5,
                "method": "balance",
            }
        }
    )
    result = apply_setting.apply(payload, target="esp32p4")
    assert result.setting.weight_split is True
    assert result.setting.weight_split_setting.interested_layers == ["/op/Conv"]
    assert result.setting.weight_split_setting.value_threshold == 1.5
    assert result.setting.weight_split_setting.method == "balance"


def test_dispatching_table_warns_unknown_op():
    payload = _baseline_payload(
        {"dispatching_table": [{"op": "/missing/Conv", "bits": 16}]}
    )
    result = apply_setting.apply(
        payload,
        target="esp32p4",
        known_op_names={"/known/Conv"},
    )
    # The dispatch is still applied but a warning is recorded.
    assert any("/missing/Conv" in w for w in result.warnings)
    # And the dispatching_table summary still reflects the request.
    assert result.summary["dispatching_table"] == [{"op": "/missing/Conv", "bits": 16}]


def test_dispatching_table_known_ops_no_warn():
    payload = _baseline_payload(
        {"dispatching_table": [{"op": "/known/Conv", "bits": 16}]}
    )
    result = apply_setting.apply(
        payload,
        target="esp32p4",
        known_op_names={"/known/Conv"},
    )
    assert result.warnings == []


def test_mutex_tqt_lsq_rejected():
    payload = _baseline_payload(
        {
            "tqt_optimization": {"enabled": True},
            "lsq_optimization": {"enabled": True},
        }
    )
    try:
        apply_setting.apply(payload, target="esp32p4")
    except ValueError as exc:
        assert "Mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected ValueError for TQT+LSQ both enabled")


def test_fusion_alignment_validation():
    payload = _baseline_payload({"fusion_alignment": {"align_concat_to": "Wrong Mode"}})
    try:
        apply_setting.apply(payload, target="esp32p4")
    except ValueError as exc:
        assert "fusion_alignment" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid fusion alignment value")


def test_fusion_alignment_applied():
    payload = _baseline_payload(
        {
            "fusion_alignment": {
                "align_elementwise_to": "Align to Large",
                "force_alignment_overlap": True,
            }
        }
    )
    result = apply_setting.apply(payload, target="esp32p4")
    assert result.setting.fusion_setting.align_elementwise_to == "Align to Large"
    assert result.setting.fusion_setting.force_alignment_overlap is True


def test_extra_pass_through_with_warning_for_unknown_attr():
    payload = _baseline_payload({"extra": {"channel_split": True, "no_such_field": 1}})
    result = apply_setting.apply(payload, target="esp32p4")
    assert result.setting.channel_split is True
    assert any("no_such_field" in w for w in result.warnings)


def test_iteration_id_required_non_negative():
    try:
        apply_setting.apply({"iteration_id": -1}, target="esp32p4")
    except ValueError as exc:
        assert "iteration_id" in str(exc)
    else:
        raise AssertionError("expected ValueError for negative iteration_id")


def test_lsq_skipped_on_power_of_two_target():
    # esp32p4 / esp32s3 / c are all POWER_OF_2 → LSQ scale training is structurally
    # disabled; the harness should disable the flag and warn rather than waste an iteration.
    for target in ("esp32p4", "esp32s3", "c", "ESP32P4"):
        payload = _baseline_payload({"lsq_optimization": {"enabled": True}})
        result = apply_setting.apply(payload, target=target)
        assert (
            result.setting.lsq_optimization is False
        ), f"target={target}: lsq_optimization should be disabled on POWER_OF_2"
        assert any(
            "lsq_optimization" in w.lower() and "power_of_2" in w.lower()
            for w in result.warnings
        ), f"target={target}: expected POWER_OF_2 warning, got {result.warnings!r}"


def test_blockwise_scale_trainable_forced_off_on_power_of_two():
    payload = _baseline_payload(
        {
            "blockwise_reconstruction": {
                "enabled": True,
                "is_scale_trainable": True,
                "steps": 100,
            }
        }
    )
    result = apply_setting.apply(payload, target="esp32p4")
    # blockwise itself remains enabled; only is_scale_trainable is reverted to False.
    assert result.setting.blockwise_reconstruction is True
    assert result.setting.blockwise_reconstruction_setting.is_scale_trainable is False
    assert any(
        "is_scale_trainable" in w and "POWER_OF_2" in w for w in result.warnings
    ), f"expected POWER_OF_2 warning, got {result.warnings!r}"


def test_tqt_unaffected_on_power_of_two():
    # TQT is POWER_OF_2-native; no warning, flag stays enabled.
    payload = _baseline_payload({"tqt_optimization": {"enabled": True, "steps": 100}})
    result = apply_setting.apply(payload, target="esp32p4")
    assert result.setting.tqt_optimization is True
    assert not any("tqt_optimization" in w.lower() for w in result.warnings)


def test_equalization_runs_with_warning_on_esp32p4():
    # On esp32p4 every Conv/ConvTranspose/Gemm uses per-channel weight quantization
    # (EspdlQuantizer._P4_PLATFORMS), and esp-ppq's official guidance is that layer-wise
    # equalization is "Not recommended" for per-channel quantization. Empirically,
    # however, some networks still benefit (the activation side stays per-tensor and can
    # gain from a narrowed range), so the harness no longer auto-disables the pass —
    # instead it logs a warning that points at safer Phase 3 levers and lets the
    # equalization run as configured. This is the **warn-only** policy: agent-friendly
    # because it surfaces the official caveat without taking the choice away. We test
    # both the lowercase target string and a casing variant to make sure normalisation
    # works.
    for target in ("esp32p4", "ESP32P4"):
        payload = _baseline_payload(
            {"equalization": {"enabled": True, "iterations": 6}}
        )
        result = apply_setting.apply(payload, target=target)
        assert (
            result.setting.equalization is True
        ), f"target={target}: equalization must remain enabled (warn-only policy)"
        assert (
            result.setting.equalization_setting.iterations == 6
        ), f"target={target}: equalization sub-fields must still be applied"
        assert any(
            "equalization" in w.lower()
            and "per-channel" in w.lower()
            and target.lower() in w.lower()
            for w in result.warnings
        ), f"target={target}: expected per-channel warning, got {result.warnings!r}"


def test_equalization_with_interested_layers_runs_on_esp32p4():
    # interested_layers is intentionally honored on esp32p4 even though esp-ppq does NOT
    # support cross-target dispatching (get_target_platform only returns same-chip-family
    # platforms). The agent might want to scope equalization to a subset of Conv chains
    # that empirically improved on this network — that's the whole point of the
    # warn-only policy: surface the caveat, but don't override the agent's decision.
    payload = _baseline_payload(
        {
            "equalization": {
                "enabled": True,
                "iterations": 6,
                "interested_layers": ["/some/Conv"],
            }
        }
    )
    result = apply_setting.apply(payload, target="esp32p4")
    assert result.setting.equalization is True, (
        "equalization must remain enabled on esp32p4 (warn-only) even when "
        "interested_layers is provided — the agent decides whether to keep it next round."
    )
    assert result.setting.equalization_setting.interested_layers == ["/some/Conv"]
    assert any(
        "equalization" in w.lower() and "per-channel" in w.lower()
        for w in result.warnings
    ), f"expected per-channel warning, got {result.warnings!r}"


def test_equalization_unaffected_on_per_tensor_targets():
    # esp32s3 / c keep weights at per-tensor quantization, so equalization is the
    # textbook fix for per-channel weight-range imbalance and should run without any
    # per-channel warning.
    for target in ("esp32s3", "c"):
        payload = _baseline_payload(
            {"equalization": {"enabled": True, "iterations": 6}}
        )
        result = apply_setting.apply(payload, target=target)
        assert (
            result.setting.equalization is True
        ), f"target={target}: equalization should remain enabled on per-tensor target"
        assert not any(
            "equalization" in w.lower() and "per-channel" in w.lower()
            for w in result.warnings
        ), f"target={target}: unexpected per-channel warning, got {result.warnings!r}"


def test_equalization_disabled_payload_emits_no_warning_on_esp32p4():
    # Sanity: when equalization is not requested at all on esp32p4, the per-channel
    # check stays silent. We do not want to clutter every iteration's warnings with a
    # message about a pass the agent did not even ask for.
    payload = _baseline_payload({"equalization": {"enabled": False}})
    result = apply_setting.apply(payload, target="esp32p4")
    assert result.setting.equalization is False
    assert not any(
        "equalization" in w.lower() and "per-channel" in w.lower()
        for w in result.warnings
    )


def _run_all():
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for f in funcs:
        try:
            f()
            print(f"PASS  {f.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append((f.__name__, exc))
            print(f"FAIL  {f.__name__}: {type(exc).__name__}: {exc}")
    print("-" * 40)
    print(f"{len(funcs) - len(failed)}/{len(funcs)} passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
