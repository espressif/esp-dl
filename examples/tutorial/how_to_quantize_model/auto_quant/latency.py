"""AutoQuant Stage 2: on-device latency benchmark.

Measures inference latency on the ESP board for every Top-K candidate
from Stage 1 and writes `topk_report.md`, which the user reads to pick
the model to ship.
"""

import os
import re
import shutil
import subprocess
import time
from typing import Any, List, Optional

import serial

from auto_quant.candidates import load_candidates, save_candidates


SERIAL_BAUD = 115200
SERIAL_TIMEOUT_S = 120
STOP_MARKER = "autoquant_latency: done"
REPORT_FILENAME = "topk_report.md"


def test_candidates_latency(run_dir: str, runtime_config: dict) -> Optional[List[dict]]:
    """Run Stage 2 on `<run_dir>` against the board described by
    `runtime_config`.

    Reads:    <run_dir>/candidates.json, runtime_config[
              "test_app_path", "port", "target", "primary_metric"]
    Writes:   <run_dir>/candidates.json   (each entry gets `latency_ms`)
              <run_dir>/topk_report.md    (Top-K table for human review)
    Returns:  the candidate list, or None if candidates.json is empty.
    """
    candidates_path = os.path.join(run_dir, "candidates.json")

    app_path = os.path.abspath(runtime_config["test_app_path"])
    port = runtime_config["port"]
    target = runtime_config["target"]
    metric_key = runtime_config["primary_metric"]

    candidates = load_candidates(candidates_path)
    if not candidates:
        print(f"[WARN] No candidates in {candidates_path}; nothing to measure.")
        return None

    _clean_workspace(app_path)
    print(f"\n[ESP] idf.py set-target {target}")
    if _idf(app_path, "set-target", target) != 0:
        raise RuntimeError("idf.py set-target failed")

    for i, record in enumerate(candidates):
        print(
            f"\n[Latency Test] [{i + 1}/{len(candidates)}] "
            f"candidate index={record['index']}"
        )
        try:
            record["latency_ms"] = _measure(app_path, port, record)
        except Exception as exc:
            print(
                f"[ERROR] measurement failed for candidate {record['index']}: "
                f"{type(exc).__name__}: {exc}"
            )
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_ms"] = None
        if record["latency_ms"] is None:
            print(f"[result] latency=N/A (measurement failed)")
        else:
            print(f"[result] latency={record['latency_ms']:.3f} ms")
        save_candidates(candidates_path, candidates)

    report_path = _write_topk_report(run_dir, candidates, metric_key, target)
    print(f"\nDONE. Top-K report written to {report_path}")
    print(
        "Review the report and pick the model whose accuracy/latency "
        "trade-off best fits your deployment."
    )

    return candidates


def _measure(app_path: str, port: str, record: dict) -> Optional[float]:
    espdl = _resolve_espdl(record)
    if not espdl or not os.path.isfile(espdl):
        print(f"[WARN] espdl not found for candidate {record['index']} ({espdl})")
        return None

    _patch_sdkconfig(app_path, espdl)
    print(f"\n[ESP] idf.py -p {port} flash")
    if _idf(app_path, "-p", port, "flash") != 0:
        print(f"[WARN] idf.py flash failed for candidate {record['index']}")
        return None

    return parse_latency(_read_serial(port))


def _resolve_espdl(record: dict) -> Optional[str]:
    folder = record.get("folder")
    if not folder:
        return None
    return os.path.abspath(os.path.join(folder, "model.espdl"))


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _write_topk_report(
    run_dir: str, candidates: List[dict], metric_key: str, target: str
) -> str:
    report_path = os.path.join(run_dir, REPORT_FILENAME)

    def fmt(v: Any, prec: int) -> str:
        return f"{v:.{prec}f}" if _is_num(v) else "—"

    def row(c: dict) -> str:
        return (
            f"| {c.get('index', 0):04d} | {c.get('folder', '—')} "
            f"| {fmt(c.get(metric_key), 4)} | {fmt(c.get('latency_ms'), 3)} |"
        )

    measured = [c for c in candidates if _is_num(c.get("latency_ms"))]
    failed = [c for c in candidates if c not in measured]
    by_metric = sorted(
        measured,
        key=lambda c: (
            c.get(metric_key) if _is_num(c.get(metric_key)) else float("-inf")
        ),
        reverse=True,
    )
    by_latency = sorted(measured, key=lambda c: c["latency_ms"])

    head = f"| index | folder | {metric_key} | latency_ms |\n|---|---|---|---|"

    def section(title: str, rows: List[dict]) -> str:
        if not rows:
            return f"## {title}\n\n_None._"
        return f"## {title}\n\n{head}\n" + "\n".join(row(c) for c in rows)

    blocks = [
        "# AutoQuant Top-K Report",
        f"- run dir: `{run_dir}`\n"
        f"- target: `{target}`\n"
        f"- primary metric: `{metric_key}`\n"
        f"- candidates: {len(candidates)} "
        f"(measured: {len(measured)}, failed: {len(failed)})",
        "Per-candidate quantization strategy and parameters are not duplicated "
        "here; see `candidates.json` (full list) and `<folder>/config.json` "
        "(per-experiment snapshot). Pick a row whose accuracy/latency "
        "trade-off you like, then ship the `model.espdl` from its `folder`.",
        section(f"Ranked by `{metric_key}` (desc)", by_metric),
        section("Ranked by `latency_ms` (asc)", by_latency),
    ]
    if failed:
        blocks.append(section("Failed measurements", failed))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")
    return report_path


def _clean_workspace(app_path: str) -> None:
    """Wipe sdkconfig + build/ + managed_components/ so a `set-target` on
    a different chip won't reuse stale, incompatible components."""
    for f in ("sdkconfig", "sdkconfig.old", "dependencies.lock"):
        p = os.path.join(app_path, f)
        if os.path.isfile(p):
            os.remove(p)
    for d in ("build", "managed_components"):
        p = os.path.join(app_path, d)
        if os.path.isdir(p):
            shutil.rmtree(p)


def _patch_sdkconfig(app_path: str, espdl_abs: str) -> None:
    sdkconfig = os.path.join(app_path, "sdkconfig")
    with open(sdkconfig, encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(r"^CONFIG_MODEL_FILE_PATH=.*$", re.MULTILINE)
    if not pattern.search(text):
        raise RuntimeError("CONFIG_MODEL_FILE_PATH not found in sdkconfig")
    text = pattern.sub(f'CONFIG_MODEL_FILE_PATH="{espdl_abs}"', text, count=1)
    with open(sdkconfig, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[ESP] CONFIG_MODEL_FILE_PATH -> {espdl_abs}")


def _idf(app_path: str, *args) -> int:
    return subprocess.run(["idf.py", *args], cwd=app_path).returncode


def _read_serial(port: str) -> str:
    print(f"\n[ESP] Reading {port} @ {SERIAL_BAUD} (timeout {SERIAL_TIMEOUT_S}s)")
    deadline = time.time() + SERIAL_TIMEOUT_S
    lines: List[str] = []
    try:
        with serial.Serial(port, SERIAL_BAUD, timeout=1.0) as sport:
            while time.time() < deadline:
                raw = sport.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                print(line, end="")
                lines.append(line)
                if STOP_MARKER in line:
                    return "".join(lines)
            print(f"\n[WARN] timeout waiting for '{STOP_MARKER}'")
    except serial.SerialException as e:
        print(f"[ERROR] serial {port}: {e}")
    return "".join(lines)


def parse_latency(log: str) -> Optional[float]:
    """Extract latency in ms. Prefers 'Latency avg: <ms>'; otherwise averages
    'run:<ms> ms' lines."""
    m = re.search(r"Latency avg:\s*([0-9.]+)", log)
    if m:
        return float(m.group(1))
    runs = [float(x) for x in re.findall(r"run:\s*(\d+)\s*ms", log)]
    return sum(runs) / len(runs) if runs else None
