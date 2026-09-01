#!/usr/bin/env python3
"""Collect operator benchmarks and compare them with the previous CI result."""

import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# Schema v2 records the per-case minimum sample ("min_us") in addition to the
# median/mean and gates the comparison on it: on-chip benchmark noise is
# one-sided (it only ever adds time), so the minimum is the most stable
# estimator across runs.
SCHEMA_VERSION = 2
BENCH_PATTERN = re.compile(
    r"BENCH name=(?P<name>\S+) iters=(?P<iters>\d+) "
    r"(?:min_us=(?P<min>[0-9]+(?:\.[0-9]+)?) )?"
    r"median_us=(?P<median>[0-9]+(?:\.[0-9]+)?) "
    r"mean_us=(?P<mean>[0-9]+(?:\.[0-9]+)?)"
)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Per-target perf gate defaults. With the min-of-N measurement these values
# were found stable enough for all three chips; keep them per-target anyway so
# each chip can be returned independently. Any pipeline can override them via
# GitLab pipeline/project CI/CD variables without a code change: a
# <NAME>_<TARGET> variable (e.g. PERF_RELATIVE_THRESHOLD_PCT_ESP32) beats the
# generic <NAME>, which beats these defaults.
DEFAULT_RELATIVE_THRESHOLD_PCT = {"esp32": 3.0, "esp32s3": 3.0, "esp32p4": 3.0}
DEFAULT_ABSOLUTE_FLOOR_US = {"esp32": 3.0, "esp32s3": 3.0, "esp32p4": 3.0}
GENERIC_RELATIVE_THRESHOLD_PCT = 3.0
GENERIC_ABSOLUTE_FLOOR_US = 3.0


def _env_float(name, target, default):
    """Resolve a perf gate threshold from environment variables.

    Checks `<NAME>_<TARGET>` (e.g. PERF_RELATIVE_THRESHOLD_PCT_ESP32) first,
    then the generic `<NAME>`, then falls back to `default`.
    """
    for key in ("{}_{}".format(name, target.upper()), name):
        value = os.environ.get(key)
        if value:
            return float(value)
    return default


def parse_benchmarks(log):
    """Return all machine-readable benchmark records from a DUT log."""
    if isinstance(log, bytes):
        log = log.decode("utf-8", errors="replace")
    log = ANSI_ESCAPE_PATTERN.sub("", log)
    return [
        {
            "name": match.group("name"),
            "iters": int(match.group("iters")),
            "min_us": float(match.group("min")) if match.group("min") else None,
            "median_us": float(match.group("median")),
            "mean_us": float(match.group("mean")),
        }
        for match in BENCH_PATTERN.finditer(log)
    ]


def _load_results(path):
    path = Path(path)
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "results": []}
    with path.open(encoding="utf-8") as result_file:
        data = json.load(result_file)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported performance result schema: {}".format(
                data.get("schema_version")
            )
        )
    if not isinstance(data.get("results"), list):
        raise RuntimeError("Performance result file has no results array")
    return data


def _result_key(result):
    return (
        result["target"],
        result["idf_version"],
        result["config"],
        result["name"],
    )


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def update_baseline_requested():
    """Return True when this MR is allowed to accept a new performance baseline."""
    if _truthy(os.environ.get("PERF_UPDATE_BASELINE", "")):
        return True
    labels = {
        item.strip().lower()
        for item in os.environ.get("CI_MERGE_REQUEST_LABELS", "").split(",")
        if item.strip()
    }
    return "update-perf-baseline" in labels


def _compared_us(result, metric):
    return float(result["min_us" if metric == "min" else "median_us"])


def compare_result(
    current,
    baseline,
    relative_threshold_pct,
    absolute_floor_us,
    allow_update=False,
):
    """Annotate one result and return an error message when it exceeds limits."""
    if baseline is None:
        current["status"] = "new"
        return None

    # Gate on the minimum sample when both records provide one: the minimum
    # converges to the noise-free compute time, so min-to-min comparisons are
    # much less sensitive to run-to-run timing fluctuation. Records from older
    # baselines without min_us fall back to the median comparison.
    metric = (
        "min"
        if current.get("min_us") is not None and baseline.get("min_us") is not None
        else "median"
    )
    current_us = _compared_us(current, metric)
    baseline_us = _compared_us(baseline, metric)
    absolute_delta_us = current_us - baseline_us
    if baseline_us == 0:
        # Sub-microsecond op: the 1 us timer resolution quantizes the baseline
        # minimum to 0 us, so the relative change is undefined. Enforce only
        # the absolute floor here; a real slowdown still shows up as a jump of
        # several us, which the floor still catches.
        delta_pct = None
        exceeds_relative = current_us > 0
    else:
        delta_pct = absolute_delta_us / baseline_us * 100.0
        exceeds_relative = abs(delta_pct) > relative_threshold_pct

    current.update(
        {
            "metric": metric,
            "baseline_min_us": baseline.get("min_us"),
            "baseline_median_us": baseline.get("median_us"),
            "delta_us": round(absolute_delta_us, 3),
            "delta_pct": round(delta_pct, 3) if delta_pct is not None else None,
        }
    )
    exceeds_limit = exceeds_relative and abs(absolute_delta_us) > absolute_floor_us
    if not exceeds_limit:
        current["status"] = "pass"
        return None
    if allow_update:
        current["status"] = "updated"
        return None
    current["status"] = "fail"
    delta_text = "n/a" if delta_pct is None else "{:+.3f}%".format(delta_pct)
    return (
        "{name}: {metric} {current:.3f} us, baseline {baseline:.3f} us, "
        "change {delta} (limit +/-{limit:.3f}%)"
    ).format(
        name=current["name"],
        metric=metric,
        current=current_us,
        baseline=baseline_us,
        delta=delta_text,
        limit=relative_threshold_pct,
    )


def _write_results(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as result_file:
        json.dump(data, result_file, indent=2, sort_keys=True)
        result_file.write("\n")
    temporary_path.replace(path)


def _write_markdown(path, data):
    lines = [
        "# ESP-DL operator performance comparison",
        "",
        "| Target | IDF | Operator | Model | Metric | Current (us) | Baseline (us) | Delta | Status |",
        "|---|---|---|---|---|---|---:|---:|---|",
    ]
    for result in sorted(data["results"], key=_result_key):
        metric = result.get("metric", "median")
        field = "min_us" if metric == "min" else "median_us"
        baseline_field = "baseline_min_us" if metric == "min" else "baseline_median_us"
        baseline = result.get(baseline_field)
        delta = result.get("delta_pct")
        lines.append(
            "| {target} | {idf_version} | {config} | {name} | {metric} | "
            "{current:.3f} | {baseline} | {delta} | {status} |".format(
                target=result["target"],
                idf_version=result["idf_version"],
                config=result["config"],
                name=result["name"],
                metric=metric,
                current=result[field],
                baseline="-" if baseline is None else "{:.3f}".format(baseline),
                delta="-" if delta is None else "{:+.3f}%".format(delta),
                status=result.get("status"),
            )
        )
    markdown_path = Path(path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_dut_log(dut):
    logfile = Path(dut.logfile)
    if logfile.is_file():
        content = logfile.read_bytes()
        if BENCH_PATTERN.search(
            ANSI_ESCAPE_PATTERN.sub("", content.decode("utf-8", errors="replace"))
        ):
            return content
    return getattr(dut.pexpect_proc, "before", b"")


def record_and_compare(dut, config, target):
    """Record one operator test's benchmarks and enforce the configured threshold."""
    benchmarks = parse_benchmarks(_read_dut_log(dut))
    if not benchmarks:
        raise AssertionError(
            "No BENCH records were found in the DUT log {}".format(dut.logfile)
        )

    idf_version = os.environ.get("IDF_VERSION", "unknown")
    result_path = Path(os.environ.get("PERF_RESULTS_FILE", "perf_results.json"))
    report_path = Path(os.environ.get("PERF_REPORT_FILE", "perf_diff.md"))
    baseline_path = Path(os.environ.get("PERF_BASELINE_FILE", "perf_baseline.json"))
    relative_threshold_pct = _env_float(
        "PERF_RELATIVE_THRESHOLD_PCT",
        target,
        DEFAULT_RELATIVE_THRESHOLD_PCT.get(target, GENERIC_RELATIVE_THRESHOLD_PCT),
    )
    absolute_floor_us = _env_float(
        "PERF_ABSOLUTE_FLOOR_US",
        target,
        DEFAULT_ABSOLUTE_FLOOR_US.get(target, GENERIC_ABSOLUTE_FLOOR_US),
    )
    allow_update = update_baseline_requested()
    print(
        "Performance gate for target {}: change limit +/-{}% and +/-{}us".format(
            target, relative_threshold_pct, absolute_floor_us
        )
    )

    results = _load_results(result_path)
    results.update(
        {
            "schema_version": SCHEMA_VERSION,
            "commit_sha": os.environ.get("PIPELINE_COMMIT_SHA", ""),
            "relative_threshold_pct": relative_threshold_pct,
            "absolute_floor_us": absolute_floor_us,
            "update_baseline": allow_update,
        }
    )
    try:
        baseline_results = _load_results(baseline_path)
    except RuntimeError as error:
        # A baseline recorded with an outdated schema (e.g. before the min_us
        # measurement) cannot be compared against. Treat it as absent: every
        # case is recorded as "new" and the publish job refreshes the baseline.
        print("Ignoring performance baseline {}: {}".format(baseline_path, error))
        baseline_results = {"schema_version": SCHEMA_VERSION, "results": []}
    baseline_by_key = {
        _result_key(result): result for result in baseline_results["results"]
    }
    current_by_key = {
        _result_key(result): result for result in results.get("results", [])
    }

    failures = []
    for benchmark in benchmarks:
        current = {
            **benchmark,
            "target": target,
            "idf_version": idf_version,
            "config": config,
        }
        key = _result_key(current)
        if key in current_by_key:
            # Multiple BENCH records share the same key (e.g. all cases exported
            # with the same graph name). Never drop a case: keep every record and
            # disambiguate the duplicates by suffixing their name.
            suffix = 2
            while True:
                current["name"] = "{}#{}".format(benchmark["name"], suffix)
                key = _result_key(current)
                if key not in current_by_key:
                    break
                suffix += 1
        error = compare_result(
            current,
            baseline_by_key.get(key),
            relative_threshold_pct,
            absolute_floor_us,
            allow_update=allow_update,
        )
        if error:
            failures.append(error)
        current_by_key[key] = current

    results["results"] = sorted(current_by_key.values(), key=_result_key)
    _write_results(result_path, results)
    _write_markdown(report_path, results)
    if failures:
        raise AssertionError(
            "Operator performance changed beyond the allowed range:\n"
            + "\n".join(failures)
        )


def fetch_baseline(api_url, project_id, token, ref, job, artifact, output):
    """Download the latest successful default-branch job artifact."""
    quoted_project = urllib.parse.quote(str(project_id), safe="")
    quoted_ref = urllib.parse.quote(ref, safe="")
    quoted_artifact = "/".join(
        urllib.parse.quote(part, safe="") for part in artifact.split("/")
    )
    query = urllib.parse.urlencode({"job": job})
    url = (
        "{api}/projects/{project}/jobs/artifacts/{ref}/raw/{artifact}?{query}"
    ).format(
        api=api_url.rstrip("/"),
        project=quoted_project,
        ref=quoted_ref,
        artifact=quoted_artifact,
        query=query,
    )
    request = urllib.request.Request(url, headers={"JOB-TOKEN": token})
    output = Path(output)
    try:
        with urllib.request.urlopen(request) as response:
            content = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            output.unlink(missing_ok=True)
            print("No performance baseline found for job {!r}".format(job))
            return False
        raise RuntimeError(
            "Performance baseline download failed with HTTP {}".format(error.code)
        ) from error

    data = json.loads(content.decode("utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        output.unlink(missing_ok=True)
        print(
            "Downloaded performance baseline has an outdated schema ({!r}); "
            "treating it as absent".format(data.get("schema_version"))
        )
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    print("Downloaded performance baseline for job {!r}".format(job))
    return True


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("--api-url", required=True)
    fetch_parser.add_argument("--project-id", required=True)
    fetch_parser.add_argument("--token", required=True)
    fetch_parser.add_argument("--ref", required=True)
    fetch_parser.add_argument("--job", required=True)
    fetch_parser.add_argument("--artifact", default="perf_results.json")
    fetch_parser.add_argument("--output", default="perf_baseline.json")

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--output", default="perf_results.json")

    args = parser.parse_args()
    if args.command == "fetch":
        fetch_baseline(
            args.api_url,
            args.project_id,
            args.token,
            args.ref,
            args.job,
            args.artifact,
            args.output,
        )
    else:
        _write_results(args.output, {"schema_version": SCHEMA_VERSION, "results": []})


if __name__ == "__main__":
    _main()
