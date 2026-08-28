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


SCHEMA_VERSION = 1
BENCH_PATTERN = re.compile(
    r"BENCH name=(?P<name>\S+) iters=(?P<iters>\d+) "
    r"median_us=(?P<median>[0-9]+(?:\.[0-9]+)?) "
    r"mean_us=(?P<mean>[0-9]+(?:\.[0-9]+)?)"
)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_benchmarks(log):
    """Return all machine-readable benchmark records from a DUT log."""
    if isinstance(log, bytes):
        log = log.decode("utf-8", errors="replace")
    log = ANSI_ESCAPE_PATTERN.sub("", log)
    return [
        {
            "name": match.group("name"),
            "iters": int(match.group("iters")),
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

    baseline_us = float(baseline["median_us"])
    current_us = float(current["median_us"])
    absolute_delta_us = current_us - baseline_us
    if baseline_us == 0:
        current["status"] = "invalid_baseline"
        return "{} has a zero-valued performance baseline".format(current["name"])

    delta_pct = absolute_delta_us / baseline_us * 100.0
    current.update(
        {
            "baseline_median_us": baseline_us,
            "delta_us": round(absolute_delta_us, 3),
            "delta_pct": round(delta_pct, 3),
        }
    )
    exceeds_limit = (
        abs(delta_pct) > relative_threshold_pct
        and abs(absolute_delta_us) > absolute_floor_us
    )
    if not exceeds_limit:
        current["status"] = "pass"
        return None
    if allow_update:
        current["status"] = "updated"
        return None
    current["status"] = "fail"
    return (
        "{name}: median {current:.3f} us, baseline {baseline:.3f} us, "
        "change {delta:+.3f}% (limit +/-{limit:.3f}%)"
    ).format(
        name=current["name"],
        current=current_us,
        baseline=baseline_us,
        delta=delta_pct,
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
        "| Target | IDF | Operator | Model | Median (us) | Baseline (us) | Delta | Status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for result in sorted(data["results"], key=_result_key):
        baseline = result.get("baseline_median_us")
        delta = result.get("delta_pct")
        lines.append(
            "| {target} | {idf_version} | {config} | {name} | {median_us:.3f} | "
            "{baseline} | {delta} | {status} |".format(
                **result,
                baseline="-" if baseline is None else "{:.3f}".format(baseline),
                delta="-" if delta is None else "{:+.3f}%".format(delta),
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
    relative_threshold_pct = float(os.environ.get("PERF_RELATIVE_THRESHOLD_PCT", "2"))
    absolute_floor_us = float(os.environ.get("PERF_ABSOLUTE_FLOOR_US", "2"))
    allow_update = update_baseline_requested()

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
    baseline_results = _load_results(baseline_path)
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
        raise RuntimeError("Downloaded performance baseline has an unsupported schema")
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
