#!/usr/bin/env python3
"""Aggregate, publish and fetch per-target ESP-DL operator performance baselines
stored in GitLab's generic package registry.

Registry layout
---------------
Every pipeline that publishes creates one immutable package version that holds
every target's file (same idea as ``espdl-op-test-cases``):

    package name : espdl-op-perf-baseline
    version      : <pipeline-id>            (e.g. 12345)
    files        : <target>_perf_results.json  (one per chip)

``fetch`` walks versions newest-first and downloads that target's file from
the first version that has it. Historical per-target versions (``<target>``
and ``<target>-<pipeline-id>``) are still recognized so existing baselines
keep working until they age out.

Versions are never overwritten or deleted by ``publish``, so concurrent MRs
can publish at the same time without corrupting each other: uploads are
single atomic PUTs of different filenames (or of unique pipeline versions),
and the last published version simply wins. Old versions are pruned by
``cleanup`` (run on protected branches only).

Update policy
-------------
Only master writes. A branch pipeline is a proposal: the gate tells it whether
its numbers are acceptable, and the ``update-perf-baseline[:<Op>]`` label can
waive that verdict for named operators, but nothing a branch measures reaches
the registry. The reference only moves on the branch everyone builds on.

``publish`` merges master's measurements into the newest existing baseline of
the target. Two independent permissions decide what it may write:

  * ``--add-unseen`` allows adding cases the baseline does not cover yet, and
    replacing a baseline whose schema is too old to compare against. Since
    schema v3 the comparison key includes the board that measured the case,
    and the CI runners hand out an arbitrary board per job, so a case is only
    gated once its own board has been recorded; adding unseen keys fills the
    per-board baselines up over a few pipelines instead of leaving those
    combinations ungated forever.
  * ``--update-ops`` names the operators this pipeline may rewrite outright:
    their entries are dropped and re-recorded from the current measurements.
    On master that comes from a human playing accept_espdl_ops_perf after
    reading the diff report of a failed gate. It is scoped to named operators
    because a change normally affects one of them, and a blanket update would
    also swallow whatever unrelated regression shares the pipeline.

Outside ``--add-unseen``, the operators in ``--update-ops`` are also the only
ones whose *unseen* entries may be added, so accepting one operator cannot
turn this pipeline into the reference for the others.

Every publish that moves anything appends a record to ``baseline_updates`` in
the file itself: what triggered it, who, which commit, and each entry's
before/after timing. Reading the baseline therefore also tells you how it got
its current values.

An in-scope operator has *every* board's entries dropped, not only those of
the boards this pipeline's shards landed on. Leaving the other boards at the
pre-optimization value fails the next pipeline that lands on one of them --
the gate is two-sided, so a speedup trips it too -- and because that failure
blocks this job, the stale entries could never be refreshed. Dropping them
instead lets each board re-record the case as it measures it again.

With neither permission, or when the merge changes nothing, it prints a skip
message and exits successfully without touching the registry.
"""

import argparse
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PACKAGE_NAME = "espdl-op-perf-baseline"
# Must stay in sync with perf_benchmark.SCHEMA_VERSION.
SCHEMA_VERSION = 3
# Must stay in sync with perf_benchmark.UPDATE_ALL_OPS.
UPDATE_ALL_OPS = "*"
# Audit trail of every publish that moved the baseline, carried along inside
# the baseline file itself so that reading the reference also tells you where
# it came from.
HISTORY_KEY = "baseline_updates"
# Bounded because the file is downloaded by every pipeline. Older records stay
# recoverable from the package versions they were published in.
HISTORY_LIMIT = 50
FILE_NAME_TEMPLATE = "{target}_perf_results.json"
# Each test_espdl_ops matrix child writes artifacts under
# ops_perf/<target>/<idf_version>/<config>/perf_results.json so they do not
# overwrite each other when GitLab extracts every child's artifacts into
# $CI_PROJECT_DIR.
PERF_ARTIFACT_ROOT = "ops_perf"
# Legacy layout from an earlier assumption that GitLab extracts matrix
# artifacts into directories named e.g. "test_espdl_ops: [esp32p4, 5.5, ...]".
MATRIX_JOB_PREFIX = "test_espdl_ops: ["
# The list/delete endpoints of the packages API can be accessed with the
# CI_JOB_TOKEN since GitLab 16.0. Older instances only allow upload/download
# with the JOB-TOKEN; in that case fetching falls back to the legacy stable
# version "<target>" and updates degrade to "always first run".
LISTING_DENIED_CODES = (401, 403)


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _result_key(result):
    """Comparison key of one measurement. Must match perf_benchmark._result_key."""
    return (
        str(result.get("target")),
        str(result.get("idf_version")),
        str(result.get("board") or "unknown"),
        str(result.get("config")),
        str(result.get("name")),
    )


def parse_update_ops(value):
    """Parse an --update-ops argument into an update scope.

    Accepts "*" for every operator, a comma-separated operator list, or an
    empty string for "replace nothing".
    """
    value = (value or "").strip()
    if not value:
        return set()
    if value == UPDATE_ALL_OPS:
        return UPDATE_ALL_OPS
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _update_allows(update_ops, result):
    """Whether this pipeline may replace `result`'s baseline entry."""
    if update_ops == UPDATE_ALL_OPS:
        return True
    return str(result.get("config", "")).lower() in update_ops


def _gate_us(result):
    """The timing the perf gate compares. Mirrors perf_benchmark._compared_us."""
    value = result.get("min_us")
    return float(value if value is not None else result["median_us"])


def _entry_label(result):
    """The part of an entry's identity worth reading in an audit record."""
    return {
        "board": result.get("board") or "unknown",
        "config": result.get("config"),
        "name": result.get("name"),
    }


def _sorted_labels(labels):
    return sorted(
        labels, key=lambda item: (item["config"], item["name"], item["board"])
    )


def _history_entry(trigger, update_ops, summary):
    """Record of who moved the baseline, and to what."""
    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # "accept" means a human read the diff report and pressed play;
        # "automatic" means the gate passed and the numbers were merely
        # recorded. That distinction is the point of this record.
        "trigger": trigger,
        "actor": os.environ.get("GITLAB_USER_LOGIN", ""),
        "commit_sha": os.environ.get("CI_COMMIT_SHA", ""),
        "commit_title": os.environ.get("CI_COMMIT_TITLE", ""),
        "ref": os.environ.get("CI_COMMIT_REF_NAME", ""),
        "pipeline_url": os.environ.get("CI_PIPELINE_URL", ""),
        "job_url": os.environ.get("CI_JOB_URL", ""),
        "update_ops": (
            UPDATE_ALL_OPS if update_ops == UPDATE_ALL_OPS else sorted(update_ops)
        ),
        # Entries that had no baseline before. A count, because the first
        # publish of a chip adds thousands of them at once.
        "added": summary["added"],
        "replaced": summary["replaced"],
        "dropped": summary["dropped"],
    }


def _file_url(api_url, project_id, target, version):
    quoted_project = urllib.parse.quote(str(project_id), safe="")
    return "{}/projects/{}/packages/generic/{}/{}/{}".format(
        api_url.rstrip("/"),
        quoted_project,
        urllib.parse.quote(PACKAGE_NAME, safe=""),
        urllib.parse.quote(version, safe=""),
        urllib.parse.quote(FILE_NAME_TEMPLATE.format(target=target), safe=""),
    )


def _upload(url, token, source):
    parsed = urllib.parse.urlsplit(url)
    connection_class = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_class(parsed.netloc, timeout=600)
    request_path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
    try:
        with open(source, "rb") as body:
            connection.request(
                "PUT",
                request_path,
                body=body,
                headers={
                    "Content-Length": str(Path(source).stat().st_size),
                    "Content-Type": "application/octet-stream",
                    "JOB-TOKEN": token,
                },
            )
            response = connection.getresponse()
            response.read()
            return response.status
    finally:
        connection.close()


def _list_packages(api_url, project_id, token, use_private_token):
    packages = []
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {
                "package_name": PACKAGE_NAME,
                "package_type": "generic",
                "order_by": "created_at",
                "page": page,
                "per_page": 100,
                "sort": "desc",
            }
        )
        project = urllib.parse.quote(str(project_id), safe="")
        url = "{}/projects/{}/packages?{}".format(api_url.rstrip("/"), project, query)
        headers = (
            {"PRIVATE-TOKEN": token} if use_private_token else {"JOB-TOKEN": token}
        )
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response:
            page_packages = json.load(response)
        if not isinstance(page_packages, list):
            raise RuntimeError("Package list response is not an array")
        packages.extend(page_packages)
        if len(page_packages) < 100:
            return packages
        page += 1


def _list_or_none(api_url, project_id, token, use_private_token):
    """List packages, or return None when the token may not list packages."""
    try:
        return _list_packages(api_url, project_id, token, use_private_token)
    except urllib.error.HTTPError as error:
        if error.code in LISTING_DENIED_CODES:
            return None
        raise


def _legacy_owner(version):
    """Return the chip a pre-shared version belongs to, or None.

    Historical layouts used ``<target>`` or ``<target>-<pipeline-id>``. Shared
    versions are a bare pipeline id (digits) and belong to every target.
    """
    version = str(version or "")
    if not version:
        return None
    if "-" in version:
        prefix, suffix = version.rsplit("-", 1)
        if prefix and prefix[0].isalpha() and suffix.isdigit():
            return prefix
        return None
    if version[0].isalpha():
        return version
    return None


def _ordered_packages(packages):
    return sorted(
        packages,
        key=lambda package: (
            str(package.get("created_at") or ""),
            int(package.get("id") or 0),
        ),
        reverse=True,
    )


def _candidate_versions(packages, target):
    """Newest-first versions that may contain ``{target}_perf_results.json``."""
    versions = []
    seen = set()
    for package in _ordered_packages(packages):
        version = package.get("version")
        if not version or version in seen:
            continue
        owner = _legacy_owner(version)
        if owner is None or owner == target:
            seen.add(version)
            versions.append(str(version))
    return versions


def _load_baseline(api_url, project_id, target, version, token):
    """Return ``(status, data)``. ``data`` is set only on HTTP 200."""
    request = urllib.request.Request(
        _file_url(api_url, project_id, target, version),
        headers={"JOB-TOKEN": token},
    )
    try:
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, None
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Baseline {}/{} is not valid JSON".format(version, target)
        ) from error
    return 200, data


def _find_latest_baseline(api_url, project_id, target, token, packages):
    """Newest existing ``{target}_perf_results.json``, or ``(None, None)``.

    When listing is denied the only probeable name is the legacy stable
    version ``<target>``.
    """
    versions = [target] if packages is None else _candidate_versions(packages, target)
    for version in versions:
        status, data = _load_baseline(api_url, project_id, target, version, token)
        if status == 404:
            continue
        if status != 200 or data is None:
            raise RuntimeError("Baseline probe failed with HTTP {}".format(status))
        return version, data
    return None, None


def _collect_perf_result_files(root, target):
    """Return every perf_results.json that belongs to `target` under `root`."""
    root = Path(root)
    found = []
    seen = set()

    def _add(path):
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        found.append(path)

    pattern = "{}/{}/**/perf_results.json".format(PERF_ARTIFACT_ROOT, target)
    for path in sorted(root.glob(pattern)):
        _add(path)
    for path in sorted(root.glob("test_espdl_ops*/perf_results.json")):
        if _matrix_child_matches(path, target):
            _add(path)
    return found


def aggregate(args):
    """Merge the perf_results.json of every test_espdl_ops matrix child of one
    target into a single <target>_perf_results.json."""
    root = Path(args.root)
    inputs = _collect_perf_result_files(root, args.target)
    if not inputs:
        leftovers = sorted(root.glob("**/perf_results.json"))
        detail = (
            "found other perf_results.json file(s): {}".format(
                ", ".join(str(path.relative_to(root)) for path in leftovers)
            )
            if leftovers
            else "no perf_results.json files were downloaded"
        )
        raise RuntimeError(
            "No perf_results.json artifacts found for target {!r} under {}: {}".format(
                args.target, root, detail
            )
        )

    merged = {}
    metadata = {}
    for path in inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported performance result schema in {}".format(path)
            )
        if not isinstance(data.get("results"), list):
            raise RuntimeError("{} has no results array".format(path))
        if not metadata:
            metadata = {key: value for key, value in data.items() if key != "results"}
        for result in data["results"]:
            # The opset matrix children re-run the same operator cases; keep the
            # first occurrence (the results are equivalent across children).
            merged.setdefault(_result_key(result), result)

    output = dict(metadata)
    output["results"] = sorted(merged.values(), key=_result_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "Aggregated {} result(s) for target {} from {} file(s)".format(
            len(output["results"]), args.target, len(inputs)
        )
    )


def fetch(args):
    """Download the latest baseline file of a target from the package registry."""
    use_private_token = bool(args.list_token)
    list_token = args.list_token or args.token
    packages = _list_or_none(
        args.api_url, args.project_id, list_token, use_private_token
    )
    version, data = _find_latest_baseline(
        args.api_url, args.project_id, args.target, args.token, packages
    )
    output = Path(args.output)
    if version is None:
        output.unlink(missing_ok=True)
        print("No performance baseline found for target {!r}".format(args.target))
        return
    if data.get("schema_version") != SCHEMA_VERSION:
        output.unlink(missing_ok=True)
        print(
            "Downloaded baseline for target {!r} (version {!r}) has an outdated "
            "schema ({!r}); ignoring it, the next publish will refresh it".format(
                args.target, version, data.get("schema_version")
            )
        )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "Downloaded performance baseline for target {!r} (version {!r})".format(
            args.target, version
        )
    )


def merge_into_baseline(baseline, current, update_ops, add_unseen):
    """Combine a baseline with new measurements.

    Entries for the operators in ``update_ops`` are dropped from the baseline
    on every board. Entries the resulting baseline does not cover are then
    added from ``current``: all of them when ``add_unseen`` is set, otherwise
    only the ones ``update_ops`` covers, so that accepting one operator cannot
    quietly turn this pipeline into the reference for unrelated ones.

    Returns the merged document plus a summary of what moved, for the audit
    record in the published file.
    """
    prior = {_result_key(result): result for result in baseline["results"]}
    # An in-scope operator is dropped on every board, not only on the boards
    # this pipeline's shards happened to land on. Keeping the other boards at
    # the pre-optimization value fails the next pipeline that lands on one of
    # them -- the gate is two-sided, so a speedup trips it too -- and that
    # failure blocks this job, so the stale entries could never be refreshed.
    # Each board re-records the case as it is measured again.
    merged = {
        key: value
        for key, value in prior.items()
        if not _update_allows(update_ops, value)
    }
    voided = {key: value for key, value in prior.items() if key not in merged}

    added = 0
    replaced = []
    for result in current["results"]:
        key = _result_key(result)
        if key in merged:
            continue
        if not add_unseen and not _update_allows(update_ops, result):
            continue
        merged[key] = result
        was = voided.pop(key, None)
        if was is None:
            added += 1
            continue
        before, after = _gate_us(was), _gate_us(result)
        replaced.append(
            {
                **_entry_label(result),
                "before_us": round(before, 3),
                "after_us": round(after, 3),
                "delta_pct": (
                    None if before == 0 else round((after - before) / before * 100.0, 3)
                ),
            }
        )

    # Carry over the newest metadata (thresholds, commit sha, ...).
    output = {key: value for key, value in current.items() if key != "results"}
    output["results"] = sorted(merged.values(), key=_result_key)
    summary = {
        "added": added,
        "replaced": _sorted_labels(replaced),
        # In scope but not measured by this pipeline, so there is nothing to
        # put in their place. The next pipeline that lands on the board
        # records them afresh.
        "dropped": _sorted_labels(_entry_label(value) for value in voided.values()),
    }
    return output, summary


def publish(args):
    """Merge this pipeline's results into the target's baseline and upload it.

    See the module docstring for what ``--add-unseen`` and ``--update`` allow.
    """
    input_path = Path(args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION or not isinstance(
        data.get("results"), list
    ):
        raise RuntimeError("Invalid performance results file: {}".format(input_path))

    update_ops = parse_update_ops(args.update_ops)
    add_unseen = _truthy(args.add_unseen)
    if not (update_ops or add_unseen):
        print(
            "This pipeline may neither add nor replace baseline cases for "
            "target {}; skipping.".format(args.target)
        )
        return

    use_private_token = bool(args.list_token)
    list_token = args.list_token or args.token
    packages = _list_or_none(
        args.api_url, args.project_id, list_token, use_private_token
    )
    latest_version, latest_data = _find_latest_baseline(
        args.api_url, args.project_id, args.target, args.token, packages
    )

    # The audit trail belongs to the baseline, not to this pipeline's
    # measurements, so it has to be carried across explicitly.
    history = list((latest_data or {}).get(HISTORY_KEY) or [])
    wholesale = {"added": len(data["results"]), "replaced": [], "dropped": []}

    if latest_version is None:
        print("No baseline for target {} yet; publishing this run.".format(args.target))
        payload, summary = data, wholesale
    elif latest_data.get("schema_version") != SCHEMA_VERSION:
        # Comparing against a baseline from an older methodology is meaningless,
        # and perf gating must not stay silently disabled until someone
        # remembers to label the MR.
        print(
            "Existing baseline for target {} (version {!r}) has an outdated "
            "schema; replacing it.".format(args.target, latest_version)
        )
        payload, summary = data, wholesale
    else:
        payload, summary = merge_into_baseline(
            latest_data, data, update_ops, add_unseen
        )
        if not (summary["added"] or summary["replaced"] or summary["dropped"]):
            print(
                "Baseline for target {} already covers every measured case and "
                "update is not requested; skipping.".format(args.target)
            )
            return
        print(
            "Merging into baseline for target {} (version {!r}): {} entry(s) "
            "added, {} replaced, {} voided without a replacement, {} total.".format(
                args.target,
                latest_version,
                summary["added"],
                len(summary["replaced"]),
                len(summary["dropped"]),
                len(payload["results"]),
            )
        )
        for change in summary["replaced"]:
            print(
                "  {config}/{name} on {board}: {before_us} -> {after_us} us "
                "({delta})".format(
                    delta=(
                        "n/a"
                        if change["delta_pct"] is None
                        else "{:+.3f}%".format(change["delta_pct"])
                    ),
                    **change,
                )
            )

    payload[HISTORY_KEY] = (
        history + [_history_entry(args.trigger, update_ops, summary)]
    )[-HISTORY_LIMIT:]

    upload_path = Path(args.merged_output) if args.merged_output else input_path
    if upload_path != input_path:
        upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pipeline_id = args.pipeline_id or os.environ.get("CI_PIPELINE_ID")
    if not pipeline_id:
        pipeline_id = time.strftime("%Y%m%d%H%M%S")
    version = str(pipeline_id)
    upload_status = _upload(
        _file_url(args.api_url, args.project_id, args.target, version),
        args.token,
        upload_path,
    )
    if upload_status in (200, 201):
        print(
            "Published baseline for target {} as {}/{}".format(
                args.target,
                version,
                FILE_NAME_TEMPLATE.format(target=args.target),
            )
        )
    elif upload_status in (400, 409):
        # A retry of the same pipeline already uploaded this exact file.
        print(
            "Baseline {}/{} already exists (idempotent retry); "
            "treating as published.".format(
                version, FILE_NAME_TEMPLATE.format(target=args.target)
            )
        )
    else:
        raise RuntimeError("Baseline upload failed with HTTP {}".format(upload_status))


def _version_recency(packages):
    return max(
        (str(package.get("created_at") or ""), int(package.get("id") or 0))
        for package in packages
    )


def _prune_versions(version_groups, keep, api_url, project_id, token):
    ordered = sorted(
        version_groups, key=lambda v: _version_recency(version_groups[v]), reverse=True
    )
    deleted = 0
    for version in ordered[keep:]:
        for package in version_groups[version]:
            deleted += _delete_package(api_url, project_id, package.get("id"), token)
    return min(len(ordered), keep), deleted


def cleanup(args):
    """Keep the newest ``keep`` shared versions, plus the newest ``keep``
    legacy per-target versions so the old layout can age out safely."""
    packages = _list_packages(
        args.api_url, args.project_id, args.token, use_private_token=True
    )
    shared = {}
    legacy = {}
    for package in packages:
        version = package.get("version")
        if not version or "id" not in package:
            raise RuntimeError("Package list entry has no version or ID")
        owner = _legacy_owner(version)
        if owner is None:
            shared.setdefault(version, []).append(package)
        else:
            legacy.setdefault(owner, {}).setdefault(version, []).append(package)

    kept_shared, deleted = _prune_versions(
        shared, args.keep, args.api_url, args.project_id, args.token
    )
    kept_legacy = 0
    for target_versions in legacy.values():
        kept, removed = _prune_versions(
            target_versions, args.keep, args.api_url, args.project_id, args.token
        )
        kept_legacy += kept
        deleted += removed
    print(
        "Perf package retention complete: kept {} shared version(s) and "
        "{} legacy version(s), deleted {} package record(s)".format(
            kept_shared, kept_legacy, deleted
        )
    )


def _delete_package(api_url, project_id, package_id, token):
    project = urllib.parse.quote(str(project_id), safe="")
    package = urllib.parse.quote(str(package_id), safe="")
    url = "{}/projects/{}/packages/{}".format(api_url.rstrip("/"), project, package)
    request = urllib.request.Request(
        url, method="DELETE", headers={"PRIVATE-TOKEN": token}
    )
    try:
        with urllib.request.urlopen(request) as response:
            response.read()
            return 1
    except urllib.error.HTTPError as error:
        if error.code not in (204, 404):
            raise RuntimeError(
                "Deleting package {} failed with HTTP {}".format(package_id, error.code)
            )
        return 0


def _matrix_child_matches(path, target):
    name = path.parent.name
    if not name.startswith(MATRIX_JOB_PREFIX):
        return False
    rest = name[len(MATRIX_JOB_PREFIX) :]
    return rest.startswith(target + ", ") or rest.startswith(target + ",")


def _add_registry_arguments(parser):
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--target", required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--root", required=True)
    aggregate_parser.add_argument("--target", required=True)
    aggregate_parser.add_argument("--output", required=True)

    fetch_parser = subparsers.add_parser("fetch")
    _add_registry_arguments(fetch_parser)
    fetch_parser.add_argument("--output", required=True)
    fetch_parser.add_argument(
        "--list-token",
        default=None,
        help="Optional PRIVATE-TOKEN used to list package versions when the "
        "JOB-TOKEN is not allowed to list. Defaults to --token.",
    )

    publish_parser = subparsers.add_parser("publish")
    _add_registry_arguments(publish_parser)
    publish_parser.add_argument("--input", required=True)
    publish_parser.add_argument(
        "--merged-output",
        default=None,
        help="Where to write the merged baseline that gets uploaded. Defaults "
        "to overwriting --input.",
    )
    publish_parser.add_argument(
        "--update-ops",
        default="",
        help='Operators whose baseline entries may be replaced: "*" for every '
        "operator, a comma-separated operator list, or empty to replace "
        "nothing. Derived from perf_benchmark.update_baseline_scope(), or "
        "from PERF_ACCEPT_OPS when a human plays accept_espdl_ops_perf.",
    )
    publish_parser.add_argument(
        "--add-unseen",
        default="0",
        help="Allow adding cases the baseline does not cover yet (and "
        "replacing a baseline with an outdated schema). Set from "
        "CI_COMMIT_REF_PROTECTED so only protected branches grow the "
        "baseline. Without it, only --update-ops operators may be added.",
    )
    publish_parser.add_argument(
        "--trigger",
        default="automatic",
        help="What caused this publish, recorded in the baseline's audit "
        'trail. "accept" for a human playing accept_espdl_ops_perf, '
        '"automatic" for a gate that simply passed.',
    )
    publish_parser.add_argument(
        "--pipeline-id",
        default=None,
        help="Shared package version for every target in this pipeline. "
        "Defaults to $CI_PIPELINE_ID or a timestamp.",
    )
    publish_parser.add_argument(
        "--list-token",
        default=None,
        help="Optional PRIVATE-TOKEN used to list package versions when the "
        "JOB-TOKEN is not allowed to list. Defaults to --token.",
    )

    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--api-url", required=True)
    cleanup_parser.add_argument("--project-id", required=True)
    cleanup_parser.add_argument(
        "--token",
        required=True,
        help="PRIVATE-TOKEN that may list and delete packages.",
    )
    cleanup_parser.add_argument("--keep", type=int, default=10)

    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args)
    elif args.command == "fetch":
        fetch(args)
    elif args.command == "publish":
        publish(args)
    elif args.command == "cleanup":
        cleanup(args)


if __name__ == "__main__":
    main()
