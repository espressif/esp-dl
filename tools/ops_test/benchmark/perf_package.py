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
``publish`` merges this pipeline's measurements into the newest existing
baseline of the target. Two independent permissions decide what it may write:

  * ``--add-unseen`` (set from ``CI_COMMIT_REF_PROTECTED``) allows adding
    cases the baseline does not cover yet, and replacing a baseline whose
    schema is too old to compare against. Since schema v3 the comparison key
    includes the board that measured the case, and the CI runners hand out an
    arbitrary board per job, so a case is only gated once its own board has
    been recorded. Letting protected branches add unseen keys fills the
    per-board baselines up over a few pipelines instead of leaving those
    combinations ungated forever, while keeping an MR from turning its own
    measurements into the reference for a board nobody has recorded yet.
  * ``--update`` (the MR carries the ``update-perf-baseline`` label, or
    PERF_UPDATE_BASELINE is set to 1) additionally allows overwriting cases
    that already have a baseline, so an unintended slowdown can never quietly
    become the new reference.

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


def merge_into_baseline(baseline, current, update):
    """Combine a baseline with new measurements.

    Cases the baseline does not cover yet are always added; cases it already
    covers are only replaced when ``update`` is set. Returns the merged
    document plus the number of added and replaced cases.
    """
    merged = {_result_key(result): result for result in baseline["results"]}
    added = 0
    replaced = 0
    for result in current["results"]:
        key = _result_key(result)
        if key not in merged:
            merged[key] = result
            added += 1
        elif update:
            merged[key] = result
            replaced += 1
    # Carry over the newest metadata (thresholds, commit sha, ...).
    output = {key: value for key, value in current.items() if key != "results"}
    output["results"] = sorted(merged.values(), key=_result_key)
    return output, added, replaced


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

    update = _truthy(args.update)
    add_unseen = update or _truthy(args.add_unseen)
    if not add_unseen:
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

    if latest_version is None:
        print("No baseline for target {} yet; publishing this run.".format(args.target))
        payload = data
    elif latest_data.get("schema_version") != SCHEMA_VERSION:
        # Comparing against a baseline from an older methodology is meaningless,
        # and perf gating must not stay silently disabled until someone
        # remembers to label the MR.
        print(
            "Existing baseline for target {} (version {!r}) has an outdated "
            "schema; replacing it.".format(args.target, latest_version)
        )
        payload = data
    else:
        payload, added, replaced = merge_into_baseline(latest_data, data, update)
        if not added and not replaced:
            print(
                "Baseline for target {} already covers every measured case and "
                "update is not requested; skipping.".format(args.target)
            )
            return
        print(
            "Merging into baseline for target {} (version {!r}): {} case(s) "
            "added, {} replaced, {} total.".format(
                args.target,
                latest_version,
                added,
                replaced,
                len(payload["results"]),
            )
        )

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
    publish_parser.add_argument("--update", default="0")
    publish_parser.add_argument(
        "--add-unseen",
        default="0",
        help="Allow adding cases the baseline does not cover yet (and "
        "replacing a baseline with an outdated schema). Set from "
        "CI_COMMIT_REF_PROTECTED so only protected branches grow the "
        "baseline. Implied by --update.",
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
