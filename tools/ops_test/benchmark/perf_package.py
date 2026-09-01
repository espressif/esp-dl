#!/usr/bin/env python3
"""Aggregate, publish and fetch per-target ESP-DL operator performance baselines
stored in GitLab's generic package registry.

Registry layout
---------------
Every publication creates an immutable package version:

    package name : espdl-op-perf-baseline
    version      : <target>-<pipeline-id>   (e.g. esp32p4-12345)
    file         : <target>_perf_results.json

The "latest" baseline of a target is the version with the newest created_at
timestamp. Versions are never overwritten or deleted by `publish`, so
concurrent MRs can publish at the same time without corrupting each other:
uploads are single atomic PUTs to unique versions, and the last published
version simply wins. Old versions are pruned by `cleanup` (run on protected
branches only).

Update policy
-------------
`publish` adds a new version only when:

  * no version exists for the target yet (first run), or
  * `--update` is passed (the MR carries the `update-perf-baseline` label, or
    PERF_UPDATE_BASELINE is set to 1), or
  * the latest existing version was recorded with an outdated schema and can
    no longer be compared against the current measurement methodology
    (e.g. after the benchmark switched from median to min). Such baselines are
    refreshed automatically so perf gating self-heals after a methodology
    change without requiring a label.

Otherwise it prints a skip message and exits successfully without touching the
registry.
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
SCHEMA_VERSION = 2
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


def _file_url(api_url, project_id, target, version):
    quoted_project = urllib.parse.quote(str(project_id), safe="")
    return "{}/projects/{}/packages/generic/{}/{}/{}".format(
        api_url.rstrip("/"),
        quoted_project,
        urllib.parse.quote(PACKAGE_NAME, safe=""),
        urllib.parse.quote(version, safe=""),
        urllib.parse.quote(FILE_NAME_TEMPLATE.format(target=target), safe=""),
    )


def _download(url, token, destination):
    """Download a generic package file; return the HTTP status code."""
    request = urllib.request.Request(url, headers={"JOB-TOKEN": token})
    try:
        with urllib.request.urlopen(request) as response:
            content = response.read()
    except urllib.error.HTTPError as error:
        return error.code
    if destination is not None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return 200


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


def _latest_version(packages, target):
    """Newest package version belonging to `target` (including the legacy
    stable version "<target>" published before the versioned scheme)."""
    candidates = [
        package
        for package in packages
        if str(package.get("version")) == target
        or str(package.get("version", "")).startswith(target + "-")
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda package: (
            str(package.get("created_at") or ""),
            int(package.get("id") or 0),
        ),
    )


def _schema_is_outdated(api_url, project_id, target, version, token):
    """Return True when the given baseline version has an older schema."""
    request = urllib.request.Request(
        _file_url(api_url, project_id, target, version),
        headers={"JOB-TOKEN": token},
    )
    try:
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return True
        raise RuntimeError(
            "Baseline probe failed with HTTP {}".format(error.code)
        ) from error
    return data.get("schema_version") != SCHEMA_VERSION


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
            key = (
                result.get("target"),
                result.get("idf_version"),
                result.get("config"),
                result.get("name"),
            )
            # The opset matrix children re-run the same operator cases; keep the
            # first occurrence (the results are equivalent across children).
            merged.setdefault(key, result)

    output = dict(metadata)
    output["results"] = sorted(
        merged.values(),
        key=lambda result: (
            str(result.get("target")),
            str(result.get("idf_version")),
            str(result.get("config")),
            str(result.get("name")),
        ),
    )
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
    """Download the latest baseline of a target from the package registry."""
    use_private_token = bool(args.list_token)
    list_token = args.list_token or args.token
    packages = _list_or_none(
        args.api_url, args.project_id, list_token, use_private_token
    )
    latest = _latest_version(packages, args.target) if packages is not None else None
    version = latest["version"] if latest is not None else args.target

    status = _download(
        _file_url(args.api_url, args.project_id, args.target, version),
        args.token,
        args.output,
    )
    if status == 404:
        Path(args.output).unlink(missing_ok=True)
        print("No performance baseline found for target {!r}".format(args.target))
        return
    if status != 200:
        raise RuntimeError("Baseline download failed with HTTP {}".format(status))
    data = json.loads(Path(args.output).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        Path(args.output).unlink(missing_ok=True)
        print(
            "Downloaded baseline for target {!r} (version {!r}) has an outdated "
            "schema ({!r}); ignoring it, the next publish will refresh it".format(
                args.target, version, data.get("schema_version")
            )
        )
        return
    print(
        "Downloaded performance baseline for target {!r} (version {!r})".format(
            args.target, version
        )
    )


def publish(args):
    """Upload a <target>_perf_results.json as a new immutable version. A new
    version is only added on the first run, when an update was explicitly
    requested, or when the latest baseline has an outdated schema."""
    input_path = Path(args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION or not isinstance(
        data.get("results"), list
    ):
        raise RuntimeError("Invalid performance results file: {}".format(input_path))

    use_private_token = bool(args.list_token)
    list_token = args.list_token or args.token
    packages = _list_or_none(
        args.api_url, args.project_id, list_token, use_private_token
    )
    latest_version = None
    if packages is not None:
        latest = _latest_version(packages, args.target)
        if latest is not None:
            latest_version = latest["version"]
    else:
        # Listing is not permitted (pre-16.0 instance): probe the legacy stable
        # version directly.
        status = _download(
            _file_url(args.api_url, args.project_id, args.target, args.target),
            args.token,
            None,
        )
        if status == 200:
            latest_version = args.target
        elif status == 404:
            pass
        else:
            raise RuntimeError("Baseline probe failed with HTTP {}".format(status))

    update = _truthy(args.update)
    if latest_version is not None and not update:
        # Refresh baselines recorded with an outdated schema automatically:
        # comparing against them is meaningless after a measurement methodology
        # change, and perf gating must not stay silently disabled until someone
        # remembers to label the MR.
        if not _schema_is_outdated(
            args.api_url, args.project_id, args.target, latest_version, args.token
        ):
            print(
                "Baseline for target {} already exists and update is not "
                "requested; skipping.".format(args.target)
            )
            return
        print(
            "Existing baseline for target {} (version {!r}) has an outdated "
            "schema; republishing.".format(args.target, latest_version)
        )

    pipeline_id = args.pipeline_id or os.environ.get("CI_PIPELINE_ID")
    if not pipeline_id:
        pipeline_id = time.strftime("%Y%m%d%H%M%S")
    version = "{}-{}".format(args.target, pipeline_id)
    upload_status = _upload(
        _file_url(args.api_url, args.project_id, args.target, version),
        args.token,
        input_path,
    )
    if upload_status in (200, 201):
        print(
            "Published baseline for target {} as version {}".format(
                args.target, version
            )
        )
    elif upload_status in (400, 409):
        # A retry of the same pipeline already uploaded this exact version.
        print(
            "Baseline version {} already exists (idempotent retry); "
            "treating as published.".format(version)
        )
    else:
        raise RuntimeError("Baseline upload failed with HTTP {}".format(upload_status))


def cleanup(args):
    """Keep only the newest `keep` versions of every target. Requires a token
    that may list and delete packages (PRIVATE-TOKEN)."""
    packages = _list_packages(
        args.api_url, args.project_id, args.token, use_private_token=True
    )
    by_target = {}
    for package in packages:
        version = str(package.get("version") or "")
        target = version.rsplit("-", 1)[0] if "-" in version else version
        by_target.setdefault(target, []).append(package)

    deleted = 0
    for target, target_packages in by_target.items():
        ordered = sorted(
            target_packages,
            key=lambda package: (
                str(package.get("created_at") or ""),
                int(package.get("id") or 0),
            ),
            reverse=True,
        )
        for package in ordered[args.keep :]:
            deleted += _delete_package(
                args.api_url, args.project_id, package.get("id"), args.token
            )
    print(
        "Perf package retention complete: kept up to {} version(s) per target, "
        "deleted {} package record(s)".format(args.keep, deleted)
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
    publish_parser.add_argument("--update", default="0")
    publish_parser.add_argument(
        "--pipeline-id",
        default=None,
        help="Unique version suffix. Defaults to $CI_PIPELINE_ID or a timestamp.",
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
