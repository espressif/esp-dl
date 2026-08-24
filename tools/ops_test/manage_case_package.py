#!/usr/bin/env python3
"""Store generated ESP-DL operator cases in GitLab's generic package registry."""

import argparse
import hashlib
import http.client
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


CACHE_SCHEMA = "1"
PACKAGE_NAME = "espdl-op-test-cases"
MANIFEST_NAME = ".case-package-manifest.json"
QUANT_SUFFIXES = {
    "w8a8": "_s8.espdl",
    "w8a16": "_w8a16.espdl",
    "w16a16": "_s16.espdl",
    "none": "_f32.espdl",
}


def _log(message):
    print(message, file=sys.stderr, flush=True)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_fingerprint(esp_ppq_version, torch_spec, generator_options, input_paths):
    digest = hashlib.sha256()
    metadata = {
        "cache_schema": CACHE_SCHEMA,
        "esp_ppq_version": esp_ppq_version,
        "generator_options": generator_options,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_spec": torch_spec,
    }
    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))

    for path in sorted(
        (Path(path) for path in input_paths), key=lambda item: item.as_posix()
    ):
        if not path.is_file():
            raise FileNotFoundError("Fingerprint input does not exist: {}".format(path))
        digest.update(b"\0")
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _package_coordinates(api_url, project_id, fingerprint, target, quant_type):
    package_version = "{}-{}".format(CACHE_SCHEMA, fingerprint)
    filename = "{}-{}.tar.gz".format(target, quant_type)
    quoted_parts = (
        urllib.parse.quote(str(project_id), safe=""),
        urllib.parse.quote(PACKAGE_NAME, safe=""),
        urllib.parse.quote(package_version, safe=""),
        urllib.parse.quote(filename, safe=""),
    )
    url = "{}/projects/{}/packages/generic/{}/{}/{}".format(
        api_url.rstrip("/"), *quoted_parts
    )
    return url, package_version, filename


def _download(url, token, destination):
    request = urllib.request.Request(url, headers={"JOB-TOKEN": token})
    try:
        with urllib.request.urlopen(request) as response:
            with open(destination, "wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        return 200
    except urllib.error.HTTPError as error:
        return error.code


def _upload(url, token, source):
    parsed = urllib.parse.urlsplit(url)
    connection_class = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_class(parsed.netloc, timeout=600)
    request_path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
    size = os.path.getsize(source)
    try:
        with open(source, "rb") as body:
            connection.request(
                "PUT",
                request_path,
                body=body,
                headers={
                    "Content-Length": str(size),
                    "Content-Type": "application/octet-stream",
                    "JOB-TOKEN": token,
                },
            )
            response = connection.getresponse()
            response.read()
            return response.status
    finally:
        connection.close()


def _case_files(output_root, target, quant_type):
    suffix = QUANT_SUFFIXES[quant_type]
    target_root = (Path(output_root) / target).resolve()
    return sorted(
        (
            path
            for path in target_root.rglob("*.espdl")
            if path.is_file() and path.name.endswith(suffix)
        ),
        key=lambda item: item.as_posix(),
    )


def _manifest(
    project_root, files, fingerprint, esp_ppq_version, torch_spec, target, quant_type
):
    entries = []
    for path in files:
        relative_path = path.relative_to(project_root).as_posix()
        entries.append(
            {
                "path": relative_path,
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return {
        "cache_schema": CACHE_SCHEMA,
        "esp_ppq_version": esp_ppq_version,
        "files": entries,
        "fingerprint": fingerprint,
        "python": platform.python_version(),
        "quant_type": quant_type,
        "target": target,
        "torch_spec": torch_spec,
    }


def create_archive(
    archive_path,
    project_root,
    output_root,
    fingerprint,
    esp_ppq_version,
    torch_spec,
    target,
    quant_type,
):
    project_root = Path(project_root).resolve()
    files = _case_files(output_root, target, quant_type)
    if not files:
        raise RuntimeError(
            "No {} test cases were generated for {}/{}".format(
                QUANT_SUFFIXES[quant_type], target, quant_type
            )
        )

    manifest = _manifest(
        project_root,
        files,
        fingerprint,
        esp_ppq_version,
        torch_spec,
        target,
        quant_type,
    )
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

    with tarfile.open(archive_path, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(project_root).as_posix())
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(manifest_bytes)
        info.mode = 0o644
        info.mtime = 0
        with tempfile.SpooledTemporaryFile() as manifest_file:
            manifest_file.write(manifest_bytes)
            manifest_file.seek(0)
            archive.addfile(info, manifest_file)
    return manifest


def _read_and_validate_archive(
    archive_path, project_root, fingerprint, target, quant_type
):
    project_root = Path(project_root).resolve()
    expected_prefix = PurePosixPath("test_apps/esp-dl/models") / target
    expected_suffix = QUANT_SUFFIXES[quant_type]

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        by_name = {member.name: member for member in members}
        if len(by_name) != len(members) or MANIFEST_NAME not in by_name:
            raise RuntimeError("Package has duplicate entries or no manifest")

        manifest_member = by_name[MANIFEST_NAME]
        manifest_file = archive.extractfile(manifest_member)
        if manifest_file is None:
            raise RuntimeError("Package manifest is not a regular file")
        manifest = json.load(manifest_file)

        expected_values = {
            "cache_schema": CACHE_SCHEMA,
            "fingerprint": fingerprint,
            "target": target,
            "quant_type": quant_type,
        }
        for key, expected in expected_values.items():
            if manifest.get(key) != expected:
                raise RuntimeError(
                    "Package manifest {} mismatch: expected {!r}, got {!r}".format(
                        key, expected, manifest.get(key)
                    )
                )

        file_entries = manifest.get("files")
        if not isinstance(file_entries, list) or not file_entries:
            raise RuntimeError("Package manifest contains no test cases")
        expected_names = {MANIFEST_NAME}
        for entry in file_entries:
            path = PurePosixPath(entry["path"])
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.is_relative_to(expected_prefix)
                or not path.name.endswith(expected_suffix)
            ):
                raise RuntimeError("Unsafe or unexpected package path: {}".format(path))
            expected_names.add(path.as_posix())

        if set(by_name) != expected_names:
            raise RuntimeError("Package contents do not match its manifest")
        for name, member in by_name.items():
            if name != MANIFEST_NAME and (not member.isfile() or member.issym()):
                raise RuntimeError(
                    "Package contains a non-regular file: {}".format(name)
                )

        for entry in file_entries:
            member = by_name[entry["path"]]
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(
                    "Could not read package entry: {}".format(member.name)
                )
            destination = (project_root / member.name).resolve()
            try:
                destination.relative_to(project_root)
            except ValueError as error:
                raise RuntimeError("Package path escapes project root") from error
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with destination.open("wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    output.write(chunk)
            if (
                destination.stat().st_size != entry["size"]
                or digest.hexdigest() != entry["sha256"]
            ):
                destination.unlink(missing_ok=True)
                raise RuntimeError("Checksum mismatch for {}".format(member.name))
    return len(file_entries)


def restore(args):
    url, package_version, filename = _package_coordinates(
        args.api_url, args.project_id, args.fingerprint, args.target, args.quant_type
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / filename
        status = _download(url, args.token, archive_path)
        if status == 404:
            _log("Case package MISS: {}/{}".format(package_version, filename))
            print("miss")
            return
        if status != 200:
            raise RuntimeError("Package download failed with HTTP {}".format(status))
        count = _read_and_validate_archive(
            archive_path,
            args.project_root,
            args.fingerprint,
            args.target,
            args.quant_type,
        )
    _log("Case package HIT: {}/{} ({} files)".format(package_version, filename, count))
    print("hit")


def publish(args):
    url, package_version, filename = _package_coordinates(
        args.api_url, args.project_id, args.fingerprint, args.target, args.quant_type
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / filename
        manifest = create_archive(
            archive_path,
            args.project_root,
            args.output_root,
            args.fingerprint,
            args.esp_ppq_version,
            args.torch_spec,
            args.target,
            args.quant_type,
        )
        status = _upload(url, args.token, archive_path)
        if status not in (200, 201):
            if status not in (400, 409):
                raise RuntimeError("Package upload failed with HTTP {}".format(status))
            _log("Concurrent package upload detected; validating the registry copy")
            remote_archive = Path(temporary_directory) / ("remote-" + filename)
            download_status = _download(url, args.token, remote_archive)
            if download_status != 200:
                raise RuntimeError(
                    "Package upload returned HTTP {} and existing package download returned HTTP {}".format(
                        status, download_status
                    )
                )
            _read_and_validate_archive(
                remote_archive,
                args.project_root,
                args.fingerprint,
                args.target,
                args.quant_type,
            )
    _log(
        "Published case package: {}/{} ({} files)".format(
            package_version, filename, len(manifest["files"])
        )
    )


def _add_registry_arguments(parser):
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--quant-type", required=True, choices=sorted(QUANT_SUFFIXES))
    parser.add_argument("--project-root", default=".")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fingerprint_parser = subparsers.add_parser("fingerprint")
    fingerprint_parser.add_argument("--esp-ppq-version", required=True)
    fingerprint_parser.add_argument("--torch-spec", required=True)
    fingerprint_parser.add_argument("--generator-options", required=True)
    fingerprint_parser.add_argument("inputs", nargs="+")

    restore_parser = subparsers.add_parser("restore")
    _add_registry_arguments(restore_parser)

    publish_parser = subparsers.add_parser("publish")
    _add_registry_arguments(publish_parser)
    publish_parser.add_argument("--output-root", required=True)
    publish_parser.add_argument("--esp-ppq-version", required=True)
    publish_parser.add_argument("--torch-spec", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "fingerprint":
        print(
            calculate_fingerprint(
                args.esp_ppq_version,
                args.torch_spec,
                args.generator_options,
                args.inputs,
            )
        )
    elif args.command == "restore":
        restore(args)
    elif args.command == "publish":
        publish(args)


if __name__ == "__main__":
    main()
