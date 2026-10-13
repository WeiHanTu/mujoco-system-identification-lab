#!/usr/bin/env python3
"""Archive explicitly selected synthetic evidence and verify every member byte.

The external JSON manifest is excluded from the archive to avoid a recursive
hash. It retains the archive SHA256, complete member inventory, original run
provenance, and producer provenance. Publication remains a separate action.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
from time import perf_counter
from typing import Any

from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mujoco-sysid-lab.evidence-archive.v1"
CHECKPOINT = Path("artifacts/milestone5/20261013_seed101")
FORBIDDEN_PARTS = frozenset({"private", ".venv", ".git", "__pycache__", ".pytest_cache"})


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError("Evidence records must be JSON objects")
    json.dumps(result, allow_nan=False)
    return result


def _artifact_path(root: Path, supplied: Path | str, *, must_exist: bool) -> Path:
    argument = Path(supplied)
    if ".." in argument.parts:
        raise ValueError("Archive paths must not contain parent traversal")
    path = argument if argument.is_absolute() else root / argument
    if not path.is_relative_to(root / "artifacts") or path == root / "artifacts":
        raise ValueError("Select a specific file/directory under ignored artifacts/, not all artifacts")
    relative = path.relative_to(root)
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        raise ValueError("Private, virtual-environment, Git, and cache paths are forbidden")
    for index in range(1, len(relative.parts) + 1):
        if (root / Path(*relative.parts[:index])).is_symlink():
            raise ValueError("Symlinks are forbidden in evidence paths")
    if not path.resolve().is_relative_to(root / "artifacts"):
        raise ValueError("Archive path resolves outside ignored artifacts")
    if must_exist and not path.exists():
        raise ValueError(f"Evidence input does not exist: {relative.as_posix()}")
    if path.exists() and not (path.is_file() or path.is_dir()):
        raise ValueError("Archive paths must be regular files or directories")
    return path


def _run_directory(root: Path, reference: Path) -> Path:
    cursor = reference if reference.is_dir() else reference.parent
    while cursor != root / "artifacts":
        candidate = cursor / "run.json"
        if candidate.exists():
            _artifact_path(root, candidate, must_exist=True)
            if not candidate.is_file():
                raise ValueError("Cached parent run.json must be a regular file")
            return cursor
        cursor = cursor.parent
        if not cursor.is_relative_to(root / "artifacts"):
            break
    raise ValueError("Cached source reference has no containing parent run.json")


def _excluded_reason(relative: Path) -> str | None:
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        return "private_or_runtime_directory"
    if relative.name == ".DS_Store" or relative.name.startswith("._"):
        return "macos_finder_or_appledouble_metadata"
    if relative.name.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
        return "existing_archive_not_nested"
    return None


def _members(root: Path, inputs: Sequence[Path]) -> tuple[list[Path], list[dict[str, str]]]:
    members: dict[str, Path] = {}
    excluded: dict[str, str] = {}
    for supplied in inputs:
        _artifact_path(root, supplied, must_exist=True)
        if supplied.is_file():
            discovered = [supplied]
        else:
            discovered = []
            for directory, names, files in os.walk(supplied, topdown=True, followlinks=False):
                parent = Path(directory)
                for name in (*names, *files):
                    if (parent / name).is_symlink():
                        raise ValueError("Symlinks are forbidden anywhere in selected evidence")
                kept = []
                for name in sorted(names):
                    path = parent / name
                    reason = _excluded_reason(path.relative_to(root))
                    if reason:
                        excluded[path.relative_to(root).as_posix()] = reason
                    else:
                        kept.append(name)
                names[:] = kept
                discovered.extend(parent / name for name in sorted(files))
        for path in discovered:
            relative = path.relative_to(root)
            reason = _excluded_reason(relative)
            if reason:
                excluded[relative.as_posix()] = reason
                continue
            _artifact_path(root, path, must_exist=True)
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("Only regular files can be archived")
            members[relative.as_posix()] = path
    return [members[name] for name in sorted(members)], [
        {"path": name, "reason": excluded[name]} for name in sorted(excluded)
    ]


def collect_inputs(root: Path, run: Path | str, includes: Sequence[Path | str]) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    """Find all cached parent runs from the immutable full-run source references."""
    selected = _artifact_path(root, run, must_exist=True)
    directory = _run_directory(root, selected)
    run_path = directory / "run.json"
    record = _read_json(run_path)
    if not isinstance(record.get("fits"), list) or not isinstance(record.get("provenance"), dict):
        raise ValueError("Selected run requires fit rows and source provenance")
    roots = {directory}
    dependencies = []
    seen = set()
    pending = [directory]
    while pending:
        current = pending.pop()
        source = _read_json(current / "run.json")
        for fit in source.get("fits", []):
            if not isinstance(fit, dict):
                raise ValueError("Run fit rows must be objects")
            reference = fit.get("source_result")
            if reference is None:
                if fit.get("reused") is True:
                    raise ValueError("Reused fit is missing its source_result reference")
                continue
            if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
                    or not isinstance(reference.get("sha256"), str)):
                raise ValueError("Every source_result must provide a file path and SHA256")
            file = _artifact_path(root, reference["path"], must_exist=True)
            if not file.is_file() or sha256_file(file) != reference["sha256"]:
                raise ValueError("Cached source_result file hash differs from the retained reference")
            parent = _run_directory(root, file)
            if parent != directory and parent not in seen:
                parent_record = _read_json(parent / "run.json")
                dependencies.append({"path": parent.relative_to(root).as_posix(),
                                     "run_sha256": sha256_file(parent / "run.json"),
                                     "provenance": parent_record.get("provenance"),
                                     "optimization_provenance": parent_record.get("optimization_provenance"),
                                     "report_provenance": parent_record.get("report_provenance")})
                seen.add(parent)
                roots.add(parent)
                pending.append(parent)
    checkpoint = _artifact_path(root, CHECKPOINT, must_exist=True)
    roots.add(checkpoint)
    for extra in includes:
        roots.add(_artifact_path(root, extra, must_exist=True))
    return sorted(roots), sorted(dependencies, key=lambda item: item["path"]), record


def verify_archive(archive: Path, inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reopen gzip/PAX archive; require exact ordered regular members and hashes."""
    expected = {item["path"]: item for item in inventory}
    if len(expected) != len(inventory) or list(expected) != sorted(expected):
        raise ValueError("Inventory must contain unique paths in exact sorted order")
    names = []
    total = 0
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            name = member.name
            if (not member.isfile() or name not in expected or name in names
                    or _excluded_reason(Path(name)) is not None
                    or Path(name).is_absolute() or ".." in Path(name).parts):
                raise ValueError("Archive contains an unexpected, duplicate, or forbidden member")
            item = expected[name]
            if member.size != item["size_bytes"]:
                raise ValueError("Archive member size differs from inventory")
            payload = stream.extractfile(member)
            if payload is None:
                raise ValueError("Archive member payload is unavailable")
            digest = hashlib.sha256()
            count = 0
            with payload:
                for block in iter(lambda: payload.read(1024 * 1024), b""):
                    digest.update(block)
                    count += len(block)
            if digest.hexdigest() != item["sha256"] or count != item["size_bytes"]:
                raise ValueError("Archive member bytes differ from inventory")
            names.append(name)
            total += count
    if names != list(expected):
        raise ValueError("Archive count/order differs from the complete inventory")
    return {"passed": True, "member_count": len(names), "uncompressed_bytes": total,
            "method": "reopened every gzip/PAX regular member; exact sorted paths, size, and SHA256; no extraction"}


def create_archive(root: Path, run: Path | str, output: Path | str, manifest: Path | str,
                   includes: Sequence[Path | str], argv: Sequence[str]) -> dict[str, Any]:
    """Create new local archive/manifest; never overwrite evidence or inputs."""
    started = perf_counter()
    root = Path(root).resolve()
    archive = _artifact_path(root, output, must_exist=False)
    sidecar = _artifact_path(root, manifest, must_exist=False)
    producer_dir = _artifact_path(root, sidecar.with_name(sidecar.stem + "_source"), must_exist=False)
    if (not archive.name.endswith(".tar.gz") or sidecar.suffix != ".json"
            or archive == sidecar or any(path.exists() for path in (archive, sidecar, producer_dir))):
        raise ValueError("Choose new .tar.gz/.json destinations; archive, manifest, and producer evidence must not already exist")
    if isinstance(argv, (str, bytes)) or not argv or any(not isinstance(value, str) for value in argv):
        raise ValueError("Archive producer must retain the exact argv sequence")
    inputs, dependencies, source_record = collect_inputs(root, run, includes)
    if any(destination == source or (source.is_dir() and destination.is_relative_to(source))
           for source in inputs for destination in (archive, sidecar, producer_dir)):
        raise ValueError("Archive destinations cannot be inside any selected input or include the archive itself")
    source_provenance = source_record["provenance"]
    config_reference = source_provenance.get("config", {})
    model_reference = source_provenance.get("model", {})
    if not isinstance(config_reference.get("path"), str) or not isinstance(model_reference.get("path"), str):
        raise ValueError("Source run must provide exact config and model paths")
    producer = collect_provenance(root, config_reference["path"], model_reference["path"], argv)
    for name in ("config", "model"):
        if producer[name]["sha256"] != source_provenance[name].get("sha256"):
            raise ValueError("Current config/model bytes differ from the archived source run")
    # Preflight all input members before creating even producer evidence.
    _members(root, inputs)
    producer_dir.mkdir(parents=True, exist_ok=False)
    if producer["git"]["dirty"]:
        producer["source_snapshot"] = snapshot_sources(root, producer_dir / "snapshot")
    write_json(producer_dir / "provenance.json", producer)
    inputs.append(producer_dir)
    members, excluded = _members(root, inputs)
    inventory = [{"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size,
                  "sha256": sha256_file(path)} for path in members]
    source_run = _run_directory(root, _artifact_path(root, run, must_exist=True)) / "run.json"
    record = {"schema_version": SCHEMA, "status": "prepared",
              "archive": {"path": archive.relative_to(root).as_posix(), "format": "gzip PAX tar; regular files only"},
              "manifest": {"path": sidecar.relative_to(root).as_posix(), "excluded_from_archive": True,
                           "reason": "External archive digest and inventory avoid a recursive self-hash."},
              "source_run": {"path": source_run.relative_to(root).as_posix(), "sha256": sha256_file(source_run)},
              "source_provenance": source_provenance,
              "source_seeds": sorted({row["seed"] for row in source_record["fits"] if isinstance(row.get("seed"), int)}),
              "cached_dependency_runs": dependencies,
              "input_roots": [path.relative_to(root).as_posix() for path in sorted(inputs)],
              "excluded": excluded, "inventory": inventory,
              "producer_provenance": producer,
              "publication": "Local only; no upload or publication performed."}
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with archive.open("xb") as destination:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as stream:
                    for path, item in zip(members, inventory, strict=True):
                        info = tarfile.TarInfo(item["path"])
                        info.size, info.mode, info.mtime = item["size_bytes"], 0o644, 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        with path.open("rb") as payload:
                            stream.addfile(info, payload)
        verification = verify_archive(archive, inventory)
        final_members, final_exclusions = _members(root, inputs)
        if (final_members != members or final_exclusions != excluded
                or any(path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]
                       for path, item in zip(members, inventory, strict=True))):
            raise ValueError("Source evidence changed while archiving; archive is not a verified snapshot")
        record.update(status="verified", verification=verification, wall_seconds=perf_counter() - started)
        record["archive"].update(sha256=sha256_file(archive), size_bytes=archive.stat().st_size)
        # Detect manifest races rather than replacing an existing evidence record.
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True, indent=2) + "\n")
        return record
    except Exception as error:
        record.update(status="failed", failure={"type": type(error).__name__, "message": str(error)},
                      wall_seconds=perf_counter() - started)
        if archive.is_file():
            record["archive"].update(sha256=sha256_file(archive), size_bytes=archive.stat().st_size)
        if not sidecar.exists():
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            with sidecar.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False, sort_keys=True, indent=2) + "\n")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Full run directory or its run.json under artifacts/")
    parser.add_argument("--output", required=True, help="New ignored .tar.gz file")
    parser.add_argument("--manifest", required=True, help="New ignored external .json manifest")
    parser.add_argument("--include", action="append", default=[], help="Additional explicit ignored file/directory; repeatable")
    args = parser.parse_args()
    executable = Path(sys.executable).absolute()
    executable_text = executable.relative_to(ROOT).as_posix() if executable.is_relative_to(ROOT) else str(executable)
    argv = [executable_text, "scripts/archive_evidence.py", *sys.argv[1:]]
    try:
        result = create_archive(ROOT, args.run, args.output, args.manifest, args.include, argv)
    except (ValueError, OSError, tarfile.TarError) as error:
        print(f"Archive failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "archive": result["archive"],
                      "manifest": result["manifest"]["path"], "verified_members": result["verification"]["member_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
