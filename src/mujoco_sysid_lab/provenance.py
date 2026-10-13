"""Small, explicit provenance records for retained synthetic experiment runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence


_EXCLUDED_PARTS = frozenset(
    {".git", ".venv", "artifacts", "private", "__pycache__", ".pytest_cache",
     ".mypy_cache", ".ruff_cache", "build", "dist"}
)
_SOURCE_SUFFIXES = frozenset(
    {".py", ".xml", ".json", ".toml", ".lock", ".md", ".yaml", ".yml", ".sh", ".txt"}
)
_SOURCE_DOTFILES = frozenset({".gitignore", ".gitattributes", ".python-version"})
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def sha256_file(path: str | Path) -> str:
    """Hash the actual bytes of a file without loading the whole file in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Write stable, strict JSON, rejecting NaN/Inf before touching the file."""
    serialized = json.dumps(value, allow_nan=False, sort_keys=True, indent=2,
                            ensure_ascii=False) + "\n"
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *arguments],
                            check=False, capture_output=True)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git provenance unavailable: {message}")
    return result.stdout


def _commit(root: Path) -> str:
    try:
        commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    except ValueError as error:
        raise ValueError("A real Git commit is required before retaining a run") from error
    if len(commit) not in (40, 64) or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Git returned an invalid commit identifier")
    return commit


def _paths(output: bytes) -> list[str]:
    return [part.decode("utf-8") for part in output.split(b"\0") if part]


def _excluded(path: str) -> bool:
    return any(part in _EXCLUDED_PARTS for part in Path(path).parts)


def _authored_source(path: str) -> bool:
    candidate = Path(path)
    if _excluded(path):
        return False
    # Never capture secret dotfiles/directories or a machine's configuration.
    if any(part.startswith(".") and part != ".github" for part in candidate.parts):
        return len(candidate.parts) == 1 and candidate.name in _SOURCE_DOTFILES
    return candidate.suffix in _SOURCE_SUFFIXES


def _git_state(root: Path) -> dict[str, Any]:
    tracked = sorted(path for path in _paths(_git(root, "diff", "HEAD", "--name-only", "-z"))
                     if not _excluded(path))
    untracked = sorted(path for path in _paths(_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
                       if not _excluded(path))
    return {"commit": _commit(root), "dirty": bool(tracked or untracked),
            "tracked_changes": tracked, "untracked_files": untracked}


def _source_paths(root: Path) -> tuple[list[str], list[str]]:
    tracked = set(_paths(_git(root, "ls-files", "--cached", "-z")))
    # Include deleted committed files so the patch records their deletion.
    tracked.update(_paths(_git(root, "ls-tree", "-r", "--name-only", "-z", "HEAD")))
    untracked = _paths(_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    return sorted(p for p in tracked if _authored_source(p)), sorted(
        p for p in untracked if _authored_source(p))


def _source_hashes(root: Path, paths: Sequence[str]) -> dict[str, str]:
    hashes = {}
    for relative in paths:
        source = root / relative
        if any((root / Path(*Path(relative).parts[:index])).is_symlink()
               for index in range(1, len(Path(relative).parts) + 1)):
            raise ValueError(f"Authored source must not be a symlink: {relative}")
        if source.is_file():
            # A symlink in a parent directory must not expose outside files either.
            if not source.resolve().is_relative_to(root):
                raise ValueError(f"Authored source resolves outside the repository: {relative}")
            hashes[relative] = sha256_file(source)
    return hashes


def snapshot_sources(root: str | Path, destination: str | Path) -> dict[str, Any]:
    """Save a HEAD patch and exact authored untracked sources under ignored artifacts.

    Reconstruct tracked source from ``base_commit`` plus ``tracked.patch`` and
    add ``untracked/``. All current authored source bytes are hashed. Generated
    outputs, virtual environments, private notes, and sensitive dotfiles are
    excluded; coverage is explicitly recorded in the manifest.
    """
    repository = Path(root).resolve()
    target = Path(destination).resolve()
    if not target.is_relative_to(repository / "artifacts"):
        raise ValueError("Source snapshots must be placed under ignored artifacts/")
    relative_target = target.relative_to(repository).as_posix()
    ignored = subprocess.run(["git", "-C", str(repository), "check-ignore", "-q",
                              relative_target], check=False, capture_output=True)
    if ignored.returncode != 0:
        raise ValueError("Source snapshot destination must be Git-ignored")
    state = _git_state(repository)
    tracked, untracked = _source_paths(repository)
    hashes = _source_hashes(repository, tracked + untracked)
    target.mkdir(parents=True, exist_ok=False)
    patch = _git(repository, "diff", "HEAD", "--binary", "--",
                 *(f":(literal){path}" for path in tracked)) if tracked else b""
    patch_path = target / "tracked.patch"
    patch_path.write_bytes(patch)
    for relative in untracked:
        output = target / "untracked" / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((repository / relative).read_bytes())
        if sha256_file(output) != hashes[relative]:
            raise ValueError("Source changed while copying the provenance snapshot")
    if (_source_paths(repository) != (tracked, untracked)
            or _source_hashes(repository, tracked + untracked) != hashes
            or _commit(repository) != state["commit"]):
        raise ValueError("Source changed while collecting the provenance snapshot")
    manifest = {
        "schema_version": 1,
        "base_commit": state["commit"],
        "dirty": state["dirty"],
        "coverage": "Authored text sources; generated/private files and sensitive dotfiles excluded",
        "source_sha256": hashes,
        "untracked_sources": untracked,
        "tracked_patch": {"path": "tracked.patch", "sha256": sha256_file(patch_path)},
    }
    manifest_path = target / "manifest.json"
    write_json(manifest_path, manifest)
    return {"path": relative_target, "base_commit": state["commit"],
            "manifest_sha256": sha256_file(manifest_path),
            "tracked_patch_sha256": sha256_file(patch_path), "source_sha256": hashes}


def _package_versions() -> dict[str, str]:
    return dict(sorted((distribution.metadata["Name"].lower(), distribution.version)
                       for distribution in importlib.metadata.distributions()
                       if distribution.metadata["Name"]))


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"logical_cores": os.cpu_count()}
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(["system_profiler", "SPHardwareDataType", "-json"],
                                    capture_output=True, check=True, timeout=15)
            hardware = json.loads(result.stdout)["SPHardwareDataType"][0]
            # Deliberately discard serial numbers, UUIDs, names, and all other keys.
            info.update({"chip": hardware.get("chip_type") or hardware.get("cpu_type"),
                         "core_description": hardware.get("number_processors"),
                         "memory": hardware.get("physical_memory"),
                         "source": "system_profiler filtered hardware fields"})
            return info
        except (OSError, subprocess.SubprocessError, KeyError, IndexError, ValueError) as error:
            reason = type(error).__name__
    else:
        reason = "system_profiler is macOS-only"
    info.update({"chip": platform.processor() or None, "memory": None,
                 "source": "platform.processor/os.cpu_count",
                 "reason": reason})
    return info


def _relative_file(repository: Path, path: str | Path) -> tuple[Path, str]:
    supplied = Path(path)
    source = supplied if supplied.is_absolute() else repository / supplied
    if not source.absolute().is_relative_to(repository):
        raise ValueError("Provenance config/model must be regular repository files")
    relative_parts = source.absolute().relative_to(repository).parts
    if (any((repository / Path(*relative_parts[:index])).is_symlink()
            for index in range(1, len(relative_parts) + 1))
            or not source.resolve().is_relative_to(repository)):
        raise ValueError("Provenance config/model must be regular repository files")
    if not source.is_file():
        raise ValueError(f"Provenance input does not exist: {source.name}")
    return source, source.resolve().relative_to(repository).as_posix()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Configuration contains non-finite JSON value: {value}")


def collect_provenance(root: str | Path, config_path: str | Path,
                       model_path: str | Path, command: Sequence[str]) -> dict[str, Any]:
    """Collect bounded run provenance; caller attaches ``snapshot_sources`` if dirty.

    ``command`` is the exact argv used to launch the run, without shell guessing.
    Executable paths inside the repository are recorded relative to the root;
    outside paths under the user's home use ``~`` to avoid publishing identity.
    """
    repository = Path(root).resolve()
    _commit(repository)  # Fail before collecting a record without a source commit.
    if isinstance(command, (str, bytes)) or not command or any(
            not isinstance(argument, str) for argument in command):
        raise ValueError("command must be a nonempty sequence of exact argv strings")
    config_file, config_relative = _relative_file(repository, config_path)
    model_file, model_relative = _relative_file(repository, model_path)
    config = json.loads(config_file.read_text(encoding="utf-8"),
                        parse_constant=_reject_json_constant)
    # Reject numbers overflowing to infinity as well as explicit NaN/Infinity.
    json.dumps(config, allow_nan=False)
    versions = _package_versions()
    state = _git_state(repository)
    tracked, untracked = _source_paths(repository)
    executable = Path(sys.executable).absolute()
    if executable.is_relative_to(repository):
        executable_text = executable.relative_to(repository).as_posix()
        executable_scope = "repository-relative"
    elif executable.is_relative_to(Path.home()):
        executable_text = "~/" + executable.relative_to(Path.home()).as_posix()
        executable_scope = "home-relative"
    else:
        executable_text = str(executable)
        executable_scope = "absolute"
    lock = repository / "uv.lock"
    return {
        "schema_version": 1,
        "command": {"argv": list(command), "working_directory": "."},
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation(),
                   "executable": executable_text, "executable_scope": executable_scope},
        "environment": {"os": {"system": platform.system(), "release": platform.release(),
                                 "version": platform.version(), "architecture": platform.machine(),
                                 "macos_version": platform.mac_ver()[0] or None},
                        "cpu": _cpu_info(),
                        "thread_settings": {name: os.environ.get(name) for name in _THREAD_VARIABLES}},
        "mujoco_version": versions.get("mujoco"),
        "package_versions": versions,
        "git": state,
        "config": {"path": config_relative, "sha256": sha256_file(config_file), "content": config},
        "model": {"path": model_relative, "sha256": sha256_file(model_file)},
        "lock": {"path": "uv.lock", "sha256": sha256_file(lock)} if lock.is_file() else
                {"path": None, "sha256": None, "reason": "uv.lock is absent"},
        "source_sha256": _source_hashes(repository, tracked + untracked),
        "source_snapshot_required": state["dirty"],
    }
