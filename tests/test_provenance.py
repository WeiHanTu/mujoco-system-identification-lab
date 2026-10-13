"""Contract tests using isolated Git repositories, not the developer's checkout."""

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from mujoco_sysid_lab import provenance


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(root), *arguments], check=True,
                          text=True, capture_output=True).stdout.strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Provenance Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    (tmp_path / ".gitignore").write_text("artifacts/\n.venv/\nprivate/\n", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"schema_version":1,"seed":101}\n', encoding="utf-8")
    (tmp_path / "model.xml").write_text("<mujoco/>\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Authored source fixture")
    monkeypatch.setattr(provenance, "_package_versions", lambda: {"mujoco": "3.15.0"})
    monkeypatch.setattr(provenance, "_cpu_info", lambda: {"chip": "fixture", "logical_cores": 1})
    return tmp_path


def collect(repository):
    return provenance.collect_provenance(repository, "config.json", "model.xml",
                                         ["python", "scripts/run_experiment.py", "--config", "config.json"])


def test_collect_requires_real_commit(tmp_path):
    git(tmp_path, "init", "-q")
    with pytest.raises(ValueError, match="real Git commit"):
        provenance.collect_provenance(tmp_path, "missing.json", "missing.xml", ["python"])


def test_collect_clean_exact_config_command_and_versions(repository):
    record = collect(repository)
    expected_bytes = (repository / "config.json").read_bytes()
    assert record["config"]["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert record["config"]["content"] == {"schema_version": 1, "seed": 101}
    assert record["command"]["argv"] == ["python", "scripts/run_experiment.py", "--config", "config.json"]
    assert record["git"]["commit"] == git(repository, "rev-parse", "HEAD")
    assert record["git"]["dirty"] is False
    assert record["source_snapshot_required"] is False
    assert record["mujoco_version"] == "3.15.0"
    assert record["source_sha256"]["src/example.py"] == provenance.sha256_file(repository / "src/example.py")
    assert record["lock"]["sha256"] == provenance.sha256_file(repository / "uv.lock")
    json.dumps(record, allow_nan=False)


def test_dirty_records_staged_unstaged_untracked_and_ignores_private(repository):
    (repository / "src/example.py").write_text("VALUE = 2\n", encoding="utf-8")
    git(repository, "add", "src/example.py")
    (repository / "model.xml").write_text("<mujoco model='changed'/>\n", encoding="utf-8")
    (repository / "new.py").write_text("NEW = True\n", encoding="utf-8")
    for directory in ("artifacts", ".venv", "private"):
        (repository / directory).mkdir()
        (repository / directory / "secret.txt").write_text("must not appear", encoding="utf-8")
    record = collect(repository)
    assert record["git"]["dirty"] is True
    assert record["git"]["tracked_changes"] == ["model.xml", "src/example.py"]
    assert record["git"]["untracked_files"] == ["new.py"]
    assert record["source_snapshot_required"] is True
    assert not any("secret" in path for path in record["source_sha256"])


def test_snapshot_reconstructs_tracked_changes_and_untracked_sources(repository):
    (repository / "src/example.py").write_text("VALUE = 3\n", encoding="utf-8")
    (repository / "model.xml").unlink()
    (repository / "scripts").mkdir()
    (repository / "scripts/new.py").write_text("print('authored')\n", encoding="utf-8")
    (repository / ".env").write_text("TOKEN=do-not-copy\n", encoding="utf-8")
    destination = repository / "artifacts/run/source_snapshot"
    snapshot = provenance.snapshot_sources(repository, destination)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert snapshot["manifest_sha256"] == provenance.sha256_file(destination / "manifest.json")
    assert manifest["untracked_sources"] == ["scripts/new.py"]
    assert (destination / "untracked/scripts/new.py").read_bytes() == (repository / "scripts/new.py").read_bytes()
    assert ".env" not in manifest["source_sha256"]
    assert not (destination / "untracked/.env").exists()
    reconstruction = repository / "artifacts/reconstruction"
    git(repository, "worktree", "add", "--detach", str(reconstruction), "HEAD")
    git(reconstruction, "apply", str(destination / "tracked.patch"))
    assert (reconstruction / "src/example.py").read_bytes() == (repository / "src/example.py").read_bytes()
    assert not (reconstruction / "model.xml").exists()


def test_snapshot_requires_ignored_artifact_destination(repository):
    with pytest.raises(ValueError, match="under ignored artifacts"):
        provenance.snapshot_sources(repository, repository / "public_source_snapshot")
    (repository / ".gitignore").write_text("private/\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Git-ignored"):
        provenance.snapshot_sources(repository, repository / "artifacts/run/source_snapshot")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_strict_json_rejects_nonfinite_before_replacing_existing_file(tmp_path, invalid):
    target = tmp_path / "run.json"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError):
        provenance.write_json(target, {"metric": invalid})
    assert target.read_text(encoding="utf-8") == "original"


def test_json_is_stable_and_parseable(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    provenance.write_json(first, {"b": 2, "a": {"seed": 101, "unavailable": None}})
    provenance.write_json(second, {"a": {"unavailable": None, "seed": 101}, "b": 2})
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8"))["a"]["seed"] == 101


@pytest.mark.parametrize("invalid_json", ['{"seed":NaN}', '{"value":1e999}'])
def test_config_nonfinite_rejected(repository, invalid_json):
    (repository / "config.json").write_text(invalid_json, encoding="utf-8")
    with pytest.raises(ValueError):
        collect(repository)


def test_source_symlinks_are_rejected(repository, tmp_path):
    secret = tmp_path / "artifacts/secret.py"
    secret.parent.mkdir()
    secret.write_text("secret", encoding="utf-8")
    (repository / "src/linked.py").symlink_to(secret)
    with pytest.raises(ValueError, match="must not be a symlink"):
        collect(repository)


def test_command_string_is_not_silently_interpreted_as_argv(repository):
    with pytest.raises(ValueError, match="exact argv"):
        provenance.collect_provenance(repository, "config.json", "model.xml", "python script.py")


def test_cpu_capture_discards_machine_identifiers(monkeypatch):
    hardware = {"chip_type": "Apple Test Chip", "number_processors": "proc 12:8:4",
                "physical_memory": "24 GB", "serial_number": "DO_NOT_RETAIN",
                "platform_UUID": "DO_NOT_RETAIN", "machine_name": "DO_NOT_RETAIN"}
    monkeypatch.setattr(provenance.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(provenance.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps({"SPHardwareDataType": [hardware]}).encode()))
    cpu = provenance._cpu_info()
    assert cpu["chip"] == "Apple Test Chip"
    assert cpu["memory"] == "24 GB"
    assert cpu["logical_cores"] == 12
    assert "DO_NOT_RETAIN" not in json.dumps(cpu)


def test_symlink_parent_cannot_expose_private_source(repository):
    private = repository / "private"
    private.mkdir()
    (private / "example.py").write_text("secret", encoding="utf-8")
    (repository / "src/example.py").unlink()
    (repository / "src").rmdir()
    (repository / "src").symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        collect(repository)
