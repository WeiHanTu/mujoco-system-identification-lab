"""Archive integrity tests use tiny isolated Git/evidence fixtures only."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from mujoco_sysid_lab import provenance


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/archive_evidence.py"
spec = importlib.util.spec_from_file_location("archive_evidence", SCRIPT)
archive_evidence = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(archive_evidence)


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False, sort_keys=True) + "\n")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Archive Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    (tmp_path / ".gitignore").write_text("artifacts/\nprivate/\n.venv/\n")
    save(tmp_path / "configs/evidence.json", {"noise_seeds": [101, 102], "frozen": True})
    (tmp_path / "model").mkdir()
    (tmp_path / "model/model.xml").write_text("<mujoco/>\n")
    (tmp_path / "uv.lock").write_text("version=1\n")
    (tmp_path / "source.py").write_text("VALUE=1\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Synthetic archive fixture")
    monkeypatch.setattr(provenance, "_package_versions", lambda: {"mujoco": "3.15.0", "pytest": "8.4.2"})
    monkeypatch.setattr(provenance, "_cpu_info", lambda: {"chip": "test fixture", "logical_cores": 1})
    original = provenance.collect_provenance(tmp_path, "configs/evidence.json", "model/model.xml", ["python", "generate_fixture.py"])
    m3 = tmp_path / "artifacts/milestone3/repaired"
    save(m3 / "run.json", {"provenance": original, "fits": []})
    (m3 / "native.pkl").write_bytes(b"fake fixture bytes; not unpickled")
    m4 = tmp_path / "artifacts/milestone4/seed101"
    save(m4 / "run.json", {"provenance": original, "fits": []})
    save(m4 / "fits/B_S0/result.json", {"estimate": 1.2})
    m6 = tmp_path / "artifacts/milestone6/seed101"
    save(m6 / "fits/C_S0/result.json", {"estimate": .024})
    save(m6 / "run.json", {"provenance": original, "fits": [{"source_result": {
        "path": (m4 / "fits/B_S0/result.json").relative_to(tmp_path).as_posix(),
        "sha256": provenance.sha256_file(m4 / "fits/B_S0/result.json")}}]})
    checkpoint = tmp_path / archive_evidence.CHECKPOINT
    save(checkpoint / "run.json", {"status": "passed", "optimization_calls": 0})
    run = tmp_path / "artifacts/full/test"
    save(run / "run.json", {"provenance": original, "fits": [
        {"seed": 101, "reused": True, "source_result": {"path": (m3 / "run.json").relative_to(tmp_path).as_posix(), "sha256": provenance.sha256_file(m3 / "run.json")}},
        {"seed": 102, "reused": True, "source_result": {"path": (m6 / "fits/C_S0/result.json").relative_to(tmp_path).as_posix(), "sha256": provenance.sha256_file(m6 / "fits/C_S0/result.json")}},
    ]})
    (run / "arrays.npz").write_bytes(b"small binary fixture\0\1\2")
    extra = tmp_path / "artifacts/curation/current"
    save(extra / "summary.json", {"scope": "synthetic"})
    return {"root": tmp_path, "run": run, "extra": extra, "m3": m3, "m4": m4, "m6": m6,
            "output": tmp_path / "artifacts/archives/test.tar.gz",
            "manifest": tmp_path / "artifacts/archives/test.json"}


def create(fixture, **kwargs):
    values = {"root": fixture["root"], "run": fixture["run"], "output": fixture["output"],
              "manifest": fixture["manifest"], "includes": [fixture["extra"]],
              "argv": ["python", "scripts/archive_evidence.py", "--run", "artifacts/full/test",
                       "--output", "artifacts/archives/test.tar.gz", "--manifest", "artifacts/archives/test.json",
                       "--include", "artifacts/curation/current"]}
    values.update(kwargs)
    return archive_evidence.create_archive(**values)


def test_exact_inventory_members_cached_dependencies_and_source_provenance(fixture):
    result = create(fixture)
    assert result["status"] == "verified"
    assert result["verification"]["passed"] is True
    assert result["archive"]["sha256"] == provenance.sha256_file(fixture["output"])
    assert result["archive"]["size_bytes"] == fixture["output"].stat().st_size
    names = [item["path"] for item in result["inventory"]]
    assert names == sorted(set(names))
    assert result["verification"]["member_count"] == len(names)
    assert "artifacts/full/test/run.json" in names
    assert "artifacts/milestone3/repaired/native.pkl" in names
    assert "artifacts/milestone4/seed101/fits/B_S0/result.json" in names
    assert "artifacts/milestone6/seed101/run.json" in names
    assert f"{archive_evidence.CHECKPOINT.as_posix()}/run.json" in names
    assert "artifacts/curation/current/summary.json" in names
    assert "artifacts/archives/test_source/provenance.json" in names
    assert "artifacts/archives/test.json" not in names
    assert "artifacts/archives/test.tar.gz" not in names
    assert result["source_seeds"] == [101, 102]
    assert len(result["cached_dependency_runs"]) == 3
    assert result["source_provenance"]["config"]["content"] == {"noise_seeds": [101, 102], "frozen": True}
    assert result["source_provenance"]["git"]["dirty"] is False
    assert result["producer_provenance"]["command"]["argv"][-1] == "artifacts/curation/current"
    assert json.loads(fixture["manifest"].read_text()) == result
    with tarfile.open(fixture["output"], "r:gz") as stream:
        for member, item in zip(stream, result["inventory"], strict=True):
            assert member.name == item["path"] and member.isfile()
            assert member.uid == member.gid == 0 and member.uname == member.gname == ""
            assert member.mtime == 0
            assert hashlib.sha256(stream.extractfile(member).read()).hexdigest() == item["sha256"]


@pytest.mark.parametrize("key", ("output", "manifest"))
def test_output_collision_preserves_existing_file_and_creates_nothing(fixture, key):
    target = fixture[key]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"existing evidence")
    with pytest.raises(ValueError, match="already exist"):
        create(fixture)
    assert target.read_bytes() == b"existing evidence"
    other = fixture["manifest" if key == "output" else "output"]
    assert not other.exists()
    assert not (target.parent / "test_source").exists()


def test_producer_directory_collision_rejected_before_outputs(fixture):
    target = fixture["manifest"].with_name("test_source")
    target.mkdir(parents=True)
    (target / "original.txt").write_text("retained")
    with pytest.raises(ValueError, match="already exist"):
        create(fixture)
    assert not fixture["output"].exists() and (target / "original.txt").read_text() == "retained"


def test_filters_private_runtime_appledouble_and_existing_archives(fixture):
    run = fixture["run"]
    for name in ("private/notes.md", ".venv/lib/token.txt", ".git/config", "__pycache__/module.pyc",
                 "._arrays.npz", ".DS_Store", "older.tar.gz", "other.zip"):
        file = run / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("must not be archived")
    result = create(fixture)
    names = [item["path"] for item in result["inventory"]]
    assert all(not any(part in archive_evidence.FORBIDDEN_PARTS for part in Path(name).parts) for name in names)
    assert not any("._" in name or name.endswith((".tar.gz", ".zip", ".DS_Store")) for name in names)
    assert len(result["excluded"]) == 8


@pytest.mark.parametrize("mode", ("member", "input_parent", "destination_parent"))
def test_symlinks_rejected_before_archive(fixture, mode):
    if mode == "member":
        (fixture["run"] / "linked.txt").symlink_to(fixture["root"] / "source.py")
        kwargs = {}
    elif mode == "input_parent":
        link = fixture["root"] / "artifacts/linked"
        link.symlink_to(fixture["extra"], target_is_directory=True)
        kwargs = {"includes": [link / "summary.json"]}
    else:
        link = fixture["root"] / "artifacts/archives"
        link.symlink_to(fixture["extra"], target_is_directory=True)
        kwargs = {}
    with pytest.raises(ValueError, match="Symlinks"):
        create(fixture, **kwargs)


@pytest.mark.parametrize("path", ("private/interview.md", ".venv/file", "../outside", "artifacts"))
def test_outside_or_broad_inputs_rejected(fixture, path):
    with pytest.raises(ValueError, match="artifacts|traversal"):
        create(fixture, includes=[path])


def test_output_inside_included_tree_rejected(fixture):
    with pytest.raises(ValueError, match="inside any selected input"):
        create(fixture, output=fixture["run"] / "nested.tar.gz")
    assert not (fixture["run"] / "nested.tar.gz").exists()


def test_input_files_and_modes_remain_unchanged(fixture):
    paths = [path for path in (fixture["root"] / "artifacts").rglob("*") if path.is_file()]
    before = {path: (path.read_bytes(), path.stat().st_mode) for path in paths}
    git_before = git(fixture["root"], "status", "--porcelain")
    create(fixture)
    assert git(fixture["root"], "status", "--porcelain") == git_before
    for path, (content, mode) in before.items():
        assert path.read_bytes() == content and path.stat().st_mode == mode


def test_dirty_producer_snapshot_saved_without_private_content(fixture):
    (fixture["root"] / "source.py").write_text("VALUE=2\n")
    (fixture["root"] / "new.py").write_text("NEW=True\n")
    private = fixture["root"] / "private/notes.md"
    private.parent.mkdir()
    private.write_text("PRIVATE_SENTINEL")
    result = create(fixture)
    producer = result["producer_provenance"]
    assert producer["git"]["dirty"] is True
    assert producer["source_snapshot"]["base_commit"] == git(fixture["root"], "rev-parse", "HEAD")
    names = {item["path"] for item in result["inventory"]}
    assert "artifacts/archives/test_source/snapshot/tracked.patch" in names
    assert "artifacts/archives/test_source/snapshot/untracked/new.py" in names
    with tarfile.open(fixture["output"], "r:gz") as stream:
        assert all(b"PRIVATE_SENTINEL" not in stream.extractfile(member).read() for member in stream)


def test_cached_reference_hash_or_missing_parent_fails(fixture):
    file = fixture["run"] / "run.json"
    record = json.loads(file.read_text())
    record["fits"][0]["source_result"]["sha256"] = "0" * 64
    save(file, record)
    with pytest.raises(ValueError, match="hash differs"):
        create(fixture)
    assert not fixture["output"].exists()


def test_missing_cached_reference_not_silently_omitted(fixture):
    file = fixture["run"] / "run.json"
    record = json.loads(file.read_text())
    record["fits"][0].pop("source_result")
    save(file, record)
    with pytest.raises(ValueError, match="missing its source_result"):
        create(fixture)


def test_config_or_model_change_rejected_before_archive(fixture):
    (fixture["root"] / "model/model.xml").write_text("<mujoco model='changed'/>\n")
    with pytest.raises(ValueError, match="config/model bytes differ"):
        create(fixture)
    assert not fixture["output"].exists()


def test_argv_string_is_rejected_instead_of_becoming_character_arguments(fixture):
    with pytest.raises(ValueError, match="exact argv"):
        create(fixture, argv="python scripts/archive_evidence.py")
    assert not fixture["output"].exists()


@pytest.mark.parametrize("tamper", ("bytes", "extra", "missing", "symlink"))
def test_verify_rejects_tampered_archive(fixture, tamper):
    result = create(fixture)
    target = fixture["root"] / "artifacts/tampered.tar.gz"
    with tarfile.open(target, "w:gz", format=tarfile.PAX_FORMAT) as stream:
        for index, item in enumerate(result["inventory"]):
            if tamper == "missing" and index == 0:
                continue
            payload = (fixture["root"] / item["path"]).read_bytes()
            if tamper == "bytes" and index == 0:
                payload = bytes((payload[0] ^ 1,)) + payload[1:]
            info = tarfile.TarInfo(item["path"])
            info.size = len(payload)
            if tamper == "symlink" and index == 0:
                info.type = tarfile.SYMTYPE
                info.linkname = "private/notes"
                info.size = 0
                stream.addfile(info)
            else:
                stream.addfile(info, io.BytesIO(payload))
        if tamper == "extra":
            info = tarfile.TarInfo("artifacts/unexpected.txt")
            info.size = 5
            stream.addfile(info, io.BytesIO(b"extra"))
    with pytest.raises(ValueError, match="bytes differ|unexpected|count/order"):
        archive_evidence.verify_archive(target, result["inventory"])


def test_changed_source_during_archive_is_retained_as_failed_evidence(fixture, monkeypatch):
    original_verify = archive_evidence.verify_archive
    def mutate_after_verification(path, inventory):
        result = original_verify(path, inventory)
        (fixture["run"] / "arrays.npz").write_bytes(b"modified during archive fixture")
        return result
    monkeypatch.setattr(archive_evidence, "verify_archive", mutate_after_verification)
    with pytest.raises(ValueError, match="changed while archiving"):
        create(fixture)
    record = json.loads(fixture["manifest"].read_text())
    assert record["status"] == "failed" and record["failure"]["type"] == "ValueError"
    assert fixture["output"].is_file() and record["archive"]["sha256"] == provenance.sha256_file(fixture["output"])


@pytest.mark.parametrize("raw", ('{"fits":[],"fits":[]}', '{"x":NaN}', '{"x":1e999}'))
def test_strict_evidence_json_rejected(fixture, raw):
    (fixture["run"] / "run.json").write_text(raw)
    with pytest.raises(ValueError):
        create(fixture)
