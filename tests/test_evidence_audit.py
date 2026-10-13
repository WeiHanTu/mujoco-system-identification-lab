"""Audit arithmetic, hash safety, and original-source reconstruction fixtures."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json
from mujoco_sysid_lab.experiment_config import config_to_dict, load_ground_truth, load_study_config


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_evidence", ROOT / "scripts/audit_evidence.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def source_tree(tmp_path):
    # Real temporary Git commit for an authored test fixture. No project
    # checkout commit or scientific evidence is created by this fixture.
    root = tmp_path / "authored_source_fixture"
    root.mkdir()
    (root / ".gitignore").write_text("artifacts/\n")
    (root / "config.json").write_text('{"test_fixture":true}\n')
    (root / "model.xml").write_text('<mujoco model="test_fixture"/>\n')
    (root / "uv.lock").write_text("authored lock fixture\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Authored test fixture",
                    "-c", "user.email=fixture@example.invalid", "commit", "-qm", "authored source fixture"], check=True)
    return root


def provenance(root):
    return collect_provenance(root, root / "config.json", root / "model.xml",
                              ["python", "-m", "pytest", "authored source verification fixture"])


def test_clean_source_verification_uses_original_git_bytes_not_current_files(source_tree):
    record = provenance(source_tree)
    assert record["git"]["dirty"] is False
    (source_tree / "model.xml").write_text("later unrelated checkout change")
    verified = audit.verify_source(source_tree, record)
    assert verified["dirty"] is False
    assert verified["source_files_verified"] == 4


def test_dirty_source_reconstructs_tracked_patch_and_authored_untracked_file(source_tree):
    (source_tree / "model.xml").write_text('<mujoco model="dirty fixture"/>\n')
    (source_tree / "helper.py").write_text("# authored untracked audit fixture\nVALUE = 2\n")
    record = provenance(source_tree)
    record["source_snapshot"] = snapshot_sources(source_tree, source_tree / "artifacts/source_snapshot")
    # Current files can subsequently diverge. The original snapshot is authority.
    (source_tree / "helper.py").write_text("later change")
    verified = audit.verify_source(source_tree, record)
    assert verified["dirty"] is True
    assert verified["source_files_verified"] == 5


def test_dirty_snapshot_tampering_is_rejected_before_reconstruction(source_tree):
    (source_tree / "model.xml").write_text('<mujoco model="dirty fixture"/>\n')
    record = provenance(source_tree)
    record["source_snapshot"] = snapshot_sources(source_tree, source_tree / "artifacts/source_snapshot")
    patch = source_tree / "artifacts/source_snapshot/tracked.patch"
    patch.write_text(patch.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch|patch"):
        audit.verify_source(source_tree, record)


def test_dirty_source_without_snapshot_is_rejected(source_tree):
    (source_tree / "model.xml").write_text("dirty fixture")
    with pytest.raises(ValueError, match="snapshot"):
        audit.verify_source(source_tree, provenance(source_tree))


def test_original_source_hash_mismatch_is_rejected(source_tree):
    record = provenance(source_tree)
    record["source_sha256"]["model.xml"] = "0" * 64
    with pytest.raises(ValueError, match="Original source mismatch"):
        audit.verify_source(source_tree, record)


@pytest.mark.parametrize("payload", ['{"bad":NaN}', '{"bad":Infinity}', '{"bad":1e999}', '{"a":1,"a":2}'])
def test_nonfinite_or_duplicate_json_is_rejected(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        audit.read_json(path)


def test_reference_hash_and_scope_are_verified(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"test_fixture":true}')
    reference = {"path": "record.json", "sha256": sha256_file(path)}
    assert audit.verified_ref(tmp_path, reference) == path
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.verified_ref(tmp_path, reference)
    with pytest.raises(ValueError, match="escapes"):
        audit.safe_path(tmp_path, "../outside.json")


def test_manual_metric_units_normalization_objective_and_window():
    times = np.arange(101) * .002
    arrays = {"timestamps": times, "measured_qpos": np.full((101, 1), .002),
              "measured_qvel": np.full((101, 1), .040)}
    study = SimpleNamespace(sensor_sigmas={"qpos": .002, "qvel": .020})
    metrics, residual = audit.manual_metrics(arrays, np.zeros((101, 2)), study)
    assert metrics["n_samples"] == 41  # Inclusive [.06,.14], no prefix/suffix scoring.
    assert metrics["qpos_rmse"] == pytest.approx(.002)
    assert metrics["qvel_rmse"] == pytest.approx(.040)
    assert metrics["normalized_rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["qpos_p95_absolute_error"] == .002
    assert metrics["qvel_p95_absolute_error"] == .040
    np.testing.assert_array_equal(residual, np.tile([1., 2.], 41))
    assert metrics["objective"] == pytest.approx(.5 * residual @ residual)
    assert metrics["objective"] == pytest.approx(metrics["n_samples"] * metrics["normalized_rmse"] ** 2)


def test_manual_metric_uses_fixed_inclusive_window_independent_of_prediction():
    times = np.arange(101) * .002
    q = np.zeros((101, 1))
    q[0] = 100.
    q[-1] = 100.
    arrays = {"timestamps": times, "measured_qpos": q, "measured_qvel": q.copy()}
    study = SimpleNamespace(sensor_sigmas={"qpos": .002, "qvel": .020})
    metrics, residual = audit.manual_metrics(arrays, np.zeros((101, 2)), study)
    assert metrics["n_samples"] == 41
    assert metrics["objective"] == 0.
    assert np.array_equal(residual, np.zeros(82))


def test_source_fit_resolves_hashed_cached_unit_run_and_individual_result(tmp_path):
    run = tmp_path / "units/run.json"
    row = {"test_fixture": True, "fit_id": "A1_seed101_S0"}
    write_json(run, {"fits": [row]})
    retained = {"fit_id": row["fit_id"], "source_result": {"path": "units/run.json", "sha256": sha256_file(run)}}
    loaded, directory = audit.source_fit(tmp_path, retained)
    assert loaded == row
    assert directory == tmp_path / "units/fits/A1_seed101_S0"
    path = tmp_path / "joint/fits/B_seed101_S0/result.json"
    write_json(path, {"test_fixture": True, "fit_id": "B_seed101_S0"})
    retained = {"fit_id": "B_seed101_S0", "source_result": {"path": path.relative_to(tmp_path).as_posix(), "sha256": sha256_file(path)}}
    _, directory = audit.source_fit(tmp_path, retained)
    assert directory == path.parent


def test_fit_refs_support_both_cached_and_new_raw_layouts(tmp_path):
    directory = tmp_path / "run/fits/B_seed101_S0"
    directory.mkdir(parents=True)
    path = directory / "residuals.npz"
    path.write_bytes(b"authored reference-path fixture")
    digest = sha256_file(path)
    assert audit.fit_ref(tmp_path, directory, {"path": "residuals.npz", "sha256": digest}) == path
    assert audit.fit_ref(tmp_path, directory, {"path": "fits/B_seed101_S0/residuals.npz", "sha256": digest}) == path


def test_incomplete_execution_cannot_be_certified(tmp_path):
    path = tmp_path / "run.json"
    write_json(path, {"schema_version": "mujoco-sysid-lab.full-run.v1", "status": "incomplete"})
    with pytest.raises(ValueError, match="incomplete"):
        audit.audit_run(tmp_path, path)


@pytest.fixture
def mechanics():
    study = load_study_config(ROOT/"configs/evidence_v1.json")
    truth = load_ground_truth(study.truth_config_path, study)
    return study, truth.model


def unit_fixture(cell, study, truth):
    """Labeled raw-layout fixture, not an optimization or evidence fit."""
    initial = config_to_dict(study.nominal)
    if cell == "A1":
        initial["damping_nm_s_per_rad"] = truth.damping_nm_s_per_rad
        free, vector = ("mass_kg",), np.array([1.1999])
    else:
        initial["mass_kg"] = truth.mass_kg
        free, vector = ("damping_nm_s_per_rad",), np.array([.15002])
    raw = {"schema_version": "mujoco-sysid-lab.training-fit.v2", "initial_physics": initial,
           "free_parameters": list(free)}
    retained = {"cell": cell, "initial_physics": dict(initial)}
    return raw, retained, free, vector


@pytest.mark.parametrize("cell", ["A1", "A2"])
def test_fresh_unit_physics_uses_native_free_value_and_known_true_nuisance(mechanics, cell):
    study, truth = mechanics
    raw, retained, free, vector = unit_fixture(cell, study, truth)
    original = json.dumps(raw, sort_keys=True)
    physical = audit.candidate_physics(retained, raw, free, vector, study, truth)
    assert physical[free[0]] == vector[0]
    nuisance = "damping_nm_s_per_rad" if cell == "A1" else "mass_kg"
    assert physical[nuisance] == getattr(truth, nuisance)
    assert physical[nuisance] != getattr(study.nominal, nuisance)
    assert json.dumps(raw, sort_keys=True) == original


@pytest.mark.parametrize("origin", ["raw", "retained"])
def test_explicit_cached_physics_must_match_native_reconstruction(mechanics, origin):
    study, truth = mechanics
    raw, retained, free, vector = unit_fixture("A1", study, truth)
    physical = audit.candidate_physics(retained, raw, free, vector, study, truth)
    record = raw if origin == "raw" else retained
    record["fitted_physics"] = dict(physical)
    assert audit.candidate_physics(retained, raw, free, vector, study, truth) == physical
    record["fitted_physics"]["mass_kg"] += .0001
    with pytest.raises(ValueError, match=f"Explicit {origin} fitted physics"):
        audit.candidate_physics(retained, raw, free, vector, study, truth)


@pytest.mark.parametrize("cell", ["A1", "A2"])
def test_nominal_unit_nuisance_cannot_be_assumed_or_certified(mechanics, cell):
    study, truth = mechanics
    raw, retained, free, vector = unit_fixture(cell, study, truth)
    nuisance = "damping_nm_s_per_rad" if cell == "A1" else "mass_kg"
    raw["initial_physics"][nuisance] = getattr(study.nominal, nuisance)
    retained["initial_physics"] = dict(raw["initial_physics"])
    with pytest.raises(ValueError, match="oracle-assisted fixed"):
        audit.candidate_physics(retained, raw, free, vector, study, truth)


def test_retained_initial_physics_must_match_hashed_raw_record(mechanics):
    study, truth = mechanics
    raw, retained, free, vector = unit_fixture("A1", study, truth)
    retained["initial_physics"]["mass_kg"] += .01
    with pytest.raises(ValueError, match="Retained/raw initial physics"):
        audit.candidate_physics(retained, raw, free, vector, study, truth)


def test_delay_only_reconstruction_preserves_training_selected_fixed_physics(mechanics):
    study, truth = mechanics
    initial = {**config_to_dict(study.nominal), "mass_kg": 1.19985, "damping_nm_s_per_rad": .15003}
    raw = {"initial_physics": initial, "free_parameters": ["delay_s"]}
    assert audit.candidate_physics({"cell": "C_delay_only"}, raw, ("delay_s",), np.array([.02398]), study, truth) == initial
