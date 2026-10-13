"""Milestone authorization, frozen inputs, and evidence-preservation guards."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from mujoco_sysid_lab.dataset import TrainingDataset, TrainingSequence
from mujoco_sysid_lab.experiment_config import load_ground_truth, load_study_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiment_runner", ROOT / "scripts/run_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_exact_registration_accepts_original_configs_and_rejects_changed_physics(tmp_path):
    config = ROOT / "configs/evidence_v1.json"
    truth = ROOT / "configs/ground_truth_v1.json"
    registration = ROOT / "docs/results/registration_v1.json"
    assert runner.validate_registration(config, truth, registration)["identification_runs"] == 0
    changed = json.loads(config.read_text())
    changed["nominal"]["mass_kg"] += 0.1
    modified = tmp_path / "evidence.json"
    modified.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Frozen evidence configuration changed"):
        runner.validate_registration(modified, truth, registration)
    with pytest.raises(ValueError, match="Frozen ground_truth configuration changed"):
        runner.validate_registration(config, modified, registration)


def test_runner_refuses_existing_output_without_touching_evidence():
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as existing:
        marker = Path(existing) / "original.txt"
        marker.write_text("preserve")
        result = subprocess.run([sys.executable, "scripts/run_experiment.py", "--output", existing],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 2
        assert "Output already exists" in result.stderr
        assert "Traceback" not in result.stderr
        assert marker.read_text() == "preserve"
        assert len(list(Path(existing).iterdir())) == 1


def test_evaluation_only_milestone_five_has_a_dedicated_command():
    result = subprocess.run([sys.executable, "scripts/run_experiment.py", "--milestone", "5"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "invalid choice" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("fit_status", [0, 1])
def test_full_command_curates_only_after_completed_fitting(monkeypatch, fit_status):
    from mujoco_sysid_lab import study_runner, final_reporting
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temporary:
        output = Path(temporary) / "new_run"
        events = []

        def fitted(root, args, target, *, through):
            assert through == 9
            assert target == output
            target.mkdir()
            (target / "run.json").write_text('{"labeled_mock":true}')
            events.append("fitting_complete" if fit_status == 0 else "fitting_incomplete")
            return fit_status

        def curated(root, source, target, command):
            assert events == ["fitting_complete"]
            assert source == output / "run.json" and source.is_file()
            assert target == output / "curation"
            assert command[1] == "scripts/run_experiment.py"
            events.append("public_curation")
            return {"status": "passed"}

        monkeypatch.setattr(study_runner, "run_study", fitted)
        monkeypatch.setattr(final_reporting, "curate_results", curated)
        monkeypatch.setattr(sys, "argv", ["scripts/run_experiment.py", "--milestone", "9", "--output", str(output)])
        assert runner.main() == fit_status
        assert events == (["fitting_complete", "public_curation"] if fit_status == 0 else ["fitting_incomplete"])


def test_noiseless_harness_simulates_only_training_controls_and_preserves_measured_inputs():
    study = load_study_config(ROOT / "configs/evidence_v1.json")
    truth = load_ground_truth(study.truth_config_path, study)
    times = np.arange(301) * study.timestep_s
    controls = (0.3 * np.sin(times * 8))[:, None]
    sequences = tuple(TrainingSequence(
        trajectory_name=name, trajectory_id=index, timestamps=times,
        controls=controls, measured_qpos=np.ones((times.size, 1)),
        measured_qvel=np.ones((times.size, 1)), initial_state=np.array([0., 0., 0.]))
        for index, name in enumerate(("train_multisine", "train_prbs")))
    measured = TrainingDataset(seed=101, sequences=sequences)
    clean = runner.make_noiseless_training(measured, truth.model, ROOT / "model/pendulum.xml")
    assert tuple(row.trajectory_name for row in clean.sequences) == ("train_multisine", "train_prbs")
    assert all(not hasattr(row, "metadata") for row in clean.sequences)
    for noisy, generated in zip(measured.sequences, clean.sequences, strict=True):
        assert np.array_equal(noisy.measured_qpos, np.ones_like(noisy.measured_qpos))
        assert np.array_equal(noisy.controls, generated.controls)
        assert generated.measured_qpos[0, 0] == generated.measured_qvel[0, 0] == 0
        assert np.isfinite(generated.measured_qpos).all()
        assert not np.array_equal(generated.measured_qpos, noisy.measured_qpos)


@pytest.mark.parametrize("phase", ["study", "curation"])
@pytest.mark.parametrize("error,exit_code,state", [(RuntimeError("injected unexpected runtime failure"), 1, "failed"),
                                                  (KeyboardInterrupt(), 130, "interrupted")])
def test_unexpected_failure_or_interruption_journals_progress_and_prints_traceback(monkeypatch, capsys, phase, error, exit_code, state):
    from mujoco_sysid_lab import study_runner, final_reporting
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temporary:
        output = Path(temporary) / "new_run"
        preserved = {"test_fixture": True, "fit_id": "B_seed101_S0", "status": "completed",
                     "optimization_returned": True, "final_objective": 2.}

        def study(root, args, target, *, through):
            target.mkdir()
            (target / "run.json").write_text(json.dumps({"status": "running" if phase == "study" else "passed",
                "fits": [preserved], "progress": {"phase": "evaluation", "active_group": "B_seed101"},
                "provenance": {"test_fixture": True}}))
            if phase == "study":
                raise error
            return 0

        def curate(*args, **kwargs):
            raise error

        monkeypatch.setattr(study_runner, "run_study", study)
        monkeypatch.setattr(final_reporting, "curate_results", curate)
        monkeypatch.setattr(sys, "argv", ["scripts/run_experiment.py", "--output", str(output)])
        assert runner.main() == exit_code
        journal = json.loads((output / "run.json").read_text())
        failure = json.loads((output / "orchestration_failure.json").read_text())
        assert journal["status"] == failure["status"] == state
        assert journal["fits"] == [preserved]
        assert failure["phase"] == phase
        assert failure["provenance"] == {"test_fixture": True}
        assert failure["evidence_preserved"] is True
        assert type(error).__name__ in (output / "orchestration_failure.txt").read_text()
        assert "Traceback" in capsys.readouterr().err


def test_collision_failure_cannot_modify_someone_elses_evidence(tmp_path):
    output = tmp_path / "already_exists"
    output.mkdir()
    marker = output / "run.json"
    marker.write_text('{"retained_fixture":true}')
    runner.record_orchestration_failure(output, FileExistsError("racing output collision"), phase="study")
    assert marker.read_text() == '{"retained_fixture":true}'
    assert list(output.iterdir()) == [marker]


def test_cli_exposes_and_forwards_explicit_fresh_and_cache_flags(monkeypatch):
    from mujoco_sysid_lab import study_runner
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temporary:
        output = Path(temporary) / "new_run"

        def mocked(root, args, target, *, through):
            assert args.fresh is False
            assert args.unit_cache_run == "artifacts/custom_units"
            assert args.physical_cache_run == "artifacts/custom_physical"
            assert args.delay_cache_run == "artifacts/custom_delay"
            return 1

        monkeypatch.setattr(study_runner, "run_study", mocked)
        monkeypatch.setattr(sys, "argv", ["scripts/run_experiment.py", "--milestone", "7", "--output", str(output),
            "--unit-cache-run", "artifacts/custom_units", "--physical-cache-run", "artifacts/custom_physical",
            "--delay-cache-run", "artifacts/custom_delay"])
        assert runner.main() == 1


@pytest.mark.parametrize("arguments", [("--fresh", "--unit-cache-run", "artifacts/custom"),
                                       ("--milestone", "3", "--fresh")])
def test_conflicting_or_inapplicable_cache_flags_fail_before_execution(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["scripts/run_experiment.py", *arguments])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


@pytest.mark.parametrize("error,code,state", [(RuntimeError("mock noiseless preparation failure"), 1, "failed"),
                                            (KeyboardInterrupt(), 130, "interrupted")])
def test_legacy_milestone3_preparation_failure_uses_same_journal_contract(monkeypatch, capsys, error, code, state):
    from mujoco_sysid_lab import dataset, provenance
    (ROOT / "artifacts").mkdir(exist_ok=True)
    study = load_study_config(ROOT / "configs/evidence_v1.json")
    record = provenance.collect_provenance(ROOT, ROOT / "configs/evidence_v1.json", ROOT / "model/pendulum.xml",
                                           [sys.executable, "-m", "pytest", "labeled M3 failure fixture"])
    times = np.arange(4001) * .002
    safe = TrainingDataset(101, tuple(TrainingSequence(name, index, times, np.zeros((4001, 1)),
                            np.zeros((4001, 1)), np.zeros((4001, 1)), np.zeros(3))
                            for index, name in enumerate(("train_multisine", "train_prbs"))))
    monkeypatch.setattr(provenance, "collect_provenance", lambda *args, **kwargs: dict(record))
    monkeypatch.setattr(provenance, "snapshot_sources", lambda *args, **kwargs: {"test_fixture": True})
    monkeypatch.setattr(dataset, "load_training_sequences", lambda *args, **kwargs: safe)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner, "make_noiseless_training", fail)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temporary:
        fixture = Path(temporary)
        manifest = fixture / "manifest.json"
        manifest.write_text(json.dumps({"test_fixture": True, "seed": 101, "scenario": "matched",
            "provenance": record, "metadata": {"truth_config_sha256": provenance.sha256_file(study.truth_config_path)}}))
        output = fixture / "new_run"
        monkeypatch.setattr(sys, "argv", ["scripts/run_experiment.py", "--milestone", "3",
            "--dataset", str(manifest), "--output", str(output)])
        assert runner.main() == code
        result = json.loads((output / "run.json").read_text())
        assert result["status"] == state and result["fits"] == []
        assert result["orchestration_failure"]["phase"] == "milestone3"
        assert type(error).__name__ in (output / "orchestration_failure.txt").read_text()
        assert "Traceback" in capsys.readouterr().err
