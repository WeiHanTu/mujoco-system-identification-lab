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
