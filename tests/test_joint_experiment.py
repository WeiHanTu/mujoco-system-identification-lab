"""Truth-loading order and honest failure accounting in joint orchestration."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from mujoco_sysid_lab import joint_experiment as experiment
from mujoco_sysid_lab import joint_identification as worker
from mujoco_sysid_lab.dataset import TrainingDataset, TrainingSequence
from mujoco_sysid_lab.provenance import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def test_joint_registration_does_not_deserialize_truth_file(monkeypatch):
    original = Path.read_text
    truth = ROOT / "configs/ground_truth_v1.json"

    def guarded(path, *args, **kwargs):
        assert path != truth, "Physical truth was deserialized before fitting"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    registration = experiment.check_registration(ROOT / "configs/evidence_v1.json", truth,
                                                 ROOT / "docs/results/registration_v1.json")
    assert registration["ground_truth_config_sha256"] == sha256_file(truth)


@pytest.mark.parametrize("valid_candidate", [False, True])
@pytest.mark.parametrize("external_interpreter", [False, True])
def test_truth_loader_runs_only_after_selection_and_failures_are_counted(monkeypatch, valid_candidate, external_interpreter):
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temporary:
        base = Path(temporary)
        if external_interpreter:
            # An interpreter outside the checkout must remain usable. This
            # labeled path exercises argv/provenance formatting; explicit data
            # and mocked fitting keep the test from launching that interpreter.
            external_path = Path(tempfile.gettempdir()) / "sysid-external-interpreter" / "python"
            assert not external_path.is_relative_to(ROOT)
            monkeypatch.setattr(experiment.sys, "executable", str(external_path))
        output = base / "run"
        manifest = base / "manifest.json"
        write_json(manifest, {
            "seed": 101, "scenario": "matched",
            "provenance": {
                "config": {"sha256": sha256_file(ROOT / "configs/evidence_v1.json")},
                "model": {"sha256": sha256_file(ROOT / "model/pendulum.xml")}},
            "metadata": {"truth_config_sha256": sha256_file(ROOT / "configs/ground_truth_v1.json")},
        })
        times = np.arange(4001) * .002
        training = TrainingDataset(seed=101, sequences=tuple(
            TrainingSequence(name, index, times, np.zeros((4001, 1)), np.zeros((4001, 1)),
                             np.zeros((4001, 1)), np.zeros(3))
            for index, name in enumerate(("train_multisine", "train_prbs"))))
        monkeypatch.setattr(experiment, "load_training_sequences", lambda path: training)
        events = []

        def mocked_starts(inputs, study, model_path, target, provenance):
            assert "truth_config" not in provenance
            events.append("three_attempts")
            rows = [{"fit_id": f"B_seed101_{start}", "start_id": start,
                     "optimization_attempted": True, "optimization_invoked": False,
                     "optimization_returned": False, "status": "optimization_failed"}
                    for start in ("S0", "S1", "S2")]
            selection = {"selected_fit_id": "B_seed101_S1" if valid_candidate else None}
            write_json(target / "selection.json", selection)
            events.append("selection_sealed")
            return {"fits": rows, "selection": selection, "status": "incomplete"}

        def guarded_truth(*args):
            assert events == ["three_attempts", "selection_sealed"]
            assert (output / "selection.json").is_file()
            events.append("truth_evaluation")
            raise RuntimeError("Stop after verifying truth-load order")

        monkeypatch.setattr(worker, "run_joint_starts", mocked_starts)
        monkeypatch.setattr(experiment, "load_ground_truth", guarded_truth)
        args = argparse.Namespace(config="configs/evidence_v1.json",
                                  registration="docs/results/registration_v1.json",
                                  dataset=manifest.relative_to(ROOT).as_posix(), milestone=4)
        assert experiment.run_milestone4(ROOT, args, output) == 1
        result = json.loads((output / "run.json").read_text())
        assert result["status"] == "failed"
        assert result["fit_counts"]["registered_slots"] == 3
        assert result["fit_counts"]["optimization_invoked"] == 0
        assert result["registered_evidence_fits_this_milestone"] == 0
        assert result["cumulative_registered_evidence_fit_count"] == 2
        assert result["heldout_evaluated"] is False
        if external_interpreter:
            assert result["provenance"]["command"]["argv"][0] == str(external_path)
            assert result["provenance"]["python"]["executable_scope"] == "absolute"
        assert events == ["three_attempts", "selection_sealed"] + (["truth_evaluation"] if valid_candidate else [])
