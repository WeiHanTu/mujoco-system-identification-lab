"""Selection contracts and reduced verification; no registered B evidence run."""

from dataclasses import replace
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from mujoco_sysid_lab.dataset import TRAIN_ARRAY_NAMES, TrainingDataset, TrainingSequence
from mujoco_sysid_lab.excitation import generate_control
from mujoco_sysid_lab.experiment_config import load_study_config
from mujoco_sysid_lab.joint_identification import run_joint_starts, select_training_winner
from mujoco_sysid_lab.provenance import collect_provenance, sha256_file
from mujoco_sysid_lab.simulation import compile_model, simulate


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "pendulum.xml"


def _row(start_id, objective, *, success=True, report_status="completed", returned=True):
    return {"cell": "B", "seed": 101, "start_id": start_id, "fit_id": f"B_seed101_{start_id}",
            "optimization_returned": returned, "optimized_vector": [1.1, .2],
            "final_objective": objective, "optimizer_success": success, "report_status": report_status}


def test_selection_includes_nonconverged_and_report_failed_numerical_fit():
    rows = [_row("S0", 9), _row("S1", 3, success=False, report_status="failed"), _row("S2", 5)]
    selection = select_training_winner(rows)
    assert selection["selected_start_id"] == "S1"
    assert selection["selected_final_objective"] == 3
    assert selection["convergence_and_reporting_are_selection_filters"] is False
    assert selection["physical_truth_evaluated"] is False


def test_exact_ties_use_registered_start_order_independent_of_input_order():
    rows = [_row("S2", 3), _row("S1", 3), _row("S0", 3)]
    assert select_training_winner(rows)["selected_start_id"] == "S0"
    rows[1]["final_objective"] = np.nextafter(3., -np.inf)
    assert select_training_winner(rows)["selected_start_id"] == "S1"


def test_selection_ignores_injected_truth_errors_and_heldout_scores():
    rows = [_row("S0", 9), _row("S1", 3), _row("S2", 5)]
    rows[0].update(relative_physical_error=0.0, heldout_rmse=0.0)
    rows[1].update(relative_physical_error=100., heldout_rmse=100.)
    assert select_training_winner(rows)["selected_start_id"] == "S1"


def test_all_invalid_selection_is_finite_explicit_failure():
    rows = [_row("S0", None, returned=False), _row("S1", np.inf), _row("S2", np.nan)]
    selection = select_training_winner(rows)
    assert selection["status"] == "all_invalid"
    assert selection["selected_fit_id"] is None
    assert all(not row["eligible"] for row in selection["candidate_eligibility"])
    json.dumps(selection, allow_nan=False)


@pytest.mark.parametrize("bad", [
    [_row("S0", 1)],
    [_row("S0", 1), _row("S0", 2), _row("S2", 3)],
    [_row("S0", 1), _row("S1", 2), dict(_row("S2", 3), seed=102)],
])
def test_invalid_selection_identity_rejected(bad):
    with pytest.raises(ValueError, match="selection"):
        select_training_winner(bad)


@pytest.fixture(scope="module")
def study():
    return load_study_config(ROOT / "configs" / "evidence_v1.json")


@pytest.fixture(scope="module")
def training(study):
    # Authored 1.2-second noisy test fixture; not an 8-second evidence dataset.
    target = replace(study.nominal, mass_kg=1.2, damping_nm_s_per_rad=.15)
    model = compile_model(MODEL, target)
    sequences = []
    for numeric_id, name in enumerate(("train_multisine", "train_prbs")):
        times, controls, _ = generate_control(study, name)
        times, controls = times[:601], controls[:601]
        latent = simulate(model, times, controls)
        random = np.random.Generator(np.random.PCG64(772 + numeric_id))
        q = latent.qpos + .002 * random.standard_normal(latent.qpos.shape)
        v = latent.qvel + .020 * random.standard_normal(latent.qvel.shape)
        sequences.append(TrainingSequence(name, numeric_id, times, controls, q, v, np.zeros(3)))
    return TrainingDataset(101, tuple(sequences))


@pytest.fixture(scope="module")
def provenance():
    return collect_provenance(ROOT, ROOT / "configs" / "evidence_v1.json", MODEL,
                              [".venv/bin/python", "-m", "pytest", "tests/test_joint_identification.py"])


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory, training, study, provenance):
    output = tmp_path_factory.mktemp("joint_verification")
    (output / "run.json").write_text('{"test_fixture":true}')
    outcome = run_joint_starts(training, study, MODEL, output, provenance)
    return output, outcome


def test_reduced_official_joint_smoke_saves_three_starts_and_sealed_selection(completed_run, study):
    output, outcome = completed_run
    assert outcome["status"] == "completed"
    assert [row["start_id"] for row in outcome["fits"]] == ["S0", "S1", "S2"]
    assert (output / "run.json").read_text() == '{"test_fixture":true}'
    assert outcome["selection_sha256"] == sha256_file(output / "selection.json")
    assert outcome["training_fits_sha256"] == sha256_file(output / "training_fits.json")
    selection = json.loads((output / "selection.json").read_text())
    assert selection["sealed_before_physical_truth_evaluation"] is True
    assert selection["run_purpose"] == "reduced_duration_verification"
    assert selection["selected_result"]["sha256"] == sha256_file(output / selection["selected_result"]["path"])
    assert selection["selected_final_objective"] == min(row["final_objective"] for row in outcome["fits"])
    for row in outcome["fits"]:
        assert row["registered_evidence_fit"] is False
        assert row["oracle_assisted"] is False and row["physical_truth_evaluated"] is False
        assert row["optimization_invoked"] and row["optimization_returned"]
        assert row["optimization_summary"]["physical_truth_evaluated"] is False
        assert not any("truth" in parameter for parameter in row["optimization_summary"]["parameter_rows"])
        assert row["optimization_summary"]["sensitivity"]["rank"] == 2
        assert row["starting_values"]["mass_kg"] == study.starts[row["start_id"]].mass_kg
        assert row["starting_values"]["damping_nm_s_per_rad"] == study.starts[row["start_id"]].damping_nm_s_per_rad
        np.testing.assert_array_equal(row["resolved_optimizer_options"]["x_scale"], [1.6, .499])
        directory = output / "fits" / row["fit_id"]
        assert row["native_result"]["sha256"] == sha256_file(directory / "native_result.pkl")
        native = pickle.loads((directory / "native_result.pkl").read_bytes())
        with np.load(directory / "residuals.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["final"], native.fun)
            np.testing.assert_array_equal(arrays["jacobian"], native.jac)
            assert arrays["jacobian"].shape == (2164, 2)
            assert float(.5 * arrays["final"] @ arrays["final"]) == pytest.approx(row["final_objective"])
        assert (directory / "official" / "report.html").is_file()
        for name in ("train_multisine", "train_prbs"):
            with np.load(directory / f"{name}_inputs.npz", allow_pickle=False) as arrays:
                assert set(arrays.files) == set(TRAIN_ARRAY_NAMES)


def test_completed_output_cannot_be_overwritten(completed_run, training, study, provenance):
    output, _ = completed_run
    sealed_sha = sha256_file(output / "selection.json")
    with pytest.raises(ValueError, match="already exists"):
        run_joint_starts(training, study, MODEL, output, provenance)
    assert sha256_file(output / "selection.json") == sealed_sha


def test_report_failure_keeps_finite_fits_eligible_and_preserves_native_outputs(tmp_path, training, study, provenance, monkeypatch):
    import mujoco_sysid_lab.fit_reporting as reporting

    def failed_report(*args, **kwargs):
        raise RuntimeError("injected report-only failure")

    monkeypatch.setattr(reporting, "save_official_fit", failed_report)
    outcome = run_joint_starts(training, study, MODEL, tmp_path, provenance)
    assert outcome["status"] == "incomplete"
    assert outcome["selection"]["status"] == "selected"
    for row in outcome["fits"]:
        assert row["report_status"] == "failed"
        assert row["selection_eligible"] is True
        assert row["optimization_returned"] is True
        assert (tmp_path / row["native_result"]["path"]).is_file()
        assert row["failure"]["message"] == "injected report-only failure"


def test_optimizer_failure_retains_all_three_attempts_partial_trace_and_actual_flags(tmp_path, training, study, provenance, monkeypatch):
    from mujoco import sysid

    def fail(*, initial_params, residual_fn, **kwargs):
        residual_fn(initial_params.as_vector(), initial_params)
        raise RuntimeError("injected optimization failure")

    monkeypatch.setattr(sysid, "optimize", fail)
    outcome = run_joint_starts(training, study, MODEL, tmp_path, provenance)
    assert outcome["selection"]["status"] == "all_invalid"
    assert len(outcome["fits"]) == 3
    for row in outcome["fits"]:
        assert row["optimization_invoked"] is True and row["optimization_returned"] is False
        assert row["call_counts"]["optimization"]["residual_calls"] == 1
        trace = json.loads((tmp_path / row["objective_trace"]["path"]).read_text())
        assert [event["event"] for event in trace] == ["initial"]
        assert row["selection_eligible"] is False


@pytest.mark.parametrize("payload", [
    {"truth_config": {"mass": 1.2}},
    {"extra": {"truth_values": {"mass_kg": 1.2}}},
    {"dataset_input": {"generation_provenance": {"truth_generator_config": {}}}},
])
def test_joint_provenance_rejects_hidden_values_before_writes(tmp_path, training, study, provenance, payload):
    with pytest.raises(ValueError, match="exclude physical truth"):
        run_joint_starts(training, study, MODEL, tmp_path, dict(provenance, **payload))
    assert list(tmp_path.iterdir()) == []


def test_joint_worker_rejects_other_seed_and_missing_provenance(tmp_path, training, study, provenance):
    with pytest.raises(ValueError, match="seed-101"):
        run_joint_starts(replace(training, seed=102), study, MODEL, tmp_path, provenance)
    with pytest.raises(ValueError, match="missing required"):
        run_joint_starts(training, study, MODEL, tmp_path, {})
    assert not any(tmp_path.iterdir())


def test_joint_worker_rejects_nonfinite_or_object_inputs_before_writes(tmp_path, training, study, provenance):
    for invalid in (np.full((601, 1), np.nan), np.full((601, 1), "bad", dtype=object)):
        sequence = replace(training.sequences[0], measured_qpos=invalid)
        inputs = replace(training, sequences=(sequence, training.sequences[1]))
        with pytest.raises(ValueError, match="finite"):
            run_joint_starts(inputs, study, MODEL, tmp_path, provenance)
    assert not any(tmp_path.iterdir())
