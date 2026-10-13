from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from mujoco_sysid_lab.advanced_data import SCENARIOS, create_scenario
from mujoco_sysid_lab.dataset import TrajectoryRecord
from mujoco_sysid_lab.evaluation import aggregate_metrics, evaluate_trajectory, relative_improvement
from mujoco_sysid_lab.experiment_config import load_ground_truth, load_study_config
from mujoco_sysid_lab.full_evaluation import (
    choose_lag, evaluate_model, phase_lag_diagnostics, phase_lag_for_signals, predict_model,
)
from mujoco_sysid_lab.simulation import compile_model, simulate


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model/pendulum.xml"


@pytest.fixture(scope="module")
def study():
    return load_study_config(ROOT / "configs/evidence_v1.json")


@pytest.fixture(scope="module")
def truth(study):
    return load_ground_truth(study.truth_config_path, study)


@pytest.fixture(scope="module")
def scenarios(study, truth):
    return {name: create_scenario(study, truth, MODEL, 101, name) for name in SCENARIOS}


@pytest.fixture(scope="module")
def matched_metrics(study, truth, scenarios):
    # This near-truth fixture is deliberately not a retained estimator result.
    physical = replace(truth.model, mass_kg=1.201, damping_nm_s_per_rad=.1502)
    return {name: evaluate_model(scenarios["matched"], candidate, MODEL, study)
            for name, candidate in (("nominal", study.nominal), ("identified_fixture", physical), ("oracle", truth.model))}


@pytest.mark.parametrize("label", ("nominal", "identified_fixture", "oracle"))
def test_every_baseline_uses_the_same_metric_contract(study, truth, scenarios, matched_metrics, label):
    result = matched_metrics[label]
    physical = replace(study.nominal, **result["physical"])
    predictions = predict_model(scenarios["matched"], physical, MODEL, study)
    for name, prediction in predictions.items():
        arrays = scenarios["matched"].records[name].arrays
        expected = evaluate_trajectory(
            arrays["timestamps"], arrays["measured_qpos"], arrays["measured_qvel"],
            prediction.qpos, prediction.qvel, .002, .02, study.evaluation_window_for(name))
        row = result["trajectory_metrics"][name]
        assert all(row[key] == value for key, value in expected.items())
        assert row["n_samples"] == 3941
        assert row["residual_count"] == 7882
    for split in ("train", "heldout"):
        rows = [row for row in result["trajectory_metrics"].values() if row["split"] == split]
        expected = aggregate_metrics(rows)
        assert all(result[split][key] == value for key, value in expected.items())
    json.dumps(result, allow_nan=False)


def test_heldout_improvement_and_oracle_noise_are_distinct_from_zero(matched_metrics):
    nominal, fitted, oracle = (matched_metrics[key] for key in ("nominal", "identified_fixture", "oracle"))
    for split in ("train", "heldout"):
        improvement = relative_improvement(fitted[split], nominal[split])
        assert improvement["qpos_rmse"] > .9 and improvement["qvel_rmse"] > .9
        assert 0 < oracle[split]["normalized_rmse"] < 1.05
        assert oracle[split]["objective"] > 0
        for key in ("qpos_rmse", "qvel_rmse", "normalized_rmse", "objective"):
            assert oracle[split][key] == oracle["generator_aware_noise_floor"][split][key]


@pytest.mark.parametrize("scenario_name", SCENARIOS)
def test_objective_matches_direct_noise_scaled_squared_error(study, truth, scenarios, scenario_name):
    data = scenarios[scenario_name]
    result = evaluate_model(data, truth.model, MODEL, study, delay_s=data.delay_s)
    predictions = predict_model(data, truth.model, MODEL, study, delay_s=data.delay_s)
    direct = {"train": 0., "heldout": 0.}
    for name, record in data.records.items():
        arrays = record.arrays
        times = arrays["timestamps"]
        lower, upper = study.evaluation_window_for(name)
        mask = (times >= lower - 1e-12) & (times <= upper + 1e-12)
        qerror = (arrays["measured_qpos"][mask] - predictions[name].qpos[mask]) / .002
        verror = (arrays["measured_qvel"][mask] - predictions[name].qvel[mask]) / .02
        cost = float((np.sum(qerror ** 2) + np.sum(verror ** 2)) / 2)
        assert result["trajectory_metrics"][name]["objective"] == pytest.approx(cost, rel=1e-14)
        direct[record.metadata["split"]] += cost
    for split in direct:
        assert result[split]["objective"] == pytest.approx(direct[split], rel=1e-14)


def test_weak_horizon_equal_trajectory_metrics_and_sample_objective(study, truth, scenarios):
    result = evaluate_model(scenarios["weak"], truth.model, MODEL, study)
    assert result["train"]["trajectory_count"] == 1
    row = result["train"]["trajectory_metrics"][0]
    assert row["trajectory_name"] == "weak_step"
    assert row["n_samples"] == 7941 and row["evaluation_window_s"] == [.06, 15.94]
    assert result["heldout"]["n_samples"] == 7882
    assert result["train"]["objective"] == row["n_samples"] * row["normalized_rmse"] ** 2


def test_delay_predictions_have_fresh_explicit_full_history(study, truth, scenarios):
    data = scenarios["delayed"]
    for delay in (0., .024, .06):
        predictions = predict_model(data, truth.model, MODEL, study, delay_s=delay)
        for name, prediction in predictions.items():
            arrays = data.records[name].arrays
            assert prediction.qpos.shape == prediction.qvel.shape == (4001, 1)
            assert prediction.prehistory_timestamps[0] == -.06
            assert prediction.prehistory_timestamps[-1] == 0
            assert np.all(prediction.prehistory_qpos == 0) and np.all(prediction.prehistory_qvel == 0)
            for sensor in ("qpos", "qvel"):
                np.testing.assert_array_equal(getattr(prediction, "physical_" + sensor), arrays["clean_" + sensor])
                assert not getattr(prediction, sensor).flags.writeable
                if delay == .024:
                    np.testing.assert_array_equal(getattr(prediction, sensor), arrays["observed_clean_" + sensor])
                if delay == 0:
                    np.testing.assert_array_equal(getattr(prediction, sensor), getattr(prediction, "physical_" + sensor))
    with pytest.raises(TypeError):
        predictions["heldout_chirp"] = predictions["heldout_pulse"]


def test_delay_corrected_oracle_equals_noise_reference_and_reduces_phase(study, truth, scenarios):
    data = scenarios["delayed"]
    zero = evaluate_model(data, truth.model, MODEL, study, delay_s=0.)
    corrected = evaluate_model(data, truth.model, MODEL, study, delay_s=.024)
    for split in ("train", "heldout"):
        assert corrected[split]["normalized_rmse"] < zero[split]["normalized_rmse"]
        assert corrected[split]["objective"] == corrected["generator_aware_noise_floor"][split]["objective"]
    before = phase_lag_diagnostics(data, truth.model, MODEL, study, delay_s=0.)
    after = phase_lag_diagnostics(data, truth.model, MODEL, study, delay_s=.024)
    for split in ("train", "heldout"):
        assert before[split]["median_absolute_lag_s"] == pytest.approx(.024)
        assert after[split]["median_absolute_lag_s"] == 0
    json.dumps(before, allow_nan=False)
    json.dumps(after, allow_nan=False)


def test_mismatch_oracle_omits_force_while_generator_reference_includes_it(study, truth, scenarios, monkeypatch):
    import mujoco_sysid_lab.advanced_data as generation
    monkeypatch.setattr(generation, "simulate_mismatch", lambda *args, **kwargs: pytest.fail("Fitting evaluator called truth-only force"))
    data = scenarios["mismatch"]
    predictions = predict_model(data, truth.model, MODEL, study)
    matched = predict_model(scenarios["matched"], truth.model, MODEL, study)
    for name, prediction in predictions.items():
        for sensor in ("qpos", "qvel"):
            np.testing.assert_array_equal(getattr(prediction, sensor), getattr(matched[name], sensor))
            assert not np.array_equal(getattr(prediction, sensor), data.records[name].arrays["clean_" + sensor])
    result = evaluate_model(data, truth.model, MODEL, study)
    for split in ("train", "heldout"):
        assert result[split]["normalized_rmse"] > result["generator_aware_noise_floor"][split]["normalized_rmse"]
        assert result["latent_clean_diagnostics"][split]["qpos_rmse"] > 0
    assert result["fitting_family_has_omitted_force"] is False


def test_changed_heldout_controls_and_measurements_cannot_affect_training_scores(study, truth, scenarios):
    data = scenarios["matched"]
    original = evaluate_model(data, truth.model, MODEL, study)
    records = dict(data.records)
    for name in ("heldout_chirp", "heldout_pulse"):
        record = records[name]
        arrays = dict(record.arrays)
        arrays.update(controls=-arrays["controls"], measured_qpos=arrays["measured_qpos"] + 10,
                      measured_qvel=arrays["measured_qvel"] - 10)
        records[name] = TrajectoryRecord(MappingProxyType(arrays), record.metadata)
    modified = replace(data, records=MappingProxyType(records))
    altered = evaluate_model(modified, truth.model, MODEL, study)
    assert original["train"] == altered["train"]
    assert original["generator_aware_noise_floor"]["train"] == altered["generator_aware_noise_floor"]["train"]
    assert original["latent_clean_diagnostics"]["train"] == altered["latent_clean_diagnostics"]["train"]
    assert original["heldout"] != altered["heldout"]


def test_predictions_ignore_truth_clean_noise_and_metadata_physics(study, truth, scenarios):
    data = scenarios["matched"]
    expected = predict_model(data, study.nominal, MODEL, study, delay_s=.03)
    records = {}
    for name, record in data.records.items():
        arrays = dict(record.arrays)
        for key in ("clean_qpos", "clean_qvel", "observed_clean_qpos", "observed_clean_qvel", "innovation_qpos", "innovation_qvel", "extra_torque_nm"):
            arrays[key] = np.full_like(arrays[key], 1e6)
        records[name] = TrajectoryRecord(MappingProxyType(arrays), record.metadata)
    modified = replace(data, records=records, metadata={"truth_generator_config": {"model": "unused"}})
    actual = predict_model(modified, study.nominal, MODEL, study, delay_s=.03)
    for name in expected:
        np.testing.assert_array_equal(actual[name].qpos, expected[name].qpos)
        np.testing.assert_array_equal(actual[name].qvel, expected[name].qvel)


def _signal_fixture(delay=0.):
    times = np.arange(4001) * .002
    def signal(t):
        return np.sin(6 * t) + .3 * np.sin(17 * t) + .1 * np.cos(32 * t)
    return times, signal(times)[:, None], signal(times - delay)[:, None]


@pytest.mark.parametrize("delay", (-.024, 0., .024, -.06, .06))
def test_phase_lag_sign_fixed_support_and_sensor_grid(study, delay):
    times, predicted, measured = _signal_fixture(delay)
    result = phase_lag_for_signals(times, predicted, measured, .002, study)
    assert result["valid"] is True
    assert result["signed_lag_s"] == pytest.approx(delay)
    assert result["absolute_lag_s"] == pytest.approx(abs(delay))
    assert result["peak_correlation"] == pytest.approx(1., abs=1e-12)
    assert result["evaluation_window_s"] == [.12, 7.88]
    assert result["n_samples"] == 3881
    assert result["lag_grid_s"] == pytest.approx(np.arange(-30, 31) * .002)


def test_phase_ties_use_global_max_absolute_then_negative_without_iterative_drift():
    assert choose_lag(np.array([-.002, .002]), np.ones(2), 1e-12) == 0
    assert choose_lag(np.array([-.002, 0., .002]), np.ones(3), 1e-12) == 1
    assert choose_lag(np.array([.04, .02, 0.]), np.array([1., 1. - .8e-12, 1. - 1.5e-12]), 1e-12) == 1


@pytest.mark.parametrize("which", ("prediction", "measurement"))
def test_phase_below_noise_signals_return_null_reason(study, which):
    times, predicted, measured = _signal_fixture()
    if which == "prediction":
        predicted *= .00001
    else:
        measured[:] = 0
    result = phase_lag_for_signals(times, predicted, measured, .002, study)
    assert result["valid"] is False and result["signed_lag_s"] is None
    assert result["absolute_lag_s"] is None and result["peak_correlation"] is None
    assert "standard deviation" in result["unavailable_reason"]
    json.dumps(result, allow_nan=False)


def test_low_correlation_returns_null_with_actual_peak(study):
    times = np.arange(4001) * .002
    result = phase_lag_for_signals(times, np.sin(2 * times)[:, None], np.sin(50 * times)[:, None], .002, study)
    assert result["valid"] is False and result["signed_lag_s"] is None
    assert result["peak_correlation"] < .2
    assert "minimum" in result["unavailable_reason"]


@pytest.mark.parametrize("delay", (-.001, .06001, np.nan, np.inf, True))
def test_invalid_delay_fails_before_prediction(study, scenarios, delay):
    with pytest.raises(ValueError, match="delay"):
        evaluate_model(scenarios["matched"], study.nominal, MODEL, study, delay_s=delay)


@pytest.mark.parametrize("field,value", (("motor_gain_nm_per_unit", 2.), ("timestep_s", .001), ("initial_qpos_rad", .1), ("mass_kg", 2.1), ("damping_nm_s_per_rad", .6)))
def test_candidate_cannot_change_registered_nuisance_or_bounds(study, scenarios, field, value):
    with pytest.raises(ValueError, match="fixed|bounds"):
        evaluate_model(scenarios["matched"], replace(study.nominal, **{field: value}), MODEL, study)


def test_nonfinite_measurement_and_wrong_initial_state_rejected(study, scenarios):
    data = scenarios["matched"]
    name = "train_multisine"
    for key, value in (("measured_qpos", np.full((4001, 1), np.nan)), ("initial_state", np.array([0., .1, 0.]))):
        records = dict(data.records)
        record = records[name]
        records[name] = TrajectoryRecord({**record.arrays, key: value}, record.metadata)
        with pytest.raises(ValueError, match="finite|initial"):
            evaluate_model(replace(data, records=records), study.nominal, MODEL, study)


def test_invalid_phase_input_or_unsafe_window_rejected(study):
    times, predicted, measured = _signal_fixture()
    with pytest.raises(ValueError, match="support"):
        phase_lag_for_signals(times, predicted, measured, .002,
                              replace(study, phase_lag={**study.phase_lag, "window_margin_s": .03}))
    with pytest.raises(ValueError, match="shape"):
        phase_lag_for_signals(times, predicted[:, 0], measured, .002, study)
    with pytest.raises(ValueError, match="finite"):
        phase_lag_for_signals(times, np.full_like(predicted, np.inf), measured, .002, study)
    with pytest.raises(ValueError, match="grid"):
        phase_lag_for_signals(times + .001, predicted, measured, .002, study)


def test_latent_secondary_metrics_do_not_apply_delay(study, truth, scenarios):
    result = evaluate_model(scenarios["delayed"], truth.model, MODEL, study, delay_s=.024)
    for split in ("train", "heldout"):
        diagnostic = result["latent_clean_diagnostics"][split]
        assert diagnostic["qpos_rmse"] == diagnostic["qvel_rmse"] == diagnostic["objective"] == 0
        assert result[split]["objective"] > 0
