import json

import numpy as np
import pytest

from mujoco_sysid_lab.evaluation import (
    aggregate_metrics,
    evaluate_trajectory,
    relative_improvement,
)


def evaluate_error(qpos, qvel, *, times=None, sigma_q=1.0, sigma_v=1.0, window=None):
    if times is None:
        times = np.arange(len(qpos), dtype=np.float64) * 0.01
    return evaluate_trajectory(
        times, np.asarray(qpos, dtype=np.float64)[:, None],
        np.asarray(qvel, dtype=np.float64)[:, None],
        np.zeros((len(times), 1)), np.zeros((len(times), 1)),
        sigma_q, sigma_v, window_s=window,
    )


def test_known_errors_have_physical_units_and_noise_weighting():
    result = evaluate_error(np.full(21, 2.0), np.full(21, -3.0), sigma_q=0.2, sigma_v=0.5)
    assert result["qpos_rmse"] == pytest.approx(2.0)
    assert result["qvel_rmse"] == pytest.approx(3.0)
    assert result["qpos_p95_absolute_error"] == pytest.approx(2.0)
    assert result["qvel_p95_absolute_error"] == pytest.approx(3.0)
    assert result["normalized_rmse"] == pytest.approx(np.sqrt((10**2 + 6**2) / 2))
    assert result["n_samples"] == 9
    assert result["residual_sign"] == "measured_minus_predicted"
    json.dumps(result, allow_nan=False)


def test_zero_error_is_zero_for_every_model_label():
    times = np.arange(101) * 0.002
    signal_q = np.sin(times)[:, None]
    signal_v = np.cos(times)[:, None]
    by_label = {
        label: evaluate_trajectory(times, signal_q, signal_v, signal_q, signal_v, 0.002, 0.020)
        for label in ("nominal", "identified", "oracle")
    }
    assert by_label["nominal"] == by_label["identified"] == by_label["oracle"]
    for key in ("qpos_rmse", "qvel_rmse", "normalized_rmse", "qpos_p95_absolute_error", "qvel_p95_absolute_error"):
        assert by_label["oracle"][key] == 0


def test_oracle_against_deterministic_gaussian_noise_approaches_noise_scales():
    times = np.arange(20001) * 0.002
    rng = np.random.Generator(np.random.PCG64(101))
    qpos = rng.normal(0, 0.002, len(times))
    qvel = rng.normal(0, 0.020, len(times))
    result = evaluate_error(qpos, qvel, times=times, sigma_q=0.002, sigma_v=0.020)
    assert result["qpos_rmse"] == pytest.approx(0.002, rel=0.025)
    assert result["qvel_rmse"] == pytest.approx(0.020, rel=0.025)
    assert result["normalized_rmse"] == pytest.approx(1.0, rel=0.025)


def test_fixed_window_excludes_extreme_boundaries_and_is_prediction_independent():
    qpos = np.ones(21)
    qpos[:6] = qpos[15:] = 1000
    result = evaluate_error(qpos, np.zeros(21))
    other = evaluate_error(qpos * 3, np.zeros(21))
    assert result["qpos_rmse"] == 1.0
    assert result["qpos_p95_absolute_error"] == 1.0
    assert result["sampled_window_s"] == [0.06, 0.14]
    assert result["n_samples"] == other["n_samples"] == 9
    assert result["evaluation_window_s"] == other["evaluation_window_s"]


def test_fixed_window_preserves_grid_endpoints_with_arithmetic_roundoff():
    times = np.arange(101) * 0.002
    result = evaluate_error(np.ones(101), np.ones(101), times=times, window=(0.06 + 1e-15, 0.14 - 1e-15))
    assert result["n_samples"] == 41
    assert result["sampled_window_s"] == [0.06, 0.14]


def test_p95_uses_absolute_samples_and_linear_interpolation():
    result = evaluate_error([-10, 0, 0, 0, 2], [0, 0, 0, 0, 0], window=(0, 0.04))
    assert result["qpos_p95_absolute_error"] == pytest.approx(8.4)
    assert result["percentile_method"] == "linear"


def test_split_rmse_weights_trajectories_equally_and_retains_p95():
    short = evaluate_error(np.full(21, 1.0), np.full(21, 2.0))
    long = evaluate_error(np.full(101, 3.0), np.full(101, 4.0))
    short["trajectory_id"] = "short"
    long["trajectory_id"] = "long"
    result = aggregate_metrics([short, long])
    assert result["qpos_rmse"] == pytest.approx(np.sqrt((1**2 + 3**2) / 2))
    assert result["qvel_rmse"] == pytest.approx(np.sqrt((2**2 + 4**2) / 2))
    assert result["normalized_rmse"] == pytest.approx(np.sqrt((1**2 + 2**2 + 3**2 + 4**2) / 4))
    assert result["trajectory_count"] == 2
    assert result["qpos_p95_absolute_error"] is None
    assert result["qvel_p95_absolute_error"] is None
    assert result["p95_unavailable_reason"]
    assert result["trajectory_metrics"] == [short, long]
    json.dumps(result, allow_nan=False)


def test_relative_improvement_fraction_sign_and_undefined_zero_baseline():
    nominal = evaluate_error(np.full(21, 4.0), np.full(21, 2.0))
    identified = evaluate_error(np.full(21, 2.0), np.full(21, 3.0))
    result = relative_improvement(identified, nominal)
    assert result["qpos_rmse"] == 0.5
    assert result["qvel_rmse"] == -0.5
    assert result["qpos_rmse_reason"] is None
    zero = evaluate_error(np.zeros(21), np.zeros(21))
    undefined = relative_improvement(zero, zero)
    for key in ("qpos_rmse", "qvel_rmse", "normalized_rmse"):
        assert undefined[key] is None
        assert "undefined" in undefined[f"{key}_reason"]
    json.dumps(undefined, allow_nan=False)


@pytest.mark.parametrize("times,error", [
    (np.array([0]), "at least two samples"),
    (np.array([[0, 0.01]]), "shape"),
    (np.array([0, 0]), "strictly increasing"),
    (np.array([0.01, 0]), "strictly increasing"),
    (np.array([0, 0.01, 0.03]), "uniformly spaced"),
    (np.array([0, np.nan]), "finite"),
    (np.array([0, np.inf]), "finite"),
    (np.array(["0", "0.01"]), "real numbers"),
])
def test_invalid_timestamp_grids_are_rejected(times, error):
    values = np.zeros((np.asarray(times).size, 1))
    with pytest.raises(ValueError, match=error):
        evaluate_trajectory(times, values, values, values, values, 1, 1, window_s=(0, 0.01))


@pytest.mark.parametrize("values,error", [
    (np.zeros(21), "common shape"),
    (np.zeros((21, 2)), "common shape"),
    (np.zeros((20, 1)), "common shape"),
    (np.full((21, 1), np.nan), "finite"),
    (np.full((21, 1), np.inf), "finite"),
    (np.full((21, 1), 1j), "real numbers"),
])
def test_sensor_array_contract_rejects_invalid_shapes_and_values(values, error):
    valid = np.zeros((21, 1))
    with pytest.raises(ValueError, match=error):
        evaluate_trajectory(np.arange(21) * 0.01, values, valid, valid, valid, 1, 1)


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), float("inf"), True, [1]])
def test_noise_scale_must_be_a_positive_finite_scalar(sigma):
    with pytest.raises(ValueError, match="sigma_qpos"):
        evaluate_error(np.zeros(21), np.zeros(21), sigma_q=sigma)


@pytest.mark.parametrize("window,error", [
    ((0.14, 0.06), "ordered"),
    ((-0.01, 0.1), "support"),
    ((0.1, 0.21), "support"),
    ((0.061, 0.069), "no timestamp"),
    ((0, float("nan")), "finite"),
    ((0,), "exactly two"),
])
def test_invalid_windows_are_rejected(window, error):
    with pytest.raises(ValueError, match=error):
        evaluate_error(np.zeros(21), np.zeros(21), window=window)


def test_default_window_rejects_too_short_trajectory():
    with pytest.raises(ValueError, match="ordered"):
        evaluate_error(np.zeros(5), np.zeros(5))


def test_finite_inputs_that_overflow_residuals_are_rejected_without_nan_json():
    times = np.arange(21) * 0.01
    positive = np.full((21, 1), 1e308)
    negative = -positive
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_trajectory(times, positive, positive, negative, negative, 1, 1)


@pytest.mark.parametrize("rows,error", [
    ([], "at least one"),
    ([{}], "missing"),
    ([None], "mappings"),
])
def test_invalid_aggregation_rows_are_rejected(rows, error):
    with pytest.raises(ValueError, match=error):
        aggregate_metrics(rows)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_aggregation_rejects_invalid_metric_values(value):
    row = evaluate_error(np.zeros(21), np.zeros(21))
    row["qpos_rmse"] = value
    with pytest.raises(ValueError):
        aggregate_metrics([row])


def test_aggregation_rejects_nonfinite_metadata():
    row = evaluate_error(np.zeros(21), np.zeros(21))
    row["diagnostic"] = float("nan")
    with pytest.raises(ValueError, match="strict JSON"):
        aggregate_metrics([row])


def test_relative_improvement_validates_metrics():
    row = evaluate_error(np.zeros(21), np.zeros(21))
    with pytest.raises(ValueError, match="nonnegative"):
        relative_improvement({**row, "qpos_rmse": -1}, row)
