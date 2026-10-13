"""Post-selection prediction metrics and registered phase diagnostics.

These functions intentionally accept the complete truth-bearing scenario and
must run only after training-only selection is sealed. They contain no fit,
state reset from observations, or omitted-force path. The fitting-family
oracle and the generator-aware noise reference are different under mismatch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import json
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from mujoco_sysid_lab.advanced_data import ScenarioData, apply_sensor_delay
from mujoco_sysid_lab.config import ModelConfig
from mujoco_sysid_lab.evaluation import aggregate_metrics, evaluate_trajectory
from mujoco_sysid_lab.experiment_config import StudyConfig, config_to_dict
from mujoco_sysid_lab.simulation import compile_model, simulate


@dataclass(frozen=True)
class PredictionRecord:
    """Full-horizon fresh fitting-model rollout and delayed sensor predictions.

``physical_qpos/qvel`` are instantaneous latent states; ``qpos/qvel`` are
observations after the candidate shared delay. Explicit prehistory is generated
from this model with zero control, never copied from a truth record.
    """

    timestamps: np.ndarray
    controls: np.ndarray
    physical_qpos: np.ndarray
    physical_qvel: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    prehistory_timestamps: np.ndarray
    prehistory_qpos: np.ndarray
    prehistory_qvel: np.ndarray


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite real scalar")
    return float(value)


def _signal(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.shape != (size, 1) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite numeric values with shape (N,1)")
    return np.asarray(array, dtype=np.float64)


def _times(value: Any, timestep: float) -> np.ndarray:
    times = np.asarray(value)
    if (times.dtype.kind not in "fiu" or times.ndim != 1 or times.size < 2
            or not np.isfinite(times).all() or times[0] != 0
            or not np.allclose(times, np.arange(times.size) * timestep, rtol=0, atol=1e-12)):
        raise ValueError("timestamps must span the increasing registered grid from zero")
    return np.asarray(times, dtype=np.float64)


def _validate_candidate(physical: ModelConfig, study: StudyConfig, delay_s: float) -> float:
    if not isinstance(physical, ModelConfig):
        raise ValueError("physical must be validated ModelConfig mechanics")
    for field in fields(ModelConfig):
        if field.name not in ("mass_kg", "damping_nm_s_per_rad"):
            if getattr(physical, field.name) != getattr(study.nominal, field.name):
                raise ValueError(f"fitting-family evaluation must preserve fixed {field.name}")
    if not study.mass_bounds_kg.contains(physical.mass_kg):
        raise ValueError("evaluation mass is outside registered bounds")
    if not study.damping_bounds_nm_s_per_rad.contains(physical.damping_nm_s_per_rad):
        raise ValueError("evaluation damping is outside registered bounds")
    delay = _finite_scalar(delay_s, "delay_s")
    if not study.delay_bounds_s.contains(delay) or delay > study.prehistory_s:
        raise ValueError("delay_s is outside registered bounds or explicit history support")
    return delay


def _record_names(scenario: ScenarioData, study: StudyConfig) -> tuple[str, ...]:
    if not isinstance(scenario, ScenarioData) or scenario.scenario not in ("matched", "delayed", "weak", "mismatch"):
        raise ValueError("evaluation requires a supported ScenarioData")
    training = ("weak_step",) if scenario.scenario == "weak" else ("train_multisine", "train_prbs")
    names = (*training, "heldout_chirp", "heldout_pulse")
    if tuple(scenario.training_ids) != training or set(scenario.records) != set(names):
        raise ValueError("scenario must preserve complete-trajectory train/held-out identities")
    for name in names:
        record = scenario.records[name]
        definition = study.trajectories[name]
        if (record.metadata.get("split") != definition.split
                or record.metadata.get("trajectory_id") != definition.trajectory_id
                or record.metadata.get("duration_s") != definition.duration_s
                or record.metadata.get("timestep_s") != study.timestep_s):
            raise ValueError("scenario trajectory identity/timing differs from registered configuration")
    return names


def predict_model(scenario: ScenarioData, physical: ModelConfig, model_path: Path,
                  study: StudyConfig, delay_s: float = 0.) -> Mapping[str, PredictionRecord]:
    """Roll out a fitting-family model independently on every complete control.

The returned full-signal arrays support plotting without resetting from noisy
states. Actual equilibrium prehistory enables even the maximum candidate lag
at t=0. No target clean states, generator force, or oracle parameter is used to
produce predictions. The truth-bearing input is for post-selection use only.
    """
    delay = _validate_candidate(physical, study, delay_s)
    names = _record_names(scenario, study)
    model = compile_model(Path(model_path), physical)
    history_count = round(study.prehistory_s / study.timestep_s)
    if history_count < 1 or not np.isclose(history_count * study.timestep_s, study.prehistory_s, rtol=0, atol=1e-12):
        raise ValueError("prehistory must be a positive integer number of sensor intervals")
    history_grid = np.arange(history_count + 1, dtype=np.float64) * study.timestep_s
    history = simulate(model, history_grid, np.zeros((history_count + 1, 1)),
                       initial_qpos=physical.initial_qpos_rad, initial_qvel=physical.initial_qvel_rad_s)
    negative_times = history_grid - study.prehistory_s
    negative_times[-1] = 0.
    negative_times.flags.writeable = False
    predictions = {}
    for name in names:
        record = scenario.records[name]
        times = _times(record.arrays["timestamps"], study.timestep_s)
        if not np.isclose(times[-1], study.trajectories[name].duration_s, rtol=0, atol=1e-12):
            raise ValueError("evaluation requires the complete registered trajectory horizon")
        state = np.asarray(record.arrays["initial_state"])
        if not np.array_equal(state, [0., physical.initial_qpos_rad, physical.initial_qvel_rad_s]):
            raise ValueError("evaluation requires the known explicit FULLPHYSICS initial state")
        controls = _signal(record.arrays["controls"], times.size, "controls")
        latent = simulate(model, times, controls, initial_qpos=physical.initial_qpos_rad,
                          initial_qvel=physical.initial_qvel_rad_s)
        observed = {
            sensor: apply_sensor_delay(times, getattr(latent, sensor), delay, negative_times,
                                       getattr(history, sensor)) for sensor in ("qpos", "qvel")
        }
        predictions[name] = PredictionRecord(
            latent.timestamps, latent.controls, latent.qpos, latent.qvel,
            observed["qpos"], observed["qvel"], negative_times, history.qpos, history.qvel,
        )
    return MappingProxyType(predictions)


def _metric_row(name: str, scenario: ScenarioData, study: StudyConfig,
                predicted_qpos: np.ndarray, predicted_qvel: np.ndarray, *,
                target_prefix: str = "measured_") -> dict[str, Any]:
    record = scenario.records[name]
    metrics = evaluate_trajectory(
        record.arrays["timestamps"], record.arrays[target_prefix + "qpos"],
        record.arrays[target_prefix + "qvel"], predicted_qpos, predicted_qvel,
        study.sensor_sigmas["qpos"], study.sensor_sigmas["qvel"],
        study.evaluation_window_for(name),
    )
    # Two channels: normalized_rmse**2 = sum(r**2)/(2*N), so the
    # official linear least-squares objective sum(r**2)/2 is N*NRMSE**2.
    return {"trajectory_name": name, "split": study.trajectories[name].split, **metrics,
            "objective": metrics["n_samples"] * metrics["normalized_rmse"] ** 2,
            "residual_count": 2 * metrics["n_samples"]}


def _splits(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for split in ("train", "heldout"):
        selected = [row for row in rows.values() if row["split"] == split]
        aggregate = aggregate_metrics(selected)
        aggregate.update(objective=float(sum(row["objective"] for row in selected)),
                         n_samples=sum(row["n_samples"] for row in selected),
                         residual_count=sum(row["residual_count"] for row in selected),
                         objective_definition="sum(noise_scaled_residual**2)/2; equal-trajectory RMSE is a separate aggregation")
        result[split] = aggregate
    return result


def evaluate_model(scenario: ScenarioData, physical: ModelConfig, model_path: Path,
                   study: StudyConfig, delay_s: float = 0.) -> dict[str, Any]:
    """Apply identical primary metrics to any nominal/identified/oracle model.

Primary scores compare candidate sensor predictions with measured data on the
fixed registered window. Generator-aware noise scores compare the measurements
with stored clean *observations*, including the generator's delay/omitted force.
Secondary latent scores compare instantaneous physical states, before delay.
    """
    predictions = predict_model(scenario, physical, model_path, study, delay_s)
    rows, noise_rows, latent_rows = {}, {}, {}
    for name, prediction in predictions.items():
        rows[name] = _metric_row(name, scenario, study, prediction.qpos, prediction.qvel)
        arrays = scenario.records[name].arrays
        noise_rows[name] = _metric_row(name, scenario, study, arrays["observed_clean_qpos"], arrays["observed_clean_qvel"])
        latent_rows[name] = _metric_row(name, scenario, study, prediction.physical_qpos,
                                        prediction.physical_qvel, target_prefix="clean_")
    result = {
        "schema_version": "mujoco-sysid-lab.prediction-evaluation.v1",
        "scenario": scenario.scenario, "seed": scenario.seed,
        "physical": config_to_dict(physical), "delay_s": float(delay_s),
        "fitting_family_has_omitted_force": False,
        **_splits(rows), "trajectory_metrics": rows,
        "generator_aware_noise_floor": {
            **_splits(noise_rows), "trajectory_metrics": noise_rows,
            "reference": "stored generator-aware clean observations after true measurement delay; includes omitted force when present",
            "finite_sample_lower_bound": False,
        },
        "latent_clean_diagnostics": {
            **_splits(latent_rows), "trajectory_metrics": latent_rows,
            "role": "secondary instantaneous physical-state diagnostic; excluded from optimization and start selection",
        },
        "prediction_semantics": "fresh full-horizon CPU rollout from known initial state; y(t)=physical(t-delay); explicit zero-control prehistory; no extrapolation",
        "oracle_semantics": "true physical parameters and true delay in the fitting family; omitted force is absent even for the oracle",
    }
    json.dumps(result, allow_nan=False)
    return result


def choose_lag(lags: np.ndarray, correlations: np.ndarray, tolerance: float) -> int:
    """Select global correlation maximum, then frozen absolute/sign tie order."""
    lags = np.asarray(lags, dtype=np.float64)
    correlations = np.asarray(correlations, dtype=np.float64)
    tolerance = _finite_scalar(tolerance, "tie_tolerance")
    if (lags.ndim != 1 or lags.size == 0 or correlations.shape != lags.shape
            or not np.isfinite(lags).all() or not np.isfinite(correlations).all() or tolerance < 0):
        raise ValueError("lag selection requires finite equal-length vectors and nonnegative tolerance")
    maximum = float(np.max(correlations))
    ties = np.flatnonzero(maximum - correlations <= tolerance)
    return int(min(ties, key=lambda index: (abs(float(lags[index])), float(lags[index]))))


def phase_lag_for_signals(times: np.ndarray, predicted: np.ndarray, measured: np.ndarray,
                          sensor_sigma: float, study: StudyConfig) -> dict[str, Any]:
    """Demeaned correlation of p(t) and y(t+lag) on one fixed safe window.

Positive reported lag means measurements lag the candidate observation.
The grid, validity thresholds, and tie policy are frozen in the study config.
This diagnostic does not estimate a model or alter its residual sample set.
    """
    timestamps = _times(times, study.timestep_s)
    prediction = _signal(predicted, timestamps.size, "phase prediction")[:, 0]
    observation = _signal(measured, timestamps.size, "phase measurement")[:, 0]
    sigma = _finite_scalar(sensor_sigma, "sensor_sigma")
    if sigma <= 0:
        raise ValueError("sensor_sigma must be positive")
    rules = study.phase_lag
    margin = float(rules["window_margin_s"])
    window = (margin, float(timestamps[-1]) - margin)
    mask = (timestamps >= window[0] - 1e-12) & (timestamps <= window[1] + 1e-12)
    if window[1] <= window[0] or np.count_nonzero(mask) < 2:
        raise ValueError("phase diagnostic fixed window requires at least two samples")
    grid = np.arange(round(rules["lag_min_s"] / rules["lag_step_s"]),
                     round(rules["lag_max_s"] / rules["lag_step_s"]) + 1) * rules["lag_step_s"]
    base = timestamps[mask]
    candidate = prediction[mask]
    centered_candidate = candidate - np.mean(candidate)
    candidate_std = float(np.std(candidate))
    default = {"signed_lag_s": None, "absolute_lag_s": None, "peak_correlation": None,
               "valid": False, "unavailable_reason": None, "prediction_std": candidate_std,
               "measurement_std": None, "sensor_noise_sigma": sigma,
               "evaluation_window_s": list(window), "n_samples": int(base.size),
               "lag_grid_s": [float(lag) for lag in grid],
               "positive_sign": rules["positive_sign"], "tie_rule": rules["tie_rule"]}
    # Every lag uses the same base samples. Explicit range checks precede
    # np.interp because its default endpoint behavior would silently pad.
    if base[0] + grid[0] < timestamps[0] or base[-1] + grid[-1] > timestamps[-1]:
        raise ValueError("phase lag queries exceed common measured support; no extrapolation")
    if not np.isfinite(candidate_std):
        raise ValueError("phase diagnostic produced non-finite standard deviation")
    if candidate_std < sigma:
        default["unavailable_reason"] = "Prediction signal standard deviation is below sensor noise sigma."
        return default
    correlations, deviations = [], []
    candidate_norm = float(np.linalg.norm(centered_candidate))
    for lag in grid:
        shifted = np.interp(base + lag, timestamps, observation)
        std = float(np.std(shifted))
        centered = shifted - np.mean(shifted)
        denominator = candidate_norm * float(np.linalg.norm(centered))
        # A constant measurement has no normalized correlation. Mark it as
        # the worst finite candidate, then use its std to return null/reason.
        correlation = float(np.dot(centered_candidate, centered) / denominator) if denominator else -1.
        if not np.isfinite([std, correlation]).all():
            raise ValueError("phase diagnostic produced NaN/Inf")
        deviations.append(std)
        correlations.append(float(np.clip(correlation, -1., 1.)))
    chosen = choose_lag(grid, np.asarray(correlations), rules["tie_tolerance"])
    default["measurement_std"] = deviations[chosen]
    if deviations[chosen] < sigma:
        default["unavailable_reason"] = "Measurement signal standard deviation is below sensor noise sigma."
        return default
    if correlations[chosen] < rules["minimum_peak_correlation"]:
        default["peak_correlation"] = correlations[chosen]
        default["unavailable_reason"] = "Peak correlation is below the registered minimum."
        return default
    lag = float(grid[chosen])
    default.update(signed_lag_s=lag, absolute_lag_s=abs(lag),
                   peak_correlation=correlations[chosen], valid=True)
    return default


def phase_lag_diagnostics(scenario: ScenarioData, physical: ModelConfig, model_path: Path,
                          study: StudyConfig, delay_s: float = 0.) -> dict[str, Any]:
    """Return per-trajectory/channel lag and median absolute valid-channel lag."""
    predictions = predict_model(scenario, physical, model_path, study, delay_s)
    rows = {}
    for name, prediction in predictions.items():
        record = scenario.records[name]
        channels = {sensor: phase_lag_for_signals(
            prediction.timestamps, getattr(prediction, sensor), record.arrays["measured_" + sensor],
            study.sensor_sigmas[sensor], study) for sensor in ("qpos", "qvel")}
        valid = [item["absolute_lag_s"] for item in channels.values() if item["valid"]]
        rows[name] = {"split": study.trajectories[name].split, "channels": channels,
                      "median_absolute_lag_s": float(np.median(valid)) if valid else None,
                      "valid_sensor_count": len(valid),
                      "unavailable_reason": None if valid else "Both sensor phase diagnostics are unavailable."}
    result = {"schema_version": "mujoco-sysid-lab.phase-lag.v1", "scenario": scenario.scenario,
              "seed": scenario.seed, "delay_s": float(delay_s), "trajectory_diagnostics": rows,
              "role": "post-selection phase diagnostic, not a delay estimator or confidence interval"}
    for split in ("train", "heldout"):
        values = [channel["absolute_lag_s"] for row in rows.values() if row["split"] == split
                  for channel in row["channels"].values() if channel["valid"]]
        result[split] = {"median_absolute_lag_s": float(np.median(values)) if values else None,
                         "valid_sensor_trajectory_count": len(values),
                         "unavailable_reason": None if values else "No valid phase diagnostics in this split."}
    json.dumps(result, allow_nan=False)
    return result
