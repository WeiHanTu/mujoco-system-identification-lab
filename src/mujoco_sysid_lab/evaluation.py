"""One metric contract for nominal, identified, and oracle predictions.

Predictions must be full rollouts from the known initial state, with any
registered observation transform already applied. This evaluator never resets
state from noisy observations, interpolates, wraps angles, or fits an offset.
Residuals have the official SysID sign: measured minus predicted. RMSE and p95
are in rad for qpos and rad/s for qvel; normalized RMSE is dimensionless.
"""

from collections.abc import Iterable, Mapping
import json
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


RMSE_KEYS = ("qpos_rmse", "qvel_rmse", "normalized_rmse")
P95_KEYS = ("qpos_p95_absolute_error", "qvel_p95_absolute_error")


def _finite_scalar(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} real scalar")
    return result


def _finite_array(value: ArrayLike, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain finite real numbers")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def _rmse(values: np.ndarray) -> float:
    """Compute RMS without unnecessarily overflowing on large finite values."""
    maximum = float(np.max(np.abs(values)))
    if maximum == 0:
        return 0.0
    result = maximum * float(np.sqrt(np.mean(np.square(values / maximum))))
    if not np.isfinite(result):
        raise ValueError("metric computation produced a non-finite result")
    return result


def evaluate_trajectory(
    times: ArrayLike,
    measured_qpos: ArrayLike,
    measured_qvel: ArrayLike,
    predicted_qpos: ArrayLike,
    predicted_qvel: ArrayLike,
    sigma_qpos: float,
    sigma_qvel: float,
    window_s: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Evaluate aligned scalar sensors on a fixed, inclusive time window.

    All four sensor arrays must have shape (N, 1); timestamps must be a
    strictly increasing, uniform (N,) grid. None uses the registered common
    window [0.060, times[-1] - 0.060] s. The same explicit window and noise
    scales must be supplied for every model/scenario being compared.

    The returned n_samples counts only the fixed-window observations. The p95
    is NumPy's linear sample percentile of absolute residuals. No model label
    is accepted: all baseline labels therefore use this identical function.
    """
    timestamps = _finite_array(times, "timestamps")
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("timestamps must have shape (N,) with at least two samples")
    intervals = np.diff(timestamps)
    if not (intervals > 0).all():
        raise ValueError("timestamps must be strictly increasing")
    if not np.allclose(intervals, intervals[0], rtol=1e-10, atol=1e-12):
        raise ValueError("timestamps must be uniformly spaced")
    arrays = [
        _finite_array(value, name)
        for value, name in (
            (measured_qpos, "measured_qpos"),
            (measured_qvel, "measured_qvel"),
            (predicted_qpos, "predicted_qpos"),
            (predicted_qvel, "predicted_qvel"),
        )
    ]
    if any(array.shape != (timestamps.size, 1) for array in arrays):
        raise ValueError("all qpos/qvel arrays must have the common shape (N, 1)")
    sigma_q = _finite_scalar(sigma_qpos, "sigma_qpos", positive=True)
    sigma_v = _finite_scalar(sigma_qvel, "sigma_qvel", positive=True)
    if window_s is None:
        window_s = (0.060, float(timestamps[-1]) - 0.060)
    if not isinstance(window_s, (tuple, list)) or len(window_s) != 2:
        raise ValueError("window_s must contain exactly two finite ordered endpoints")
    lower = _finite_scalar(window_s[0], "window_s lower endpoint")
    upper = _finite_scalar(window_s[1], "window_s upper endpoint")
    if lower > upper:
        raise ValueError("window_s endpoints must be ordered")
    # Arithmetic tolerance only: preserve intended grid-boundary samples.
    # This is far below a sensor timestep and is independent of predictions.
    tolerance = 1e-12 + 8 * np.finfo(np.float64).eps * timestamps.size * max(
        1.0, abs(float(timestamps[-1]))
    )
    if lower < timestamps[0] - tolerance or upper > timestamps[-1] + tolerance:
        raise ValueError("window_s must lie inside the common timestamp support")
    mask = (timestamps >= lower - tolerance) & (timestamps <= upper + tolerance)
    if not mask.any():
        raise ValueError("window_s contains no timestamp samples")
    measured_q, measured_v, predicted_q, predicted_v = arrays
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error_q = measured_q[mask, 0] - predicted_q[mask, 0]
            error_v = measured_v[mask, 0] - predicted_v[mask, 0]
            normalized = np.column_stack((error_q / sigma_q, error_v / sigma_v))
            metrics = {
                "qpos_rmse": _rmse(error_q),
                "qvel_rmse": _rmse(error_v),
                "normalized_rmse": _rmse(normalized),
                "qpos_p95_absolute_error": float(np.percentile(np.abs(error_q), 95, method="linear")),
                "qvel_p95_absolute_error": float(np.percentile(np.abs(error_v), 95, method="linear")),
            }
    except FloatingPointError as exc:
        raise ValueError("metric computation produced a non-finite result") from exc
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("metric computation produced a non-finite result")
    return {
        **metrics,
        "n_samples": int(np.count_nonzero(mask)),
        "total_samples": int(timestamps.size),
        "evaluation_window_s": [lower, upper],
        "sampled_window_s": [float(timestamps[mask][0]), float(timestamps[mask][-1])],
        "residual_sign": "measured_minus_predicted",
        "percentile_method": "linear",
    }


def _metric_value(row: Mapping[str, Any], key: str) -> float:
    if key not in row:
        raise ValueError(f"metric row is missing {key}")
    result = _finite_scalar(row[key], key)
    if result < 0:
        raise ValueError(f"{key} must be nonnegative")
    return result


def aggregate_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Give every trajectory equal weight, regardless of its sample count.

    Split RMSE = sqrt(mean(per-trajectory RMSE**2)) for each sensor and for
    normalized RMSE. Per-trajectory p95 values remain in trajectory_metrics.
    There is deliberately no split p95: a mean of p95 values is not a pooled
    percentile, and this function has no raw residual samples to pool.
    """
    retained = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("metric rows must be mappings")
        for key in (*RMSE_KEYS, *P95_KEYS):
            _metric_value(row, key)
        copied = dict(row)
        try:
            json.dumps(copied, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metric rows must be strict JSON-compatible values") from exc
        retained.append(copied)
    if not retained:
        raise ValueError("at least one trajectory metric row is required")
    result = {
        key: _rmse(np.array([_metric_value(row, key) for row in retained]))
        for key in RMSE_KEYS
    }
    return {
        **result,
        "trajectory_count": len(retained),
        "aggregation": "equal_trajectory_root_mean_squared_rmse",
        "qpos_p95_absolute_error": None,
        "qvel_p95_absolute_error": None,
        "p95_unavailable_reason": "Per-trajectory percentiles are retained; raw residuals are required for a pooled percentile.",
        "trajectory_metrics": retained,
    }


def relative_improvement(
    identified: Mapping[str, Any], nominal: Mapping[str, Any]
) -> dict[str, Any]:
    """Return 1 - identified_RMSE / nominal_RMSE for the three RMSE fields.

    Positive fractions mean improvement; negative fractions mean degradation.
    For a zero nominal RMSE, return null and a reason instead of NaN or Inf.
    This works for either one trajectory or equal-trajectory split aggregates.
    """
    result: dict[str, Any] = {}
    for key in RMSE_KEYS:
        fitted = _metric_value(identified, key)
        baseline = _metric_value(nominal, key)
        if baseline == 0:
            result[key] = None
            result[f"{key}_reason"] = "Nominal RMSE is zero; relative improvement is undefined."
        else:
            improvement = 1.0 - fitted / baseline
            if not np.isfinite(improvement):
                raise ValueError("relative improvement produced a non-finite result")
            result[key] = improvement
            result[f"{key}_reason"] = None
    return result
