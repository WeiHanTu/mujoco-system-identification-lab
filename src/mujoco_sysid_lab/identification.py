"""Official MuJoCo SysID plumbing with a training-only input contract.

This module never opens a dataset, truth config, or held-out trajectory. The
oracle-assisted A1/A2 harness supplies its fixed nuisance in ``physical``;
only named free parameters enter ParameterDict. Optimization stays inside
``mujoco.sysid.optimize`` and its released ``scipy_parallel_fd`` backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import mujoco
from mujoco import sysid
import numpy as np
from scipy.optimize import OptimizeResult

from mujoco_sysid_lab.api_check import make_aligned_observation_hook
from mujoco_sysid_lab.config import ModelConfig
from mujoco_sysid_lab.dataset import (
    DATASET_SCHEMA_VERSION, TRAIN_TRAJECTORIES, TRAJECTORY_IDS,
    TrainingDataset, TrainingSequence,
)
from mujoco_sysid_lab.experiment_config import StudyConfig
from mujoco_sysid_lab.simulation import load_spec


PHYSICAL_PARAMETERS = ("mass_kg", "damping_nm_s_per_rad")
_COUNTER_KEYS = ("residual_calls", "candidate_columns", "batched_residual_calls",
                 "logical_rollouts", "logical_perturbed_rollouts")


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _flat(residuals: list[np.ndarray]) -> np.ndarray:
    result = np.concatenate(residuals)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("single-candidate official residual must be finite and flat")
    return result


def _objective(residual: np.ndarray) -> float:
    value = float(0.5 * np.dot(residual, residual))
    if not np.isfinite(value):
        raise ValueError("least-squares objective must be finite")
    return value


class _ResidualCounter:
    def __init__(self, official: Callable, n_trajectories: int,
                 bounds: tuple[np.ndarray, np.ndarray]):
        self.official = official
        self.n_trajectories = n_trajectories
        self.bounds = bounds
        self.reset()

    def reset(self) -> None:
        self.phase = "external"
        self.counts: dict[str, dict[str, int]] = {
            phase: dict.fromkeys(_COUNTER_KEYS, 0)
            for phase in ("initial", "optimization", "final", "external")
        }

    def snapshot(self) -> dict[str, dict[str, int]]:
        result = {phase: dict(counts) for phase, counts in self.counts.items()}
        result["total"] = {key: sum(row[key] for row in self.counts.values())
                           for key in _COUNTER_KEYS}
        return result

    def snapshot_counts(self) -> dict[str, dict[str, int]]:
        return self.snapshot()

    def __call__(self, x, params, **kwargs):
        point = np.asarray(x, dtype=np.float64)
        lower, upper = self.bounds
        if (point.ndim not in (1, 2) or point.shape[0] != lower.size
                or not np.isfinite(point).all() or (point.ndim == 2 and point.shape[1] < 1)):
            raise ValueError("parameter candidate has invalid dimensions or non-finite values")
        candidates = point[:, None] if point.ndim == 1 else point
        if (candidates < lower[:, None]).any() or (candidates > upper[:, None]).any():
            raise ValueError("parameter candidate is outside registered bounds")
        n_candidates = candidates.shape[1]
        batched = int(point.ndim == 2)
        counts = self.counts.setdefault(self.phase, dict.fromkeys(_COUNTER_KEYS, 0))
        counts["residual_calls"] += 1
        counts["candidate_columns"] += n_candidates
        counts["batched_residual_calls"] += batched
        counts["logical_rollouts"] += n_candidates * self.n_trajectories
        counts["logical_perturbed_rollouts"] += batched * n_candidates * self.n_trajectories
        residuals, predicted, measured = self.official(point, params, **kwargs)
        if any(not np.isfinite(array).all() for array in residuals):
            raise ValueError("official residual produced NaN/Inf")
        return residuals, predicted, measured


@dataclass(frozen=True)
class FitExecution:
    optimized_params: sysid.ParameterDict
    optimize_result: OptimizeResult
    initial_residual: np.ndarray
    final_residual: np.ndarray
    initial_predictions: list
    final_predictions: list
    initial_records: list
    final_records: list
    call_counts: dict[str, dict[str, int]]
    objective_trace: tuple[dict[str, Any], ...]
    callback_events: int
    runtime_seconds: float

    @property
    def calls(self):
        return self.call_counts

    @property
    def runtime(self):
        return self.runtime_seconds

    @property
    def initial_residuals(self):
        return self.initial_residual

    @property
    def final_residuals(self):
        return self.final_residual


@dataclass
class FitProblem:
    spec: mujoco.MjSpec
    params: sysid.ParameterDict
    sequences: list[sysid.ModelSequences]
    residual_fn: Callable
    optimizer_options: dict[str, Any]
    free_parameters: tuple[str, ...]
    training_ids: tuple[str, ...]
    training_seed: int
    _counter: _ResidualCounter = field(repr=False)
    last_trace: list[dict[str, Any]] = field(default_factory=list, repr=False)
    optimization_invoked: bool = field(default=False, init=False)

    @property
    def initial_params(self):
        return self.params

    @property
    def models_sequences(self):
        return self.sequences

    @property
    def call_counts(self):
        return self._counter.snapshot()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate one candidate without mutating the starting ParameterDict."""
        point = np.asarray(x, dtype=np.float64)
        if point.ndim != 1:
            raise ValueError("evaluate requires a flat single-candidate parameter vector")
        residuals, _, _ = self.residual_fn(point, self.params.copy())
        return _readonly(_flat(residuals))

    def run(self) -> FitExecution:
        """Run the frozen official optimizer; capture callback and call counts.

        Counts are reset here so prior diagnostic evaluations do not become
        optimizer work. External reporting calls remain visible in the live
        problem ledger, while the returned optimization ledger is a snapshot.
        """
        self._counter.reset()
        self.last_trace = []
        self.optimization_invoked = False
        started = perf_counter()
        start_vector = self.params.as_vector().copy()
        callback_count = 0
        self._counter.phase = "initial"
        try:
            initial_raw, initial_pred, initial_measured = self.residual_fn(
                start_vector, self.params.copy(),
            )
            initial = _flat(initial_raw)
            self.last_trace.append({"event": "initial", "callback_event": None,
                                    "candidate": start_vector.tolist(),
                                    "objective": _objective(initial), "native_nfev": None})
        except BaseException:
            self._counter.phase = "external"
            raise

        def callback(intermediate_result):
            nonlocal callback_count
            callback_count += 1
            cost = float(intermediate_result.cost)
            if not np.isfinite(cost):
                raise ValueError("optimizer callback objective must be finite")
            self.last_trace.append({
                "event": "callback", "callback_event": callback_count,
                "candidate": np.asarray(intermediate_result.x).tolist(),
                "objective": cost, "native_nfev": int(intermediate_result.nfev),
            })

        try:
            self._counter.phase = "optimization"
            self.optimization_invoked = True
            optimized, result = sysid.optimize(
                initial_params=self.params.copy(), residual_fn=self.residual_fn,
                optimizer="scipy_parallel_fd", verbose=False, check_conditioning=False,
                callback=callback, **self.optimizer_options,
            )
            self._counter.phase = "final"
            final_raw, final_pred, final_measured = self.residual_fn(
                optimized.as_vector(), optimized,
            )
            final = _flat(final_raw)
            final_objective = _objective(final)
            if (not math.isclose(final_objective, float(result.cost), rel_tol=1e-10, abs_tol=1e-8)
                    or not np.allclose(final, result.fun, rtol=1e-10, atol=1e-8)):
                raise ValueError("recomputed official residual/objective disagrees with optimizer result")
            self.last_trace.append({"event": "final", "callback_event": None,
                                    "candidate": optimized.as_vector().tolist(),
                                    "objective": final_objective, "native_nfev": int(result.nfev)})
            # The official HTML reads this optional native-report extension.
            # The source of these entries is explicit: initial/callback/final,
            # rather than claiming an upstream full iteration history.
            result.extras = {
                "objective": [row["objective"] for row in self.last_trace],
                "candidate": [np.array(row["candidate"]) for row in self.last_trace],
            }
            return FitExecution(
                optimized, result, _readonly(initial), _readonly(final),
                initial_pred, final_pred, initial_measured, final_measured,
                self._counter.snapshot(), tuple(dict(row) for row in self.last_trace),
                callback_count, perf_counter() - started,
            )
        finally:
            self._counter.phase = "external"


def _copy_build_model(params: sysid.ParameterDict, spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Supported builder hook avoids upstream in-place mutation of shared spec."""
    return sysid.apply_param_modifiers(params, spec.copy())


def _validate_training(training: TrainingDataset, physical: ModelConfig,
                       study: StudyConfig) -> None:
    if type(training) is not TrainingDataset:
        raise ValueError("fitting requires the sanitized TrainingDataset interface")
    if (type(training.dataset_schema_version) is not int
            or training.dataset_schema_version != DATASET_SCHEMA_VERSION):
        raise ValueError("unsupported training dataset schema version")
    if any(type(sequence) is not TrainingSequence for sequence in training.sequences):
        raise ValueError("training inputs must use the sanitized TrainingSequence interface")
    names = tuple(sequence.trajectory_name for sequence in training.sequences)
    if names != TRAIN_TRAJECTORIES:
        raise ValueError("fit input must contain exactly the two registered training trajectories")
    if training.seed not in study.noise_seeds:
        raise ValueError("training seed is outside registered seeds")
    for sequence in training.sequences:
        if sequence.trajectory_id != TRAJECTORY_IDS[sequence.trajectory_name]:
            raise ValueError("training trajectory identity is inconsistent")
        times = sequence.timestamps
        if (not isinstance(times, np.ndarray) or times.dtype.kind not in "fiu"
                or times.ndim != 1 or times.size < 2 or not np.isfinite(times).all()
                or not np.array_equal(times, np.arange(times.size) * physical.timestep_s)):
            raise ValueError("training timestamps must use the complete fixed grid from zero")
        if not np.array_equal(sequence.initial_state, [0.0, 0.0, 0.0]):
            raise ValueError("only the registered exact equilibrium initial state is supported")
        for name in ("controls", "measured_qpos", "measured_qvel"):
            values = getattr(sequence, name)
            if (not isinstance(values, np.ndarray) or values.dtype.kind not in "fiu"
                    or values.shape != (times.size, 1) or not np.isfinite(values).all()):
                raise ValueError(f"training {name} must be finite with shape (N,1)")
        if np.max(np.abs(sequence.controls)) * physical.motor_gain_nm_per_unit > physical.control_limit_nm:
            raise ValueError("training control exceeds the registered torque limit")
    # Only the requested physical nuisance may differ from nominal. Other
    # constants cannot change as a side effect of the oracle-assisted harness.
    for name in ("timestep_s", "link_length_m", "link_radius_m", "motor_gain_nm_per_unit",
                 "control_limit_nm", "initial_qpos_rad", "initial_qvel_rad_s"):
        if getattr(physical, name) != getattr(study.nominal, name):
            raise ValueError(f"fitting physical config changes fixed {name}")


def build_problem(training: TrainingDataset, model_path: Path, physical: ModelConfig,
                  study: StudyConfig, free_parameters: tuple[str, ...],
                  start: Mapping[str, float]) -> FitProblem:
    """Build a bounded official fit using only typed training observations.

    ``physical`` stamps every fixed quantity, including A1/A2's known nuisance.
    ``start`` must contain exactly the requested free physical parameter names.
    No delay fitting is implemented before its separate milestone gate.
    """
    if mujoco.__version__ != "3.15.0":
        raise RuntimeError("identification requires released MuJoCo 3.15.0")
    _validate_training(training, physical, study)
    if (not isinstance(free_parameters, tuple) or not free_parameters
            or len(set(free_parameters)) != len(free_parameters)
            or any(name not in PHYSICAL_PARAMETERS for name in free_parameters)):
        raise ValueError("free_parameters must name unique mass and/or damping parameters")
    if not isinstance(start, Mapping) or set(start) != set(free_parameters):
        raise ValueError("start must contain exactly the free parameter names")
    bounds = {"mass_kg": study.mass_bounds_kg,
              "damping_nm_s_per_rad": study.damping_bounds_nm_s_per_rad}
    for name in free_parameters:
        value = start[name]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int))
                or not np.isfinite(value) or not bounds[name].contains(value)):
            raise ValueError(f"start {name} must be finite and inside registered bounds")
    starting_physics = replace(physical, **start)
    spec = load_spec(model_path, starting_physics)
    model = spec.compile()
    if (model.nq, model.nv, model.na, model.nu, model.nsensordata) != (1, 1, 0, 1, 2):
        raise ValueError("fit model must implement the one-hinge no-activation sensor contract")
    params = sysid.ParameterDict()
    for name in free_parameters:
        bound = bounds[name]
        if name == "mass_kg":
            params.add(sysid.body_inertia_param(
                spec, model, "link", inertia_type=sysid.InertiaType.Mass,
                scale_rot_inertia=True,
                mass_bound_mult=np.array([bound.lower, bound.upper]) / start[name],
                param_name=name,
            ))
        else:
            def damping_modifier(candidate_spec, parameter):
                candidate_spec.joint("hinge").damping = [float(parameter.value[0]), 0.0, 0.0]

            params.add(sysid.Parameter(name, start[name], bound.lower, bound.upper,
                                       modifier=damping_modifier))
    controls, observations, initial_states = [], [], []
    for sequence in training.sequences:
        initial = sysid.create_initial_state(model, sequence.initial_state[1:2],
                                            sequence.initial_state[2:3])
        if not np.array_equal(initial, sequence.initial_state):
            raise ValueError("official initial-state helper disagrees with explicit known state")
        lower, upper = study.prehistory_s, float(sequence.timestamps[-1]) - study.prehistory_s
        if lower >= upper:
            raise ValueError("training horizon must exceed twice the fixed observation margin")
        mask = (sequence.timestamps >= lower - 1e-12) & (sequence.timestamps <= upper + 1e-12)
        if not mask.any():
            raise ValueError("fixed training observation window is empty")
        controls.append(sysid.TimeSeries.from_control_names(
            sequence.timestamps.copy(), sequence.controls.copy(), model, ["torque"],
        ))
        observations.append(sysid.TimeSeries.from_names(
            sequence.timestamps[mask].copy(),
            np.column_stack((sequence.measured_qpos[mask, 0], sequence.measured_qvel[mask, 0])),
            model, ["qpos", "qvel"],
        ))
        initial_states.append(initial)
    sequences = [sysid.ModelSequences(
        "pendulum", spec, list(TRAIN_TRAJECTORIES), initial_states, controls, observations,
    )]
    hook = make_aligned_observation_hook(initial_states[0], study.sensor_sigmas)
    official = sysid.build_residual_fn(
        models_sequences=sequences, modify_residual=hook,
        build_model=_copy_build_model, n_threads=1,
    )
    counter = _ResidualCounter(official, len(training.sequences), params.get_bounds())
    optimizer = study.optimizer
    if (optimizer["backend"] != "scipy_parallel_fd" or optimizer["rollout_workers"] != 1
            or optimizer["amplitude_normalization"] is not False):
        raise ValueError("unsupported departure from registered optimizer/residual contract")
    options = {name: optimizer[name] for name in ("method", "loss", "max_iters", "ftol",
                                                "xtol", "gtol", "diff_step")}
    options["x_scale"] = params.get_bounds()[1] - params.get_bounds()[0]
    return FitProblem(spec, params, sequences, counter, options, free_parameters,
                      TRAIN_TRAJECTORIES, training.seed, counter)
