"""Strict preregistration config; physical truth has a separate loader.

StudyConfig is safe to pass to a future estimator: it contains nominal physics,
known constants, and a path to truth, never the generator's physical values.
"""

from dataclasses import dataclass, fields, is_dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from mujoco_sysid_lab.config import ModelConfig, ParameterBounds


STUDY_SCHEMA = "mujoco-sysid-lab.evidence.v1"
TRUTH_SCHEMA = "mujoco-sysid-lab.ground-truth.v1"
CORE_TRAJECTORY_IDS = (
    "train_multisine", "train_prbs", "heldout_chirp", "heldout_pulse",
)


def _object(raw: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    if set(raw) != expected:
        raise ValueError(
            f"{name} fields invalid: missing={sorted(expected-set(raw))}, "
            f"unknown={sorted(set(raw)-expected)}"
        )
    return raw


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _grid(value: float, dt: float, name: str) -> None:
    if not math.isclose(value / dt, round(value / dt), rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{name} must be an integer multiple of timestep_s")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def config_to_dict(value: Any) -> Any:
    """Return JSON-safe copies; useful for exact per-run config snapshots."""
    if is_dataclass(value):
        return {field.name: config_to_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: config_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value} is forbidden")


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant, object_pairs_hook=_unique_object,
    )
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")
    # JSON exponents can overflow without being a named NaN/Infinity constant.
    def check(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("config contains a non-finite number")
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(raw)
    return raw


@dataclass(frozen=True)
class TrajectoryConfig:
    trajectory_id: int
    split: str
    kind: str
    duration_s: float
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class Initialization:
    mass_kg: float
    damping_nm_s_per_rad: float
    delay_s: float


@dataclass(frozen=True)
class StudyConfig:
    schema_version: str
    experiment_id: str
    truth_config_path: Path
    nominal: ModelConfig
    mass_bounds_kg: ParameterBounds
    damping_bounds_nm_s_per_rad: ParameterBounds
    delay_bounds_s: ParameterBounds
    duration_s: float
    zero_control_prefix_s: float
    prehistory_s: float
    evaluation_window_s: tuple[float, float]
    noise_seeds: tuple[int, ...]
    primary_seed: int
    sensor_sigmas: Mapping[str, float]
    noise: Mapping[str, Any]
    trajectories: Mapping[str, TrajectoryConfig]
    starts: Mapping[str, Initialization]
    optimizer: Mapping[str, Any]
    phase_lag: Mapping[str, Any]
    metrics: Mapping[str, Any]
    criteria: Mapping[str, Any]
    evidence_matrix: tuple[Mapping[str, Any], ...]
    registration: Mapping[str, Any]

    @property
    def timestep_s(self) -> float:
        return self.nominal.timestep_s

    def evaluation_window_for(self, trajectory_id: str) -> tuple[float, float]:
        return self.prehistory_s, self.trajectories[trajectory_id].duration_s - self.prehistory_s


@dataclass(frozen=True)
class TruthConfig:
    schema_version: str
    model: ModelConfig
    matched_delay_s: float
    delay_study_s: float
    mismatch: Mapping[str, Any]


def _model(raw: Any, name: str) -> ModelConfig:
    return ModelConfig(**_object(raw, {field.name for field in fields(ModelConfig)}, name))


def _bounds(raw: Any, name: str) -> ParameterBounds:
    return ParameterBounds(**_object(raw, {"lower", "upper"}, name))


def _trajectory(raw: Any, name: str, dt: float, prefix: float) -> TrajectoryConfig:
    raw = _object(raw, {"trajectory_id", "split", "kind", "duration_s", "settings"}, name)
    numeric_id = _integer(raw["trajectory_id"], f"{name}.trajectory_id")
    if raw["split"] not in ("train", "heldout"):
        raise ValueError(f"{name}.split must be train or heldout")
    duration = _number(raw["duration_s"], f"{name}.duration_s", positive=True)
    _grid(duration, dt, f"{name}.duration_s")
    if duration <= prefix:
        raise ValueError(f"{name}.duration_s must exceed zero_control_prefix_s")
    kind = raw["kind"]
    settings = raw["settings"]
    required = {
        "multisine": {"frequencies_hz", "amplitudes_nm", "phases_rad", "active_rms_nm"},
        "prbs": {"choices", "block_duration_s", "active_rms_nm", "seed_sequence"},
        "chirp": {"start_frequency_hz", "end_frequency_hz", "amplitude_nm", "onset_s"},
        "pulse": {"pulses"},
        "step": {"amplitude_nm"},
    }
    if kind not in required:
        raise ValueError(f"{name}.kind is unsupported")
    _object(settings, required[kind], f"{name}.settings")
    if kind == "multisine":
        vectors = [settings[key] for key in ("frequencies_hz", "amplitudes_nm", "phases_rad")]
        if not all(isinstance(vector, list) and vector for vector in vectors) or len({len(v) for v in vectors}) != 1:
            raise ValueError(f"{name} multisine vectors require matching nonzero lengths")
        for key, vector in zip(("frequencies_hz", "amplitudes_nm", "phases_rad"), vectors):
            for item in vector:
                number = _number(item, f"{name}.{key}", positive=key != "phases_rad")
                if key == "frequencies_hz" and number >= 0.5 / dt:
                    raise ValueError(f"{name} frequency must be below Nyquist")
        _number(settings["active_rms_nm"], f"{name}.active_rms_nm", positive=True)
    elif kind == "prbs":
        choices = settings["choices"]
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"{name}.choices needs at least two values")
        for item in choices:
            _number(item, f"{name}.choices")
        if len(set(choices)) != len(choices) or not any(choices):
            raise ValueError(f"{name}.choices requires distinct nonzero excitation")
        block = _number(settings["block_duration_s"], f"{name}.block_duration_s", positive=True)
        _grid(block, dt, f"{name}.block_duration_s")
        if block > duration - prefix:
            raise ValueError(f"{name}.block_duration_s exceeds active duration")
        _number(settings["active_rms_nm"], f"{name}.active_rms_nm", positive=True)
        if not isinstance(settings["seed_sequence"], list) or len(settings["seed_sequence"]) != 3:
            raise ValueError(f"{name}.seed_sequence needs three integer keys")
        for value in settings["seed_sequence"]:
            _integer(value, f"{name}.seed_sequence")
    elif kind == "chirp":
        for key in ("start_frequency_hz", "end_frequency_hz", "amplitude_nm", "onset_s"):
            number = _number(settings[key], f"{name}.{key}", positive=True)
            if "frequency" in key and number >= 0.5 / dt:
                raise ValueError(f"{name} frequency must be below Nyquist")
        if settings["end_frequency_hz"] <= settings["start_frequency_hz"]:
            raise ValueError(f"{name} chirp must have increasing frequency")
        if settings["onset_s"] > duration - prefix:
            raise ValueError(f"{name}.onset_s exceeds active duration")
        _grid(settings["onset_s"], dt, f"{name}.onset_s")
    elif kind == "pulse":
        pulses = settings["pulses"]
        if not isinstance(pulses, list) or not pulses:
            raise ValueError(f"{name}.pulses must be nonempty")
        previous_end = prefix
        for pulse in pulses:
            _object(pulse, {"start_s", "end_s", "amplitude_nm"}, f"{name}.pulse")
            start = _number(pulse["start_s"], f"{name}.start_s")
            end = _number(pulse["end_s"], f"{name}.end_s")
            _number(pulse["amplitude_nm"], f"{name}.amplitude_nm")
            if not previous_end <= start < end <= duration:
                raise ValueError(f"{name} pulses must be ordered, disjoint, and within active support")
            _grid(start, dt, f"{name}.start_s")
            _grid(end, dt, f"{name}.end_s")
            previous_end = end
    else:
        _number(settings["amplitude_nm"], f"{name}.amplitude_nm", positive=True)
    return TrajectoryConfig(numeric_id, raw["split"], kind, duration, _freeze(settings))


def load_study_config(path: Path) -> StudyConfig:
    """Load and validate frozen evidence knobs without reading physical truth."""
    path = Path(path).resolve()
    raw = _read_json(path)
    expected = {field.name for field in fields(StudyConfig)} - {"truth_config_path"}
    expected.add("truth_config_file")
    _object(raw, expected, "study")
    if raw["schema_version"] != STUDY_SCHEMA:
        raise ValueError("unsupported study config schema")
    if not isinstance(raw["experiment_id"], str) or not raw["experiment_id"]:
        raise ValueError("experiment_id must be a nonempty string")
    truth_file = raw["truth_config_file"]
    if not isinstance(truth_file, str) or Path(truth_file).name != truth_file or not truth_file.endswith(".json"):
        raise ValueError("truth_config_file must be an adjacent JSON filename")
    nominal = _model(raw["nominal"], "nominal")
    mass_bounds = _bounds(raw["mass_bounds_kg"], "mass_bounds_kg")
    damping_bounds = _bounds(raw["damping_bounds_nm_s_per_rad"], "damping_bounds_nm_s_per_rad")
    delay_bounds = _bounds(raw["delay_bounds_s"], "delay_bounds_s")
    if mass_bounds.lower <= 0 or damping_bounds.lower < 0 or delay_bounds.lower != 0:
        raise ValueError("physical bounds require positive mass, nonnegative damping, and delay lower=0")
    if not mass_bounds.contains(nominal.mass_kg) or not damping_bounds.contains(nominal.damping_nm_s_per_rad):
        raise ValueError("nominal physical parameters must lie within bounds")
    if nominal.initial_qpos_rad != 0 or nominal.initial_qvel_rad_s != 0:
        raise ValueError("registered equilibrium prehistory requires zero initial state")
    duration = _number(raw["duration_s"], "duration_s", positive=True)
    prefix = _number(raw["zero_control_prefix_s"], "zero_control_prefix_s", positive=True)
    prehistory = _number(raw["prehistory_s"], "prehistory_s", positive=True)
    for name, value in (("duration_s", duration), ("zero_control_prefix_s", prefix), ("prehistory_s", prehistory), ("delay upper", delay_bounds.upper)):
        _grid(value, nominal.timestep_s, name)
    if not duration > 2 * prehistory or not prefix >= prehistory >= delay_bounds.upper:
        raise ValueError("timing requires duration>2*prehistory and prefix>=prehistory>=maximum delay")
    window = raw["evaluation_window_s"]
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError("evaluation_window_s requires two endpoints")
    for endpoint in window:
        _number(endpoint, "evaluation_window_s")
    if window != [prehistory, duration - prehistory]:
        raise ValueError("evaluation_window_s must be the fixed safe [prehistory,duration-prehistory] window")
    seeds = raw["noise_seeds"]
    if not isinstance(seeds, list) or len(seeds) != 5:
        raise ValueError("noise_seeds requires exactly five seeds")
    for seed in seeds:
        _integer(seed, "noise_seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("noise_seeds must be unique")
    primary_seed = _integer(raw["primary_seed"], "primary_seed")
    if primary_seed not in seeds:
        raise ValueError("primary_seed must belong to noise_seeds")
    sigmas = _object(raw["sensor_sigmas"], {"qpos", "qvel"}, "sensor_sigmas")
    for name, value in sigmas.items():
        _number(value, f"sensor_sigmas.{name}", positive=True)
    noise = _object(raw["noise"], {"distribution", "bit_generator", "seed_sequence", "sensor_ids", "paired_group_id", "weak_group_id", "process_noise"}, "noise")
    _object(noise["sensor_ids"], {"qpos", "qvel"}, "noise.sensor_ids")
    for key in ("paired_group_id", "weak_group_id"):
        _integer(noise[key], f"noise.{key}")
    for key, value in noise["sensor_ids"].items():
        _integer(value, f"noise.sensor_ids.{key}")
    _boolean(noise["process_noise"], "noise.process_noise")
    if noise != {"distribution": "independent_zero_mean_gaussian", "bit_generator": "PCG64", "seed_sequence": ["noise_seed", "noise_group_id", "trajectory_id", "sensor_id"], "sensor_ids": {"qpos": 0, "qvel": 1}, "paired_group_id": 0, "weak_group_id": 1, "process_noise": False}:
        raise ValueError("noise scheme must match the registered PCG64 independent paired-noise contract")
    trajectory_raw = _object(raw["trajectories"], set(CORE_TRAJECTORY_IDS) | {"weak_step"}, "trajectories")
    trajectories = {name: _trajectory(value, name, nominal.timestep_s, prefix) for name, value in trajectory_raw.items()}
    for index, name in enumerate((*CORE_TRAJECTORY_IDS, "weak_step")):
        config = trajectories[name]
        expected_split = "heldout" if name.startswith("heldout") else "train"
        if config.trajectory_id != index or config.split != expected_split:
            raise ValueError(f"{name} numeric ID/split disagrees with registered trajectory partition")
        if name != "weak_step" and config.duration_s != duration:
            raise ValueError(f"{name} duration must match rich study duration_s")
    if trajectories["weak_step"].duration_s != 2 * duration:
        raise ValueError("weak_step duration must match total rich training duration")
    start_raw = _object(raw["starts"], {"S0", "S1", "S2"}, "starts")
    starts = {}
    for name, value in start_raw.items():
        _object(value, {"mass_kg", "damping_nm_s_per_rad", "delay_s"}, f"starts.{name}")
        for key, number in value.items():
            _number(number, f"starts.{name}.{key}")
        start = Initialization(**value)
        if not mass_bounds.contains(start.mass_kg) or not damping_bounds.contains(start.damping_nm_s_per_rad) or not delay_bounds.lower < start.delay_s < delay_bounds.upper:
            raise ValueError(f"starts.{name} must lie within physical bounds and strictly inside delay bounds")
        starts[name] = start
    if starts["S0"].mass_kg != nominal.mass_kg or starts["S0"].damping_nm_s_per_rad != nominal.damping_nm_s_per_rad:
        raise ValueError("starts.S0 physical parameters must equal nominal")
    optimizer = _object(raw["optimizer"], {"backend", "method", "loss", "max_iters", "budget_meaning", "ftol", "xtol", "gtol", "x_scale", "diff_step", "finite_difference_semantics", "rollout_workers", "numerical_library_threads", "delay_only_start_s", "selection", "objective", "residual_sign", "amplitude_normalization"}, "optimizer")
    if (optimizer["backend"], optimizer["method"], optimizer["loss"]) != ("scipy_parallel_fd", "trf", "linear"):
        raise ValueError("optimizer must use the official scipy_parallel_fd/trf/linear backend")
    for key in ("max_iters", "rollout_workers", "numerical_library_threads"):
        _integer(optimizer[key], f"optimizer.{key}", minimum=1)
    if optimizer["rollout_workers"] != 1 or optimizer["numerical_library_threads"] != 1:
        raise ValueError("registered CPU execution requires one rollout worker and one numerical-library thread")
    _boolean(optimizer["amplitude_normalization"], "optimizer.amplitude_normalization")
    for key in ("ftol", "xtol", "gtol", "diff_step", "delay_only_start_s"):
        _number(optimizer[key], f"optimizer.{key}", positive=True)
    if not 0 < optimizer["delay_only_start_s"] < delay_bounds.upper:
        raise ValueError("optimizer.delay_only_start_s must be strictly inside delay bounds")
    fixed_optimizer = {"budget_meaning": "scipy_max_nfev", "x_scale": "free_parameter_bound_widths", "finite_difference_semantics": "eps*max(1,abs(x)); sign toward bounds midpoint; clipped candidate", "selection": "lowest_finite_training_objective_including_nonconverged", "objective": "0.5*sum(noise_normalized_residual**2)", "residual_sign": "measured_minus_prediction", "amplitude_normalization": False}
    if any(optimizer[key] != value for key, value in fixed_optimizer.items()):
        raise ValueError("optimizer conventions disagree with registered objective/selection/finite-difference rules")
    phase_lag = _object(raw["phase_lag"], {"window_margin_s", "lag_min_s", "lag_max_s", "lag_step_s", "tie_tolerance", "tie_rule", "minimum_peak_correlation", "minimum_signal_std", "positive_sign"}, "phase_lag")
    for key in ("window_margin_s", "lag_min_s", "lag_max_s", "lag_step_s", "tie_tolerance", "minimum_peak_correlation"):
        _number(phase_lag[key], f"phase_lag.{key}")
    if not phase_lag["window_margin_s"] >= 2*delay_bounds.upper or phase_lag["lag_min_s"] != -delay_bounds.upper or phase_lag["lag_max_s"] != delay_bounds.upper or phase_lag["lag_step_s"] != nominal.timestep_s:
        raise ValueError("phase_lag must use fixed safe window and registered delay-grid support")
    if phase_lag["tie_tolerance"] < 0 or not 0 <= phase_lag["minimum_peak_correlation"] <= 1:
        raise ValueError("phase_lag tolerance/correlation invalid")
    if (phase_lag["tie_rule"], phase_lag["minimum_signal_std"], phase_lag["positive_sign"]) != ("smallest_absolute_lag_then_negative", "sensor_noise_sigma", "measurements_lag_candidate_prediction"):
        raise ValueError("phase_lag diagnostic semantics invalid")
    if 2*phase_lag["window_margin_s"] >= duration:
        raise ValueError("phase_lag fixed window must contain samples")
    metrics = _object(raw["metrics"], {"primary", "per_trajectory", "normalized_rmse", "split_aggregation", "relative_improvement", "bound_hit_fraction", "consistent_bound_hit_seed_count", "latent_clean", "seed_aggregation", "missing_diagnostics"}, "metrics")
    expected_metrics = {
        "primary": "heldout_trajectory_prediction",
        "per_trajectory": ["qpos_rmse", "qvel_rmse", "qpos_p95_absolute_error", "qvel_p95_absolute_error", "noise_normalized_rmse"],
        "normalized_rmse": "sqrt(mean((qpos_error/sigma_qpos)**2+(qvel_error/sigma_qvel)**2)/2)",
        "split_aggregation": "sqrt(mean(per_trajectory_rmse**2)); equal_trajectory_weight",
        "relative_improvement": "1-identified_rmse/nominal_rmse; null_reason_if_zero_denominator",
        "bound_hit_fraction": 0.001,
        "consistent_bound_hit_seed_count": 3,
        "latent_clean": "secondary_diagnostic_only",
        "seed_aggregation": ["median", "minimum", "maximum", "finite_count", "failures", "nonconvergence"],
        "missing_diagnostics": "null_with_reason",
    }
    _number(metrics["bound_hit_fraction"], "metrics.bound_hit_fraction", positive=True)
    _integer(metrics["consistent_bound_hit_seed_count"], "metrics.consistent_bound_hit_seed_count", minimum=1)
    if metrics != expected_metrics:
        raise ValueError("metrics must preserve the registered metric formulas and diagnostics")
    criteria = _object(raw["criteria"], {"physical_relative_error_max", "minimum_recovery_seed_count", "fixed_seed_denominator", "heldout_median_improvement_min", "require_nonnegative_each_heldout_median", "delay_absolute_error_max_s", "smoke_runtime_target_s", "full_runtime_target_s", "noiseless_single_parameter_relative_error_max", "delay_physical_eligibility_relative_error_max", "delay_joint_requires_successful_delay_only", "weak_or_mismatch_positive_result_required", "missing_required_seed_counts_as_failure"}, "criteria")
    for key in ("physical_relative_error_max", "heldout_median_improvement_min", "delay_absolute_error_max_s", "smoke_runtime_target_s", "full_runtime_target_s", "noiseless_single_parameter_relative_error_max", "delay_physical_eligibility_relative_error_max"):
        _number(criteria[key], f"criteria.{key}", positive=True)
    for key in ("minimum_recovery_seed_count", "fixed_seed_denominator"):
        _integer(criteria[key], f"criteria.{key}", minimum=1)
    for key in ("require_nonnegative_each_heldout_median", "delay_joint_requires_successful_delay_only", "weak_or_mismatch_positive_result_required", "missing_required_seed_counts_as_failure"):
        _boolean(criteria[key], f"criteria.{key}")
    for key in ("physical_relative_error_max", "heldout_median_improvement_min", "noiseless_single_parameter_relative_error_max", "delay_physical_eligibility_relative_error_max"):
        if criteria[key] > 1:
            raise ValueError(f"criteria.{key} must be <= 1")
    if criteria["fixed_seed_denominator"] != len(seeds) or criteria["minimum_recovery_seed_count"] > len(seeds):
        raise ValueError("criteria fixed denominator/recovery count must agree with registered noise seeds")
    if criteria["delay_absolute_error_max_s"] != nominal.timestep_s:
        raise ValueError("delay recovery threshold must equal one timestep")
    matrix = raw["evidence_matrix"]
    if not isinstance(matrix, list) or len(matrix) != 8:
        raise ValueError("evidence_matrix requires all eight registered experiment cells")
    matrix_ids = set()
    for row in matrix:
        _object(row, {"id", "scenario", "training_ids", "free_parameters", "noise_seeds", "start_ids", "fits", "prerequisite"}, "evidence_matrix row")
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError("evidence_matrix ID must be a nonempty string")
        if row["id"] in matrix_ids:
            raise ValueError("evidence_matrix IDs must be unique")
        matrix_ids.add(row["id"])
        if row["scenario"] not in ("matched", "delayed", "mismatch"):
            raise ValueError("evidence_matrix scenario is unsupported")
        for key in ("training_ids", "free_parameters", "noise_seeds", "start_ids"):
            if not isinstance(row[key], list) or not row[key]:
                raise ValueError(f"evidence_matrix {key} requires a nonempty list")
            if len(set(row[key])) != len(row[key]):
                raise ValueError(f"evidence_matrix {key} values must be unique")
        if any(name not in ("mass_kg", "damping_nm_s_per_rad", "delay_s") for name in row["free_parameters"]):
            raise ValueError("evidence_matrix may only fit the registered free parameters")
        if not row["training_ids"] or any(name not in trajectories or trajectories[name].split != "train" for name in row["training_ids"]):
            raise ValueError("evidence_matrix may only optimize training trajectories")
        if not row["noise_seeds"] or any(seed not in seeds for seed in row["noise_seeds"]):
            raise ValueError("evidence_matrix seeds must be registered")
        if not row["start_ids"] or any(name not in starts for name in row["start_ids"]):
            raise ValueError("evidence_matrix starts must be registered")
        _integer(row["fits"], "evidence_matrix fits", minimum=1)
        if row["fits"] != len(row["noise_seeds"])*len(row["start_ids"]):
            raise ValueError("evidence_matrix fit count must equal seeds times starts")
    if sum(row["fits"] for row in matrix) != 82:
        raise ValueError("evidence_matrix must preserve all 82 proposed evidence fits")
    if matrix_ids != {"A1", "A2", "B", "weak", "delay_excluded", "C_delay_only", "C_joint", "E"}:
        raise ValueError("evidence_matrix must preserve the eight registered experiment IDs")
    rich = ["train_multisine", "train_prbs"]
    physical = ["mass_kg", "damping_nm_s_per_rad"]
    all_starts = ["S0", "S1", "S2"]
    registered_cells = {
        "A1": ("matched", rich, ["mass_kg"], [primary_seed], ["S0"], "oracle_assisted_fixed_true_damping"),
        "A2": ("matched", rich, ["damping_nm_s_per_rad"], [primary_seed], ["S0"], "oracle_assisted_fixed_true_mass"),
        "B": ("matched", rich, physical, seeds, all_starts, None),
        "weak": ("matched", ["weak_step"], physical, seeds, all_starts, None),
        "delay_excluded": ("delayed", rich, physical, seeds, all_starts, None),
        "C_delay_only": ("delayed", rich, ["delay_s"], seeds, ["S0"], "same_seed_B_training_selected_winner_both_physical_errors_le_0.1"),
        "C_joint": ("delayed", rich, physical+["delay_s"], seeds, all_starts, "same_seed_C_delay_only_finite_abs_delay_error_le_one_timestep_and_timing_tests_pass"),
        "E": ("mismatch", rich, physical, seeds, all_starts, None),
    }
    for row in matrix:
        conventions = tuple(row[key] for key in ("scenario", "training_ids", "free_parameters", "noise_seeds", "start_ids", "prerequisite"))
        if conventions != registered_cells[row["id"]]:
            raise ValueError(f"evidence_matrix {row['id']} disagrees with its registered experiment/prerequisite")
    registration = _object(raw["registration"], {"status", "version_change_policy", "default_dataset_trajectory_ids", "record_all_failures", "truth_visibility", "measurement_delay_convention", "terminal_control", "gravity_m_s2", "integrator", "armature", "dry_friction", "contacts", "angle_convention", "initial_state_estimation", "estimator_inputs"}, "registration")
    if registration["default_dataset_trajectory_ids"] != list(CORE_TRAJECTORY_IDS):
        raise ValueError("default dataset must contain the four registered core trajectories")
    for key in ("record_all_failures", "contacts", "initial_state_estimation"):
        _boolean(registration[key], f"registration.{key}")
    if registration["record_all_failures"] is not True or registration["contacts"] is not False or registration["initial_state_estimation"] is not False:
        raise ValueError("registration requires all failures recorded, no contacts, and fixed known initial state")
    for key in ("armature", "dry_friction"):
        if _number(registration[key], f"registration.{key}") != 0:
            raise ValueError(f"registration.{key} must be zero in the matched model")
    if registration["gravity_m_s2"] != [0, 0, -9.81] or registration["integrator"] != "implicitfast" or registration["angle_convention"] != "unwrapped_radians_q0_downward":
        raise ValueError("registration known mechanical conventions are invalid")
    if registration["status"] != "frozen_before_first_identification_evidence_run" or registration["measurement_delay_convention"] != "measured_y(t)=clean_y(t-delay)+noise(t); no_extrapolation":
        raise ValueError("registration must preserve the frozen timing/study contract")
    converted = dict(raw)
    converted.pop("truth_config_file")
    converted.update(truth_config_path=path.parent/truth_file, nominal=nominal, mass_bounds_kg=mass_bounds, damping_bounds_nm_s_per_rad=damping_bounds, delay_bounds_s=delay_bounds, duration_s=duration, zero_control_prefix_s=prefix, prehistory_s=prehistory, evaluation_window_s=tuple(window), noise_seeds=tuple(seeds), primary_seed=primary_seed, sensor_sigmas=_freeze(sigmas), noise=_freeze(noise), trajectories=MappingProxyType(trajectories), starts=MappingProxyType(starts), optimizer=_freeze(optimizer), phase_lag=_freeze(phase_lag), metrics=_freeze(metrics), criteria=_freeze(criteria), evidence_matrix=tuple(_freeze(row) for row in matrix), registration=_freeze(registration))
    return StudyConfig(**converted)


def load_ground_truth(path: Path, study: StudyConfig) -> TruthConfig:
    """Generator/evaluation-only truth; never needed by load_study_config."""
    raw = _object(_read_json(Path(path)), {"schema_version", "model", "matched_delay_s", "delay_study_s", "mismatch"}, "ground_truth")
    if raw["schema_version"] != TRUTH_SCHEMA:
        raise ValueError("unsupported ground truth schema")
    model = _model(raw["model"], "ground_truth.model")
    if not study.mass_bounds_kg.contains(model.mass_kg) or not study.damping_bounds_nm_s_per_rad.contains(model.damping_nm_s_per_rad):
        raise ValueError("ground truth physical parameters must be within registered bounds")
    for field in fields(ModelConfig):
        if field.name not in ("mass_kg", "damping_nm_s_per_rad") and getattr(model, field.name) != getattr(study.nominal, field.name):
            raise ValueError(f"ground truth and nominal must share fixed {field.name}")
    for key in ("matched_delay_s", "delay_study_s"):
        _number(raw[key], f"ground_truth.{key}")
        if not study.delay_bounds_s.contains(raw[key]):
            raise ValueError(f"ground_truth.{key} is outside delay bounds")
        _grid(raw[key], study.timestep_s, f"ground_truth.{key}")
    if raw["matched_delay_s"] != 0:
        raise ValueError("matched data must have zero sensor delay")
    mismatch = _object(raw["mismatch"], {"kind", "torque_nm", "velocity_scale_rad_s", "fitting_model_has_effect"}, "ground_truth.mismatch")
    if mismatch["kind"] != "smooth_coulomb_torque" or mismatch["fitting_model_has_effect"] is not False:
        raise ValueError("mismatch must be generator-only smooth Coulomb torque")
    _number(mismatch["torque_nm"], "mismatch.torque_nm", positive=True)
    _number(mismatch["velocity_scale_rad_s"], "mismatch.velocity_scale_rad_s", positive=True)
    return TruthConfig(raw["schema_version"], model, raw["matched_delay_s"], raw["delay_study_s"], _freeze(mismatch))
