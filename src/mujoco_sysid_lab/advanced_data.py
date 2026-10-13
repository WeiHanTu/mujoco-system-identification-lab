"""Paired synthetic scenarios for delay, excitation, and omitted-force studies.

Generator/evaluation records contain physical truth. ``ScenarioData.training``
and ``load_scenario_training`` expose only the five known training input arrays.
This format is separate from the immutable Milestone 2 dataset schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import mujoco
import numpy as np

from mujoco_sysid_lab.dataset import (
    ARRAY_NAMES as M2_ARRAY_NAMES, CORE_TRAJECTORIES, TRAIN_ARRAY_NAMES,
    TrajectoryRecord, TrainingDataset, TrainingSequence,
    _initial_state, _owned_readonly, _reject_constant, _snapshot, _strict_json_object,
    _unique_object, _valid_sha256, array_sha256, generate_dataset,
)
from mujoco_sysid_lab.excitation import generate_control
from mujoco_sysid_lab.experiment_config import StudyConfig, TruthConfig
from mujoco_sysid_lab.provenance import sha256_file, write_json
from mujoco_sysid_lab.simulation import Trajectory, compile_model, simulate


SCENARIO_SCHEMA_VERSION = 1
SCENARIOS = ("matched", "delayed", "weak", "mismatch")
ARRAY_NAMES = M2_ARRAY_NAMES | {
    "observed_clean_qpos", "observed_clean_qvel", "extra_torque_nm",
}
_IDS = {**dict(zip(CORE_TRAJECTORIES, range(4), strict=True)), "weak_step": 4}
_HELDOUT_IDS = CORE_TRAJECTORIES[2:]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ScenarioData:
    scenario: str
    seed: int
    records: Mapping[str, TrajectoryRecord]
    metadata: Mapping[str, Any]
    scenario_schema_version: int = SCENARIO_SCHEMA_VERSION

    @property
    def training_ids(self) -> tuple[str, ...]:
        return ("weak_step",) if self.scenario == "weak" else CORE_TRAJECTORIES[:2]

    @property
    def delay_s(self) -> float:
        return float(self.metadata["shared_sensor_delay_s"])

    @property
    def training(self) -> TrainingDataset:
        return TrainingDataset(self.seed, tuple(
            TrainingSequence(name, _IDS[name], **{
                key: self.records[name].arrays[key] for key in TRAIN_ARRAY_NAMES
            }) for name in self.training_ids
        ))

    @property
    def heldout(self) -> Mapping[str, TrajectoryRecord]:
        return MappingProxyType({name: self.records[name] for name in _HELDOUT_IDS})


def apply_sensor_delay(times: np.ndarray, values: np.ndarray, delay_s: float,
                       prehistory_times: np.ndarray, prehistory_values: np.ndarray) -> np.ndarray:
    """Apply positive lag y(t)=clean(t-delay), without padding/extrapolation.

    The explicit prehistory includes t=0 and must join continuously to the
    trajectory. Query support is checked before NumPy's interpolator is used.
    """
    arrays = (times, values, prehistory_times, prehistory_values)
    if any(not isinstance(array, np.ndarray) or array.dtype.kind not in "fiu"
           or not np.isfinite(array).all() for array in arrays):
        raise ValueError("delay inputs must be finite numeric arrays")
    if (times.ndim != 1 or times.size < 2 or times[0] != 0
            or not np.all(np.diff(times) > 0) or values.shape != (times.size, 1)):
        raise ValueError("delay trajectory must have increasing times from zero and values (N,1)")
    if (prehistory_times.ndim != 1 or prehistory_times.size < 2
            or prehistory_times[0] >= 0 or prehistory_times[-1] != 0
            or not np.all(np.diff(prehistory_times) > 0)
            or prehistory_values.shape != (prehistory_times.size, 1)):
        raise ValueError("delay requires explicit increasing prehistory ending at zero")
    if not np.array_equal(prehistory_values[-1], values[0]):
        raise ValueError("prehistory must join the clean initial observation continuously")
    if (isinstance(delay_s, (bool, np.bool_)) or not isinstance(delay_s, (float, int))
            or not np.isfinite(delay_s) or delay_s < 0):
        raise ValueError("sensor delay must be finite and nonnegative")
    support_times = np.concatenate((prehistory_times[:-1], times))
    support_values = np.concatenate((prehistory_values[:-1, 0], values[:, 0]))
    query = times - delay_s
    if query[0] < support_times[0] or query[-1] > support_times[-1]:
        raise ValueError("delay query is outside explicit history support; extrapolation/padding forbidden")
    return _owned_readonly(np.interp(query, support_times, support_values)[:, None])


def simulate_mismatch(model: mujoco.MjModel, times: np.ndarray, controls: np.ndarray,
                      torque_nm: float, velocity_scale_rad_s: float) -> tuple[Trajectory, np.ndarray]:
    """Run a truth-only smooth Coulomb-like force on local MjData, with fresh sensors.

    Force is -torque_nm*tanh(v/velocity_scale) at the start of each interval.
    The final recorded force is the computed but unused terminal value.
    No global callback, MjModel mutation, dry-friction field, or contact is used.
    """
    if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
            or not np.isfinite(value) for value in (torque_nm, velocity_scale_rad_s))
            or torque_nm < 0 or velocity_scale_rad_s <= 0):
        raise ValueError("mismatch torque must be nonnegative and velocity scale positive")
    # Reuse the strict standard simulator's input/model/control-limit contract.
    # Its equilibrium result is not used as target data.
    checked = simulate(model, times, controls)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.time = 0.0
    data.ctrl[:] = checked.controls[0]
    mujoco.mj_forward(model, data)
    qpos = np.empty_like(checked.qpos)
    qvel = np.empty_like(checked.qvel)
    sensors = np.empty_like(checked.sensordata)
    extra = np.empty_like(checked.qvel)
    for index in range(checked.timestamps.size):
        qpos[index] = data.qpos
        qvel[index] = data.qvel
        sensors[index] = data.sensordata
        extra[index, 0] = -torque_nm * np.tanh(data.qvel[0] / velocity_scale_rad_s)
        if index + 1 < checked.timestamps.size:
            data.ctrl[:] = checked.controls[index]
            data.qfrc_applied[0] = extra[index, 0]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    if any(not np.isfinite(array).all() for array in (qpos, qvel, sensors, extra)):
        raise RuntimeError("mismatch simulation produced NaN/Inf")
    trajectory = Trajectory(checked.timestamps, checked.controls,
                            _owned_readonly(qpos), _owned_readonly(qvel), _owned_readonly(sensors))
    return trajectory, _owned_readonly(extra)


def _weak_record(study: StudyConfig, truth: TruthConfig, model_path: Path, seed: int,
                 history_template: TrajectoryRecord) -> TrajectoryRecord:
    name = "weak_step"
    times, controls, diagnostics = generate_control(study, name)
    target = simulate(compile_model(model_path, truth.model), times, controls)
    nominal = simulate(compile_model(model_path, study.nominal), times, controls)
    arrays = {"timestamps": times, "controls": controls,
              "clean_qpos": target.qpos, "clean_qvel": target.qvel,
              "nominal_qpos": nominal.qpos, "nominal_qvel": nominal.qvel,
              "initial_state": _initial_state(compile_model(model_path, truth.model), 0., 0.)}
    noise = {}
    for sensor_id, sensor in enumerate(("qpos", "qvel")):
        entropy = [seed, 1, 4, sensor_id]
        innovation = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence(entropy))).standard_normal((times.size, 1))
        arrays["innovation_" + sensor] = innovation
        arrays["measured_" + sensor] = arrays["clean_" + sensor] + study.sensor_sigmas[sensor] * innovation
        noise[sensor] = {"sigma": study.sensor_sigmas[sensor], "seed_sequence_entropy": entropy,
                         "innovation_sha256": array_sha256(innovation)}
    arrays.update({key: value for key, value in history_template.arrays.items() if key.startswith("prehistory_")})
    metadata = {"trajectory_id": 4, "split": "train", "duration_s": study.trajectories[name].duration_s,
                "timestep_s": study.timestep_s, "prehistory_s": study.prehistory_s,
                "excitation_config": _snapshot(study.trajectories[name]),
                "excitation_diagnostics": diagnostics, "noise": noise,
                "initial_state_spec": "mjSTATE_FULLPHYSICS",
                "initial_state_order": ["time_s", "qpos_rad", "qvel_rad_s"],
                "initial_state_source": "known exact reset, not measured/refitted"}
    return TrajectoryRecord(arrays, metadata)


def create_scenario(study: StudyConfig, truth: TruthConfig, model_path: Path,
                    seed: int, scenario: str) -> ScenarioData:
    """Generate paired matched/delayed/weak/mismatch trajectories from frozen inputs."""
    if scenario not in SCENARIOS:
        raise ValueError("unsupported synthetic scenario")
    # The matched generator supplies exactly the registered group-0 innovations,
    # authored controls, histories, and clean nominal/family-oracle trajectories.
    base = generate_dataset(study, truth, model_path, seed)
    records = dict(base.records)
    if scenario == "weak":
        records = {"weak_step": _weak_record(study, truth, model_path, seed,
                                               base.records["train_multisine"]),
                   **{name: base.records[name] for name in _HELDOUT_IDS}}
    delay = truth.delay_study_s if scenario == "delayed" else truth.matched_delay_s
    if not 0 <= delay <= study.delay_bounds_s.upper:
        raise ValueError("generator sensor delay is outside registered support")
    target_model = compile_model(model_path, truth.model)
    finalized = {}
    for name, record in records.items():
        arrays = dict(record.arrays)
        metadata = _snapshot(record.metadata)
        extra = np.zeros_like(arrays["clean_qvel"])
        if scenario == "mismatch":
            trajectory, extra = simulate_mismatch(
                target_model, arrays["timestamps"], arrays["controls"],
                truth.mismatch["torque_nm"], truth.mismatch["velocity_scale_rad_s"],
            )
            arrays["clean_qpos"], arrays["clean_qvel"] = trajectory.qpos, trajectory.qvel
        for sensor in ("qpos", "qvel"):
            observed = apply_sensor_delay(
                arrays["timestamps"], arrays["clean_" + sensor], delay,
                arrays["prehistory_timestamps"], arrays["prehistory_" + sensor],
            )
            arrays["observed_clean_" + sensor] = observed
            arrays["measured_" + sensor] = observed + study.sensor_sigmas[sensor] * arrays["innovation_" + sensor]
        arrays["extra_torque_nm"] = extra
        metadata.update(scenario=scenario, sensor_delay_s=delay,
                        noise_group_id=1 if name == "weak_step" else 0,
                        extra_torque_terminal_unused=True,
                        truth_force_path="local qfrc_applied per interval" if scenario == "mismatch" else "none",
                        fitting_model_has_extra_torque=False)
        finalized[name] = TrajectoryRecord(
            MappingProxyType({key: _owned_readonly(value) for key, value in arrays.items()}),
            _freeze(metadata),
        )
    metadata = _snapshot(base.metadata)
    metadata.update(scenario=scenario, training_ids=["weak_step"] if scenario == "weak" else list(CORE_TRAJECTORIES[:2]),
                    heldout_ids=list(_HELDOUT_IDS), shared_sensor_delay_s=delay,
                    applied_mismatch=_snapshot(truth.mismatch) if scenario == "mismatch" else None,
                    mismatch_force_semantics="-torque_nm*tanh(pre_interval_qvel/velocity_scale); held per interval; terminal unused",
                    mismatch_isolation="local MjData only; no callbacks, MjModel changes, or fitting-model force",
                    weak_noise_group_id=1, paired_noise_group_id=0,
                    observed_clean_semantics="noise-free physical signals after the measurement delay")
    result = ScenarioData(scenario, seed, MappingProxyType(finalized), _freeze(metadata))
    validate_scenario(result)
    return result


def _record_names(scenario: str) -> tuple[str, ...]:
    return ("weak_step", *_HELDOUT_IDS) if scenario == "weak" else CORE_TRAJECTORIES


def _validate_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], *,
                     training_only: bool = False) -> None:
    names = set(TRAIN_ARRAY_NAMES) if training_only else ARRAY_NAMES
    if set(arrays) != names:
        raise ValueError("scenario arrays do not match the versioned schema")
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError(f"scenario {name} must be finite float64")
    times = arrays["timestamps"]
    if (times.ndim != 1 or times.size < 2 or not np.array_equal(times, np.arange(times.size) * metadata["timestep_s"])
            or not np.isclose(times[-1], metadata["duration_s"], rtol=0, atol=1e-12)):
        raise ValueError("scenario timestamps must span the complete fixed grid")
    if not np.array_equal(arrays["initial_state"], [0., 0., 0.]):
        raise ValueError("scenario requires exact FULLPHYSICS equilibrium initialization")
    for name, array in arrays.items():
        if name not in {"timestamps", "initial_state"} and not name.startswith("prehistory_"):
            if array.shape != (times.size, 1):
                raise ValueError(f"scenario {name} must have shape (N,1)")
    if training_only:
        return
    pre_times = arrays["prehistory_timestamps"]
    if (pre_times.ndim != 1 or pre_times.size < 2
            or not np.allclose(pre_times, np.arange(pre_times.size) * metadata["timestep_s"] - metadata["prehistory_s"], rtol=0, atol=1e-12)
            or pre_times[-1] != 0):
        raise ValueError("scenario prehistory must span [-H,0] on the fixed grid")
    if not np.array_equal(arrays["prehistory_initial_state"], [-metadata["prehistory_s"], 0., 0.]):
        raise ValueError("scenario prehistory must record its negative initial time")
    for name in ("prehistory_qpos", "prehistory_qvel", "prehistory_controls"):
        if arrays[name].shape != (pre_times.size, 1) or np.any(arrays[name] != 0):
            raise ValueError("scenario prehistory must be explicit zero-control equilibrium")


def validate_scenario(data: ScenarioData) -> None:
    if type(data.scenario_schema_version) is not int or data.scenario_schema_version != SCENARIO_SCHEMA_VERSION:
        raise ValueError("unsupported scenario schema")
    if data.scenario not in SCENARIOS or set(data.records) != set(_record_names(data.scenario)):
        raise ValueError("scenario trajectory split is invalid")
    if type(data.seed) is not int or data.seed < 0:
        raise ValueError("scenario seed must be a nonnegative integer")
    global_metadata = _strict_json_object(data.metadata, "scenario metadata")
    if (global_metadata.get("scenario") != data.scenario
            or global_metadata.get("training_ids") != list(data.training_ids)
            or global_metadata.get("heldout_ids") != list(_HELDOUT_IDS)
            or not isinstance(global_metadata.get("study_config"), dict)
            or data.seed not in global_metadata["study_config"].get("noise_seeds", [])):
        raise ValueError("scenario global identity/config split is invalid")
    for name, record in data.records.items():
        metadata = _strict_json_object(record.metadata, "trajectory metadata")
        _validate_identity(name, metadata, data.scenario, data.seed)
        _validate_arrays(record.arrays, metadata)
        if metadata["sensor_delay_s"] != global_metadata.get("shared_sensor_delay_s"):
            raise ValueError("scenario sensor delay must be shared by every trajectory")
        for sensor in ("qpos", "qvel"):
            noise = metadata["noise"][sensor]
            innovation = record.arrays["innovation_" + sensor]
            if array_sha256(innovation) != noise["innovation_sha256"]:
                raise ValueError("scenario innovation hash mismatch")
            if not np.array_equal(record.arrays["measured_" + sensor],
                                  record.arrays["observed_clean_" + sensor] + noise["sigma"] * innovation):
                raise ValueError("scenario noise must be added after the clean observation transform")
            expected = apply_sensor_delay(
                record.arrays["timestamps"], record.arrays["clean_" + sensor], metadata["sensor_delay_s"],
                record.arrays["prehistory_timestamps"], record.arrays["prehistory_" + sensor],
            )
            if not np.array_equal(expected, record.arrays["observed_clean_" + sensor]):
                raise ValueError("scenario clean observation must match the supported positive delay transform")
        torque = record.arrays["extra_torque_nm"]
        if data.scenario != "mismatch" and np.any(torque != 0):
            raise ValueError("extra force must be absent outside the mismatch generator")
        if data.scenario == "mismatch":
            force = global_metadata.get("applied_mismatch")
            if not isinstance(force, dict) or force.get("fitting_model_has_effect") is not False:
                raise ValueError("mismatch force must be isolated from the fitting model")
            expected = -force["torque_nm"] * np.tanh(record.arrays["clean_qvel"] / force["velocity_scale_rad_s"])
            if not np.array_equal(torque, expected) or np.any(torque * record.arrays["clean_qvel"] > 0):
                raise ValueError("mismatch force must match the registered dissipative torque")


def _validate_identity(name: str, metadata: Mapping[str, Any], scenario: str, seed: int) -> None:
    if type(metadata.get("trajectory_id")) is not int or metadata["trajectory_id"] != _IDS[name]:
        raise ValueError("scenario trajectory identity is invalid")
    split = "heldout" if name in _HELDOUT_IDS else "train"
    if metadata.get("split") != split or metadata.get("scenario") != scenario:
        raise ValueError("scenario split identity is invalid")
    for key in ("timestep_s", "duration_s", "prehistory_s", "sensor_delay_s"):
        value = metadata.get(key)
        if (isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value)
                or value < 0 or (key != "sensor_delay_s" and value == 0)):
            raise ValueError(f"scenario metadata has invalid {key}")
    noise = metadata.get("noise")
    if not isinstance(noise, dict) or set(noise) != {"qpos", "qvel"}:
        raise ValueError("scenario noise must describe both sensors")
    for sensor_id, sensor in enumerate(("qpos", "qvel")):
        descriptor = noise[sensor]
        if (not isinstance(descriptor, dict) or set(descriptor) != {"sigma", "seed_sequence_entropy", "innovation_sha256"}
                or descriptor["seed_sequence_entropy"] != [seed, 1 if name == "weak_step" else 0, _IDS[name], sensor_id]
                or not _valid_sha256(descriptor["innovation_sha256"])
                or not isinstance(descriptor["sigma"], (float, int)) or isinstance(descriptor["sigma"], bool)
                or not np.isfinite(descriptor["sigma"]) or descriptor["sigma"] <= 0):
            raise ValueError("scenario noise descriptor/seed identity is invalid")


def write_scenario(output: Path, data: ScenarioData, provenance: Mapping[str, Any]) -> Path:
    """Persist a new scenario manifest and hashed NPZs; never reuse a directory."""
    validate_scenario(data)
    from mujoco_sysid_lab.dataset import _validate_provenance
    provenance_record = _validate_provenance(provenance, data)
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise ValueError("Scenario output already exists; preserve evidence")
    target.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name in _record_names(data.scenario):
        record = data.records[name]
        file = target / f"{name}.npz"
        np.savez_compressed(file, **record.arrays)
        entries[name] = {"file": file.name, "file_sha256": sha256_file(file),
                         "metadata": _snapshot(record.metadata),
                         "arrays": {key: {"shape": list(array.shape), "dtype": array.dtype.str,
                                          "sha256": array_sha256(array)} for key, array in sorted(record.arrays.items())}}
    manifest = {"scenario_schema_version": data.scenario_schema_version, "scenario": data.scenario,
                "seed": data.seed, "metadata": _snapshot(data.metadata),
                "provenance": provenance_record, "trajectories": entries}
    path = target / "manifest.json"
    write_json(path, manifest)
    return path


def _read_manifest(path: Path) -> tuple[Path, dict]:
    file = Path(path) / "manifest.json" if Path(path).is_dir() else Path(path)
    if file.is_symlink():
        raise ValueError("scenario manifest must be a regular file")
    with file.open(encoding="utf-8") as stream:
        raw = json.load(stream, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    raw = _strict_json_object(raw, "scenario manifest")
    if set(raw) != {"scenario_schema_version", "scenario", "seed", "metadata", "provenance", "trajectories"}:
        raise ValueError("scenario manifest fields do not match schema")
    if type(raw["scenario_schema_version"]) is not int or raw["scenario_schema_version"] != SCENARIO_SCHEMA_VERSION:
        raise ValueError("unsupported scenario schema")
    if raw["scenario"] not in SCENARIOS or type(raw["seed"]) is not int or raw["seed"] < 0:
        raise ValueError("invalid scenario or seed")
    entries = raw["trajectories"]
    if not isinstance(entries, dict) or set(entries) != set(_record_names(raw["scenario"])):
        raise ValueError("scenario trajectory split is invalid")
    for name, entry in entries.items():
        if (not isinstance(entry, dict) or set(entry) != {"file", "file_sha256", "metadata", "arrays"}
                or entry["file"] != name + ".npz" or not _valid_sha256(entry["file_sha256"])
                or not isinstance(entry["arrays"], dict) or set(entry["arrays"]) != ARRAY_NAMES):
            raise ValueError("invalid scenario trajectory file/array manifest")
        _validate_identity(name, _strict_json_object(entry["metadata"], "trajectory metadata"), raw["scenario"], raw["seed"])
        for descriptor in entry["arrays"].values():
            if (not isinstance(descriptor, dict) or set(descriptor) != {"shape", "dtype", "sha256"}
                    or descriptor["dtype"] != np.dtype(np.float64).str
                    or not isinstance(descriptor["shape"], list)
                    or any(type(size) is not int or size < 1 for size in descriptor["shape"])
                    or not _valid_sha256(descriptor["sha256"])):
                raise ValueError("invalid scenario array descriptor")
    return file.parent, raw


def _read_arrays(directory: Path, entry: dict, *, training_only: bool) -> Mapping[str, np.ndarray]:
    file = directory / entry["file"]
    if file.is_symlink() or not file.is_file() or sha256_file(file) != entry["file_sha256"]:
        raise ValueError("scenario NPZ file missing, unsafe, or hash mismatch")
    names = TRAIN_ARRAY_NAMES if training_only else sorted(ARRAY_NAMES)
    arrays = {}
    with np.load(file, allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_NAMES or len(archive.files) != len(ARRAY_NAMES):
            raise ValueError("scenario NPZ array keys do not match schema")
        for name in names:
            array = archive[name]
            descriptor = entry["arrays"][name]
            if (list(array.shape) != descriptor["shape"] or array.dtype.str != descriptor["dtype"]
                    or array_sha256(array) != descriptor["sha256"]):
                raise ValueError("scenario array shape/dtype/hash mismatch")
            arrays[name] = _owned_readonly(array)
    _validate_arrays(arrays, entry["metadata"], training_only=training_only)
    return MappingProxyType(arrays)


def load_scenario(path: Path) -> ScenarioData:
    """Full truth/evaluation reader: verify every scenario file and array hash."""
    directory, raw = _read_manifest(path)
    records = {name: TrajectoryRecord(_read_arrays(directory, entry, training_only=False),
                                     _freeze(entry["metadata"]))
               for name, entry in raw["trajectories"].items()}
    data = ScenarioData(raw["scenario"], raw["seed"], MappingProxyType(records), _freeze(raw["metadata"]))
    validate_scenario(data)
    return data


def load_scenario_training(path: Path) -> TrainingDataset:
    """Load only whitelisted training files/arrays, never latent or held-out arrays."""
    directory, raw = _read_manifest(path)
    names = ("weak_step",) if raw["scenario"] == "weak" else CORE_TRAJECTORIES[:2]
    sequences = tuple(TrainingSequence(name, _IDS[name], **_read_arrays(
        directory, raw["trajectories"][name], training_only=True)) for name in names)
    return TrainingDataset(raw["seed"], sequences)
